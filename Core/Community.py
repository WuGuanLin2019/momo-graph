"""社区发现（多层级 Louvain）与社区摘要刷新。

结构每次全量重算：Louvain 是本地计算、毫秒级、零 LLM 成本。
摘要只在成员变化时更新，而且每次都 append 一条 Description——和实体/边的
「只增不减」哲学一致，旧摘要留作时间线。

社区本体复用 Party：entity_ids 是成员，description_record 是摘要历史。
"""

from concurrent.futures import ThreadPoolExecutor
from time import time

import networkx as nx

from MomoGraph.Core.Embedding import EmbeddingClient, cosine
from MomoGraph.Data.DataStructure import Description, GraphNodeType, Party

MAX_LEVEL = 2  # 最多三层：level 0 最细，level 2 最宏观
MIN_ENTITIES = 3  # 实体太少，划分没有意义
LOUVAIN_SEED = 42  # 固定种子，同一张图每次划分结果一致
MATCH_JACCARD = 0.5  # 新社区与旧 Party 匹配的最低相似度
REFRESH_EVERY_N_CHUNKS = 20  # 距上次刷新累计新增这么多 chunk，才考虑重写摘要
MAX_WORKERS = 4  # 同层社区没依赖，并发调 LLM；别开太大，容易被限流
SMALL_COMMUNITY = 2  # 两个成员以内的社区不调 LLM，按规则拼摘要

MAX_LEAF_EDGES = 40  # 叶子摘要看内部边，按权重取前 N 条
MAX_LEAF_CHUNKS = 5  # 叶子摘要只看最近 N 条原文
MAX_UPPER_EDGES = 20  # 上层摘要看子社区之间的边，取前 N 条

DISSOLVED_TEXT = "该社区已解散。"


# ---------- 结构：逐层粗化的 Louvain ----------

def _entity_graph(graph) -> nx.Graph:
    """实体当节点、边权当权重；同一对实体有多条边时权重累加。"""
    nx_graph = nx.Graph()
    for edge in graph.edges.values():
        source, target = edge.source_entity_id, edge.target_entity_id
        if source == target:
            continue
        if source not in graph.entities or target not in graph.entities:
            continue
        weight = edge.weight or 0.5
        if nx_graph.has_edge(source, target):
            nx_graph[source][target]["weight"] += weight
        else:
            nx_graph.add_edge(source, target, weight=weight)
    return nx_graph


def _sorted_members(communities) -> list[list[int]]:
    """社区内部按 id 排序、社区之间按首个成员排序，保证结果可复现。"""
    members = [sorted(community) for community in communities if community]
    members.sort(key=lambda ids: ids[0])
    return members


def _louvain(nx_graph) -> list[list[int]]:
    return _sorted_members(
        nx.community.louvain_communities(nx_graph, weight="weight", seed=LOUVAIN_SEED)
    )


def _contract(nx_graph, communities: list[list[int]]) -> nx.Graph:
    """把每个社区缩成一个节点：社区间边权取和，社区内部的边丢掉。"""
    node_of = {}
    for index, members in enumerate(communities):
        for entity_id in members:
            node_of[entity_id] = index

    contracted = nx.Graph()
    contracted.add_nodes_from(range(len(communities)))
    for source, target, data in nx_graph.edges(data=True):
        left, right = node_of.get(source), node_of.get(target)
        if left is None or right is None or left == right:
            continue
        weight = data.get("weight", 1.0)
        if contracted.has_edge(left, right):
            contracted[left][right]["weight"] += weight
        else:
            contracted.add_edge(left, right, weight=weight)
    return contracted


def detect_levels(graph) -> list[list[list[int]]]:
    """多层级划分：返回每一层的社区成员集合，level 0 最细。

    Louvain 原生只出一层，上层靠「社区缩成节点 → 再跑一次」得到。
    """
    nx_graph = _entity_graph(graph)
    # 孤立实体不进社区（Louvain 会把它们各自分一簇，白白多出一堆单点社区）
    working = nx_graph.subgraph(
        [node for node, degree in nx_graph.degree() if degree > 0]
    ).copy()
    if working.number_of_nodes() < MIN_ENTITIES:
        return []

    current = _louvain(working)
    levels = [current]
    while len(levels) <= MAX_LEVEL and len(current) > 1:
        coarser = _louvain(_contract(working, current))
        if len(coarser) >= len(current):
            break  # 粗化后没合并，再往上只是重复同一层
        current = _sorted_members(
            [set(entity_id for index in group for entity_id in current[index]) for group in coarser]
        )
        levels.append(current)
    return levels


# ---------- 摘要输入：把一簇记忆拼成可读文本 ----------

def _latest_attributes(entity) -> dict[str, str]:
    latest = {}
    for attribute in entity.attributes:
        items = [item for item in attribute.items if item.content]
        if not items:
            continue
        newest = max(items, key=lambda item: max(item.activate_time or [item.create_time]))
        latest[attribute.index.name] = newest.content
    return latest


def _leaf_context(graph, members: list[int]) -> str:
    lines = ["【成员】"]
    for entity_id in members:
        entity = graph.entities.get(entity_id)
        if entity is None:
            continue
        parts = [entity.index.name]
        if entity.description:
            parts.append(entity.description)
        attributes = _latest_attributes(entity)
        if attributes:
            parts.append("属性：" + "、".join(f"{name}={value}" for name, value in attributes.items()))
        lines.append("· " + "；".join(parts))

    member_set = set(members)
    internal = [
        edge
        for edge in graph.edges.values()
        if edge.source_entity_id in member_set and edge.target_entity_id in member_set
    ]
    internal.sort(key=lambda edge: edge.weight, reverse=True)
    if internal:
        lines.append("【内部关系】")
        for edge in internal[:MAX_LEAF_EDGES]:
            lines.append(
                f"· {graph.entities[edge.source_entity_id].index.name} --{edge.index.name}→ "
                f"{graph.entities[edge.target_entity_id].index.name}（权重 {round(edge.weight, 2)}）"
            )

    chunk_ids: list[int] = []
    for entity_id in members:
        entity = graph.entities.get(entity_id)
        if entity is not None:
            chunk_ids.extend(entity.index.relate_memory_chunk_id)
    chunks = [
        graph.memory_chunks[chunk_id]
        for chunk_id in dict.fromkeys(chunk_ids)
        if chunk_id in graph.memory_chunks
    ]
    chunks.sort(key=lambda chunk: chunk.create_time, reverse=True)
    if chunks:
        lines.append("【最近原文】")
        lines.extend("· " + chunk.content for chunk in chunks[:MAX_LEAF_CHUNKS])

    return "\n".join(lines)


def _upper_context(graph, members: list[int], children: list[dict]) -> str:
    lines = ["【子社区】"]
    for child in children:
        lines.append(f"· {child['name']}：{child['summary']}")

    child_of: dict[int, int] = {}
    for index, child in enumerate(children):
        for entity_id in child["entity_ids"]:
            child_of[entity_id] = index

    # 只看跨子社区的边：子社区内部的关系已经在各自的摘要里说过了
    cross = []
    member_set = set(members)
    for edge in graph.edges.values():
        left = child_of.get(edge.source_entity_id)
        right = child_of.get(edge.target_entity_id)
        if left is None or right is None or left == right:
            continue
        if edge.source_entity_id not in member_set or edge.target_entity_id not in member_set:
            continue
        cross.append(
            (
                edge.weight,
                f"· {children[left]['name']} 的 {graph.entities[edge.source_entity_id].index.name} "
                f"--{edge.index.name}→ {children[right]['name']} 的 "
                f"{graph.entities[edge.target_entity_id].index.name}",
            )
        )
    cross.sort(key=lambda pair: pair[0], reverse=True)
    if cross:
        lines.append("【子社区之间】")
        lines.extend(text for _, text in cross[:MAX_UPPER_EDGES])

    return "\n".join(lines)


def _community_context(graph, members, level, party_ids, levels) -> str:
    if level == 0:
        return _leaf_context(graph, members)

    member_set = set(members)
    children = []
    for child_members in levels[level - 1]:
        if not member_set.issuperset(child_members):
            continue
        party = _party_by_members(graph, party_ids, child_members)
        if party is None or not party.description_record:
            continue
        children.append(
            {
                "name": party.name,
                "summary": party.description_record[-1].content,
                "entity_ids": child_members,
            }
        )
    if not children:
        # 子社区一个都没匹配上（理论上不会发生），退回叶子写法保证还能出摘要
        return _leaf_context(graph, members)
    return _upper_context(graph, members, children)


def _party_by_members(graph, party_ids, members):
    party_id = party_ids.get(frozenset(members))
    return graph.parties.get(party_id) if party_id else None


# ---------- 摘要写入与刷新编排 ----------

def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _write_summary(graph, party_id: int, party, name: str, summary: str, members, now: int, client) -> None:
    party.name = name or party.name or "、".join(
        graph.entities[entity_id].index.name for entity_id in members[:2] if entity_id in graph.entities
    )
    party.entity_ids = sorted(members)
    party.description_record.append(Description(content=summary, create_time=now))
    vector = client.embed_one(f"{party.name}。{summary}".strip("。")) if summary else []
    party.latest_description_embedding = vector or []
    if vector:
        graph.embedding_dict_for_party[party_id] = vector


def _last_refresh_time(graph) -> int:
    """上一次刷新摘要的时间：取所有社区最后一条摘要时间里的最大值。"""
    last = 0
    for party in graph.parties.values():
        if party.description_record:
            last = max(last, party.description_record[-1].create_time)
    return last


def _chunks_since_last_refresh(graph) -> int:
    """距上次刷新的新增 chunk 数：上次刷新时间取各社区最后一条摘要的时间。"""
    last = _last_refresh_time(graph)
    return sum(1 for chunk in graph.memory_chunks.values() if chunk.create_time > last)


def _touched_entities(graph, since: int) -> set[int]:
    """最近新增的 chunk 碰到了哪些实体——只有它们所在的社区需要重写摘要。"""
    chunk_ids = {
        chunk.unique_id for chunk in graph.memory_chunks.values() if chunk.create_time > since
    }
    if not chunk_ids:
        return set()
    return {
        entity_id
        for entity_id, entity in graph.entities.items()
        if chunk_ids & entity.index.relate_memory_chunk_id
    }


def _small_summary(graph, members) -> tuple[str, str]:
    """两个成员以内的社区没什么可总结的，按名字和内部关系拼，省掉一次 LLM。"""
    names = [
        graph.entities[entity_id].index.name
        for entity_id in members
        if entity_id in graph.entities
    ]
    member_set = set(members)
    relations = [
        edge
        for edge in graph.edges.values()
        if edge.source_entity_id in member_set and edge.target_entity_id in member_set
    ]
    relations.sort(key=lambda edge: edge.weight, reverse=True)
    name = " 与 ".join(names) or "小社区"
    lines = [f"这一簇只有{len(names)}个实体：{'、'.join(names)}。"]
    if relations:
        lines.append(
            "内部关系："
            + "；".join(
                f"{graph.entities[edge.source_entity_id].index.name}"
                f"--{edge.index.name}→"
                f"{graph.entities[edge.target_entity_id].index.name}"
                for edge in relations[:3]
            )
            + "。"
        )
    return name, "".join(lines)


def _run_summary(llm, job) -> dict:
    """一个社区一次调用：已有摘要就增量改写，没有就全量生成。"""
    if job["old_summary"]:
        return llm.rewrite_community_summary(job["old_summary"], job["context"])
    return llm.summarize_community(job["context"], is_leaf=job["level"] == 0)


def refresh_communities(graph, llm, client: EmbeddingClient | None = None, force: bool = False) -> dict:
    """划分一次并刷新摘要，返回本次的统计。

    匹配按「同层、存活、成员 Jaccard 最高」：
    成员不变就跳过（不花 LLM）；部分重叠就增量改写；对不上就新建，旧 Party 置为解散。
    成本控制三条：同层并发调 LLM、只有关联到新 chunk 的社区才重写、两个成员以内不调 LLM。
    """
    report = {"parties": 0, "created": 0, "updated": 0, "skipped": 0, "dissolved": 0, "llm_calls": 0}
    levels = detect_levels(graph)
    if not levels:
        report["parties"] = sum(1 for party in graph.parties.values() if party.active)
        return report

    # 新增 chunk 攒够阈值时，只把「关联到新内容」的社区标脏；force 才是全体重写
    stale_entities: set[int] = set()
    if not force and _chunks_since_last_refresh(graph) >= REFRESH_EVERY_N_CHUNKS:
        stale_entities = _touched_entities(graph, _last_refresh_time(graph))

    client = client or EmbeddingClient()
    now = int(time())
    party_ids: dict[frozenset[int], int] = {}

    for level, communities in enumerate(levels):
        old_parties = {
            party_id: party
            for party_id, party in graph.parties.items()
            if party.level == level and party.active
        }
        used: set[int] = set()
        jobs: list[dict] = []

        for members in communities:
            member_set = set(members)
            best_id, best_score = 0, 0.0
            for party_id, party in old_parties.items():
                if party_id in used:
                    continue
                score = _jaccard(member_set, set(party.entity_ids))
                if score > best_score:
                    best_id, best_score = party_id, score

            if best_id and best_score >= MATCH_JACCARD:
                used.add(best_id)
                party = old_parties[best_id]
                party_ids[frozenset(member_set)] = best_id
                # 成员没变、也没碰到新内容：一个 token 都不花
                if (
                    not force
                    and member_set == set(party.entity_ids)
                    and not (member_set & stale_entities)
                ):
                    report["skipped"] += 1
                    continue
                jobs.append(
                    {
                        "party_id": best_id,
                        "party": party,
                        "members": member_set,
                        "level": level,
                        "old_summary": (
                            party.description_record[-1].content if party.description_record else ""
                        ),
                        "context": _community_context(graph, members, level, party_ids, levels),
                        "kind": "updated",
                    }
                )
                continue

            # 对不上任何旧社区：先分配 id（上层社区要靠子社区的 id 找摘要），摘要稍后再写
            party_id = graph._alloc_id(GraphNodeType.party)
            party = Party(level=level)
            graph.parties[party_id] = party
            party_ids[frozenset(member_set)] = party_id
            jobs.append(
                {
                    "party_id": party_id,
                    "party": party,
                    "members": member_set,
                    "level": level,
                    "old_summary": "",
                    "context": _community_context(graph, members, level, party_ids, levels),
                    "kind": "created",
                }
            )

        small_jobs = [job for job in jobs if len(job["members"]) <= SMALL_COMMUNITY]
        big_jobs = [job for job in jobs if len(job["members"]) > SMALL_COMMUNITY]

        for job in small_jobs:
            name, summary = _small_summary(graph, job["members"])
            if job["kind"] == "updated" and job["old_summary"] == summary:
                # 规则拼出来的文本没变，就不必往历史里再塞一条
                report["skipped"] += 1
                continue
            _write_summary(
                graph, job["party_id"], job["party"], name, summary, job["members"], now, client
            )
            report[job["kind"]] += 1

        if big_jobs:
            # 同层社区互不依赖，可以并发；上层要等下层摘要写完，所以并发只发生在层内
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(_run_summary, llm, job) for job in big_jobs]
                results = [future.result() for future in futures]
            report["llm_calls"] += len(big_jobs)
            for job, result in zip(big_jobs, results):
                _write_summary(
                    graph,
                    job["party_id"],
                    job["party"],
                    result.get("name", ""),
                    result.get("summary", ""),
                    job["members"],
                    now,
                    client,
                )
                report[job["kind"]] += 1

        # 这一层没匹配上的旧社区：保留但不显示，历史摘要还能查
        for party_id, party in old_parties.items():
            if party_id in used:
                continue
            party.active = False
            party.description_record.append(Description(content=DISSOLVED_TEXT, create_time=now))
            report["dissolved"] += 1

    report["parties"] = sum(1 for party in graph.parties.values() if party.active)
    return report


def search_communities(graph, query_vector: list[float], top_k: int = 3) -> list[dict]:
    """用 query 向量找最相关的社区摘要（只找存活的）。"""
    if not query_vector:
        return []
    scored = []
    for party_id, party in graph.parties.items():
        if not party.active or not party.description_record:
            continue
        vector = party.latest_description_embedding or graph.embedding_dict_for_party.get(party_id)
        if not vector:
            continue
        scored.append(
            {
                "id": party_id,
                "name": party.name,
                "level": party.level,
                "summary": party.description_record[-1].content,
                "score": round(cosine(query_vector, vector), 4),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
