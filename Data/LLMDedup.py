"""LLM 实体去重与消歧：候选簇构建 + 裁决落地 + 存疑文档。

确定性规则（字符串相等 / 字符串相似 / 余弦）只负责召回，合并与否交给 LLM；
代码只负责把裁决安全落地：结构级联复用 `Deduplicator.merge_entity`，
文本由 LLM 重写，代码只做「别名不丢、名字不撞车」的兜底。
"""

import hashlib
import json
import os
from datetime import datetime
from time import time

from rapidfuzz.distance import JaroWinkler, Levenshtein

from MomoGraph.Core.DataSave import TEMP_DIR
from MomoGraph.Core.Embedding import EmbeddingClient, cosine, update_embeddings
from MomoGraph.Core.LLM import LLM_Adapter
from MomoGraph.Data.DataStructure import Entity, Graph
from MomoGraph.Data.Dedup import ENTITY_SIM_THRESHOLD, Deduplicator

# 灰度带：余弦落在这里只召回、不判定，交 LLM；≥ ENTITY_SIM_THRESHOLD 才由代码直合
GRAY_LOW = 0.7
GRAY_HIGH = ENTITY_SIM_THRESHOLD
# 单次 LLM 调用最多带多少个实体；簇不跨批拆开
BATCH_MAX_ENTITIES = 20
DOUBT_PATH = os.path.join(TEMP_DIR, "entity_doubts.json")

# 召回用的名字归一：书名号、间隔号、括号这类排版符号不参与比较（判断仍交 LLM）
RECALL_DROP_CHARS = "《》〈〉「」『』“”‘’\"'()（）[]【】{}<>·•,，.。:：;；!！?？-—_~～/\\|*#…"
# 名字包含召回：太短的串容易误召回（V ⊂ Vedal、魂 ⊂ 灵魂），另要求较短串占较长串的一半以上
RECALL_CONTAIN_MIN_LEN = 2
RECALL_CONTAIN_MIN_RATIO = 0.5
# 拼写级相似召回：JW 认姓名拼写变化、Levenshtein 认少量错字、n-gram 认局部重叠
RECALL_SIM_MIN_LEN = 3
RECALL_JARO_WINKLER = 0.88
RECALL_LEVENSHTEIN = 0.8
RECALL_NGRAM_DICE = 0.7


def recall_keys(index) -> set[str]:
    """名字/别名去掉排版符号后的召回键：只用于找候选，不作为合并判据。"""
    keys = set()
    for name in [index.name, *index.relate_name]:
        if not name:
            continue
        cleaned = name.translate(str.maketrans("", "", RECALL_DROP_CHARS))
        key = Deduplicator.normalize(cleaned)
        if key:
            keys.add(key)
    return keys


def ngram_dice(left: str, right: str, size: int = 2) -> float:
    """字符 n-gram 的 Dice 系数：名字短于 size 时退化成单字。"""
    if not left or not right:
        return 0.0
    size = min(size, len(left), len(right))
    left_grams = {left[start:start + size] for start in range(len(left) - size + 1)}
    right_grams = {right[start:start + size] for start in range(len(right) - size + 1)}
    if not left_grams or not right_grams:
        return 0.0
    return 2 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


class DoubtStore:
    """存疑文档：记录没裁决完的簇、人工补答，以及安全闸的审计流水。"""

    def __init__(self, path: str = DOUBT_PATH):
        self.path = path
        self._data = self._load()

    @staticmethod
    def key_of(member_ids) -> str:
        return "|".join(str(i) for i in sorted({int(i) for i in member_ids}))

    def _load(self) -> dict:
        empty = {"doubts": [], "decided": {}, "audit": []}
        if not os.path.exists(self.path):
            return empty
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(data, dict):
            return empty
        # 旧文件可能缺字段，补齐即可，不做迁移
        data.setdefault("doubts", [])
        data.setdefault("decided", {})
        data.setdefault("audit", [])
        return data

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def _find(self, key: str) -> dict | None:
        return next((r for r in self._data["doubts"] if r.get("key") == key), None)

    def upsert(self, member_ids, question: str) -> dict:
        """同一成员集合（id 排序后作为 key）已有记录就更新，不重复追加。"""
        key = self.key_of(member_ids)
        now = int(time())
        members = [int(i) for i in sorted({int(i) for i in member_ids})]
        record = self._find(key)
        if record is None:
            # 成员被合并后 id 会变：沿用包含旧成员的那条形疑记录，避免同一问题记两遍
            record = next(
                (
                    r
                    for r in self._data["doubts"]
                    if r.get("status") == "open"
                    and set(r.get("members") or []) <= set(members)
                ),
                None,
            )
        if record is None:
            record = {
                "id": len(self._data["doubts"]) + 1,
                "key": key,
                "members": members,
                "question": question,
                "status": "open",
                "answer": "",
                "create_time": now,
                "update_time": now,
            }
            self._data["doubts"].append(record)
        else:
            record["key"] = key
            record["members"] = members
            if question:
                record["question"] = question
            record["status"] = "open"
            record["update_time"] = now
        return record

    def record_answer(self, key_or_id, answer: str) -> dict | None:
        """人工/外部补答入口：写进记录并重新置为 open，下次裁决当证据注入。"""
        record = self._find(str(key_or_id))
        if record is None:
            try:
                target = int(key_or_id)
            except (TypeError, ValueError):
                return None
            record = next((r for r in self._data["doubts"] if r.get("id") == target), None)
        if record is None:
            return None
        record["answer"] = answer
        record["status"] = "open"
        record["update_time"] = int(time())
        self.save()
        return record

    def open_records(self) -> list[dict]:
        return [record for record in self._data["doubts"] if record.get("status") == "open"]

    def has_answer(self, member_ids) -> bool:
        current = {int(i) for i in member_ids}
        return any(
            record.get("answer") and set(record.get("members") or []) <= current
            for record in self.open_records()
        )

    def resolve(self, member_ids) -> None:
        """裁决成功：把成员集合被当前簇覆盖的存疑记录全部置 resolved。"""
        current = {int(i) for i in member_ids}
        now = int(time())
        for record in self._data["doubts"]:
            if record.get("status") == "open" and set(record.get("members") or []) <= current:
                record["status"] = "resolved"
                record["update_time"] = now

    def append_audit(self, member_ids, text: str) -> None:
        self._data["audit"].append(
            {"key": self.key_of(member_ids), "text": text, "create_time": int(time())}
        )
        # 审计只用于回溯，不需要无限长；保留最近 200 条
        if len(self._data["audit"]) > 200:
            self._data["audit"] = self._data["audit"][-200:]

    def is_decided(self, fingerprint: str) -> bool:
        return fingerprint in self._data["decided"]

    def mark_decided(self, fingerprint: str, action: str) -> None:
        self._data["decided"][fingerprint] = {"action": action, "create_time": int(time())}


def doubts_to_string(graph: Graph, doubts: DoubtStore | None = None) -> str:
    """把待确认的存疑记录连同实体当前信息渲染成中文，供人工预览。"""
    doubts = doubts or DoubtStore()
    records = doubts.open_records()
    if not records:
        return "当前没有待确认的存疑实体。"

    def shorten(text: str, limit: int = 90) -> str:
        return text if len(text) <= limit else text[:limit] + "…"

    lines = [f"存疑实体：{len(records)} 条待确认", "-" * 34]
    for record in records:
        updated = datetime.fromtimestamp(record.get("update_time", 0)).strftime("%Y-%m-%d %H:%M")
        lines.append(f"#{record.get('id')} 更新于 {updated}")
        lines.append(f"  问题：{record.get('question') or '（LLM 未写问题）'}")
        for entity_id in record.get("members") or []:
            entity = graph.entities.get(entity_id)
            if entity is None:
                lines.append(f"  - {entity_id}：（已被合并或删除）")
                continue
            type_text = f"（{entity.entity_type}）" if entity.entity_type else ""
            aliases = sorted(
                alias for alias in entity.index.relate_name
                if alias and alias != entity.index.name
            )
            alias_text = f" 别名：{'、'.join(aliases)}" if aliases else ""
            lines.append(f"  - {entity_id}:{entity.index.name}{type_text}{alias_text}")
            if entity.description:
                lines.append(f"    描述：{shorten(entity.description)}")
        lines.append(f"  补答：{record.get('answer') or '无'}")
    return "\n".join(lines)


class LLMDisambiguator:
    """把候选实体簇交给 LLM 裁决，并按裁决安全落地。"""

    def __init__(
        self,
        graph: Graph,
        llm: LLM_Adapter,
        client: EmbeddingClient | None = None,
        doubts: DoubtStore | None = None,
    ):
        self.graph = graph
        self.llm = llm
        self.client = client or EmbeddingClient()
        self.deduplicator = Deduplicator(graph, self.client)
        self.doubts = doubts or DoubtStore()

    def run(self, force: bool = False) -> dict:
        report = {
            "enabled": True,
            "clusters": 0,
            "merged": 0,
            "distinguished": 0,
            "uncertain": 0,
            "skipped": 0,
            "llm_calls": 0,
        }
        if not self.graph.entities:
            return report

        update_embeddings(self.graph, self.client)
        clusters = self._build_clusters()
        report["clusters"] = len(clusters)
        print(f"[LLMDedup] 候选簇 {len(clusters)} 个：")
        for cluster in clusters:
            members = "、".join(f"{entity['id']}:{entity['name']}" for entity in cluster["entities"])
            print(f"  - 簇#{cluster['id']}: {members}")

        fingerprints: dict[int, str] = {}
        pending: list[dict] = []
        for cluster in clusters:
            member_ids = [entity["id"] for entity in cluster["entities"]]
            fingerprint = self._fingerprint(cluster)
            fingerprints[cluster["id"]] = fingerprint
            # 指纹没变且没有新增人工答案：直接跳过，避免反复调 LLM
            if (
                not force
                and self.doubts.is_decided(fingerprint)
                and not self.doubts.has_answer(member_ids)
            ):
                report["skipped"] += 1
                continue
            pending.append(cluster)

        print(f"[LLMDedup] 本轮回合：待裁决 {len(pending)} 个簇，指纹跳过 {report['skipped']} 个")
        if not pending:
            self.doubts.save()
            return report

        report["llm_calls"] = len(LLM_Adapter.batch_clusters(pending, BATCH_MAX_ENTITIES))
        print(f"[LLMDedup] 调用 LLM {report['llm_calls']} 次（单批实体上限 {BATCH_MAX_ENTITIES}）")
        decisions = self.llm.judge_entity_clusters(
            pending, self.doubts.open_records(), batch_max=BATCH_MAX_ENTITIES
        )
        print(f"[LLMDedup] LLM 返回 {len(decisions)} 条裁决")
        by_id = {cluster["id"]: cluster for cluster in pending}
        processed: set[int] = set()
        for decision in decisions:
            cluster_id = decision.get("cluster_id")
            cluster = by_id.get(cluster_id)
            if cluster is None:
                print(f"[LLMDedup] 忽略越界裁决：cluster_id={cluster_id}")
                continue
            if cluster_id in processed:
                print(f"[LLMDedup] 忽略重复裁决：cluster_id={cluster_id}")
                continue
            processed.add(cluster_id)

            # 上一簇的合并可能让本簇成员消失，落地前重新确认还活着
            alive_ids = [
                entity["id"] for entity in cluster["entities"] if entity["id"] in self.graph.entities
            ]
            members = "、".join(
                f"{entity['id']}:{entity['name']}"
                for entity in cluster["entities"]
                if entity["id"] in self.graph.entities
            )
            groups = decision.get("groups") or []
            doubts = decision.get("doubts") or []
            print(
                f"[LLMDedup] 裁决 簇#{cluster_id} → "
                f"{len(groups)} 个结果组、{len(doubts)} 条存疑：{members}"
            )
            if decision.get("reason"):
                print(f"           依据：{decision['reason']}")

            result = self._apply_groups(groups, doubts, alive_ids)
            if not result["applied"]:
                report["uncertain"] += 1
                print("           未落地：结果组没有完整覆盖本簇，转存疑")
                fallback = alive_ids or [entity["id"] for entity in cluster["entities"]]
                self.doubts.upsert(fallback, "LLM 没有给出完整的分组，请确认这簇实体的关系。")
                self.doubts.mark_decided(fingerprints[cluster_id], "uncertain")
                continue

            report["merged"] += result["merged"]
            report["distinguished"] += result["distinguished"]
            report["uncertain"] += result["uncertain"]
            if result["uncertain"]:
                action = "uncertain"
            elif result["merged"]:
                action = "merge"
            else:
                action = "distinguish"
            self.doubts.mark_decided(self._post_fingerprint(cluster), action)

        self.doubts.save()
        return report

    # ---- 候选簇 ----

    def _build_clusters(self) -> list[dict]:
        """用并查集把「可能同一」的实体连成簇：字符串相等，或余弦落在 [0.7, 0.9)。"""
        entities = [entity for entity in self.graph.entities.values() if entity.index.unique_id > 0]
        parent = {entity.index.unique_id: entity.index.unique_id for entity in entities}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for index, left in enumerate(entities):
            for right in entities[index + 1:]:
                reason = self._related_reason(left, right)
                if reason:
                    print(
                        f"[LLMDedup] 召回 {left.index.unique_id}:{left.index.name} "
                        f"↔ {right.index.unique_id}:{right.index.name}（{reason}）"
                    )
                    union(left.index.unique_id, right.index.unique_id)

        groups: dict[int, list[Entity]] = {}
        for entity in entities:
            groups.setdefault(find(entity.index.unique_id), []).append(entity)

        clusters = []
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda entity: entity.index.unique_id)
            clusters.append({"entities": [self._entity_payload(entity) for entity in members]})
        # 排序后分配 id，保证同一张图每次跑出来的簇 id 一致
        clusters.sort(key=lambda cluster: cluster["entities"][0]["id"])
        for index, cluster in enumerate(clusters):
            cluster["id"] = index
        return clusters

    def _related(self, left: Entity, right: Entity) -> bool:
        return self._related_reason(left, right) is not None

    def _related_reason(self, left: Entity, right: Entity) -> str | None:
        """返回这一对实体被召回的原因（不相关返回 None），同时用于打印。"""
        # 字符串相等始终进候选，类型冲突也拦不住（作品/角色这种必须交 LLM 区分）
        if Deduplicator.string_match(left.index, right.index):
            return "名字/别名字符串相等"
        # 书名号、间隔号、括号剥掉后再比：「《哈利·波特》」=「哈利波特」、「邓布利多」⊂「阿不思·邓布利多」
        overlap = self._name_overlap_reason(left, right)
        if overlap:
            return overlap
        # 拼写级相似：拼错的字、缩写、局部重叠
        spell = self._name_similarity_reason(left, right)
        if spell:
            return spell
        # 余弦只召回灰度带；类型冲突或画像不全都不进
        if not self.deduplicator._entities_compatible(left, right):
            return None
        left_vector = self.graph.embedding_dict_for_entity.get(left.index.unique_id)
        right_vector = self.graph.embedding_dict_for_entity.get(right.index.unique_id)
        if not left_vector or not right_vector:
            return None
        similarity = cosine(left_vector, right_vector)
        if GRAY_LOW <= similarity < GRAY_HIGH:
            return f"画像余弦 {similarity:.2f}（灰度带）"
        return None

    @staticmethod
    def _name_overlap_reason(left: Entity, right: Entity) -> str | None:
        """名字/别名（去排版符号后）相等或包含，就召回；包含只用于召回，不判合并。"""
        left_keys = recall_keys(left.index)
        right_keys = recall_keys(right.index)
        for left_key in left_keys:
            for right_key in right_keys:
                if left_key == right_key:
                    return f"名字去符号后相等（{left_key}）"
                short, long = sorted((left_key, right_key), key=len)
                if (
                    len(short) >= RECALL_CONTAIN_MIN_LEN
                    and short in long
                    and len(short) / len(long) >= RECALL_CONTAIN_MIN_RATIO
                ):
                    return f"名字包含（{short} ⊂ {long}）"
        return None

    @staticmethod
    def _name_similarity_reason(left: Entity, right: Entity) -> str | None:
        """拼写级相似召回：JW 认姓名变体、Levenshtein 认少量错字、n-gram 认局部重叠。"""
        left_keys = recall_keys(left.index)
        right_keys = recall_keys(right.index)
        for left_key in left_keys:
            for right_key in right_keys:
                if len(left_key) < RECALL_SIM_MIN_LEN or len(right_key) < RECALL_SIM_MIN_LEN:
                    continue
                # 顺序按「最像名字」到「最像局部重叠」：JW 对前缀敏感，出错字也能认出来
                jaro = JaroWinkler.similarity(left_key, right_key)
                if jaro >= RECALL_JARO_WINKLER:
                    return f"Jaro-Winkler {jaro:.2f}（{left_key} / {right_key}）"
                edit = Levenshtein.normalized_similarity(left_key, right_key)
                if edit >= RECALL_LEVENSHTEIN:
                    return f"Levenshtein {edit:.2f}（{left_key} / {right_key}）"
                dice = ngram_dice(left_key, right_key)
                if dice >= RECALL_NGRAM_DICE:
                    return f"n-gram {dice:.2f}（{left_key} / {right_key}）"
        return None

    @staticmethod
    def _entity_payload(entity: Entity) -> dict:
        return {
            "id": entity.index.unique_id,
            "name": entity.index.name,
            "aliases": sorted(
                alias for alias in entity.index.relate_name if alias and alias != entity.index.name
            ),
            "entity_type": entity.entity_type,
            "description": entity.description,
        }

    @staticmethod
    def _fingerprint(cluster: dict) -> str:
        payload = json.dumps(cluster["entities"], ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _post_fingerprint(self, cluster: dict) -> str:
        """落地后重新算指纹：下次冥想时若簇还是这个样子，直接跳过。"""
        alive = [
            self._entity_payload(self.graph.entities[entity["id"]])
            for entity in cluster["entities"]
            if entity["id"] in self.graph.entities
        ]
        alive.sort(key=lambda item: item["id"])
        return self._fingerprint({"entities": alive})

    # ---- 裁决落地 ----

    def _apply_groups(
        self,
        groups: list[dict],
        doubts: list[dict],
        alive_ids: list[int],
    ) -> dict:
        """按结果实体组落地：先校验覆盖，再合并结构，最后统一改写画像并写存疑。"""
        failed = {"applied": False, "merged": 0, "distinguished": 0, "uncertain": 0}
        if not alive_ids:
            return failed

        # 1) 每个存活实体必须恰好出现在一个组里，且主实体必须是组成员
        normalized = []
        covered: set[int] = set()
        primaries: set[int] = set()
        for group in groups:
            member_ids = list(
                dict.fromkeys(
                    entity_id
                    for entity_id in (group.get("member_ids") or [])
                    if entity_id in alive_ids
                )
            )
            if not member_ids:
                print("           结果组没有有效成员，放弃本簇裁决")
                return failed
            primary_id = group.get("primary_id", 0)
            if primary_id not in member_ids:
                print(f"           主实体 {primary_id} 不在组成员 {member_ids} 里，放弃本簇裁决")
                return failed
            if primary_id in primaries or set(member_ids) & covered:
                print("           结果组之间出现重复成员，放弃本簇裁决")
                return failed
            primaries.add(primary_id)
            covered.update(member_ids)
            normalized.append(
                {
                    "primary_id": primary_id,
                    "member_ids": member_ids,
                    "final": group.get("final") or {},
                }
            )
        missing = sorted(set(alive_ids) - covered)
        if missing:
            names = "、".join(
                f"{entity_id}:{self.graph.entities[entity_id].index.name}"
                for entity_id in missing
                if entity_id in self.graph.entities
            )
            print(f"           结果组漏掉实体：{names}，放弃本簇裁决")
            return failed

        # 2) 存疑成员一律换算成组的主实体 id，避免合并后 id 丢失
        id_to_primary = {
            entity_id: group["primary_id"]
            for group in normalized
            for entity_id in group["member_ids"]
        }
        normalized_doubts = []
        uncertain_ids: set[int] = set()
        for doubt in doubts:
            members = []
            for entity_id in doubt.get("members") or []:
                primary_id = id_to_primary.get(entity_id)
                if primary_id is not None and primary_id not in members:
                    members.append(primary_id)
            if len(members) < 2:
                print(f"           存疑成员不足两个，忽略：{doubt.get('members')}")
                continue
            normalized_doubts.append(
                {
                    "members": members,
                    "question": doubt.get("question") or "这些实体是否为同一实体？",
                }
            )
            uncertain_ids.update(members)

        # 之前的存疑记录按本轮簇成员统一收口；之后再写新记录的 primary_id
        self.doubts.resolve(alive_ids)

        records = []
        for group in normalized:
            primary_id = group["primary_id"]
            old_texts = set()
            for entity_id in group["member_ids"]:
                entity = self.graph.entities.get(entity_id)
                if entity is None:
                    continue
                old_texts.add(entity.index.name)
                old_texts.update(entity.index.relate_name)
            records.append(
                {
                    "primary_id": primary_id,
                    "members": group["member_ids"],
                    "final": group["final"],
                    "rewrite": primary_id not in uncertain_ids,
                    "old_texts": old_texts,
                }
            )

        # 3) 先做结构合并；存疑组只合结构，不改画像
        for record in records:
            if len(record["members"]) < 2:
                continue
            keep = self.graph.entities[record["primary_id"]]
            drops = [
                self.graph.entities[entity_id]
                for entity_id in record["members"]
                if entity_id != record["primary_id"] and entity_id in self.graph.entities
            ]
            joined = "、".join(f"{drop.index.unique_id}:{drop.index.name}" for drop in drops)
            print(f"[LLMDedup] 合并：保留 {keep.index.unique_id}:{keep.index.name}，并入 {joined}")
            for drop in drops:
                print(
                    f"           级联 {drop.index.unique_id}:{drop.index.name}："
                    f"出边 {len(drop.out_edges)}、入边 {len(drop.in_edges)}、"
                    f"属性 {len(drop.attributes)}、chunk {len(drop.index.relate_memory_chunk_id)}"
                )
                self.deduplicator.merge_entity(keep, drop)

        # 4) 算每个结果实体的新画像
        proposed: dict[int, dict] = {}
        for record in records:
            entity = self.graph.entities.get(record["primary_id"])
            if entity is None:
                continue
            final = record["final"]
            if record["rewrite"]:
                name = (final.get("name") or "").strip() or entity.index.name
                aliases = [
                    alias.strip()
                    for alias in (final.get("aliases") or [])
                    if alias.strip()
                ]
                description = (final.get("description") or "").strip() or entity.description
                entity_type = (final.get("entity_type") or "").strip() or entity.entity_type
            else:
                name = entity.index.name
                aliases = sorted(entity.index.relate_name)
                description = entity.description
                entity_type = entity.entity_type
            proposed[record["primary_id"]] = {
                "name": name,
                "aliases": [alias for alias in dict.fromkeys(aliases) if alias and alias != name],
                "description": description,
                "entity_type": entity_type,
            }

        # 5) 名字安全闸：簇内现有名字先全部锁住，簇外重名保留原名
        locked_names = {
            Deduplicator.normalize(self.graph.entities[record["primary_id"]].index.name): record["primary_id"]
            for record in records
            if record["primary_id"] in self.graph.entities
        }
        for record in records:
            if not record["rewrite"]:
                continue
            entity_id = record["primary_id"]
            current = self.graph.entities[entity_id]
            item = proposed[entity_id]
            if item["name"] == current.index.name:
                continue
            blocked = self._name_conflict(item["name"], entity_id, set(alive_ids))
            locked_key = Deduplicator.normalize(item["name"])
            locked_by = locked_names.get(locked_key)
            if blocked is None and locked_by is not None and locked_by != entity_id:
                blocked = self.graph.entities[locked_by]
            if blocked is None:
                continue
            print(
                f"           名字冲突：「{item['name']}」已被 "
                f"{blocked.index.unique_id}:{blocked.index.name} 占用，保留原名「{current.index.name}」"
            )
            self.doubts.append_audit(
                alive_ids,
                f"分组时「{item['name']}」与「{blocked.index.name}」(id={blocked.index.unique_id}) 重名，保留原名「{current.index.name}」",
            )
            item["name"] = current.index.name
            # 撞车的是原目标写法，别把它留在别名里
            item["aliases"] = [
                alias for alias in item["aliases"] if Deduplicator.normalize(alias) != locked_key
            ]

        # 组与组之间也不能撞名；后者保留原名
        seen_names: dict[str, int] = {}
        for record in records:
            entity_id = record["primary_id"]
            item = proposed.get(entity_id)
            if item is None:
                continue
            key = Deduplicator.normalize(item["name"])
            owner = seen_names.get(key)
            if owner is not None and owner != entity_id:
                current = self.graph.entities[entity_id]
                print(
                    f"           组间名字冲突：「{item['name']}」同时被 {owner} 和 "
                    f"{entity_id} 使用，{entity_id} 保留原名「{current.index.name}」"
                )
                self.doubts.append_audit(
                    alive_ids,
                    f"分组时组间名字冲突「{item['name']}」，{entity_id} 保留原名「{current.index.name}」",
                )
                item["name"] = current.index.name
                item["aliases"] = [
                    alias for alias in item["aliases"] if Deduplicator.normalize(alias) != key
                ]
                key = Deduplicator.normalize(item["name"])
            seen_names[key] = entity_id

        # 6) 别名安全闸：旧叫法被删且没迁移到同簇其它组，就保留
        owners: dict[str, set[int]] = {}
        for entity_id, item in proposed.items():
            for text in [item["name"], *item["aliases"]]:
                if text:
                    owners.setdefault(Deduplicator.normalize(text), set()).add(entity_id)

        for record in records:
            if not record["rewrite"]:
                continue
            entity_id = record["primary_id"]
            item = proposed[entity_id]
            own = {Deduplicator.normalize(item["name"])} | {
                Deduplicator.normalize(alias) for alias in item["aliases"]
            }
            kept, migrated = [], []
            for text in sorted(record["old_texts"]):
                key = Deduplicator.normalize(text) if text else ""
                if not key or key in own:
                    continue
                others = owners.get(key, set()) - {entity_id}
                if others:
                    migrated.append(text)
                else:
                    kept.append(text)
                    item["aliases"].append(text)
                    owners.setdefault(key, set()).add(entity_id)
            if kept:
                print(f"           安全闸：{entity_id} 的旧叫法被删，已保留：{'、'.join(kept)}")
                self.doubts.append_audit(
                    alive_ids, f"分组时保留 LLM 漏写的旧叫法：{'、'.join(kept)}"
                )
            if migrated:
                print(f"           迁移：{entity_id} 的旧叫法交给同簇其它组：{'、'.join(migrated)}")
                self.doubts.append_audit(
                    alive_ids, f"旧叫法已迁移到同簇其它组：{'、'.join(migrated)}"
                )

        # 7) 写回画像
        for record in records:
            entity_id = record["primary_id"]
            entity = self.graph.entities.get(entity_id)
            item = proposed.get(entity_id)
            if entity is None or item is None:
                continue
            before = (
                entity.index.name,
                sorted(entity.index.relate_name),
                entity.description,
                entity.entity_type,
            )
            if not record["rewrite"]:
                print(f"           存疑：{entity_id}:{entity.index.name} 画像保持原样")
                continue
            aliases = [alias for alias in dict.fromkeys(item["aliases"]) if alias and alias != item["name"]]
            entity.index.name = item["name"]
            entity.index.relate_name = set(aliases)
            if item["description"]:
                entity.description = item["description"]
            if item["entity_type"]:
                entity.entity_type = item["entity_type"]
            after = (
                entity.index.name,
                sorted(entity.index.relate_name),
                entity.description,
                entity.entity_type,
            )
            if before == after:
                print(f"           id={entity_id}:{entity.index.name}：独立实体，字段未改")
                continue
            changes = []
            if before[0] != after[0]:
                changes.append(f"name {before[0]!r} → {after[0]!r}")
            if before[1] != after[1]:
                changes.append(f"aliases {before[1]} → {after[1]}")
            if before[3] != after[3]:
                changes.append(f"type {before[3]!r} → {after[3]!r}")
            if before[2] != after[2]:
                changes.append(f"desc {before[2]!r} → {after[2]!r}")
            print(f"           id={entity_id}：" + "；".join(changes))

        # 8) 写存疑：members 已是各组主实体，合并后不会丢 id
        for doubt in normalized_doubts:
            members_text = "、".join(
                f"{entity_id}:{self.graph.entities[entity_id].index.name}"
                for entity_id in doubt["members"]
                if entity_id in self.graph.entities
            )
            print(f"           存疑：{members_text} → {doubt['question']}")
            self.doubts.upsert(doubt["members"], doubt["question"])

        return {
            "applied": True,
            "merged": sum(1 for record in records if len(record["members"]) > 1),
            "distinguished": sum(1 for record in records if len(record["members"]) == 1),
            "uncertain": len(normalized_doubts),
        }

    def _name_conflict(self, name: str, entity_id: int, allowed_ids: set[int]) -> Entity | None:
        """新名字（归一化后）与簇外实体的名字/别名撞车就返回那个实体。"""
        target = Deduplicator.normalize(name)
        if not target:
            return None
        for other in self.graph.entities.values():
            if other.index.unique_id == entity_id or other.index.unique_id in allowed_ids:
                continue
            if Deduplicator.normalize(other.index.name) == target:
                return other
            if any(Deduplicator.normalize(alias) == target for alias in other.index.relate_name):
                return other
        return None
