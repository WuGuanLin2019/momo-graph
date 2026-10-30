"""把检索结果转写成给人/模型读的记忆文本。

不叫 TextNormalise：这里不做文本清洗，而是把 chains / chunks 渲染成
一段可以直接塞进 prompt 的上下文。

链条统一写成「路径标题 + 单向事实行 + 属性行」：事实行一律 A --关系→ B，
反向的边直接调换两端，不再用反向箭头，模型不用猜箭头朝哪边。
"""

from datetime import datetime


def format_time(timestamp: int) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def time_text(timestamp: int, tag: str = "") -> str:
    """括号里的时间后缀，例如「（2026-09-20，较新）」；没有时间就返回空串。"""
    text = format_time(timestamp)
    return f"（{text}{tag}）" if text else ""


def recency_tags(times: list[int]) -> list[str]:
    """同一组记录的时间先后标记。

    只有同组记录的时间确实分得出先后才标（第一条/最新的标「较新」，其余标「较早」）；
    全部相同时不标，避免瞎猜。
    """
    if len(times) < 2 or len(set(times)) < 2:
        return ["" for _ in times]
    newest = max(times)
    return ["，较新" if time_value == newest else "，较早" for time_value in times]


def format_chain(chain: dict, index: int) -> str:
    """一条链：路径标题 + 画像 + 事实行 + 属性行。"""
    nodes = chain.get("nodes") or []
    edges = chain.get("edges") or []
    names = {node["id"]: node["name"] for node in nodes}

    lines = [f"{index}. 路径：" + " / ".join(node["name"] for node in nodes)]

    portraits = [f"{node['name']} = {node['description']}" for node in nodes if node.get("description")]
    if portraits:
        lines.append("   · 画像：" + "；".join(portraits))

    for edge in edges:
        # 主边 + 同端点的其它边（喜欢 / 不喜欢这类摇摆）放在同一组，统一标先后
        records = [{"name": edge["name"], "time": edge.get("time", 0)}]
        records.extend(edge.get("siblings") or [])
        records.sort(key=lambda item: item["time"], reverse=True)
        tags = recency_tags([record["time"] for record in records])
        for record, tag in zip(records, tags):
            lines.append(
                f"   · {names.get(edge['source'], '?')} --{record['name']}→ "
                f"{names.get(edge['target'], '?')}{time_text(record['time'], tag)}"
            )

    for node in nodes:
        for name, items in (node.get("attributes") or {}).items():
            tags = recency_tags([item["time"] for item in items])
            values = " / ".join(
                item["value"] + time_text(item["time"], tag) for item, tag in zip(items, tags)
            )
            lines.append(f"   · 属性：{node['name']} {name} = {values}")

    return "\n".join(lines)


def build_prompt(query: str, result: dict) -> str:
    """拼出完整的用户侧 prompt：问题 + 相关社区 + 记忆链条 + 记忆原文。"""
    chains = result.get("chains") or []
    chunks = result.get("chunks") or []
    communities = result.get("communities") or []

    blocks = [f"记忆检索字段：{query}", ""]
    # 社区摘要是"整体在讲什么"，放在链条之前当背景，具体关系再由链条补
    if communities:
        blocks.append("【相关社区】")
        blocks.extend(
            f"{index}. {item.get('name') or '社区'}：{item.get('summary', '')}"
            for index, item in enumerate(communities, 1)
        )
        blocks.append("")
    if chains:
        blocks.append("【记忆链条】")
        blocks.extend(format_chain(chain, index) for index, chain in enumerate(chains, 1))
        blocks.append("")
    if chunks:
        blocks.append("【记忆原文】")
        blocks.extend(f"[#{chunk['id']}] {chunk['content']}" for chunk in chunks)
        blocks.append("")
    if not chains and not chunks and not communities:
        blocks.append("（没有检索到相关记忆）")

    blocks.append("请依据以上记忆回答，记不得就说记不得。")
    return "\n".join(blocks)
