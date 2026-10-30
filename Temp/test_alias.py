from MomoGraph.Data.DataHandle import DataHandler
from MomoGraph.Data.DataStructure import Graph
from MomoGraph.Data.Dedup import Deduplicator

graph = Graph()
chunk1 = graph.create_chunk("吴今天喝了奶茶。", "user", 1)
# 先有"吴"这个节点，模拟历史数据里名字写错的情况
old = graph.get_or_create_entity("吴", chunk1, "一个人类。", "人物")

# 后来输入："吴是Wu的别名"，同时提到 Wu 喜欢奶茶
chunk2 = graph.create_chunk("吴是 Wu 的别名，Wu 喜欢奶茶。", "user", 2)
handler = DataHandler()
handler._make_up_entity_and_edge(
    {
        "link_data": [
            {"chunk_index": 0, "entity_source": "Wu", "entity_target": "奶茶",
             "edge_str": "喜欢", "edge_kind": "态度",
             "attribute_name": "", "attribute_value": "", "attribute_entity": ""},
        ],
        "entities": [
            {"name": "Wu", "aliases": ["吴"], "entity_type": "人物", "description": "一个人类。"},
        ],
    },
    graph, "user", [chunk2],
)

print("合并前实体:", [(e.index.name, sorted(e.index.relate_name)) for e in graph.entities.values()])

dedup = Deduplicator(graph)
wu = graph.find_entity("Wu")
wu2 = [e for e in graph.entities.values() if e.index.name == "吴"][0]
print("吴 → Wu 合并判定:", dedup.compare_index_base(wu2.index, wu.index, "entity"))
print("Wu → 吴 合并判定:", dedup.compare_index_base(wu.index, wu2.index, "entity"))

dedup.merge_entity(wu, wu2)
print("合并后实体:", [(e.index.name, sorted(e.index.relate_name),
                     sorted(e.index.relate_memory_chunk_id)) for e in graph.entities.values()])
print("再次查 吴:", graph.find_entity("吴").index.name)

# 真实抽取：模型是否会把别名放进 aliases、而不是抽成边
from MomoGraph.Core.LLM import LLM_Adapter

print("\n真实抽取:", LLM_Adapter().preheat_memories(["吴是 Wu 的别名，两个都指同一个人。"]))
