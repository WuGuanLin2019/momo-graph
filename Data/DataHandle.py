from time import time

from MomoGraph.Core.LLM import LLM_Adapter
from MomoGraph.Core.MemoryExtractor import MemoryExtractor
from MomoGraph.Data.Dedup import Deduplicator
from MomoGraph.Data.DataStructure import (
    Graph,
    MemoryChunk,
)


class DataHandler:
    def __init__(self):
        self.LLM = LLM_Adapter()
        self.extractor = MemoryExtractor(self.LLM)

    def input_sentence(self, sentence:str, graph:Graph, provider:str="user", create_time:int|None=None):
        # 输入：自然语言
        # ① 放宽：记忆提取器只做整理（去噪/切块/指代消解），片段先入库
        fragments = self.extractor.extract(sentence)
        chunks = self._make_up_chunks(fragments,graph,provider,create_time)
        # ② 收窄：再从已入库的片段里抽实体/边/属性；这一步失败也不影响记忆
        self._make_up_entity_and_edge(self.LLM.preheat_memories(fragments),graph,provider,chunks)

    def import_memories(self, memories:list[dict], graph:Graph)->list[MemoryChunk]:
        """导入 mem0 这类已经蒸馏好的记忆：一条记忆=一个 chunk，不拆分、不改写原文。

        memories: [{"memory": 原文, "provider": 谁说的, "create_time": 记忆时间}]
        一批只调一次 LLM（见 LLM_Adapter.preheat_memories）。
        """
        texts = [item["memory"] for item in memories]
        LLM_result = self.LLM.preheat_memories(texts)
        chunks = [
            graph.create_chunk(
                content=item["memory"],
                provider=item.get("provider", "user"),
                create_time=item.get("create_time") or int(time()),
            )
            for item in memories
        ]
        self._make_up_entity_and_edge(LLM_result, graph, "mem0", chunks)
        return chunks

    def _make_up_entity_and_edge(self,LLM_result:dict,graph:Graph,provider:str,chunks:list[MemoryChunk]):
        if not chunks:
            return
        # 实体画像按名字建索引；名字先做归一化，避免全角/空格差异对不上
        profiles = {
            Deduplicator.normalize(desc.get("name", "")): desc
            for desc in LLM_result.get("entities", [])
            if desc.get("name")
        }
        for link in LLM_result.get("link_data", []):
            entity_source_str = link.get("entity_source", "").strip()
            entity_target_str = link.get("entity_target", "").strip()
            edge_str = link.get("edge_str", "").strip()
            attribute_name = link.get("attribute_name", "").strip()
            attribute_value = link.get("attribute_value", "").strip()
            attribute_entity = link.get("attribute_entity", "").strip()

            # 实体名缺失的片段无法建边，直接跳过，避免造出无名实体
            if not entity_source_str:
                continue

            # 每条关系按 chunk_index 找回来源文本块；越界就退回第 0 块
            chunk_index = link.get("chunk_index", 0)
            if not (0 <= chunk_index < len(chunks)):
                chunk_index = 0
            chunk = chunks[chunk_index]

            source = graph.get_or_create_entity(entity_source_str, chunk, *self._profile_of(profiles, entity_source_str))

            # 纯属性句：只有主体和属性，不建边
            if not entity_target_str:
                if attribute_name and attribute_value:
                    graph.add_attribute(source, attribute_name, attribute_value, create_time=chunk.create_time)
                continue

            target = graph.get_or_create_entity(entity_target_str, chunk, *self._profile_of(profiles, entity_target_str))

            # 只有常驻关系才落边：事件型、自环、超长边名都只留下两端的实体
            kind = link.get("edge_kind", "事件")
            if (
                kind != "事件"
                and edge_str
                and len(edge_str) <= 4
                and source.index.unique_id != target.index.unique_id
            ):
                # 边和属性的时间跟着来源 chunk 走：导入历史记忆时就是那条记忆的时间
                graph.get_or_create_edge(source, target, edge_str, chunk, create_time=chunk.create_time)

            # 属性按 attribute_entity 决定挂在源实体还是目标实体上
            if attribute_name and attribute_value:
                owner = target if attribute_entity == "target" else source
                graph.add_attribute(owner, attribute_name, attribute_value, create_time=chunk.create_time)

        # 别名最后挂：只登记叫法，实体间的合并留给 Deduplicator
        self._apply_aliases(graph, LLM_result.get("entities", []))

    @staticmethod
    def _apply_aliases(graph:Graph, entities:list[dict]):
        """把抽取出的别名挂到主体实体上。

        只写 relate_name，不动别的实体：如果图里已经有一个叫"吴"的节点，
        下一次 Deduplicator 归并时会按别名规则把它并进 Wu（方向由别名声明决定）。
        """
        for desc in entities:
            name = desc.get("name", "").strip()
            if not name:
                continue
            entity = graph.find_entity(name)
            if entity is None:
                continue
            for alias in desc.get("aliases") or []:
                if alias and alias != name:
                    entity.index.relate_name.add(alias)


    @staticmethod
    def _profile_of(profiles:dict, name:str)->tuple[str, str]:
        desc = profiles.get(Deduplicator.normalize(name)) or {}
        return desc.get("description", ""), desc.get("entity_type", "")

    def _make_up_chunks(self, fragments:list[str], graph:Graph, provider:str, create_time:int|None=None)->list[MemoryChunk]:
        # create_time 传了就用它（导入历史记忆时由外部给），否则记当下
        moment = int(create_time if create_time is not None else time())
        return [
            graph.create_chunk(content=content,provider=provider,create_time=moment)
            for content in fragments
        ]


