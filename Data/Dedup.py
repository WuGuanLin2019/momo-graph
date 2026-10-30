import unicodedata

from MomoGraph.Core.Embedding import (
    EmbeddingClient,
    cosine,
    index_profile_text,
    update_embeddings,
)
from MomoGraph.Data.DataStructure import (
    Attribute,
    Edge,
    Entity,
    Graph,
    IndexBase,
)

# 实体画像向量阈值：实测不同人之间最高约 0.79，同一个人换个叫法通常 > 0.93
ENTITY_SIM_THRESHOLD = 0.9
# 属性名/边名都很短，实测同义项在 0.86~0.99（体重/重量 0.87、关系不错/关系很好 0.99），
# 而反义项只有 0.63（喜欢/讨厌），0.85 正好卡在中间
NAME_SIM_THRESHOLD = 0.85


class Deduplicator:
    """实体 / 边 / 属性的离线归并器。"""

    def __init__(self, graph: Graph, client: EmbeddingClient | None = None):
        self.graph = graph
        self.client = client or EmbeddingClient()
        self._vectors_ready = False
        # 属性向量在 embedding_dict_for_attribute 里，缺号时退回按 index 对象缓存
        self._index_vectors: dict[int, list[float]] = {}

    def run(self) -> dict:
        e = self.dedup_entities()
        a = self.dedup_attributes()
        d = self.dedup_edges()
        return {"merged_entities": e, "merged_attributes": a, "merged_edges": d}

    # ---- 候选查找：通用 helper ----

    def _find_similar(self, items, key=lambda x: x.index, kind: str = "attribute") -> list[tuple]:
        pairs = []
        for keep in items:
            for drop in items:
                if keep is drop:
                    continue
                if self.compare_index_base(key(keep), key(drop), kind):
                    pairs.append((keep, drop))
        return pairs

    def _find_similar_entities(self) -> list[tuple[Entity, Entity]]:
        return self._find_similar(self.graph.entities.values(), kind="entity")

    def _find_similar_attributes(self, entity: Entity) -> list[tuple[Attribute, Attribute]]:
        return self._find_similar(entity.attributes, kind="attribute")

    def _find_similar_edges(self, edges: list[Edge]) -> list[tuple[Edge, Edge]]:
        return self._find_similar(edges, kind="edge")

    @staticmethod
    def _entities_compatible(keep: Entity, drop: Entity) -> bool:
        """语义合并前的硬门槛：两边都得有画像，类型不能冲突。"""
        if not (keep.description and drop.description):
            return False
        if keep.entity_type and drop.entity_type and keep.entity_type != drop.entity_type:
            return False
        return True

    def _index_vector(self, index: IndexBase, kind: str) -> list[float] | None:
        """按类型取向量：先查持久化字典，缺了再按当前文本临时算一次。"""
        if kind == "entity":
            return self.graph.embedding_dict_for_entity.get(index.unique_id)
        if kind == "edge":
            return self.graph.embedding_dict_for_edge.get(index.unique_id)
        if index.unique_id > 0:
            vector = self.graph.embedding_dict_for_attribute.get(index.unique_id)
            if vector:
                return vector
        # 属性刚合并/还没 reindex 时字典里没有，按 index 对象缓存避免重复请求
        key = id(index)
        if key not in self._index_vectors:
            self._index_vectors[key] = self.client.embed_one(index_profile_text(index))
        return self._index_vectors[key]

    def compare_index_base(self, keep: IndexBase, drop: IndexBase, kind: str = "attribute") -> bool:
        """判断两个索引是否相似：先比文本，不中再比向量。"""
        # unique_id > 0 时才做自比较保护（Attribute.index 默认 unique_id=0，不能拿它去重）
        if keep.unique_id > 0 and keep.unique_id == drop.unique_id:
            return False

        # 方向：keep 是 drop 的别名时，这一对是反的，交给反向那一对处理，
        # 这样"吴是 Wu 的别名"合并后留下的名字一定是 Wu
        if keep.name and keep.name in drop.relate_name and drop.name not in keep.relate_name:
            return False

        # 规范化名命中（大小写/全半角/空格归一）
        if self.normalize(keep.name) == self.normalize(drop.name):
            return True
        # 别名互含：A 的本名在 B 的别名里，或反过来
        if keep.name and keep.name in drop.relate_name:
            return True
        if drop.name and drop.name in keep.relate_name:
            return True
        # 别名集合有交集
        for cur_name in keep.relate_name:
            if cur_name in drop.relate_name:
                return True

        # 文本不像了再比向量：实体有画像、阈值更严；边和属性名字短，阈值更松
        if kind == "entity":
            keep_entity = self.graph.entities.get(keep.unique_id)
            drop_entity = self.graph.entities.get(drop.unique_id)
            if (
                keep_entity is None
                or drop_entity is None
                or not self._entities_compatible(keep_entity, drop_entity)
            ):
                return False
        keep_vector = self._index_vector(keep, kind)
        drop_vector = self._index_vector(drop, kind)
        if not keep_vector or not drop_vector:
            return False
        threshold = ENTITY_SIM_THRESHOLD if kind == "entity" else NAME_SIM_THRESHOLD
        return cosine(keep_vector, drop_vector) >= threshold

    # ---- 流程入口 ----

    def dedup_entities(self) -> int:
        self._ensure_embeddings()
        merged = 0
        for keep, drop in self._find_similar_entities():
            if keep.index.unique_id in self.graph.entities and drop.index.unique_id in self.graph.entities:
                self.merge_entity(keep, drop)
                merged += 1
        return merged

    def _ensure_embeddings(self) -> None:
        """语义比较要用实体/边/属性向量，缺了就补一次（数据量小，代价可忽略）。"""
        if self._vectors_ready:
            return
        update_embeddings(self.graph, self.client)
        self._vectors_ready = True

    def dedup_attributes(self) -> int:
        merged = 0
        for entity in list(self.graph.entities.values()):
            merged += self.dedup_attributes_by_entity(entity)
        return merged

    def dedup_edges(self) -> int:
        self._ensure_embeddings()
        groups: dict[tuple[int, int], list[Edge]] = {}
        for edge in self.graph.edges.values():
            groups.setdefault((edge.source_entity_id, edge.target_entity_id), []).append(edge)
        merged = 0
        for edges in groups.values():
            merged += self.dedup_edges_in_group(edges)
        return merged

    def dedup_attributes_by_entity(self, entity: Entity) -> int:
        merged = 0
        changed = True
        while changed:
            changed = False
            pairs = self._find_similar_attributes(entity)
            if not pairs:
                break
            keep, drop = pairs[0]
            self.merge_attribute(keep, drop)
            entity.attributes.remove(drop)
            changed = True
            merged += 1
        return merged

    def dedup_edges_in_group(self, edges: list[Edge]) -> int:
        merged = 0
        changed = True
        while changed:
            changed = False
            pairs = self._find_similar_edges(edges)
            if not pairs:
                break
            keep, drop = pairs[0]
            self.merge_edge(keep, drop)
            edges = [e for e in edges if e.index.unique_id != drop.index.unique_id]
            changed = True
            merged += 1
        return merged

    # ---- 工具 ----

    @staticmethod
    def normalize(name: str) -> str:
        # NFKC 把全角转半角，再去空格、统一小写：" ＡＢＣ " 和 " abc " 视为同一个名字
        return unicodedata.normalize("NFKC", name).strip().lower().replace(" ", "")

    # ---- 合并原语 ----

    def merge_index_base(self, keep: IndexBase, drop: IndexBase):
        keep.relate_name.update(drop.relate_name)
        keep.relate_memory_chunk_id.update(drop.relate_memory_chunk_id)

    def merge_entity(self, keep: Entity, drop: Entity):
        if keep is drop or drop.index.unique_id not in self.graph.entities:
            return
        self.merge_index_base(keep.index, drop.index)
        if drop.index.name:
            keep.index.relate_name.add(drop.index.name)
        # 画像缺的补上，两边都有就保留 keep 的（不覆盖，避免向量来回漂）
        if not keep.description:
            keep.description = drop.description
        if not keep.entity_type:
            keep.entity_type = drop.entity_type
        # 属性先迁移再归并
        keep.attributes.extend(drop.attributes)
        self.dedup_attributes_by_entity(keep)
        # 邻接表合并
        keep.in_edges.update(drop.in_edges)
        keep.out_edges.update(drop.out_edges)

        # 边端点改写（先快照，别边遍历边改）
        for edge_id in list(drop.in_edges) + list(drop.out_edges):
            edge = self.graph.edges.get(edge_id)
            if edge is None:
                continue
            if edge.source_entity_id == drop.index.unique_id:
                edge.source_entity_id = keep.index.unique_id
            if edge.target_entity_id == drop.index.unique_id:
                edge.target_entity_id = keep.index.unique_id

        # 端点改写后可能产生重复边，分两组各自收口
        self.dedup_edges_in_group(
            [self.graph.edges[eid] for eid in list(keep.out_edges) if eid in self.graph.edges]
        )
        self.dedup_edges_in_group(
            [self.graph.edges[eid] for eid in list(keep.in_edges) if eid in self.graph.edges]
        )

        # 注销 drop
        self.graph.entities.pop(drop.index.unique_id, None)
        self.graph.embedding_dict_for_entity.pop(drop.index.unique_id, None)
        for party in self.graph.parties.values():
            party.entity_ids = [
                keep.index.unique_id if eid == drop.index.unique_id else eid
                for eid in party.entity_ids
            ]

    def merge_edge(self, keep: Edge, drop: Edge):
        self.merge_index_base(keep.index, drop.index)
        keep.weight = (keep.weight + drop.weight) / 2  # TODO: 权重语义待定，先保留原逻辑
        keep.activate_time.extend(drop.activate_time)
        keep.create_time = min(keep.create_time, drop.create_time)
        self.remove_edge(drop)

    def merge_attribute(self, keep: Attribute, drop: Attribute):
        self.merge_index_base(keep.index, drop.index)
        # AttributeItem 是 dataclass，用属性访问；按最近激活倒序
        keep.items.extend(drop.items)
        keep.items.sort(
            key=lambda item: max(item.activate_time) if item.activate_time else item.create_time,
            reverse=True,
        )
        # 名字/别名变了，旧向量作废：drop 的直接删，keep 的重算一次
        self._index_vectors.pop(id(keep.index), None)
        self._index_vectors.pop(id(drop.index), None)
        self.graph.embedding_dict_for_attribute.pop(drop.index.unique_id, None)
        if keep.index.unique_id > 0:
            self.graph.embedding_dict_for_attribute[keep.index.unique_id] = self.client.embed_one(
                index_profile_text(keep.index)
            )

    def remove_edge(self, edge: Edge):
        # 从 edges 字典、两端邻接表、embedding 字典中摘除
        self.graph.edges.pop(edge.index.unique_id, None)
        self.graph.embedding_dict_for_edge.pop(edge.index.unique_id, None)
        source = self.graph.entities.get(edge.source_entity_id)
        if source is not None:
            source.out_edges.discard(edge.index.unique_id)
        target = self.graph.entities.get(edge.target_entity_id)
        if target is not None:
            target.in_edges.discard(edge.index.unique_id)
