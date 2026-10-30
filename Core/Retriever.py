"""记忆检索。

两条路：
1. chunk 混合检索：BM25（jieba 分词）+ 语义余弦，RRF 融合。
2. 图检索：实体画像混合检索取种子，再做 beam search 扩展成链条；
   链条上的节点再反哺 chunk，最后三路 RRF 融合出 chunk 结果。
"""

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from MomoGraph.Core.Community import search_communities as community_search
from MomoGraph.Core.Embedding import EmbeddingClient, cosine, update_embeddings

RRF_K = 60
NAME_BOOST = 0.35  # query 里直接出现实体名/别名时给它的加成
COMMUNITY_TOP_K = 3  # 社区摘要只取最相关的几个，多了会淹没链条


class Retriever:
    def __init__(self, graph, client: EmbeddingClient | None = None):
        self.graph = graph
        self.client = client or EmbeddingClient()
        self._bm25: BM25Okapi | None = None
        self._bm25_chunk_ids: list[int] = []
        self._synced_chunks = 0

    # ---------- 索引 ----------

    def _ensure_index(self) -> None:
        """向量缺失就补（增量），chunk 数量变了就重建 BM25。"""
        chunk_count = len(self.graph.memory_chunks)
        if self._synced_chunks != chunk_count:
            update_embeddings(self.graph, self.client)
            self._synced_chunks = chunk_count
        if self._bm25 is None or len(self._bm25_chunk_ids) != chunk_count:
            self._build_bm25()

    def _build_bm25(self) -> None:
        chunk_ids, corpus = [], []
        for chunk in self.graph.memory_chunks.values():
            tokens = self._tokenize(chunk.content)
            if not tokens:
                continue
            chunk_ids.append(chunk.unique_id)
            corpus.append(tokens)
        self._bm25_chunk_ids = chunk_ids
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # 只保留带字母/数字/汉字的词，丢掉标点和空白
        return [
            token.strip().lower()
            for token in jieba.lcut(text or "")
            if token.strip() and any(char.isalnum() for char in token)
        ]

    @staticmethod
    def _rrf(rankings: list[list[int]]) -> dict[int, float]:
        """Reciprocal Rank Fusion：只吃排名，免疫 BM25 与余弦的量纲差异。"""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, item_id in enumerate(ranking):
                fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        return fused

    # ---------- chunk 检索 ----------

    def search_chunks(self, query: str, top_k: int = 6, recall: int = 20) -> list[dict]:
        query = (query or "").strip()
        if not query or not self.graph.memory_chunks:
            return []
        self._ensure_index()

        bm25_scores, semantic_scores, bm25_rank, semantic_rank = self._chunk_rankings(query, recall)
        fused = self._rrf([bm25_rank, semantic_rank])

        results = []
        for chunk_id, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
            chunk = self.graph.memory_chunks.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                {
                    "id": chunk_id,
                    "content": chunk.content,
                    "score": round(score, 6),
                    "bm25": round(bm25_scores.get(chunk_id, 0.0), 4),
                    "semantic": round(semantic_scores.get(chunk_id, 0.0), 4),
                    "from": self._hit_sources(chunk_id, bm25_scores, semantic_scores, {}),
                }
            )
        return results

    def _chunk_rankings(self, query: str, recall: int):
        bm25_scores = self._bm25_scores(query)
        bm25_rank = sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:recall]
        semantic_scores = self._semantic_scores(query)
        semantic_rank = sorted(semantic_scores, key=semantic_scores.get, reverse=True)[:recall]
        return bm25_scores, semantic_scores, bm25_rank, semantic_rank

    def _bm25_scores(self, query: str) -> dict[int, float]:
        tokens = self._tokenize(query)
        if not self._bm25 or not tokens:
            return {}
        scores = self._bm25.get_scores(tokens)
        return {
            chunk_id: float(score)
            for chunk_id, score in zip(self._bm25_chunk_ids, scores)
            if score > 0
        }

    def _semantic_scores(self, query: str) -> dict[int, float]:
        chunk_ids = [
            chunk.unique_id
            for chunk in self.graph.memory_chunks.values()
            if self.graph.embedding_dict_for_memory_chunk.get(chunk.unique_id)
        ]
        if not chunk_ids:
            return {}
        query_vector = np.asarray(self.client.embed_one(query), dtype=np.float32)
        matrix = np.asarray(
            [self.graph.embedding_dict_for_memory_chunk[chunk_id] for chunk_id in chunk_ids],
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1.0
        similarities = matrix @ query_vector / norms
        return {chunk_id: float(score) for chunk_id, score in zip(chunk_ids, similarities)}

    # ---------- 实体检索 ----------

    def search_entities(self, query: str, top_k: int = 3) -> list[dict]:
        """画像向量余弦 + 名字/别名命中加成。名字太短时余弦不可靠，加成很关键。"""
        query = (query or "").strip()
        if not query or not self.graph.entities:
            return []
        self._ensure_index()

        query_vector = np.asarray(self.client.embed_one(query), dtype=np.float32)
        query_lower = query.lower()
        scored = []
        for entity in self.graph.entities.values():
            vector = self.graph.embedding_dict_for_entity.get(entity.index.unique_id)
            if not vector:
                continue
            semantic = cosine(query_vector.tolist(), vector)
            names = [entity.index.name, *entity.index.relate_name]
            name_hit = any(name and name.lower() in query_lower for name in names)
            scored.append(
                {
                    "id": entity.index.unique_id,
                    "name": entity.index.name,
                    "semantic": round(semantic, 4),
                    "name_hit": name_hit,
                    "score": round(semantic + (NAME_BOOST if name_hit else 0.0), 4),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    # ---------- 图检索 ----------

    def search_graph(
        self,
        query: str,
        seed_k: int = 3,
        depth: int = 2,
        beam_width: int = 4,
        decay: float = 0.6,
    ) -> dict:
        """从实体种子出发做 beam search，返回链条和反哺的 chunk 权重。"""
        query = (query or "").strip()
        seeds = self.search_entities(query, seed_k)
        if not seeds:
            return {"seeds": [], "chains": [], "chunk_boost": {}}

        query_vector = np.asarray(self.client.embed_one(query), dtype=np.float32)
        query_list = query_vector.tolist()
        max_weight = max((edge.weight for edge in self.graph.edges.values()), default=1.0) or 1.0

        paths = [
            {"nodes": [seed["id"]], "edges": [], "score": seed["score"]}
            for seed in seeds
        ]

        for _ in range(depth):
            candidates = []
            for path in paths:
                current_id = path["nodes"][-1]
                current = self.graph.entities.get(current_id)
                if current is None:
                    continue
                for edge_id in [*current.out_edges, *current.in_edges]:
                    edge = self.graph.edges.get(edge_id)
                    if edge is None:
                        continue
                    other_id = (
                        edge.target_entity_id
                        if edge.source_entity_id == current_id
                        else edge.source_entity_id
                    )
                    if other_id in path["nodes"]:
                        continue
                    other = self.graph.entities.get(other_id)
                    if other is None:
                        continue
                    neighbor_vector = self.graph.embedding_dict_for_entity.get(other_id)
                    edge_vector = self.graph.embedding_dict_for_edge.get(edge_id)
                    if not neighbor_vector or not edge_vector:
                        continue
                    step = (
                        (edge.weight / max_weight)
                        * cosine(query_list, neighbor_vector)
                        * cosine(query_list, edge_vector)
                    )
                    hop = len(path["nodes"])  # 第 hop 跳，越远衰减越大
                    candidates.append(
                        {
                            "nodes": [*path["nodes"], other_id],
                            "edges": [*path["edges"], edge_id],
                            "score": path["score"] + (decay ** (hop - 1)) * step,
                        }
                    )
            if not candidates:
                break
            # 同一组节点只留一条链路，避免不同顺序走同一条路
            candidates.sort(key=lambda item: item["score"], reverse=True)
            seen, paths = set(), []
            for candidate in candidates:
                key = frozenset(candidate["nodes"])
                if key in seen:
                    continue
                seen.add(key)
                paths.append(candidate)
                if len(paths) >= beam_width:
                    break

        # 链条上的节点反哺 chunk：越靠前、路径分越高，权重越大
        chunk_boost: dict[int, float] = {}
        for path in paths:
            for hop, node_id in enumerate(path["nodes"]):
                entity = self.graph.entities.get(node_id)
                if entity is None:
                    continue
                value = path["score"] * (0.9 ** hop)
                for chunk_id in entity.index.relate_memory_chunk_id:
                    chunk_boost[chunk_id] = max(chunk_boost.get(chunk_id, 0.0), value)

        return {
            "seeds": seeds,
            "chains": [self._format_chain(path) for path in paths],
            "chunk_boost": chunk_boost,
        }

    # ---------- 社区摘要检索 ----------

    def search_communities(self, query: str, top_k: int = COMMUNITY_TOP_K) -> list[dict]:
        """社区摘要一路：query 向量与各社区最新摘要向量余弦。"""
        query = (query or "").strip()
        if not query or not self.graph.parties:
            return []
        return community_search(self.graph, self.client.embed_one(query), top_k)

    # ---------- 汇总 ----------

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        seed_k: int = 3,
        depth: int = 2,
        beam_width: int = 4,
        recall: int = 20,
    ) -> dict:
        query = (query or "").strip()
        if not query:
            return {"query": "", "chains": [], "chunks": [], "communities": []}
        self._ensure_index()

        bm25_scores, semantic_scores, bm25_rank, semantic_rank = self._chunk_rankings(query, recall)
        graph_part = self.search_graph(query, seed_k=seed_k, depth=depth, beam_width=beam_width)
        chunk_boost = graph_part["chunk_boost"]
        graph_rank = sorted(chunk_boost, key=chunk_boost.get, reverse=True)

        fused = self._rrf([bm25_rank, semantic_rank, graph_rank])
        chunks = []
        for chunk_id, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
            chunk = self.graph.memory_chunks.get(chunk_id)
            if chunk is None:
                continue
            chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk.content,
                    "score": round(score, 6),
                    "bm25": round(bm25_scores.get(chunk_id, 0.0), 4),
                    "semantic": round(semantic_scores.get(chunk_id, 0.0), 4),
                    "graph": round(chunk_boost.get(chunk_id, 0.0), 4),
                    "from": self._hit_sources(chunk_id, bm25_scores, semantic_scores, chunk_boost),
                }
            )

        return {
            "query": query,
            "chains": graph_part["chains"],
            "chunks": chunks,
            "communities": self.search_communities(query),
        }

    # ---------- 输出格式 ----------

    @staticmethod
    def _hit_sources(chunk_id: int, bm25_scores: dict, semantic_scores: dict, chunk_boost: dict) -> list[str]:
        return [
            name
            for name, hits in (
                ("bm25", bm25_scores),
                ("semantic", semantic_scores),
                ("graph", chunk_boost),
            )
            if chunk_id in hits
        ]

    def _format_chain(self, path: dict) -> dict:
        nodes = []
        for node_id in path["nodes"]:
            entity = self.graph.entities[node_id]
            attributes = {}
            for attribute in entity.attributes:
                items = [
                    {"value": item.content, "time": self._activation_time(item)}
                    for item in attribute.items
                    if item.content
                ]
                if not items:
                    continue
                # 最近激活的排前面，渲染层按顺序把第一条标成【最新】
                items.sort(key=lambda item: item["time"], reverse=True)
                attributes[attribute.index.name] = items
            nodes.append(
                {
                    "id": node_id,
                    "name": entity.index.name,
                    "description": entity.description,
                    "attributes": attributes,
                }
            )

        edges = []
        for edge_id in path["edges"]:
            edge = self.graph.edges[edge_id]
            # 同向同端点的其它边（喜欢 / 不喜欢这类摇摆），一起交出去让模型看时间线
            siblings = [
                {"name": other.index.name, "time": self._activation_time(other)}
                for other in self.graph.edges.values()
                if other.index.unique_id != edge_id
                and other.source_entity_id == edge.source_entity_id
                and other.target_entity_id == edge.target_entity_id
            ]
            siblings.sort(key=lambda item: item["time"], reverse=True)
            edges.append(
                {
                    "name": edge.index.name,
                    "weight": edge.weight,
                    "source": edge.source_entity_id,
                    "target": edge.target_entity_id,
                    "chunks": sorted(edge.index.relate_memory_chunk_id),
                    "time": self._activation_time(edge),
                    "siblings": siblings,
                }
            )

        chunk_ids = sorted(
            {
                chunk_id
                for node_id in path["nodes"]
                for chunk_id in self.graph.entities[node_id].index.relate_memory_chunk_id
            }
        )
        return {"score": round(path["score"], 4), "nodes": nodes, "edges": edges, "chunks": chunk_ids}

    @staticmethod
    def _activation_time(record) -> int:
        """最近一次激活时间；没有激活记录时退回创建时间（旧数据兜底）。"""
        times = list(getattr(record, "activate_time", None) or [])
        return max(times) if times else getattr(record, "create_time", 0)
