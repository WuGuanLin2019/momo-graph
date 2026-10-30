"""本地 BGE-M3 向量服务客户端，以及图谱向量回填。

服务地址默认 http://localhost:4242/v1（OpenAI 兼容 /v1/embeddings），
可用环境变量覆盖：EMBED_BASE_URL / EMBED_MODEL。
"""

import json
import os
import urllib.error
import urllib.request

from MomoGraph.Data.DataStructure import GraphNodeType

DEFAULT_BASE_URL = "http://localhost:4242/v1"
DEFAULT_MODEL = "BAAI/bge-m3"


class EmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        batch_size: int = 64,
    ):
        self.base_url = (base_url or os.getenv("EMBED_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.getenv("EMBED_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量取向量，返回顺序与输入一致；空字符串会被替换成单个空格避免服务报错。"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._request([text or " " for text in texts[start:start + self.batch_size]]))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return vectors[0] if vectors else []

    def _request(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"input": texts, "model": self.model}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError(
                f"无法连接向量服务 {self.base_url}，请先启动本地 bge-m3 服务（{error}）"
            ) from error
        # OpenAI 协议返回的 data 带 index，按 index 排序才和输入一一对应
        items = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
        return [list(item.get("embedding") or []) for item in items]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def entity_profile_text(entity) -> str:
    """实体画像文本：名字 + 别名 + 稳定描述。刻意不拼边和属性（它们会变，向量会漂）。"""
    aliases = sorted(name for name in entity.index.relate_name if name and name != entity.index.name)
    parts = [entity.index.name]
    if aliases:
        parts.append("别名：" + "、".join(aliases))
    if entity.description:
        parts.append(entity.description)
    return "，".join(parts)


def index_profile_text(index) -> str:
    """索引的画像文本：名字 + 关联名字（边和属性都用它）。"""
    aliases = sorted(name for name in index.relate_name if name and name != index.name)
    parts = [index.name]
    if aliases:
        parts.append("别名：" + "、".join(aliases))
    return "，".join(parts)


def edge_profile_text(edge) -> str:
    """边画像文本：名字 + 关联名字（不拼端点，端点会随建图变化）。"""
    return index_profile_text(edge.index)


def update_embeddings(graph, client: EmbeddingClient | None = None) -> dict:
    """补齐图谱里的向量。

    chunk 内容不会变，只在缺失时补；实体画像、边、属性会随写入变化，每次重算（数据量小，可忽略）。
    """
    client = client or EmbeddingClient()
    report = {"chunks": 0, "entities": 0, "edges": 0, "attributes": 0}

    pending_chunks = [
        chunk.unique_id
        for chunk in graph.memory_chunks.values()
        if chunk.content and not graph.embedding_dict_for_memory_chunk.get(chunk.unique_id)
    ]
    if pending_chunks:
        vectors = client.embed([graph.memory_chunks[chunk_id].content for chunk_id in pending_chunks])
        for chunk_id, vector in zip(pending_chunks, vectors):
            graph.embedding_dict_for_memory_chunk[chunk_id] = vector
        report["chunks"] = len(pending_chunks)

    if graph.entities:
        entity_ids = list(graph.entities)
        vectors = client.embed([entity_profile_text(graph.entities[entity_id]) for entity_id in entity_ids])
        for entity_id, vector in zip(entity_ids, vectors):
            graph.embedding_dict_for_entity[entity_id] = vector
        report["entities"] = len(entity_ids)

    if graph.edges:
        edge_ids = list(graph.edges)
        vectors = client.embed([edge_profile_text(graph.edges[edge_id]) for edge_id in edge_ids])
        for edge_id, vector in zip(edge_ids, vectors):
            graph.embedding_dict_for_edge[edge_id] = vector
        report["edges"] = len(edge_ids)

    # 属性挂在实体下面，按 unique_id 存进 embedding_dict_for_attribute
    attributes = []
    for entity in graph.entities.values():
        for attribute in entity.attributes:
            # 旧数据里的属性没有 id，先补一个，否则向量全挤在键 0 上互相覆盖
            if attribute.index.unique_id <= 0:
                attribute.index.unique_id = graph._alloc_id(GraphNodeType.attribute)
            attributes.append(attribute)
    if attributes:
        vectors = client.embed([index_profile_text(attribute.index) for attribute in attributes])
        for attribute, vector in zip(attributes, vectors):
            graph.embedding_dict_for_attribute[attribute.index.unique_id] = vector
        report["attributes"] = len(attributes)

    return report
