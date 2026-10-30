"""Graph 的可逆序列化：内存对象 <-> JSON 友好的 dict。

为什么单独拆出来：DataSave 负责落盘，Viewer 的实时服务需要把文件还原成 Graph
再转成前端视图。两边都要用同一套编解码，放这里避免循环 import。
"""

from dataclasses import fields, is_dataclass

from MomoGraph.Data.DataStructure import (
    Attribute,
    AttributeItem,
    Description,
    Edge,
    Entity,
    Graph,
    GraphNodeType,
    IndexBase,
    MemoryChunk,
    Party,
)

FORMAT = "momo-graph"
VERSION = 1


# ---------- 编码 ----------

def _encode(value):
    """递归转成 JSON 可写的结构：set -> 排序后的 list，dataclass -> dict。"""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        # JSON 的 key 只能是字符串，embedding 字典的 int key 会被转成 "1"
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    return value


def encode_graph(graph: Graph) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        # id_counter 的 key 是枚举，存 name（"memory_chunk"），还原时按 name 查回枚举
        "id_counter": {node_type.name: count for node_type, count in graph.id_counter.items()},
        "memory_chunks": [_encode(chunk) for chunk in graph.memory_chunks.values()],
        "entities": [_encode(entity) for entity in graph.entities.values()],
        "edges": [_encode(edge) for edge in graph.edges.values()],
        "parties": [
            {"id": party_id, **_encode(party)} for party_id, party in graph.parties.items()
        ],
        "embedding_dict_for_memory_chunk": _encode(graph.embedding_dict_for_memory_chunk),
        "embedding_dict_for_entity": _encode(graph.embedding_dict_for_entity),
        "embedding_dict_for_attribute": _encode(graph.embedding_dict_for_attribute),
        "embedding_dict_for_edge": _encode(graph.embedding_dict_for_edge),
        "embedding_dict_for_party": _encode(graph.embedding_dict_for_party),
    }


# ---------- 解码 ----------

def _index_from(data: dict) -> IndexBase:
    return IndexBase(
        unique_id=data.get("unique_id", 0),
        name=data.get("name", ""),
        relate_name=set(data.get("relate_name") or []),
        embedding=list(data.get("embedding") or []),
        relate_memory_chunk_id=set(data.get("relate_memory_chunk_id") or []),
    )


def _item_from(data: dict) -> AttributeItem:
    return AttributeItem(
        content=data.get("content", ""),
        activate_time=list(data.get("activate_time") or []),
        create_time=data.get("create_time", 0),
    )


def _attribute_from(data: dict) -> Attribute:
    return Attribute(
        index=_index_from(data.get("index") or {}),
        items=[_item_from(item) for item in data.get("items") or []],
    )


def _chunk_from(data: dict) -> MemoryChunk:
    return MemoryChunk(
        unique_id=data.get("unique_id", 0),
        content=data.get("content", ""),
        embedding=list(data.get("embedding") or []),
        provider=data.get("provider", ""),
        create_time=data.get("create_time", 0),
    )


def _entity_from(data: dict) -> Entity:
    return Entity(
        index=_index_from(data.get("index") or {}),
        attributes=[_attribute_from(attr) for attr in data.get("attributes") or []],
        in_edges=set(data.get("in_edges") or []),
        out_edges=set(data.get("out_edges") or []),
        description=data.get("description", ""),
        entity_type=data.get("entity_type", ""),
    )


def _edge_from(data: dict) -> Edge:
    return Edge(
        index=_index_from(data.get("index") or {}),
        weight=data.get("weight", 0.5),
        source_entity_id=data.get("source_entity_id", 0),
        target_entity_id=data.get("target_entity_id", 0),
        activate_time=list(data.get("activate_time") or []),
        create_time=data.get("create_time", 0),
    )


def _description_from(data: dict) -> Description:
    return Description(
        content=data.get("content", ""),
        create_time=data.get("create_time", 0),
    )


def _party_from(data: dict) -> Party:
    return Party(
        latest_description_embedding=list(data.get("latest_description_embedding") or []),
        description_record=[_description_from(item) for item in data.get("description_record") or []],
        entity_ids=list(data.get("entity_ids") or []),
        # 旧文件没有这三个字段，用默认值兜底，不需要迁移
        name=data.get("name", ""),
        level=data.get("level", 0),
        active=data.get("active", True),
    )


def _embedding_dict_from(data: dict | None) -> dict[int, list[float]]:
    return {int(key): list(value) for key, value in (data or {}).items()}


def _restore_counter(graph: Graph) -> None:
    """id_counter 缺失时按现有最大 id 兜底，避免下次分配撞号。"""
    if graph.id_counter:
        return
    graph.id_counter = {
        GraphNodeType.memory_chunk: max(graph.memory_chunks, default=0),
        GraphNodeType.entity: max(graph.entities, default=0),
        GraphNodeType.edge: max(graph.edges, default=0),
        GraphNodeType.party: max(graph.parties, default=0),
        # 属性没有独立字典，得从各实体的属性列表里找最大值
        GraphNodeType.attribute: max(
            (
                attribute.index.unique_id
                for entity in graph.entities.values()
                for attribute in entity.attributes
            ),
            default=0,
        ),
    }


def decode_graph(payload: dict) -> Graph:
    graph = Graph()
    graph.id_counter = {
        GraphNodeType[name]: count
        for name, count in (payload.get("id_counter") or {}).items()
        if name in GraphNodeType.__members__
    }
    graph.memory_chunks = {
        chunk.unique_id: chunk
        for chunk in (_chunk_from(item) for item in payload.get("memory_chunks") or [])
    }
    graph.entities = {
        entity.index.unique_id: entity
        for entity in (_entity_from(item) for item in payload.get("entities") or [])
    }
    graph.edges = {
        edge.index.unique_id: edge
        for edge in (_edge_from(item) for item in payload.get("edges") or [])
    }
    graph.parties = {
        item.get("id", index): _party_from(item)
        for index, item in enumerate(payload.get("parties") or [], start=1)
    }
    graph.embedding_dict_for_memory_chunk = _embedding_dict_from(payload.get("embedding_dict_for_memory_chunk"))
    graph.embedding_dict_for_entity = _embedding_dict_from(payload.get("embedding_dict_for_entity"))
    graph.embedding_dict_for_attribute = _embedding_dict_from(payload.get("embedding_dict_for_attribute"))
    graph.embedding_dict_for_edge = _embedding_dict_from(payload.get("embedding_dict_for_edge"))
    graph.embedding_dict_for_party = _embedding_dict_from(payload.get("embedding_dict_for_party"))
    _restore_counter(graph)
    return graph


def is_graph_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("format") == FORMAT
