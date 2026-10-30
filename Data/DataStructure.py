from dataclasses import dataclass, field
from enum import Enum
from time import time

#备注一下，activate_time是激活时间，为了记录一些“摇摆的态度”，比如我上两个月喜欢你，但上个月又讨厌你，但现在又喜欢你了

@dataclass
class IndexBase:
    unique_id:int = 0
    name:str = ""
    relate_name:set[str] = field(default_factory=set)
    embedding:list[float] = field(default_factory=list)
    relate_memory_chunk_id:set[int] = field(default_factory=set)

@dataclass
class MemoryChunk():
    unique_id:int = 0
    content:str = ""
    embedding:list[float] = field(default_factory=list)
    provider:str = ""
    create_time:int = 0

@dataclass
class AttributeItem():
    content:str = ""
    activate_time:list[int] = field(default_factory=list)
    create_time:int = 0

@dataclass
class Attribute():
    index:IndexBase = field(default_factory=IndexBase)
    items:list[AttributeItem] = field(default_factory=list)

@dataclass
class Entity():
    index:IndexBase = field(default_factory=IndexBase)
    attributes:list[Attribute] = field(default_factory=list)
    in_edges:set[int] = field(default_factory=set)
    out_edges:set[int] = field(default_factory=set)
    # 实体画像：只写稳定身份（是谁/是什么），状态类信息走 attributes
    description:str = ""
    entity_type:str = ""  # person / pet / work / place / org / object

@dataclass
class Edge():
    index:IndexBase = field(default_factory=IndexBase)
    weight:float = 0.5
    source_entity_id:int = 0
    target_entity_id:int = 0
    activate_time:list[int] = field(default_factory=list)
    create_time:int = 0

@dataclass
class Description():
    content:str = ""
    create_time:int = 0

@dataclass
class Party():
    # Party 就是社区：entity_ids 是成员，description_record 是摘要历史（只追加不覆盖）
    latest_description_embedding:list[float] = field(default_factory=list)
    description_record:list[Description] = field(default_factory=list)
    entity_ids:list[int] = field(default_factory=list)
    name:str = ""
    level:int = 0  # 0 最细，越大越宏观
    active:bool = True  # 解散后置 False，保留历史不删除

class GraphNodeType(Enum):
    memory_chunk = 0
    entity = 1
    edge = 2
    party = 3
    attribute = 4

@dataclass
class Graph():
    id_counter:dict[GraphNodeType,int] = field(default_factory=dict)

    memory_chunks:dict[int,MemoryChunk] = field(default_factory=dict)
    entities:dict[int,Entity] = field(default_factory=dict)
    edges:dict[int,Edge] = field(default_factory=dict)
    parties:dict[int,Party] = field(default_factory=dict)

    embedding_dict_for_memory_chunk:dict[int,list[float]] = field(default_factory=dict)
    embedding_dict_for_entity:dict[int,list[float]] = field(default_factory=dict)
    embedding_dict_for_attribute:dict[int,list[float]] = field(default_factory=dict)
    embedding_dict_for_edge:dict[int,list[float]] = field(default_factory=dict)
    embedding_dict_for_party:dict[int,list[float]] = field(default_factory=dict)

    def _alloc_id(self, node_type:GraphNodeType)-> int:
        if node_type not in self.id_counter:
            self.id_counter[node_type] = 0
        self.id_counter[node_type] += 1
        return self.id_counter[node_type]


    def create_chunk(self, content:str, provider:str, create_time:int = time())->MemoryChunk:
        chunk = MemoryChunk(content = content, provider = provider, create_time = create_time)
        chunk.unique_id = self._alloc_id(GraphNodeType.memory_chunk)
        self.memory_chunks[chunk.unique_id] = chunk

        #todo去重？
        return chunk

    def find_entity(self, name:str)->Entity|None:
        """按名字或别名找实体；找不到返回 None。"""
        for entity in self.entities.values():
            #todo 相同的判断过于草率，无法判断小米公司和粮食小米
            if entity.index.name == name or name in entity.index.relate_name:
                return entity
        return None

    def get_or_create_entity(self, name:str, chunk:MemoryChunk, description:str = "", entity_type:str = "")->Entity:
        # 先按 name / relate_name 找已有实体，命中就复用，避免重复建点
        entity = self.find_entity(name)
        if entity is not None:
            entity.index.relate_memory_chunk_id.add(chunk.unique_id)
            # 旧数据没有画像时补上，已有内容不覆盖（避免向量来回漂）
            if description and not entity.description:
                entity.description = description
            if entity_type and not entity.entity_type:
                entity.entity_type = entity_type
            return entity

        entity = Entity()
        entity.index.unique_id = self._alloc_id(GraphNodeType.entity)
        entity.index.name = name
        entity.index.relate_memory_chunk_id.add(chunk.unique_id)
        entity.description = description
        entity.entity_type = entity_type
        self.entities[entity.index.unique_id] = entity

        #todo去重？

        return entity

    def get_or_create_edge(self, source:Entity, target:Entity, name:str, chunk:MemoryChunk, create_time:int|None=None)->Edge:
        # 同源、同目标、同名视为同一条边；重复出现只累加权重，不重复建边
        # create_time 传了就用它（导入历史数据时按记忆本身的时间走），否则用当下
        now = int(create_time if create_time is not None else time())
        for edge_id in source.out_edges:
            edge = self.edges[edge_id]
            if edge.target_entity_id == target.index.unique_id and edge.index.name == name:
                #todo 暂用魔法数字
                edge.weight += 0.5
                edge.activate_time.append(now)
                edge.index.relate_memory_chunk_id.add(chunk.unique_id)
                return edge

        edge = Edge()
        edge.index.unique_id = self._alloc_id(GraphNodeType.edge)
        edge.index.name = name
        edge.index.relate_memory_chunk_id.add(chunk.unique_id)
        edge.source_entity_id = source.index.unique_id
        edge.target_entity_id = target.index.unique_id
        edge.weight = 0.5
        edge.create_time = now
        # 首次出现也算一次激活，否则时间线没有起点
        edge.activate_time = [now]
        self.edges[edge.index.unique_id] = edge
        source.out_edges.add(edge.index.unique_id)
        target.in_edges.add(edge.index.unique_id)
        return edge

    def add_attribute(self, entity:Entity, name:str, value:str, create_time:int|None=None):
        # 同一实体下按属性名归类：味道 -> [好吃, 很甜]；时间线只追加，矛盾的值共存
        # create_time 传了就用它（导入历史数据时按记忆本身的时间走），否则用当下
        now = int(create_time if create_time is not None else time())
        for attribute in entity.attributes:
            if attribute.index.name == name:
                for item in attribute.items:
                    if item.content == value:
                        # 同一个值再次出现：只补激活时间，不重复存内容
                        item.activate_time.append(now)
                        return
                attribute.items.append(
                    AttributeItem(content=value, activate_time=[now], create_time=now)
                )
                return
        attribute = Attribute(
            index=IndexBase(name=name),
            items=[AttributeItem(content=value, activate_time=[now], create_time=now)],
        )
        # 属性也要 id：embedding_dict_for_attribute 用它当键
        attribute.index.unique_id = self._alloc_id(GraphNodeType.attribute)
        entity.attributes.append(attribute)
