# MomoGraph 待办清单

> 优先级：P0（挡路/必崩）> P1（核心功能）> P2（工程完善）
> 相关文件：Core/MomoGraph.py、Core/DataImport.py、Data/DataStructure.py、main.py

## 次日待办（2026-10-29）

- [ ] DashBoard 优化：实时刷新别重置视图和布局、边权重可视化（粗细/颜色）、属性多时节点自适应
- [ ] 社区发现（用模型）：先做无向单层社区划分，再让 LLM 按社区写摘要（对应下面 P1 的社区发现条目）
- [ ] 去重：实体/边/属性三级；合并时级联更新边端点、chunk 引用、邻接表、社区成员（骨架已搭：`Data/Dedup.py` 的 `Deduplicator`，`meditate()` 已接入口，判定与合并逻辑待填）
- [ ] 矛盾处理：同一属性出现相反值（比如 喜欢 ↔ 讨厌）、同一关系正反冲突时怎么标记和展示
- [ ] 检索：query → 实体召回 → BFS 邻域（跳数上限）→ 证据 chunk + 社区摘要 → 拼成自然语言回答

## P0 立刻修（现在 import 就会抛异常）

- [ ] `IndexBase.embeding`、`MemoryChunk.embeding` 的 `= []` 改为 `field(default_factory=list)`（DataStructure.py:9、16）
- [ ] `AttributeItem.activate_time`、`Edge.activate_time` 的 `= []` 改为 `field(default_factory=list)`（DataStructure.py:23、44）
- [ ] 全项目拼写统一：`embeding` → `embedding`；`entitity_ids` → `entity_ids`
- [ ] 修完后跑一次 `import DataStructure` 确认能正常加载

## P0 数据结构补全

- [ ] `Graph` 里补上原始文本存储 `chunks: dict[int, MemoryChunk]`（现在 chunk 没地方存，但 `relate_memory_chunk_id` 又指向它）
- [ ] embedding 只留一个数据源：`Entity.index.embeding` 与 `Graph.embeding_dict_for_entity` 二选一；若保留字典，明确它是"派生缓存"并注明何时重建，避免两处不一致
- [ ] 加 ID 分配器：`next_entity_id` / `next_edge_id` / `next_party_id` / `next_chunk_id`，并注明各 ID 空间相互独立
- [ ] `in_edges` / `out_edges` 改用 `set[int]` 防重复；增删边时同步维护邻接表
- [ ] 确认 `Edge` 是否真需要 embedding（GraphRAG 通常只召回实体和社区摘要，边未必需要）

## P1 建图 remember

- [ ] 定义 LLM 抽取的 JSON schema + few-shot 示例 + 抽取结果校验（实体类型、关系类型约束）
- [ ] 确定切块策略与 chunk 边界（多大、怎么切）
- [ ] 指代消解落地：代词/省略主语 → 解析到具体实体（决定是否建虚拟实体节点）
- [ ] chunk 指纹去重，保证重复导入幂等
- [ ] 空/噪音输入分支：返回"没有值得记下的"

## P1 去重与 meditate

- [ ] 实体合并的级联更新：边端点、chunk 引用、社区成员 ID、邻接表全部迁移（最易出 bug，单独写 + 测试）
- [ ] 去重判定规则：name 精确 + relate_name + embedding 相似度阈值；防"苹果(公司)/苹果(水果)"误合并
- [ ] 社区发现：先做无向、单层；有向边如何转无向（取 max/求和）要定义
- [ ] 社区摘要幂等：记录"已处理版本"，避免反复 meditate 重复生成 / 摘要过期

## P1 检索 remind

- [ ] 建向量索引（先用 numpy 矩阵做点积，规模大了再换 FAISS）
- [ ] 局部检索：query → top-k 实体 → 沿边 BFS（跳数上限固定，如 2 跳）→ hub 节点剪枝 → 回溯证据 chunk 原文
- [ ] 全局检索：召回社区摘要 → 汇总生成
- [ ] 查询路由：先判断是局部问题还是全局问题
- [ ] 明确返回结构（实体描述 + 关系链 + 社区报告 + 证据 chunk）

## P2 持久化与工程

- [ ] import/export 格式定为 JSON；处理 `set` 字段序列化（set 不能直接 JSON，要转 list）
- [ ] 导入时 ID 冲突处理（重新分配或命名空间隔离）
- [ ] embedding 记录模型名/维度版本号；换模型需全量 re-embed
- [ ] 时序语义定义：`activate_time` 是点时刻还是区间，空值代表"永远有效"还是"未知"
- [ ] main.py 入口与依赖装配（现在还是空的）
- [ ] 并发/线程安全（学习阶段可先跳过）
