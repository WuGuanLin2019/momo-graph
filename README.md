# MomoGraph

一个面向「长期记忆」的图谱实验项目：把自然语言记忆抽成实体 / 关系 / 属性，存进一张可检索、可可视化的知识图谱，再用它回答问题。

> 项目仍在开发中，入口 `main.py` 尚未接线，进度与待办见 [Doc/Todo.md](Doc/Todo.md)。

## 核心能力

| 能力 | 入口 | 说明 |
| --- | --- | --- |
| 记住 | `GraphHandler.remember()` | 原文 → LLM 整理成记忆片段 → 抽取实体/关系/属性 → 建图并落盘 |
| 消化 | `GraphHandler.meditate()` | 实体/边/属性去重合并、社区发现、LLM 生成社区摘要 |
| 回忆 | `GraphHandler.remind()` | BM25 + 向量 + 社区摘要混合召回，拼成 prompt 交给 LLM 作答 |
| 导入 | `GraphHandler.import_memories()` | 批量导入 mem0 等外部工具导出的记忆，支持断点续跑 |
| 可视化 | `DashBoard/Viewer.py` | 生成单文件 HTML 图谱看板，双击即可打开；支持实时刷新 |

## 目录结构

```
MomoGraph/
├── main.py                 # 程序入口（待实现）
├── Core/
│   ├── Graph.py            # 门面类 GraphHandler：remember / remind / meditate / import
│   ├── LLM.py              # 模型适配层：抽取、画像、社区摘要、作答的 prompt 与 schema
│   ├── MemoryExtractor.py  # 原文 → 记忆片段（去噪、切块、指代消解）
│   ├── Embedding.py        # OpenAI 兼容 /v1/embeddings 客户端（默认本地 BGE-M3）
│   ├── Retriever.py        # 检索：BM25 + 向量召回 + 沿边扩散 + 社区摘要
│   ├── Community.py        # 社区发现与社区摘要刷新
│   ├── DataImport.py       # 外部记忆批量导入
│   └── DataSave.py         # 图谱落盘
├── Data/
│   ├── DataStructure.py    # 数据结构：Entity / Edge / Party / MemoryChunk / Community
│   ├── DataHandle.py       # 建图写入（抽图结果 → 图谱）
│   ├── Dedup.py            # 实体/边/属性三级去重与合并
│   ├── GraphIO.py          # JSON 序列化 / 反序列化
│   └── MemoryText.py       # 检索结果 → 自然语言 prompt
├── DashBoard/
│   ├── Viewer.py           # 生成 HTML 看板 / --live 实时模式
│   ├── dashboard.html      # 导出产物
│   ├── vendor/             # 前端依赖（cytoscape 及其布局插件）
│   └── start.bat           # Windows 快捷启动
└── Doc/
    ├── Todo.md             # 待办清单
    └── 设计图.png
```

## 运行环境

- Python 3.11+
- 主要依赖：`langchain-core`、`langchain-openai`、`pydantic`、`networkx`、`numpy`、`jieba`、`rank-bm25`

```bash
pip install langchain-core langchain-openai pydantic networkx numpy jieba rank-bm25
```

## 配置

通过环境变量配置，代码里不写死密钥：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | LLM 密钥，必填 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 任意 OpenAI 风格接口 |
| `LLM_MODEL` | `deepseek-chat` | 模型名 |
| `LLM_TEMPERATURE` | `0` | 采样温度 |
| `EMBED_BASE_URL` | `http://localhost:4242/v1` | 向量服务地址（OpenAI 兼容） |
| `EMBED_MODEL` | `BAAI/bge-m3` | 向量模型名 |

## 使用

项目内部使用 `MomoGraph.xxx` 包路径导入，运行前需把**项目父目录**加入 `PYTHONPATH`：

```bash
# PowerShell
$env:PYTHONPATH = "E:\Study\GitProject"
& "E:\Study\GitProject\MomoGraph\.venv\Scripts\python.exe" -c "from MomoGraph.Core.Graph import GraphHandler; ..."
```

图谱预览看板：

```bash
# 直接跑 demo：生成并打开 HTML
python MomoGraph/DashBoard/Viewer.py

# 实时模式：每 1.5 秒拉一次 MomoGraph/Temp/graph.json，有变化自动重绘
python MomoGraph/DashBoard/Viewer.py --live
```

Windows 用户可直接双击 `DashBoard/start.bat`。

## 数据文件

图谱默认落在 `MomoGraph/Temp/graph.json`（已被 `.gitignore` 忽略）。这是本地个人记忆数据，不随仓库分发。

## License

暂未指定。
