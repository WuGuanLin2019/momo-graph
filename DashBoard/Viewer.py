"""MomoGraph 图谱预览 Dashboard。

用法一（直接跑 demo，生成并打开 HTML）：
    & ".venv\\Scripts\\python.exe" MomoGraph/DashBoard/Viewer.py

用法二（导出真实图）：
    from MomoGraph.DashBoard.Viewer import build_dashboard
    build_dashboard(graph, "MomoGraph/DashBoard/dashboard.html")

用法三（实时看板，边记忆边刷新）：
    & ".venv\\Scripts\\python.exe" MomoGraph/DashBoard/Viewer.py --live
    页面每 1.5 秒拉一次 MomoGraph/Temp/graph.json，有变化自动重绘。

生成的是单文件 HTML，数据内嵌，双击即可在浏览器打开，无需起服务。
"""

import json
import os
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from MomoGraph.Data.GraphIO import decode_graph, is_graph_payload

EMPTY_DASHBOARD = {
    "title": "MomoGraph · 图谱预览",
    "entities": [],
    "edges": [],
    "chunks": {},
    "communities": [],
    "stats": {"entities": 0, "edges": 0, "chunks": 0, "parties": 0},
}


def graph_to_dict(graph) -> dict:
    """把内存里的 Graph 转成前端需要的 JSON 结构（鸭子类型，无需 import Graph）。"""
    entities = []
    for entity in graph.entities.values():
        index = entity.index
        aliases = sorted(name for name in index.relate_name if name and name != index.name)
        attributes = []
        for attribute in entity.attributes:
            values = [item.content for item in attribute.items if item.content]
            if values:
                attributes.append(
                    {
                        "name": attribute.index.name,
                        "values": values,
                        # 详情卡片要展示每个值的历史，原始记录一并带过去
                        "items": [
                            {
                                "content": item.content,
                                "activate_time": list(item.activate_time),
                                "create_time": item.create_time,
                            }
                            for item in attribute.items
                            if item.content
                        ],
                    }
                )
        entities.append(
            {
                "id": index.unique_id,
                "name": index.name,
                "aliases": aliases,
                "description": entity.description,
                "entity_type": entity.entity_type,
                "attributes": attributes,
                "chunks": sorted(index.relate_memory_chunk_id),
            }
        )

    edges = []
    for edge in graph.edges.values():
        edges.append(
            {
                "id": edge.index.unique_id,
                "source": edge.source_entity_id,
                "target": edge.target_entity_id,
                "name": edge.index.name,
                "weight": edge.weight,
                "chunks": sorted(edge.index.relate_memory_chunk_id),
                "activate_time": list(edge.activate_time),
                "create_time": edge.create_time,
            }
        )

    # 社区（Party）：解散的默认不显示，保留的历史只留在 graph.json 里
    communities = []
    for party_id, party in graph.parties.items():
        if not party.active:
            continue
        communities.append(
            {
                "id": party_id,
                "name": party.name,
                "level": party.level,
                "members": [
                    graph.entities[entity_id].index.name
                    for entity_id in party.entity_ids
                    if entity_id in graph.entities
                ],
                "summary": party.description_record[-1].content if party.description_record else "",
                "history": [
                    {"content": record.content, "create_time": record.create_time}
                    for record in party.description_record
                ],
            }
        )
    communities.sort(key=lambda item: (item["level"], -len(item["members"]), item["id"]))

    return {
        "title": "MomoGraph · 图谱预览",
        "entities": entities,
        "edges": edges,
        # 详情卡片要展示关联记忆的原文，按 id 建索引方便前端查
        "chunks": {
            str(chunk.unique_id): {
                "content": chunk.content,
                "provider": chunk.provider,
                "create_time": chunk.create_time,
            }
            for chunk in graph.memory_chunks.values()
        },
        "communities": communities,
        "stats": {
            "entities": len(entities),
            "edges": len(edges),
            "chunks": len(graph.memory_chunks),
            "parties": len(communities),
        },
    }


DEMO_DATA = {
    "title": "MomoGraph · 图谱预览",
    "entities": [
        {"id": 1, "name": "我", "aliases": [], "description": "MOMO，Wu 创造的 AI。", "entity_type": "object",
         "attributes": [], "chunks": [1]},
        {"id": 2, "name": "罗小姐", "aliases": ["小罗", "罗罗"], "description": "Wu 的朋友。", "entity_type": "person",
         "attributes": [], "chunks": [1, 2]},
        {"id": 3, "name": "奶茶", "aliases": [], "description": "一种饮品。", "entity_type": "object",
         "attributes": [{"name": "味道", "values": ["好喝"]}], "chunks": [2]},
    ],
    "edges": [
        {"id": 1, "source": 1, "target": 2, "name": "喜欢", "weight": 1},
        {"id": 2, "source": 1, "target": 2, "name": "一起去玩", "weight": 1},
        {"id": 3, "source": 2, "target": 3, "name": "喜欢", "weight": 1},
    ],
    "stats": {"entities": 3, "edges": 3, "chunks": 2, "parties": 1},
    "communities": [
        {"id": 1, "name": "喝奶茶日常", "level": 0, "members": ["我", "罗小姐", "奶茶"],
         "summary": "这簇讲的是我和罗小姐一起喝奶茶的日常，罗小姐喜欢阿嬷手作的奶茶。",
         "history": [{"content": "这簇讲的是我和罗小姐一起喝奶茶的日常，罗小姐喜欢阿嬷手作的奶茶。", "create_time": 0}]},
    ],
    "chunks": {
        "1": {"content": "今天和罗小姐一起去喝奶茶。", "provider": "demo", "create_time": 0},
        "2": {"content": "原来她喜欢阿嬷手作的奶茶，味道好喝。", "provider": "demo", "create_time": 0},
    },
}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --ink: #2b2b2b;
    --line: #3a3a3a;
    --paper: #ffffff;
    --muted: #999999;
    --shadow: 0 10px 30px rgba(0, 0, 0, .12);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; background: var(--paper); color: var(--ink);
    font-family: "Songti SC", "Noto Serif SC", "Source Han Serif SC", Georgia, serif; }
  /* 横向布局：左边画布自适应撑满，右边详情栏常驻 340px */
  #app { position: relative; display: flex; width: 100%; height: 100%; overflow: hidden; }
  header { position: absolute; top: 0; left: 0; right: 340px; height: 58px; display: flex;
    align-items: center; gap: 12px; padding: 0 22px; z-index: 10; pointer-events: none; }
  header .title { font-size: 18px; font-weight: 700; letter-spacing: .5px; pointer-events: auto; }
  header .stats { font-size: 13px; color: var(--muted); pointer-events: auto; }
  header .spacer { flex: 1; }
  header .btn, header select.btn { pointer-events: auto; border: 1.5px solid var(--line);
    background: #fff; border-radius: 10px; padding: 6px 12px; font: inherit; font-size: 13px;
    color: var(--ink); cursor: pointer; }
  header .btn:hover { background: #f6f6f6; }
  #graph { position: relative; flex: 1 1 auto; min-width: 0; height: 100%; background: var(--paper); }
  #graph canvas { display: block; }
  #detail { flex: 0 0 340px; width: 340px; height: 100%; display: flex; flex-direction: column;
    background: var(--paper); border-left: 1.6px solid var(--line); overflow: hidden;
    position: relative; z-index: 30; }
  #detail .d-head { font-size: 17px; font-weight: 700; text-align: center; padding: 16px 12px;
    border-bottom: 1.6px solid var(--line); }
  #detail .d-body { flex: 1; overflow: auto; }
  #detail .d-sec { padding: 10px 15px; border-bottom: 1.4px solid var(--line); font-size: 14px;
    line-height: 1.7; }
  #detail .d-sec:last-child { border-bottom: none; }
  #detail .d-key { color: var(--muted); font-size: 12px; margin-right: 8px; }
  #detail .d-placeholder { padding: 46px 20px; text-align: center; color: var(--muted); font-size: 14px; }
  #detail details { margin-top: 6px; }
  #detail summary { cursor: pointer; color: var(--muted); font-size: 12px; outline: none; }
  #detail .rec { font-size: 12px; color: #444; margin-top: 4px; padding-left: 12px; line-height: 1.7; }
  #detail .chunk-box { max-height: 190px; overflow: auto; margin-top: 4px; }
  #detail .chunk-id { color: var(--muted); margin-right: 4px; }
  #detail .close { display: none; position: absolute; top: 9px; right: 13px; cursor: pointer;
    color: var(--muted); font-size: 18px; line-height: 1; }
  .empty-hint { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; color: var(--muted); font-size: 15px; pointer-events: none; z-index: 5; }
  #notice { position: absolute; left: 50%; top: 70px; transform: translateX(-50%); z-index: 15;
    display: none; background: #fff; border: 1.4px solid var(--line); border-radius: 8px;
    padding: 6px 14px; font-size: 12px; color: #555; max-width: 70%; }
  #communities { position: absolute; top: 72px; left: 22px; width: 272px;
    max-height: calc(100% - 100px); overflow: auto; background: #fff;
    border: 1.8px solid var(--line); border-radius: 12px; box-shadow: var(--shadow);
    z-index: 20; display: none; }
  #communities .c-head { font-size: 15px; font-weight: 700; padding: 10px 15px;
    border-bottom: 1.6px solid var(--line); }
  #communities .c-list { padding: 4px 10px 10px; }
  .c-card { margin: 8px 0; border: 1.3px solid #ddd; border-radius: 9px; padding: 7px 11px; }
  .c-card summary { cursor: pointer; outline: none; font-size: 14px; font-weight: 700;
    display: flex; align-items: center; gap: 6px; }
  .c-card .c-name { cursor: pointer; }
  .c-card .c-name:hover { text-decoration: underline; }
  .c-card .c-level { font-size: 11px; font-weight: 400; color: var(--muted); }
  .c-card .c-focus { margin-left: auto; font-size: 11px; font-weight: 400; color: #666;
    border: 1px solid #ddd; border-radius: 6px; padding: 1px 7px; cursor: pointer; }
  .c-card .c-focus:hover { background: #f3f3f3; }
  .c-card .c-body { font-size: 13px; line-height: 1.7; margin-top: 6px; }
  .c-card .c-key { color: var(--muted); font-size: 12px; margin-top: 6px; }
  .c-card .c-members { color: #333; }
  .c-card .rec { font-size: 12px; color: #444; margin-top: 4px; padding-left: 10px; }
  /* 窄屏：详情栏改成覆盖在画布上，不挤压画布宽度 */
  @media (max-width: 900px) {
    header { right: 0; }
    header .stats { display: none; }
    #detail { position: absolute; top: 0; right: 0; bottom: 0; width: min(340px, 86vw);
      height: auto; display: none; box-shadow: var(--shadow);
      border-left: 1.6px solid var(--line); }
    #detail.open { display: flex; }
    #detail .close { display: block; }
    #communities { width: 232px; }
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="title">__TITLE__</div>
    <div class="stats" id="stats"></div>
    <div class="spacer"></div>
    <select class="btn" id="layout-mode" title="布局切换">
      <option value="community">社区分组</option>
      <option value="concentric">同心圆</option>
      <option value="circle">环形</option>
    </select>
    <input type="file" id="file" accept="application/json" style="display:none">
    <button class="btn" id="load">导入 JSON</button>
    <button class="btn" id="relayout">重新布局</button>
    <button class="btn" id="fit">适应视图</button>
  </header>
  <div id="graph"></div>
  <div id="communities">
    <div class="c-head">社区 <span id="c-count"></span></div>
    <div class="c-list" id="c-list"></div>
  </div>
  <div id="notice"></div>
  <div id="detail">
    <span class="close">×</span>
    <div class="d-head">详情</div>
    <div class="d-body"><div class="d-placeholder">点一个节点看详情</div></div>
  </div>
</div>

__SCRIPTS__
<script>
(function () {
  const graphEl = document.getElementById("graph");
  function showNotice(text) {
    const el = document.getElementById("notice");
    el.textContent = text;
    el.style.display = "block";
  }
  if (typeof cytoscape === "undefined") {
    graphEl.insertAdjacentHTML("beforeend",
      '<div class="empty-hint">cytoscape 加载失败，请联网后刷新页面</div>');
    return;
  }
  // fcose 复合布局插件；没加载到就退回内置 cose，至少图还能出来
  const HAS_FCOSE = typeof cytoscapeFcose !== "undefined";
  if (HAS_FCOSE) cytoscape.use(cytoscapeFcose);

  let DATA = __DATA__;
  // 实时模式：非 null 时轮询这个路径的 JSON，数据一变就重绘
  const LIVE_URL = __LIVE__;
  let cy = null;
  let mode = "community";  // community | concentric | circle
  // communities.members 存的是名字，这里维护 名字 → cytoscape 节点 id
  let nameToId = new Map();

  function nid(id) { return "n" + String(id); }
  function eid(id) { return "e" + String(id); }

  function textWidth(s) {
    let w = 0;
    for (const ch of String(s)) w += /[\u4e00-\u9fff]/.test(ch) ? 16 : 8.2;
    return w;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // 磁盘上存的是完整图结构（format: momo-graph），导入时要先翻译成视图结构
  function fromGraphPayload(p) {
    const entities = (p.entities || []).map(e => {
      const idx = e.index || {};
      const name = idx.name || "";
      return {
        id: idx.unique_id,
        name: name,
        aliases: (idx.relate_name || []).filter(a => a && a !== name),
        description: e.description || "",
        entity_type: e.entity_type || "",
        attributes: (e.attributes || []).map(a => {
          const items = (a.items || []).filter(i => i && i.content).map(i => ({
            content: i.content,
            create_time: i.create_time || 0,
            activate_time: i.activate_time || [],
          }));
          return { name: (a.index || {}).name || "", values: items.map(i => i.content), items: items };
        }).filter(a => a.values.length),
        chunks: idx.relate_memory_chunk_id || [],
      };
    });
    const edges = (p.edges || []).map(e => ({
      id: (e.index || {}).unique_id,
      source: e.source_entity_id,
      target: e.target_entity_id,
      name: (e.index || {}).name || "",
      weight: e.weight,
      chunks: (e.index || {}).relate_memory_chunk_id || [],
      activate_time: e.activate_time || [],
      create_time: e.create_time || 0,
    }));
    const chunks = {};
    (p.memory_chunks || []).forEach(c => {
      chunks[String(c.unique_id)] = {
        content: c.content || "",
        provider: c.provider || "",
        create_time: c.create_time || 0,
      };
    });
    // 社区（Party）：解散的默认不显示
    const nameOf = new Map(entities.map(e => [e.id, e.name]));
    const communities = (p.parties || [])
      .filter(pt => pt.active !== false)
      .map(pt => {
        const history = (pt.description_record || []).map(d => ({
          content: d.content || "",
          create_time: d.create_time || 0,
        }));
        return {
          id: pt.id,
          name: pt.name || "",
          level: pt.level || 0,
          members: (pt.entity_ids || []).map(id => nameOf.get(id) || ("#" + id)),
          summary: history.length ? history[history.length - 1].content : "",
          history: history,
        };
      });
    return {
      title: "MomoGraph · 图谱预览",
      entities: entities,
      edges: edges,
      chunks: chunks,
      communities: communities,
      stats: {
        entities: entities.length,
        edges: edges.length,
        chunks: (p.memory_chunks || []).length,
        parties: communities.length,
      },
    };
  }

  const SERIF = '"Songti SC","Noto Serif SC","Source Han Serif SC",Georgia,serif';
  // 黑白报刊感的样式表：白底、3a3a3a 描边、宋体字
  const STYLE = [
    { selector: "node", style: {
      "shape": "round-rectangle",
      "background-color": "#ffffff",
      "border-width": 1.8,
      "border-color": "#3a3a3a",
      "label": "data(label)",
      "font-family": SERIF,
      "font-size": 12,
      "color": "#2b2b2b",
      "text-valign": "center",
      "text-halign": "center",
    } },
    // 叶子节点（实体）才有固定尺寸；复合节点由 cytoscape 按子节点自动撑开
    { selector: "node:childless", style: {
      "width": "data(w)",
      "height": "data(h)",
      "text-wrap": "ellipsis",
      "text-max-width": "data(tw)",
    } },
    { selector: "node:selected", style: {
      "border-width": 3,
      "background-color": "#f3f3ef",
    } },
    { selector: "node.community", style: {
      "background-color": "#fafaf7",
      "background-opacity": 0.9,
      "border-width": 1.2,
      "border-color": "#9a9a9a",
      "border-style": "dashed",
      "padding": "18px",
      "font-size": 11,
      "font-weight": "bold",
      "color": "#7a7a7a",
      "text-valign": "top",
      "text-halign": "left",
      "text-margin-x": 8,
      "text-margin-y": 6,
    } },
    { selector: "node.community:selected", style: {
      "border-width": 2.4,
      "border-color": "#3a3a3a",
    } },
    { selector: "node.faded", style: {
      "opacity": 0.15,
      "text-opacity": 0.15,
    } },
    // 边：贝塞尔 + 箭头，关系名带白底，图密集时也能读
    { selector: "edge", style: {
      "curve-style": "bezier",
      "line-color": "#3a3a3a",
      "target-arrow-color": "#3a3a3a",
      "target-arrow-shape": "triangle",
      "arrow-scale": 0.7,
      "width": 1.4,
      "label": "data(label)",
      "font-family": SERIF,
      "font-size": 9,
      "color": "#222222",
      "text-rotation": "none",
      "text-background-color": "#ffffff",
      "text-background-opacity": 1,
      "text-background-padding": "2px",
      "text-background-shape": "round-rectangle",
    } },
    { selector: "edge:selected", style: {
      "width": 2.6,
      "line-color": "#000000",
      "target-arrow-color": "#000000",
    } },
    { selector: "edge.faded", style: {
      "opacity": 0.15,
      "text-opacity": 0.15,
    } },
  ];

  // 布局参数：默认社区分组（fcose + 复合节点）
  function layoutOptions() {
    if (mode === "concentric") {
      return {
        name: "concentric", animate: false, fit: true, padding: 40, avoidOverlap: true,
        concentric: node => node.data("degree") || 0,
        levelWidth: () => 2, minNodeSpacing: 24,
      };
    }
    if (mode === "circle") {
      return { name: "circle", animate: false, fit: true, padding: 40, spacingFactor: 1.2 };
    }
    if (!HAS_FCOSE) {
      showNotice("fcose 布局加载失败，已退回内置力导向布局；联网后刷新可恢复");
      return {
        name: "cose", animate: false, fit: true, padding: 40,
        nodeRepulsion: 20000, idealEdgeLength: 90, gravity: 0.3, numIter: 2000,
      };
    }
    return {
      name: "fcose", quality: "default", animate: false, fit: true, padding: 40,
      randomize: true, idealEdgeLength: 90, edgeElasticity: 0.45,
      gravity: 0.3, gravityRange: 3.8, numIter: 2500,
      nodeSeparation: 90, packComponents: true,
      // 度数大的 hub 推得远一点，小节点之间也要分得开
      nodeRepulsion: node => {
        const d = node.data("degree") || 0;
        return d >= 20 ? 26000 : (d >= 5 ? 14000 : 9000);
      },
    };
  }

  // 实体按度数分 3 档尺寸，hub 一眼就能看到
  function nodeMetrics(name, degree) {
    const base = Math.max(46, Math.min(140, textWidth(name) + 18));
    if (degree >= 20) return { w: Math.round(Math.min(140, base * 1.35)), h: 34 };
    if (degree >= 5) return { w: Math.round(Math.min(140, base * 1.15)), h: 32 };
    return { w: Math.round(Math.max(46, base)), h: 30 };
  }

  // 把视图数据翻译成 cytoscape 元素；社区分组模式下用 level=0 的社区当复合节点
  function buildElements(data) {
    const entities = (data.entities || []).filter(e => e && e.id !== undefined && e.id !== null);
    const byId = new Map(entities.map(e => [String(e.id), e]));
    nameToId = new Map();
    entities.forEach(e => { if (e.name) nameToId.set(e.name, nid(e.id)); });

    const rawEdges = (data.edges || [])
      .filter(l => l && byId.has(String(l.source)) && byId.has(String(l.target)));

    const degree = new Map();
    entities.forEach(e => degree.set(nid(e.id), 0));
    rawEdges.forEach(l => {
      const s = nid(l.source), t = nid(l.target);
      degree.set(s, (degree.get(s) || 0) + 1);
      degree.set(t, (degree.get(t) || 0) + 1);
    });

    const elements = [];
    const parentOf = new Map();
    if (mode === "community") {
      (data.communities || []).forEach(c => {
        // 只把 level 0 的社区画成组；更上层社区会嵌套爆炸，留给左侧卡片
        if ((c.level || 0) !== 0 || c.active === false) return;
        const ids = [];
        (c.members || []).forEach(name => {
          const id = nameToId.get(name);
          // 一个实体只能挂一个父节点
          if (id && !parentOf.has(id) && ids.indexOf(id) < 0) ids.push(id);
        });
        if (ids.length < 2) return;
        const pid = "c" + String(c.id);
        ids.forEach(id => parentOf.set(id, pid));
        elements.push({ group: "nodes", data: {
          id: pid,
          label: (c.name || ("社区 " + c.id)) + " · " + ids.length,
          isCommunity: true,
          members: ids,
          degree: 0,
        } });
      });
    }

    entities.forEach(e => {
      const id = nid(e.id);
      const deg = degree.get(id) || 0;
      const metrics = nodeMetrics(e.name || ("实体" + e.id), deg);
      const nodeData = {
        id: id,
        label: e.name || ("实体" + e.id),
        name: e.name || ("实体" + e.id),
        aliases: e.aliases || [],
        description: e.description || "",
        entity_type: e.entity_type || "",
        attributes: (e.attributes || []).filter(a => a && (a.values || []).length),
        chunks: e.chunks || [],
        degree: deg,
        w: metrics.w,
        h: metrics.h,
        tw: Math.max(30, metrics.w - 12),
      };
      if (parentOf.has(id)) nodeData.parent = parentOf.get(id);
      elements.push({ group: "nodes", data: nodeData });
    });

    rawEdges.forEach((l, i) => {
      elements.push({ group: "edges", data: {
        id: eid(l.id === undefined || l.id === null ? "x" + i : l.id),
        source: nid(l.source),
        target: nid(l.target),
        label: l.name || "",
        name: l.name || "",
        weight: l.weight,
        chunks: l.chunks || [],
        activate_time: l.activate_time || [],
        create_time: l.create_time || 0,
      } });
    });

    return { entities, elements };
  }

  function fmtTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    const p = n => String(n).padStart(2, "0");
    return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
      " " + p(d.getHours()) + ":" + p(d.getMinutes());
  }

  function openPanel(title) {
    const panel = document.getElementById("detail");
    panel.querySelector(".d-head").textContent = title || "详情";
    const body = panel.querySelector(".d-body");
    body.innerHTML = "";
    body.scrollTop = 0;
    // 宽屏本来就常驻；窄屏靠这个类把它覆盖显示出来
    panel.classList.add("open");
    return body;
  }

  function addSection(body, html) {
    const sec = document.createElement("div");
    sec.className = "d-sec";
    sec.innerHTML = html;
    body.appendChild(sec);
    return sec;
  }

  function chunkText(id) {
    const c = (DATA.chunks || {})[String(id)];
    return c && c.content ? c.content : "";
  }

  // 关联记忆：折叠展示，展开后是原始文本，方便核对抽取有没有错
  function addChunksSection(body, ids) {
    const sec = document.createElement("div");
    sec.className = "d-sec";
    if (!ids || !ids.length) {
      sec.innerHTML = '<span class="d-key">关联记忆</span>—';
      body.appendChild(sec);
      return;
    }
    const details = document.createElement("details");
    details.open = true;  // 默认展开，直接看到原文
    const summary = document.createElement("summary");
    summary.textContent = "关联记忆 (" + ids.length + ")";
    details.appendChild(summary);
    const box = document.createElement("div");
    box.className = "chunk-box";
    ids.forEach(id => {
      const rec = document.createElement("div");
      rec.className = "rec";
      rec.innerHTML = '<span class="chunk-id">#' + escapeHtml(String(id)) + "</span>" +
        escapeHtml(chunkText(id) || "（原文缺失）");
      box.appendChild(rec);
    });
    details.appendChild(box);
    sec.appendChild(details);
    body.appendChild(sec);
  }

  function showNodeDetail(d) {
    const body = openPanel(d.name);
    addSection(body, '<span class="d-key">描述</span>' +
      (d.description ? escapeHtml(d.description) : "—") +
      (d.entity_type ? ' <span class="d-key">· ' + escapeHtml(d.entity_type) + "</span>" : ""));
    addSection(body, '<span class="d-key">别名</span>' +
      (d.aliases.length ? escapeHtml(d.aliases.join("，")) : "—"));
    if (!d.attributes.length) {
      addSection(body, '<span class="d-key">属性</span>—');
    } else {
      d.attributes.forEach(a => {
        const sec = addSection(body, '<span class="d-key">' + escapeHtml(a.name || "属性") +
          "</span>" + escapeHtml(a.values.join("、")));
        const items = a.items || [];
        if (!items.length) return;
        // 属性值的历史记录收在折叠栏里，平时不占地方
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "激活记录 (" + items.length + ")";
        details.appendChild(summary);
        items.forEach(item => {
          const rec = document.createElement("div");
          rec.className = "rec";
          const acts = (item.activate_time || []).map(fmtTime);
          rec.innerHTML = escapeHtml(item.content) + " · 创建 " + fmtTime(item.create_time) +
            (acts.length ? " · 激活 " + escapeHtml(acts.join("、")) : " · 激活 —");
          details.appendChild(rec);
        });
        sec.appendChild(details);
      });
    }
    addChunksSection(body, d.chunks);
  }

  function showEdgeDetail(edge) {
    const d = edge.data();
    const s = edge.source(), t = edge.target();
    const body = openPanel(d.name || "关系");
    addSection(body, '<span class="d-key">方向</span>' +
      escapeHtml((s ? s.data("name") : "?") + " → " + (t ? t.data("name") : "?")));
    addSection(body, '<span class="d-key">权重</span>' +
      (d.weight === undefined ? "—" : escapeHtml(String(d.weight))));
    addSection(body, '<span class="d-key">创建</span>' + fmtTime(d.create_time));
    const acts = (d.activate_time || []).map(fmtTime);
    addSection(body, '<span class="d-key">激活</span>' +
      (acts.length ? escapeHtml(acts.join("、")) : "—"));
    addChunksSection(body, d.chunks);
  }

  // 悬停节点：其它元素淡出，只留节点自己、它的边和邻居
  function highlightNode(node) {
    if (!cy) return;
    cy.batch(() => {
      cy.elements().addClass("faded");
      node.removeClass("faded");
      node.connectedEdges().removeClass("faded");
      node.connectedEdges().connectedNodes().removeClass("faded");
      const parent = node.parent();
      if (parent && parent.length) parent.removeClass("faded");
    });
  }

  function clearHighlight() {
    if (!cy) return;
    cy.elements().removeClass("faded");
  }

  // 重绘时先销毁旧实例，避免 live 刷新叠出多个 cy、事件重复绑定
  function render(data) {
    if (cy) { cy.destroy(); cy = null; }
    const hint = graphEl.querySelector(".empty-hint");
    if (hint) hint.remove();

    const built = buildElements(data);
    if (!built.entities.length) {
      graphEl.insertAdjacentHTML("beforeend",
        '<div class="empty-hint">图里还没有实体，先 remember 点东西吧</div>');
      return;
    }

    cy = cytoscape({
      container: graphEl,
      elements: built.elements,
      style: STYLE,
      layout: layoutOptions(),
      wheelSensitivity: 0.25,
      minZoom: 0.1,
      maxZoom: 3,
      boxSelectionEnabled: false,
    });
    bindEvents();
  }

  function bindEvents() {
    cy.on("tap", "node", event => {
      const node = event.target;
      // 复合节点没有实体详情，点它就把成员框出来
      if (node.data("isCommunity")) { focusNodes(node.data("members") || []); return; }
      showNodeDetail(node.data());
    });
    cy.on("tap", "edge", event => showEdgeDetail(event.target));
    cy.on("mouseover", "node", event => {
      const node = event.target;
      if (node.data("isCommunity")) return;
      highlightNode(node);
    });
    cy.on("mouseout", "node", () => clearHighlight());
    // 双击画布空白处：恢复所有元素
    cy.on("dbltap", event => { if (event.target === cy) clearHighlight(); });
  }

  function focusNodes(ids) {
    if (!cy || !ids || !ids.length) return;
    const eles = cy.collection();
    ids.forEach(id => {
      const node = cy.getElementById(id);
      if (node && node.length) eles.merge(node);
    });
    if (!eles.length) return;
    cy.elements().unselect();
    eles.select();
    cy.fit(eles, 70);
    clearHighlight();
  }

  function focusMembers(names) {
    if (!cy) return;
    focusNodes((names || []).map(name => nameToId.get(name)).filter(Boolean));
  }

  // 社区卡片：默认收起，展开后看成员和摘要历史
  function renderCommunities() {
    const box = document.getElementById("communities");
    const list = document.getElementById("c-list");
    const items = DATA.communities || [];
    document.getElementById("c-count").textContent = items.length ? "(" + items.length + ")" : "";
    list.innerHTML = "";
    if (!items.length) {
      box.style.display = "none";
      return;
    }
    box.style.display = "block";
    items.forEach(c => {
      const card = document.createElement("details");
      card.className = "c-card";
      const head = document.createElement("summary");
      head.innerHTML = '<span class="c-name">' + escapeHtml(c.name || ("社区 " + c.id)) + "</span>" +
        '<span class="c-level">L' + (c.level || 0) + "</span>" +
        '<span class="c-focus">定位</span>';
      // 点社区名或“定位”：选中该社区成员并 fit 到它们
      const focus = event => {
        event.preventDefault();
        event.stopPropagation();
        focusMembers(c.members);
      };
      head.querySelector(".c-name").addEventListener("click", focus);
      head.querySelector(".c-focus").addEventListener("click", focus);
      card.appendChild(head);

      const body = document.createElement("div");
      body.className = "c-body";
      body.innerHTML = '<div class="c-sum">' +
        escapeHtml(c.summary || "（还没有摘要）") + "</div>" +
        '<div class="c-key">成员 (' + (c.members || []).length + ")</div>" +
        '<div class="c-members">' + escapeHtml((c.members || []).join("、") || "—") + "</div>";

      const history = c.history || [];
      if (history.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "摘要历史 (" + history.length + ")";
        details.appendChild(summary);
        // 最新的排最前
        history.slice().reverse().forEach(record => {
          const rec = document.createElement("div");
          rec.className = "rec";
          rec.innerHTML = escapeHtml(fmtTime(record.create_time)) + " · " + escapeHtml(record.content);
          details.appendChild(rec);
        });
        body.appendChild(details);
      }
      card.appendChild(body);
      list.appendChild(card);
    });
  }

  function updateStats(data) {
    const s = data.stats || {
      entities: (data.entities || []).length,
      edges: (data.edges || []).length,
      chunks: 0,
    };
    let text = "实体 " + s.entities + " · 关系 " + s.edges + " · 文本块 " + (s.chunks || 0);
    if (s.parties !== undefined) text += " · 社区 " + s.parties;
    document.getElementById("stats").textContent = text;
  }

  function fitView() {
    if (!cy || !cy.nodes().length) return;
    cy.fit(undefined, 40);
  }

  function start() {
    updateStats(DATA);
    render(DATA);
    renderCommunities();
  }

  document.getElementById("fit").onclick = fitView;
  document.getElementById("relayout").onclick = () => {
    if (!cy || !cy.nodes().length) return;
    cy.layout(layoutOptions()).run();
  };
  // 换布局整图重建：只有社区分组模式才挂复合父节点
  document.getElementById("layout-mode").onchange = event => {
    mode = event.target.value;
    clearHighlight();
    render(DATA);
  };
  document.getElementById("load").onclick = () => document.getElementById("file").click();
  document.getElementById("file").onchange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        let data = JSON.parse(reader.result);
        if (data && data.format === "momo-graph") data = fromGraphPayload(data);
        if (!data.entities) throw new Error("缺少 entities 字段");
        DATA = data;
        start();
      } catch (err) {
        alert("JSON 解析失败：" + err.message);
      }
    };
    reader.readAsText(file, "utf-8");
  };
  document.querySelector("#detail .close").onclick = () => {
    document.getElementById("detail").classList.remove("open");
  };

  start();

  if (LIVE_URL) {
    let last = JSON.stringify(DATA);
    const poll = async () => {
      try {
        const res = await fetch(LIVE_URL + "?t=" + Date.now(), { cache: "no-store" });
        if (!res.ok) return;
        const next = await res.json();
        if (!next || !next.entities) return;
        const raw = JSON.stringify(next);
        if (raw === last) return;
        last = raw;
        DATA = next;
        start();
      } catch (err) {
        console.warn("实时数据拉取失败：", err);
      }
    };
    poll();
    setInterval(poll, 1500);
  }
})();
</script>
</body>
</html>
"""


# 图谱引擎和布局插件：优先内联本地 vendor 的文件（看板要能离线双击打开，不依赖 CDN），
# 缺文件时才退回 CDN。顺序不能乱：fcose 依赖 cose-base，cose-base 依赖 layout-base。
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
VENDOR_SCRIPTS = [
    ("cytoscape.min.js", "https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"),
    ("layout-base.js", "https://cdn.jsdelivr.net/npm/layout-base@2.0.1/layout-base.js"),
    ("cose-base.js", "https://cdn.jsdelivr.net/npm/cose-base@2.2.0/cose-base.js"),
    ("cytoscape-fcose.js", "https://cdn.jsdelivr.net/npm/cytoscape-fcose@2.2.0/cytoscape-fcose.js"),
]


def _scripts_block() -> str:
    """把 vendor 里的 JS 内联进页面；读不到就退回 CDN 标签。"""
    tags = []
    for filename, cdn in VENDOR_SCRIPTS:
        try:
            with open(os.path.join(VENDOR_DIR, filename), "r", encoding="utf-8") as handle:
                code = handle.read()
        except OSError:
            tags.append(f'<script src="{cdn}"></script>')
            continue
        # 源码里出现 </script 会提前闭合标签
        tags.append("<script>" + code.replace("</script", "<\\/script") + "</script>")
    return "\n".join(tags)


def render_dashboard(data: dict, title: str | None = None, live_url: str | None = None) -> str:
    if title:
        data = {**data, "title": title}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    title_text = data.get("title", "MomoGraph · 图谱预览")
    live = "null" if not live_url else json.dumps(live_url, ensure_ascii=False)
    return (
        HTML_TEMPLATE.replace("__DATA__", payload)
        .replace("__LIVE__", live)
        .replace("__TITLE__", title_text)
        .replace("__SCRIPTS__", _scripts_block())
    )


def build_dashboard(data, out_path: str) -> str:
    """data 可以是 Graph 对象，也可以是 graph_to_dict 产出的 dict。"""
    if not isinstance(data, dict):
        data = graph_to_dict(data)
    html = render_dashboard(data)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return out_path


def _load_dashboard_data(json_path: str) -> dict:
    """从磁盘读文件，还原成 Graph 再转前端视图；旧版 dashboard 快照则原样返回。"""
    if not os.path.exists(json_path):
        return EMPTY_DASHBOARD
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return EMPTY_DASHBOARD

    if is_graph_payload(payload):
        return graph_to_dict(decode_graph(payload))
    if isinstance(payload, dict) and "entities" in payload:
        return payload  # 兼容旧版直接存视图的文件
    return EMPTY_DASHBOARD


def serve_dashboard(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """实时模式：起一个本地服务，页面每 1.5 秒拉一次 /graph_view。

    为什么必须起服务：浏览器在 file:// 下不允许 fetch 本地 JSON（CORS 限制），
    所以实时刷新只能走 http。
    /graph_view 是动态端点：磁盘上是完整 Graph，浏览器要的是视图，
    转换放在服务端做，这样存储只有一份、无需冗余字段。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)  # MomoGraph/ 作为服务根目录
    json_path = os.path.join(root, "Temp", "graph.json")
    out = os.path.join(here, "dashboard_live.html")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root, **kwargs)

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/graph_view":
                body = json.dumps(_load_dashboard_data(json_path), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *args):
            pass  # 轮询很频繁，不打日志

    # 初次打开时内嵌当前数据，之后靠 /graph_view 增量刷新
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(render_dashboard(_load_dashboard_data(json_path), live_url="/graph_view"))

    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/DashBoard/dashboard_live.html"
    print(f"实时看板：{url}  （Ctrl+C 退出）")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    if "--live" in sys.argv:
        live_port = 8765
        if "--port" in sys.argv:
            live_port = int(sys.argv[sys.argv.index("--port") + 1])
        serve_dashboard(port=live_port)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "dashboard.html")
        build_dashboard(DEMO_DATA, out)
        print("已生成:", out)
        webbrowser.open("file:///" + out.replace("\\", "/"))
