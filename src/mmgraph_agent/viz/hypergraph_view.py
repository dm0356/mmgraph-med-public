"""Interactive, self-contained hypergraph viewer.

Adapted from the canvas force-directed viewer in
``ai-knowledge-graph`` (Apache-2.0), reworked for *n-ary multimodal hypergraphs*:
the page renders a bipartite graph of

* **entity** nodes (blue),
* **unit / hyperedge** nodes, coloured by modality (text / table / diagram),
* **image** nodes (green),

with membership edges (entity–unit) and modality edges (unit–image, dashed).
The output is a single HTML file with embedded JSON: no external assets, so it
can be opened offline, attached to slides, or screenshotted for a paper.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..hypergraph import Store

_MAX_LABEL = 18


def _short(text: str, n: int = _MAX_LABEL) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_view_data(store: Store) -> dict:
    """Turn a :class:`Store` into ``{nodes, links, meta}`` for the viewer."""
    nodes: list[dict] = []
    links: list[dict] = []

    for ent in store.entities:
        nodes.append(
            {
                "id": ent["id"],
                "type": "entity",
                "label": _short(ent["name"]),
                "desc": (ent.get("description") or "")[:400],
                "kind": ent.get("type", "concept"),
            }
        )

    for edge in store.hyperedges:
        nodes.append(
            {
                "id": edge["id"],
                "type": "unit",
                "modality": edge["modality"],
                "label": _short(edge["text"]),
                "desc": (edge["text"] or "")[:800],
                "doc_id": edge.get("doc_id", ""),
            }
        )

    for image in store.images:
        nodes.append(
            {
                "id": image["id"],
                "type": "image",
                "label": _short(image.get("caption", "")),
                "desc": (image.get("caption") or "")[:400],
                "doc_id": image.get("doc_id", ""),
            }
        )

    node_ids = {node["id"] for node in nodes}
    for edge in store.hyperedges:
        for entity_id in edge.get("entities", []):
            if entity_id in node_ids:
                links.append({"source": entity_id, "target": edge["id"], "kind": "member"})
        for image_id in edge.get("images", []):
            if image_id in node_ids:
                links.append({"source": edge["id"], "target": image_id, "kind": "modality"})

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "entities": len(store.entities),
            "units": len(store.hyperedges),
            "images": len(store.images),
            "edges": len(links),
        },
    }


def render_hypergraph_html(store: Store, output_path: str | Path, title: str = "Multimodal Knowledge Hypergraph") -> dict:
    """Render ``store`` to a standalone interactive HTML file; return stats."""
    data = build_view_data(store)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return data["meta"]


def render_store_file(store_path: str | Path, output_path: str | Path, title: str = "Multimodal Knowledge Hypergraph") -> dict:
    from ..hypergraph import Store as _Store

    return render_hypergraph_html(_Store.load(store_path), output_path, title)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f7f8fa; --fg:#222; }
  * { box-sizing: border-box; }
  body { margin:0; overflow:hidden; font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",sans-serif; background:var(--bg); color:var(--fg); }
  #c { display:block; width:100vw; height:100vh; }
  #tip { position:fixed; pointer-events:none; background:rgba(30,30,30,.92); color:#fff; padding:7px 10px; border-radius:6px; font-size:12px; max-width:360px; display:none; z-index:9; white-space:pre-wrap; }
  #panel { position:fixed; left:12px; top:12px; background:rgba(255,255,255,.96); border:1px solid #e2e5ea; border-radius:10px; padding:12px 14px; font-size:13px; width:310px; max-height:86vh; overflow:auto; z-index:8; box-shadow:0 4px 18px rgba(0,0,0,.08); }
  #panel h2 { margin:0 0 4px; font-size:15px; }
  .dim { color:#888; font-size:12px; }
  .legend { display:flex; flex-wrap:wrap; gap:7px 10px; margin:8px 0; }
  .legend label { display:inline-flex; align-items:center; gap:4px; font-size:12px; cursor:pointer; }
  .sw { width:10px; height:10px; border-radius:50%; display:inline-block; flex:none; }
  #search { width:100%; padding:6px 8px; border:1px solid #d5dae0; border-radius:6px; font-size:12px; margin:6px 0; }
  #detail { margin-top:10px; border-top:1px dashed #e2e5ea; padding-top:10px; font-size:12px; line-height:1.6; }
  #detail h3 { margin:0 0 6px; font-size:13px; word-break:break-word; }
  #detail .tag { display:inline-block; padding:1px 6px; border-radius:10px; background:#eef1f5; font-size:11px; margin:0 4px 4px 0; }
  #detail p { margin:4px 0; word-break:break-word; }
  #detail ul { margin:4px 0 0; padding-left:16px; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="tip"></div>
<div id="panel">
  <h2>__TITLE__</h2>
  <div class="dim" id="stat"></div>
  <div class="legend" id="legend"></div>
  <input id="search" placeholder="搜索实体 / 单元 / 图像…">
  <div class="dim">拖动节点 · 滚轮缩放 · 空白拖动平移 · 点击看详情</div>
  <div id="detail"></div>
</div>
<script>
const DATA = __DATA__;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const tip = document.getElementById('tip');
const detail = document.getElementById('detail');
const stat = document.getElementById('stat');
const search = document.getElementById('search');
const legend = document.getElementById('legend');

const nodes = DATA.nodes.map(n => ({...n, x:(Math.random()-0.5)*600, y:(Math.random()-0.5)*400, vx:0, vy:0, deg:0}));
const byId = {}; nodes.forEach(n => byId[n.id] = n);
const links = DATA.links.filter(l => byId[l.source] && byId[l.target]).map(l => ({...l, s:byId[l.source], t:byId[l.target]}));
links.forEach(l => { l.s.deg++; l.t.deg++; });
const adj = {}; nodes.forEach(n => adj[n.id] = new Set());
links.forEach(l => { adj[l.s.id].add(l.t.id); adj[l.t.id].add(l.s.id); });

stat.textContent = DATA.meta.entities + ' 个实体 · ' + DATA.meta.units + ' 个知识单元 · ' + DATA.meta.images + ' 个图像 · ' + DATA.meta.edges + ' 条关系';

const MODALITY_COLORS = { text:'#f28e2b', table:'#76b7b2', diagram:'#b07aa1' };
function color(n){
  if (n.type === 'image') return '#59a14f';
  if (n.type === 'unit') return MODALITY_COLORS[n.modality] || '#f28e2b';
  return '#4e79a7';
}

const filters = { entity:true, unit:true, image:true, text:true, table:true, diagram:true };
function active(n){
  if (n.type === 'entity') return filters.entity;
  if (n.type === 'image') return filters.image;
  return filters.unit && filters[n.modality] !== false;
}

[['entity','#4e79a7','实体'],['text','#f28e2b','文本单元'],['table','#76b7b2','表格单元'],['diagram','#b07aa1','图表单元'],['image','#59a14f','图像']].forEach(([key,col,label]) => {
  const el = document.createElement('label');
  el.innerHTML = '<input type="checkbox" data-key="' + key + '" checked><span class="sw" style="background:' + col + '"></span>' + label;
  legend.appendChild(el);
});
legend.addEventListener('change', e => {
  const key = e.target.dataset.key;
  if (!key) return;
  filters[key] = e.target.checked;
  draw();
});

let query = null;
search.addEventListener('input', () => { query = search.value.trim().toLowerCase() || null; draw(); });
function matches(n){ return !!(query && (n.label || '').toLowerCase().includes(query)); }

let alpha = 1, T = {s:1, tx:0, ty:0}, hover = null, selected = null, dragNode = null, panning = false, moved = 0, lastMouse = {x:0, y:0};

function resize(){ canvas.width = innerWidth; canvas.height = innerHeight; fit(); draw(); }
addEventListener('resize', resize);

function fit(){
  const vis = nodes.filter(active);
  if (!vis.length) return;
  let minX=1e9, minY=1e9, maxX=-1e9, maxY=-1e9;
  vis.forEach(n => { minX=Math.min(minX,n.x); maxX=Math.max(maxX,n.x); minY=Math.min(minY,n.y); maxY=Math.max(maxY,n.y); });
  if (maxX-minX < 1 || maxY-minY < 1) return;
  const s = Math.min(innerWidth/(maxX-minX), innerHeight/(maxY-minY)) * 0.8;
  T.s = s; T.tx = innerWidth/2 - (minX+maxX)/2*s; T.ty = innerHeight/2 - (minY+maxY)/2*s;
}
const toScreen = p => ({x:p.x*T.s+T.tx, y:p.y*T.s+T.ty});
const toWorld  = p => ({x:(p.x-T.tx)/T.s, y:(p.y-T.ty)/T.s});

function step(){
  const vis = nodes.filter(active);
  for (let i=0;i<vis.length;i++) for (let j=i+1;j<vis.length;j++){
    const a=vis[i], b=vis[j];
    let dx=a.x-b.x, dy=a.y-b.y, d2=dx*dx+dy*dy;
    if (d2 < 0.01){ dx=Math.random()-0.5; dy=Math.random()-0.5; d2=1; }
    const f=Math.min(2600/d2,40)*alpha, d=Math.sqrt(d2);
    a.vx+=dx/d*f; a.vy+=dy/d*f; b.vx-=dx/d*f; b.vy-=dy/d*f;
  }
  for (const l of links){
    if (!active(l.s) || !active(l.t)) continue;
    const dx=l.t.x-l.s.x, dy=l.t.y-l.s.y, d=Math.sqrt(dx*dx+dy*dy)||1;
    const f=(d-120)*0.03*alpha;
    l.s.vx+=dx/d*f; l.s.vy+=dy/d*f; l.t.vx-=dx/d*f; l.t.vy-=dy/d*f;
  }
  for (const n of nodes){
    if (!active(n)) continue;
    n.vx += -n.x*0.012*alpha; n.vy += -n.y*0.012*alpha;
    if (n !== dragNode){ n.vx*=0.8; n.vy*=0.8; n.x+=n.vx; n.y+=n.vy; }
  }
  alpha *= 0.985;
  fit(); draw();
  if (alpha > 0.02) requestAnimationFrame(step);
}

function radius(n){
  if (n.type === 'entity') return 6 + Math.min(n.deg,12)*0.7;
  if (n.type === 'image') return 5 + Math.min(n.deg,8)*0.4;
  return 5 + Math.min(n.deg,12)*0.45;
}

function draw(){
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.setTransform(T.s,0,0,T.s,T.tx,T.ty);
  const focus = hover || selected;
  const focusAdj = focus ? adj[focus.id] : null;

  for (const l of links){
    if (!active(l.s) || !active(l.t)) continue;
    const on = focus && (l.s === focus || l.t === focus);
    if (l.kind === 'modality'){
      ctx.strokeStyle = on ? '#59a14f' : (focus ? 'rgba(89,161,79,.15)' : 'rgba(89,161,79,.45)');
      ctx.setLineDash([5/T.s, 4/T.s]);
    } else {
      ctx.strokeStyle = on ? '#999' : (focus ? 'rgba(150,160,170,.18)' : 'rgba(150,160,170,.42)');
      ctx.setLineDash([]);
    }
    ctx.lineWidth = (on ? 2 : 1) / T.s;
    ctx.beginPath(); ctx.moveTo(l.s.x,l.s.y); ctx.lineTo(l.t.x,l.t.y); ctx.stroke();
    ctx.setLineDash([]);
  }

  for (const n of nodes){
    if (!active(n)) continue;
    const dim = (focus && n !== focus && !focusAdj.has(n.id)) || (query && !matches(n));
    ctx.globalAlpha = dim ? 0.16 : 1;
    const r = radius(n);
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI*2);
    ctx.fillStyle = color(n); ctx.fill();
    if (matches(n)){ ctx.lineWidth = 3/T.s; ctx.strokeStyle = '#e0b100'; ctx.stroke(); }
    else if (n === focus){ ctx.lineWidth = 2.5/T.s; ctx.strokeStyle = '#333'; ctx.stroke(); }
    ctx.font = (12/T.s) + 'px "Microsoft YaHei",sans-serif';
    ctx.lineWidth = 3.5/T.s; ctx.strokeStyle = 'rgba(255,255,255,.85)';
    ctx.strokeText(n.label, n.x, n.y + r + 14/T.s);
    ctx.fillStyle = '#222'; ctx.fillText(n.label, n.x, n.y + r + 14/T.s);
    ctx.globalAlpha = 1;
  }
}

function nearest(sp, maxPx){
  let best = null, bd = maxPx;
  for (const n of nodes){
    if (!active(n)) continue;
    const p = toScreen(n), d = Math.hypot(p.x-sp.x, p.y-sp.y);
    if (d < bd){ bd = d; best = n; }
  }
  return best;
}

canvas.addEventListener('mousedown', e => {
  moved = 0; lastMouse = {x:e.clientX, y:e.clientY};
  dragNode = nearest({x:e.clientX, y:e.clientY}, 15);
  if (!dragNode) panning = true;
});
addEventListener('mousemove', e => {
  const dx = e.clientX-lastMouse.x, dy = e.clientY-lastMouse.y;
  moved += Math.abs(dx) + Math.abs(dy);
  if (dragNode){
    const w = toWorld({x:e.clientX, y:e.clientY});
    dragNode.x = w.x; dragNode.y = w.y; draw();
  } else if (panning){
    T.tx += dx; T.ty += dy; lastMouse = {x:e.clientX, y:e.clientY}; draw();
  } else {
    const n = nearest({x:e.clientX, y:e.clientY}, 15);
    if (n !== hover){ hover = n; draw(); }
    if (n){
      tip.style.display = 'block';
      tip.style.left = (e.clientX+14)+'px'; tip.style.top = (e.clientY+14)+'px';
      tip.textContent = n.label + (n.desc ? '\n' + n.desc.slice(0, 180) : '');
      canvas.style.cursor = 'pointer';
    } else { tip.style.display = 'none'; canvas.style.cursor = 'default'; }
  }
  if (dragNode || panning) lastMouse = {x:e.clientX, y:e.clientY};
});
addEventListener('mouseup', e => {
  if (dragNode || panning){
    if (moved < 4 && dragNode){ selected = dragNode; showDetail(dragNode); }
    dragNode = null; panning = false; draw();
  }
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const k = e.deltaY < 0 ? 1.12 : 1/1.12;
  const m = toWorld({x:e.clientX, y:e.clientY});
  T.s *= k; T.tx = e.clientX - m.x*T.s; T.ty = e.clientY - m.y*T.s;
  draw();
}, {passive:false});

function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function showDetail(n){
  const typeLabel = n.type === 'entity' ? '实体' : n.type === 'image' ? '图像' : '知识单元';
  const mod = n.type === 'unit' ? ' · ' + n.modality : '';
  let h = '<h3>' + esc(n.label) + '</h3>';
  h += '<span class="tag">' + typeLabel + mod + '</span>';
  if (n.kind) h += '<span class="tag">' + esc(n.kind) + '</span>';
  if (n.doc_id) h += '<span class="tag">' + esc(n.doc_id) + '</span>';
  if (n.desc) h += '<p>' + esc(n.desc) + '</p>';
  const neighbors = [...adj[n.id]].map(id => byId[id]).filter(Boolean);
  if (neighbors.length){
    h += '<p class="dim">关联：</p><ul>';
    neighbors.forEach(m => { h += '<li>' + esc(m.label) + '</li>'; });
    h += '</ul>';
  }
  detail.innerHTML = h;
  draw();
}

resize();
step();
</script>
</body>
</html>
"""
