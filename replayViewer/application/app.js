// 太郎リプレイビューア v2 — ダークHUD風。素のJS+SVG。
// 体ビュー：臓器は形で描き、ラベルは左右の余白に逃がして重ならないようにする。
// ネットワーク図：向き付き（矢印＋流れアニメ）で情報の流れを示す。

// 体ビュー：mx,my=体上のマーカー位置／side=ラベルを出す側／ly=ラベルのy／
// organ=専用の形を持つ(HTMLに描画済み)ならtrue、なければ小さな丸マーカー。
const BODY = {
  cortex:        { label:"大脳皮質", mx:240, my:52,  side:"L", ly:46,  organ:true  },
  critic:        { label:"クリティック", mx:224, my:56, side:"L", ly:92, organ:false },
  basal_ganglia: { label:"基底核",   mx:238, my:64,  side:"L", ly:138, organ:false },
  insula:        { label:"島皮質",   mx:240, my:126, side:"L", ly:186, organ:false },
  lungs:         { label:"肺",       mx:222, my:152, side:"L", ly:232, organ:true  },
  hippocampus:   { label:"海馬",     mx:256, my:56,  side:"R", ly:46,  organ:false },
  cerebellum:    { label:"小脳",     mx:252, my:82,  side:"R", ly:92,  organ:true  },
  locus:         { label:"青斑核",   mx:240, my:98,  side:"R", ly:138, organ:false },
  vocal:         { label:"声道",     mx:240, my:110, side:"R", ly:186, organ:true  },
  stomach:       { label:"胃",       mx:252, my:196, side:"R", ly:232, organ:true  },
};

// ネットワーク図：情報の流れに沿った配置。大脳皮質→声道が島皮質を貫かない
// よう、島皮質を左下・声道を右下に振り分ける。上り/下りは二車線で分ける。
const NET = {
  cerebellum:    { label:"小脳",   x:230, y:38  },
  locus:         { label:"青斑核", x:95,  y:58  },
  critic:        { label:"クリティック", x:372, y:58  },
  hippocampus:   { label:"海馬",   x:66,  y:128 },
  basal_ganglia: { label:"基底核", x:396, y:128 },
  cortex:        { label:"大脳皮質", x:230, y:120 },
  insula:        { label:"島皮質", x:158, y:206 },
  vocal:         { label:"声道",   x:344, y:212 },
  stomach:       { label:"胃",     x:96,  y:262 },
  lungs:         { label:"肺",     x:250, y:266 },
};
const NET_EDGES = [
  ["stomach","insula"], ["lungs","insula"], ["insula","cortex"],
  ["cortex","vocal"], ["locus","cortex"], ["cortex","critic"],
  ["hippocampus","cortex"], ["cortex","basal_ganglia"], ["cerebellum","cortex"],
  ["insula","critic"],
];

const GAUGES = [
  { key:"hunger", label:"空腹" }, { key:"ne", label:"探索(NE)" },
  { key:"dopamine", label:"ドーパミン分泌量" }, { key:"happiness", label:"幸福度" },
];

const SVGNS = "http://www.w3.org/2000/svg";
const TOTAL_T = 2 * 365 * 86400;
let moments = [], idx = 0, timer = null, speed = 1;

const el = id => document.getElementById(id);
function mk(name, attrs, parent) {
  const e = document.createElementNS(SVGNS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function fmtDate(t) {
  const days = Math.floor(t / 86400), y = Math.floor(days / 365), rem = days % 365;
  return `${y}年 ${Math.floor(rem / 30)}か月 ${rem % 30}日`;
}
function cx(id){ return NET[id].x; } function cy(id){ return NET[id].y; }
// 線分の両端を縮め（矢印がノード/臓器に被らないよう手前で止め）、
// 進行方向の右側へ off だけずらす（＝上り線と下り線が別々の車線に分かれる）。
function seg(x1,y1,x2,y2,p1,p2,off){
  const dx=x2-x1, dy=y2-y1, L=Math.hypot(dx,dy)||1;
  const ux=dx/L, uy=dy/L;
  off = off || 0;
  const ox=-uy*off, oy=ux*off; // 進行方向の右側
  return [x1+ux*p1+ox, y1+uy*p1+oy, x2-ux*p2+ox, y2-uy*p2+oy];
}

// ---- 体ビュー：ラベル・引き出し線・丸マーカーを生成 ----
function buildBody() {
  const labels = el("bodyLabels"), leaders = el("bodyLeaders"), markers = el("bodyMarkers");
  for (const id in BODY) {
    const m = BODY[id];
    const lx = m.side === "L" ? 14 : 466;
    const anchor = m.side === "L" ? "start" : "end";
    const lineX = m.side === "L" ? 118 : 362;
    mk("line", { x1:lineX, y1:m.ly, x2:m.mx, y2:m.my, class:"leader", id:"leader_"+id }, leaders);
    const t = mk("text", { x:lx, y:m.ly+4, "text-anchor":anchor, class:"blabel", id:"blabel_"+id }, labels);
    t.textContent = m.label;
    if (!m.organ) mk("circle", { cx:m.mx, cy:m.my, r:6, class:"organ marker", id:"organ_"+id }, markers);
  }
}

// ---- ネットワーク図：エッジ・ノード生成 ----
function buildNet() {
  const edges = el("netEdges"), nodes = el("netNodes");
  NET_EDGES.forEach(([a,b]) =>
    mk("line", { x1:cx(a), y1:cy(a), x2:cx(b), y2:cy(b), class:"net-edge",
      "data-edge":a+"|"+b }, edges));
  for (const id in NET) {
    const n = NET[id];
    const g = mk("g", { class:"nnode", id:"nn_"+id }, nodes);
    mk("circle", { cx:n.x, cy:n.y, r:15 }, g);
    const t = mk("text", { x:n.x, y:n.y+30, "text-anchor":"middle" }, g);
    t.textContent = n.label;
  }
}

function clearFlows() {
  el("bodyFlows").innerHTML = "";
  el("netFlows").innerHTML = "";
  document.querySelectorAll(".net-edge.on").forEach(e => e.classList.remove("on"));
  document.querySelectorAll(".fire-badge").forEach(e => e.remove());
}

function render() {
  if (!moments.length) return;
  const m = moments[idx];
  const active = new Set(m.active || []);

  for (const id in BODY) {
    const on = active.has(id);
    el("organ_"+id).classList.toggle("on", on);
    el("leader_"+id).classList.toggle("on", on);
    el("blabel_"+id).classList.toggle("on", on);
  }
  for (const id in NET) el("nn_"+id).classList.toggle("on", active.has(id));

  clearFlows();
  (m.flows || []).forEach(([a,b]) => {
    // 体ビューの向き付き矢印（両端をマーカーの手前で止める）
    if (BODY[a] && BODY[b]) {
      const c = seg(BODY[a].mx, BODY[a].my, BODY[b].mx, BODY[b].my, 13, 15);
      mk("line", { x1:c[0], y1:c[1], x2:c[2], y2:c[3], class:"flow", "marker-end":"url(#arrowF)" }, el("bodyFlows"));
    }
    // ネットワーク図の向き付き流れ（手前で止め＋右側へ寄せて上り/下りを二車線に）
    if (NET[a] && NET[b]) {
      const c = seg(cx(a), cy(a), cx(b), cy(b), 17, 23, 6);
      mk("line", { x1:c[0], y1:c[1], x2:c[2], y2:c[3], class:"net-flow", "marker-end":"url(#arrowF)" }, el("netFlows"));
    }
    document.querySelectorAll(".net-edge").forEach(e => {
      const [ea,eb] = e.getAttribute("data-edge").split("|");
      if ((ea===a&&eb===b)||(ea===b&&eb===a)) e.classList.add("on");
    });
  });

  if (active.size) {
    const f = [...active][0];
    if (NET[f]) {
      const badge = mk("text", { x:NET[f].x, y:NET[f].y-22, "text-anchor":"middle", class:"fire-badge" }, el("netNodes"));
      badge.textContent = "発火!";
    }
  }

  GAUGES.forEach(g => {
    const v = (m.gauges && m.gauges[g.key] != null) ? m.gauges[g.key] : 0;
    el("gauge_"+g.key).style.width = Math.round(v*100)+"%";
    el("gval_"+g.key).textContent = Math.round(v*100)+"%";
  });

  el("sceneLabel").textContent = m.kind || "";
  el("dateLabel").textContent = fmtDate(m.t);
  el("utterVal").textContent = m.utter ? m.utter : "—";
  el("seek").value = idx;
  el("seekReadout").textContent = (idx+1)+" / "+moments.length;
}

function go(i){ idx = Math.max(0, Math.min(moments.length-1, i)); render(); }
function stop(){ if (timer){ clearInterval(timer); timer=null; } setPlayIcon(false); }
function play(){ stop(); timer = setInterval(() => {
  if (idx >= moments.length-1){ stop(); return; } go(idx+1); }, 1400/speed); setPlayIcon(true); }
function setPlayIcon(on){ el("playBtn").textContent = on ? "⏸ 停止" : "▶ 再生"; }

function parseTrace(text) {
  try { const arr = JSON.parse(text); if (Array.isArray(arr)) return arr; } catch(e) {}
  const out = [];
  let gauges = { hunger:0, ne:0, dopamine:0, happiness:0 };
  text.split(/\r?\n/).forEach(line => {
    line = line.trim(); if (!line) return;
    let o; try { o = JSON.parse(line); } catch(e){ return; }
    if (o.type === "snap") gauges = { hunger:o.hunger, ne:o.ne, dopamine:o.dopamine, happiness:o.happiness };
    else if (o.type === "event") {
      // イベント行が数値を自分で持っていればそれを使い、無ければ直近のsnapを使う
      const g = (o.hunger != null)
        ? { hunger:o.hunger, ne:o.ne, dopamine:o.dopamine, happiness:o.happiness }
        : Object.assign({}, gauges);
      out.push({ t:o.t, kind:o.kind, active:o.modules||[], flows:o.flows||[], utter:o.utter||"", gauges:g });
    }
    else if (o.active) out.push(o);
  });
  return out;
}
function loadMoments(list){ moments = list; idx = 0; el("seek").max = Math.max(0, moments.length-1); render(); }

// ---- パネルを自由に動かす（ヘッダをドラッグ）。リサイズはCSS(resize:both) ----
function makeDraggable(panel, handle) {
  let sx, sy, ol, ot, drag = false;
  handle.addEventListener("pointerdown", e => {
    drag = true; sx = e.clientX; sy = e.clientY;
    const r = panel.getBoundingClientRect(); ol = r.left; ot = r.top;
    handle.setPointerCapture(e.pointerId); e.preventDefault();
  });
  handle.addEventListener("pointermove", e => {
    if (!drag) return;
    const nl = Math.max(0, Math.min(window.innerWidth - 60, ol + (e.clientX - sx)));
    const nt = Math.max(0, Math.min(window.innerHeight - 40, ot + (e.clientY - sy)));
    panel.style.left = nl + "px"; panel.style.top = nt + "px";
    panel.style.zIndex = 3;
  });
  handle.addEventListener("pointerup", () => { drag = false; });
}
function place(id, l, t, w, h) {
  const p = el(id); p.style.left = l+"px"; p.style.top = t+"px"; p.style.width = w+"px"; p.style.height = h+"px";
}
function initPanels() {
  const W = window.innerWidth, H = window.innerHeight;
  const top = 28, bottomH = 96, avail = H - top - bottomH - 8;
  const leftW = Math.round(W * 0.53), rightX = leftW + 12, rightW = W - rightX - 8;
  const netH = Math.round(avail * 0.60);
  place("panel_body",   8,      top,          leftW - 8, avail);
  place("panel_net",    rightX, top,          rightW,    netH);
  place("panel_metrics",rightX, top + netH+8, rightW,    avail - netH - 8);
  ["panel_body","panel_net","panel_metrics"].forEach(id => {
    const p = el(id); makeDraggable(p, p.querySelector(".bar"));
  });
}

function init() {
  buildBody();
  buildNet();
  initPanels();
  fetch("sample_trace.json").then(r => r.text()).then(t => loadMoments(parseTrace(t)))
    .catch(() => { el("sceneLabel").textContent = "sample_trace.json を読めません（サーバー経由で開いてください）"; });

  el("playBtn").onclick = () => timer ? stop() : play();
  el("prevBtn").onclick = () => { stop(); go(idx-1); };
  el("nextBtn").onclick = () => { stop(); go(idx+1); };
  el("seek").oninput = e => { stop(); go(parseInt(e.target.value,10)); };
  el("speedSel").onchange = e => { speed = parseFloat(e.target.value); if (timer) play(); };
  el("fileInput").onchange = e => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = () => { const l = parseTrace(r.result); if (l.length) loadMoments(l); };
    r.readAsText(f);
  };
}
document.addEventListener("DOMContentLoaded", init);
