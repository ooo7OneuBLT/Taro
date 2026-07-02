// 太郎リプレイビューア v3 — 多粒度対応。
// 概観（時/日/月/年バケツ）を軽く表示し、粒度切替と生ログ(詳細)を行き来する。
// マイルストーン（節目）を旗で表示し、クリックでその時点へジャンプする。

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
const NET = {
  cerebellum:    { label:"小脳",   x:230, y:38  }, locus:{ label:"青斑核", x:95, y:58 },
  critic:        { label:"クリティック", x:372, y:58 }, hippocampus:{ label:"海馬", x:66, y:128 },
  basal_ganglia: { label:"基底核", x:396, y:128 }, cortex:{ label:"大脳皮質", x:230, y:120 },
  insula:        { label:"島皮質", x:158, y:206 }, vocal:{ label:"声道", x:344, y:212 },
  stomach:       { label:"胃",     x:96,  y:262 }, lungs:{ label:"肺", x:250, y:266 },
};
const NET_EDGES = [
  ["stomach","insula"],["lungs","insula"],["insula","cortex"],["cortex","vocal"],
  ["locus","cortex"],["cortex","critic"],["hippocampus","cortex"],["cortex","basal_ganglia"],
  ["cerebellum","cortex"],["insula","critic"],
];
// イベント種別→発火部品（概観バケツで「どの部品がよく働いたか」を出すのに使う。
// core_b/parent_sim_b の TRACE_MAP と対応）
const KIND_MODULES = {
  babble:["locus","cortex","vocal"], babble_response:["cortex","vocal"],
  word_request:["stomach","insula","cortex","vocal"], feed:["stomach","insula","cortex","critic"],
  comfort:["cortex","insula"], cry:["stomach","insula","cortex","lungs"], sleep:["hippocampus","cortex"],
};
const GAUGES = [
  { key:"hunger", label:"空腹" }, { key:"ne", label:"探索(NE)" },
  { key:"dopamine", label:"ドーパミン分泌量" }, { key:"happiness", label:"幸福度" },
];
const GRANS = [["year","年"],["month","月"],["day","日"],["hour","時"],["raw","生"]];

const SVGNS = "http://www.w3.org/2000/svg";
let datasets = {};      // gran -> array of items (buckets or raw moments)
let milestones = [];
let activeGran = null;
let items = [];         // 現在表示中のデータ列
let idx = 0, timer = null, speed = 1;
let TOTAL_T = 2 * 365 * 86400;

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
function seg(x1,y1,x2,y2,p1,p2,off){
  const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L; off=off||0;
  const ox=-uy*off,oy=ux*off;
  return [x1+ux*p1+ox,y1+uy*p1+oy,x2-ux*p2+ox,y2-uy*p2+oy];
}

function buildBody() {
  const labels=el("bodyLabels"),leaders=el("bodyLeaders"),markers=el("bodyMarkers");
  for (const id in BODY) {
    const m=BODY[id], lx=m.side==="L"?14:466, anchor=m.side==="L"?"start":"end", lineX=m.side==="L"?118:362;
    mk("line",{x1:lineX,y1:m.ly,x2:m.mx,y2:m.my,class:"leader",id:"leader_"+id},leaders);
    const t=mk("text",{x:lx,y:m.ly+4,"text-anchor":anchor,class:"blabel",id:"blabel_"+id},labels);
    t.textContent=m.label;
    if(!m.organ) mk("circle",{cx:m.mx,cy:m.my,r:6,class:"organ marker",id:"organ_"+id},markers);
  }
}
function buildNet() {
  const edges=el("netEdges"),nodes=el("netNodes");
  NET_EDGES.forEach(([a,b])=>mk("line",{x1:cx(a),y1:cy(a),x2:cx(b),y2:cy(b),class:"net-edge","data-edge":a+"|"+b},edges));
  for (const id in NET) {
    const n=NET[id], g=mk("g",{class:"nnode",id:"nn_"+id},nodes);
    mk("circle",{cx:n.x,cy:n.y,r:15},g);
    const t=mk("text",{x:n.x,y:n.y+30,"text-anchor":"middle"},g); t.textContent=n.label;
  }
}
function clearFlows() {
  el("bodyFlows").innerHTML=""; el("netFlows").innerHTML="";
  document.querySelectorAll(".net-edge.on").forEach(e=>e.classList.remove("on"));
  document.querySelectorAll(".fire-badge").forEach(e=>e.remove());
}
function setModules(activeSet) {
  for (const id in BODY){ const on=activeSet.has(id);
    el("organ_"+id).classList.toggle("on",on); el("leader_"+id).classList.toggle("on",on); el("blabel_"+id).classList.toggle("on",on); }
  for (const id in NET) el("nn_"+id).classList.toggle("on",activeSet.has(id));
}
function setGauges(g) {
  GAUGES.forEach(x=>{ const v=(g&&g[x.key]!=null)?g[x.key]:0;
    el("gauge_"+x.key).style.width=Math.round(v*100)+"%"; el("gval_"+x.key).textContent=Math.round(v*100)+"%"; });
}

function renderMoment(m) {
  const active=new Set(m.active||[]); setModules(active); clearFlows();
  (m.flows||[]).forEach(([a,b])=>{
    if(BODY[a]&&BODY[b]){ const c=seg(BODY[a].mx,BODY[a].my,BODY[b].mx,BODY[b].my,13,15);
      mk("line",{x1:c[0],y1:c[1],x2:c[2],y2:c[3],class:"flow","marker-end":"url(#arrowF)"},el("bodyFlows")); }
    if(NET[a]&&NET[b]){ const c=seg(cx(a),cy(a),cx(b),cy(b),17,23,6);
      mk("line",{x1:c[0],y1:c[1],x2:c[2],y2:c[3],class:"net-flow","marker-end":"url(#arrowF)"},el("netFlows")); }
    document.querySelectorAll(".net-edge").forEach(e=>{const[ea,eb]=e.getAttribute("data-edge").split("|");
      if((ea===a&&eb===b)||(ea===b&&eb===a))e.classList.add("on");});
  });
  if(active.size){ const f=[...active][0]; if(NET[f]){ const badge=mk("text",{x:NET[f].x,y:NET[f].y-22,"text-anchor":"middle",class:"fire-badge"},el("netNodes")); badge.textContent="発火!"; } }
  setGauges(m.gauges);
  el("utterVal").textContent=m.utter?m.utter:"—";
}
function renderBucket(b) {
  clearFlows();
  // イベント回数から「どの部品がその期間よく働いたか」を集計
  const act={}; const counts=b.counts||{};
  for (const kind in counts) (KIND_MODULES[kind]||[]).forEach(m=>act[m]=(act[m]||0)+counts[kind]);
  const activeSet=new Set(Object.keys(act).filter(m=>act[m]>0));
  setModules(activeSet);
  setGauges(b.gauges);
  // 発声欄に主なイベント回数を出す
  const top=Object.entries(counts).sort((a,c)=>c[1]-a[1]).slice(0,4)
    .map(([k,v])=>`${k}${v}`).join(" / ");
  el("utterVal").textContent = top || "—";
}

function render() {
  if (!items.length) return;
  const it=items[idx];
  el("seek").value=idx; el("seekReadout").textContent=(idx+1)+" / "+items.length;
  el("dateLabel").textContent=fmtDate(it.t);
  if (it.counts) { el("sceneLabel").textContent=`【${granLabel(activeGran)}集約】`; renderBucket(it); }
  else { el("sceneLabel").textContent=it.kind||""; renderMoment(it); }
  updatePlayhead();
}
function granLabel(g){ const f=GRANS.find(x=>x[0]===g); return f?f[1]:g; }

function updatePlayhead(){
  const it=items[idx]; if(!it) return;
  el("playhead").setAttribute("x", 8 + (it.t/TOTAL_T)*684 - 1);
}
function drawMilestones(){
  const g=el("msFlags"); g.innerHTML="";
  milestones.forEach(ms=>{
    const x=8+(ms.t/TOTAL_T)*684;
    const fl=mk("g",{class:"ms","transform":`translate(${x},0)`},g);
    mk("line",{x1:0,y1:4,x2:0,y2:18,class:"ms-line"},fl);
    mk("path",{d:"M0,4 L8,7 L0,10 Z",class:"ms-flag"},fl);
    const t=mk("title",{},fl); t.textContent=`${ms.name}（${fmtDate(ms.t)}）`;
    fl.style.cursor="pointer";
    fl.addEventListener("click",()=>jumpToTime(ms.t));
  });
}
function jumpToTime(t){
  stop();
  let best=0,bd=Infinity;
  items.forEach((it,i)=>{ const d=Math.abs(it.t-t); if(d<bd){bd=d;best=i;} });
  go(best);
}

function go(i){ idx=Math.max(0,Math.min(items.length-1,i)); render(); }
function stop(){ if(timer){clearInterval(timer);timer=null;} setPlayIcon(false); }
function play(){ stop(); timer=setInterval(()=>{ if(idx>=items.length-1){stop();return;} go(idx+1); },1000/speed); setPlayIcon(true); }
function setPlayIcon(on){ el("playBtn").textContent=on?"⏸ 停止":"▶ 再生"; }

function setActive(gran){
  if(!datasets[gran]||!datasets[gran].length) return;
  activeGran=gran; items=datasets[gran]; idx=0;
  TOTAL_T=Math.max(1, ...items.map(x=>x.t)) || TOTAL_T;
  el("seek").max=Math.max(0,items.length-1);
  el("axisEnd").textContent=fmtDate(TOTAL_T);
  GRANS.forEach(([g])=>{ const btn=el("gran_"+g); if(btn) btn.classList.toggle("sel",g===gran); });
  drawMilestones(); render();
}

// ---- パース ----
function parseAny(text){
  const out=[]; let gauges={hunger:0,ne:0,dopamine:0,happiness:0};
  // 配列JSON（sample）
  try{ const arr=JSON.parse(text); if(Array.isArray(arr)) return arr; }catch(e){}
  text.split(/\r?\n/).forEach(line=>{
    line=line.trim(); if(!line) return;
    let o; try{o=JSON.parse(line);}catch(e){return;}
    if(o.type==="bucket") out.push(o);
    else if(o.type==="snap") gauges={hunger:o.hunger,ne:o.ne,dopamine:o.dopamine,happiness:o.happiness};
    else if(o.type==="event"){ const g=(o.hunger!=null)?{hunger:o.hunger,ne:o.ne,dopamine:o.dopamine,happiness:o.happiness}:Object.assign({},gauges);
      out.push({t:o.t,kind:o.kind,active:o.modules||[],flows:o.flows||[],utter:o.utter||"",gauges:g}); }
    else if(o.active) out.push(o);
  });
  return out;
}
function routeFile(name, text){
  name=name.toLowerCase();
  if(name.includes("milestone")){ try{ milestones=JSON.parse(text)||[]; }catch(e){} return; }
  let gran=null;
  for(const [g] of GRANS) if(name.includes("overview_"+g)) gran=g;
  if(!gran && name.includes("trace")) gran="raw";
  if(!gran) return;
  datasets[gran]=parseAny(text);
}

function afterLoad(){
  const order=["day","month","hour","year","raw"];
  const g=order.find(x=>datasets[x]&&datasets[x].length);
  if(g) setActive(g);
}

// ---- 初期化 ----
function init(){
  buildBody(); buildNet();

  // 既定：同じフォルダに置かれた概観/生ログ/マイルストーンを取りに行く。
  // 無ければ sample_trace.json（生の見本）を使う。
  const files=["trace_overview_year.jsonl","trace_overview_month.jsonl","trace_overview_day.jsonl",
               "trace_overview_hour.jsonl","milestones.json","trace.jsonl"];
  Promise.allSettled(files.map(f=>fetch(f).then(r=>r.ok?r.text():Promise.reject()).then(t=>routeFile(f,t))))
    .then(()=>{
      if(Object.keys(datasets).length) afterLoad();
      else fetch("sample_trace.json").then(r=>r.text()).then(t=>{ datasets.raw=parseAny(t); setActive("raw"); })
        .catch(()=>{ el("sceneLabel").textContent="ログが読めません（サーバー経由で開く／フォルダを選択）"; });
    });

  el("playBtn").onclick=()=>timer?stop():play();
  el("prevBtn").onclick=()=>{stop();go(idx-1);};
  el("nextBtn").onclick=()=>{stop();go(idx+1);};
  el("seek").oninput=e=>{stop();go(parseInt(e.target.value,10));};
  el("speedSel").onchange=e=>{speed=parseFloat(e.target.value); if(timer)play();};
  GRANS.forEach(([g])=>{ const b=el("gran_"+g); if(b) b.onclick=()=>setActive(g); });

  el("folderInput").onchange=e=>{
    datasets={}; milestones=[];
    const fs=[...e.target.files]; let pending=fs.length;
    if(!pending) return;
    fs.forEach(f=>{ const r=new FileReader(); r.onload=()=>{ routeFile(f.name,r.result); if(--pending===0) afterLoad(); }; r.readAsText(f); });
  };
}
document.addEventListener("DOMContentLoaded", init);

// パネル移動（ヘッダをドラッグ）／リサイズ(CSS)
function makeDraggable(panel,handle){
  let sx,sy,ol,ot,drag=false;
  handle.addEventListener("pointerdown",e=>{drag=true;sx=e.clientX;sy=e.clientY;const r=panel.getBoundingClientRect();ol=r.left;ot=r.top;handle.setPointerCapture(e.pointerId);e.preventDefault();});
  handle.addEventListener("pointermove",e=>{if(!drag)return;const nl=Math.max(0,Math.min(innerWidth-60,ol+(e.clientX-sx))),nt=Math.max(0,Math.min(innerHeight-40,ot+(e.clientY-sy)));panel.style.left=nl+"px";panel.style.top=nt+"px";panel.style.zIndex=3;});
  handle.addEventListener("pointerup",()=>{drag=false;});
}
function place(id,l,t,w,h){const p=el(id);p.style.left=l+"px";p.style.top=t+"px";p.style.width=w+"px";p.style.height=h+"px";}
(function(){ // パネル初期配置（DOM後）
  document.addEventListener("DOMContentLoaded",()=>{
    const W=innerWidth,H=innerHeight,top=28,bottomH=120,avail=H-top-bottomH-8;
    const leftW=Math.round(W*0.53),rightX=leftW+12,rightW=W-rightX-8,netH=Math.round(avail*0.60);
    place("panel_body",8,top,leftW-8,avail);
    place("panel_net",rightX,top,rightW,netH);
    place("panel_metrics",rightX,top+netH+8,rightW,avail-netH-8);
    ["panel_body","panel_net","panel_metrics"].forEach(id=>{const p=el(id);makeDraggable(p,p.querySelector(".bar"));});
  });
})();
