// 太郎リプレイビューア v3 — 多粒度対応。
// 概観（時/日/月/年バケツ）を軽く表示し、粒度切替と生ログ(詳細)を行き来する。
// マイルストーン（節目）を旗で表示し、クリックでその時点へジャンプする。

const BODY = {
  cortex:        { label:"大脳皮質",   mx:150, my:54,  side:"L", organ:true  },
  critic:        { label:"クリティック", mx:124, my:62,  side:"L", organ:false },
  basal_ganglia: { label:"基底核",     mx:150, my:78,  side:"L", organ:false },
  insula:        { label:"島皮質",     mx:128, my:160, side:"L", organ:false },
  lungs:         { label:"肺",         mx:141, my:166, side:"L", organ:true  },
  hippocampus:   { label:"海馬",       mx:176, my:60,  side:"R", organ:false },
  cerebellum:    { label:"小脳",       mx:174, my:90,  side:"R", organ:true  },
  locus:         { label:"青斑核",     mx:166, my:118, side:"R", organ:false },
  vocal:         { label:"声道",       mx:150, my:116, side:"R", organ:true  },
  stomach:       { label:"胃",         mx:162, my:216, side:"R", organ:true  },
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
let chatMsgs = [];      // {t, who:'parent'|'taro', text, cry}
let bodyBox = {x:0,y:0,w:300,h:360};   // 体ビューの現在のviewBox（ズーム/パンで変わる）
let activeMods = new Set();             // 今光っている部品

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
  const markers=el("bodyMarkers");
  for (const id in BODY) if(!BODY[id].organ)
    mk("circle",{cx:BODY[id].mx,cy:BODY[id].my,r:5.5,class:"organ marker",id:"organ_"+id},markers);
  layoutBodyLabels();
}
// 引き出し線＋名前を、今見えている範囲（bodyBox）に合わせて動的に配置する。
// ・見えている臓器だけ名前を出す（ズーム時に画面外は隠す）
// ・各辺に均等に並べて重ならないようにする
// ・文字/線は画面上ほぼ一定サイズになるよう viewBox 幅で逆スケール
function layoutBodyLabels(){
  const labels=el("bodyLabels"),leaders=el("bodyLeaders");
  if(!labels||!leaders) return;
  labels.innerHTML=""; leaders.innerHTML="";
  const b=bodyBox, x0=b.x, x1=b.x+b.w, y0=b.y, y1=b.y+b.h;
  const font=12*(b.w/300), margin=b.w*0.02, pad=b.h*0.05;
  const inside=m=>m.mx>=x0&&m.mx<=x1&&m.my>=y0&&m.my<=y1;
  ["L","R"].forEach(side=>{
    const ids=Object.keys(BODY).filter(id=>BODY[id].side===side&&inside(BODY[id]))
              .sort((a,c)=>BODY[a].my-BODY[c].my);
    if(!ids.length) return;
    const lx=side==="L"?x0+margin:x1-margin, anchor=side==="L"?"start":"end";
    const top=y0+pad, bot=y1-pad, span=bot-top;
    ids.forEach((id,i)=>{
      const ly=ids.length===1?(top+bot)/2:top+span*i/(ids.length-1);
      mk("line",{x1:lx,y1:ly,x2:BODY[id].mx,y2:BODY[id].my,class:"leader",id:"leader_"+id},leaders);
      const t=mk("text",{x:lx,y:ly+font*0.34,"text-anchor":anchor,class:"blabel",id:"blabel_"+id,"font-size":font},labels);
      t.textContent=BODY[id].label;
    });
  });
  applyActive();
}
function applyActive(){
  for(const id in BODY){ const on=activeMods.has(id);
    const o=el("organ_"+id); if(o) o.classList.toggle("on",on);
    const l=el("leader_"+id); if(l) l.classList.toggle("on",on);
    const t=el("blabel_"+id); if(t) t.classList.toggle("on",on);
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
  activeMods=activeSet; applyActive();
  for (const id in NET) el("nn_"+id).classList.toggle("on",activeSet.has(id));
}
function setGauges(g) {
  GAUGES.forEach(x=>{ const v=(g&&g[x.key]!=null)?g[x.key]:0;
    el("gauge_"+x.key).style.width=Math.round(v*100)+"%"; el("gval_"+x.key).textContent=Math.round(v*100)+"%"; });
}

function renderMoment(m) {
  const active=new Set(m.active||[]); setModules(active); clearFlows();
  (m.flows||[]).forEach(([a,b])=>{
    if(BODY[a]&&BODY[b]){ const c=seg(BODY[a].mx,BODY[a].my,BODY[b].mx,BODY[b].my,8,9);
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

// ---- チャット（親↔太郎の会話） ----
function escapeHtml(s){ return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
// 1イベント→0〜2の発話。say=親の言葉（あれば）、utter=太郎の発声（あれば）。
function messagesOf(ev){
  const out=[];
  const say=ev.say||ev.parent;
  if(say) out.push({t:ev.t, who:"parent", text:say});
  const u=ev.utter;
  if(u) out.push({t:ev.t, who:"taro", text:u, cry:/泣/.test(u)});
  return out;
}
// 会話の元データは常に生イベント列（バケツには発話が無いため）。
function chatSource(){
  if(datasets.raw && datasets.raw.length) return datasets.raw;
  return items.filter(it=>!it.counts);
}
function rebuildChat(){
  const box=el("chatList"); if(!box) return;
  chatMsgs=[];
  chatSource().forEach(ev=>{ if(!ev.counts) messagesOf(ev).forEach(m=>chatMsgs.push(m)); });
  chatMsgs.sort((a,b)=>a.t-b.t);
  box.innerHTML="";
  if(!chatMsgs.length){ box.innerHTML='<div class="chat-empty">この期間に発話ログはありません</div>'; return; }
  chatMsgs.forEach(m=>{
    const d=document.createElement("div");
    d.className="msg "+m.who+(m.cry?" cry":"");
    d.dataset.t=m.t;
    d.innerHTML='<span class="who">'+(m.who==="parent"?"親":"太郎")+" ・ "+fmtDate(m.t)+"</span>"+escapeHtml(m.text);
    box.appendChild(d);
  });
}
function updateChat(curT){
  const box=el("chatList"); if(!box||!chatMsgs.length) return;
  const kids=box.children; let last=-1;
  for(let i=0;i<kids.length;i++){
    const seen=(+kids[i].dataset.t)<=curT+0.5;
    kids[i].classList.toggle("seen",seen);
    kids[i].classList.remove("cur");
    if(seen) last=i;
  }
  if(last>=0){ kids[last].classList.add("cur"); kids[last].scrollIntoView({block:"nearest"}); }
}

function render() {
  if (!items.length) return;
  const it=items[idx];
  el("seek").value=idx; el("seekReadout").textContent=(idx+1)+" / "+items.length;
  el("dateLabel").textContent=fmtDate(it.t);
  if (it.counts) { el("sceneLabel").textContent=`【${granLabel(activeGran)}集約】`; renderBucket(it); }
  else { el("sceneLabel").textContent=it.kind||""; renderMoment(it); }
  updatePlayhead();
  updateChat(it.t);
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
  drawMilestones(); rebuildChat(); render();
}

// ---- ビュー操作：ホイールで拡大縮小／右ドラッグで移動（パン）／ダブルクリックで戻す ----
function enableZoom(panelId, vb, onChange){
  const panel=el(panelId); if(!panel) return;
  const svg=panel.querySelector("svg"); if(!svg) return;
  const base={x:vb[0],y:vb[1],w:vb[2],h:vb[3]};
  let box=Object.assign({},base);
  const apply=()=>{ svg.setAttribute("viewBox",`${box.x} ${box.y} ${box.w} ${box.h}`);
    if(onChange) onChange(Object.assign({},box)); };
  svg.style.cursor="grab";
  svg.addEventListener("wheel",e=>{
    e.preventDefault();
    const r=svg.getBoundingClientRect();
    const mx=box.x+(e.clientX-r.left)/r.width*box.w;
    const my=box.y+(e.clientY-r.top)/r.height*box.h;
    const f=e.deltaY<0?0.88:1.14;
    const nw=Math.min(base.w*1.4,Math.max(base.w*0.22,box.w*f));
    const nh=Math.min(base.h*1.4,Math.max(base.h*0.22,box.h*f));
    box.x=mx-(mx-box.x)*(nw/box.w); box.y=my-(my-box.y)*(nh/box.h);
    box.w=nw; box.h=nh; apply();
  },{passive:false});
  // 右ドラッグでパン（左ドラッグはパネル移動に使うので右のみ）
  let pan=false,px,py,bx,by;
  svg.addEventListener("contextmenu",e=>e.preventDefault());
  svg.addEventListener("pointerdown",e=>{
    if(e.button!==2) return;
    pan=true; px=e.clientX; py=e.clientY; bx=box.x; by=box.y;
    svg.style.cursor="grabbing"; svg.setPointerCapture(e.pointerId); e.preventDefault();
  });
  svg.addEventListener("pointermove",e=>{
    if(!pan) return;
    const r=svg.getBoundingClientRect();
    box.x=bx-(e.clientX-px)/r.width*box.w;
    box.y=by-(e.clientY-py)/r.height*box.h; apply();
  });
  const end=e=>{ if(pan){pan=false; svg.style.cursor="grab"; try{svg.releasePointerCapture(e.pointerId);}catch(_){}}};
  svg.addEventListener("pointerup",end); svg.addEventListener("pointercancel",end);
  svg.addEventListener("dblclick",()=>{ box=Object.assign({},base); apply(); });
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
  enableZoom("panel_body",[0,0,300,360],box=>{ bodyBox=box; layoutBodyLabels(); });
  enableZoom("panel_net",[0,0,460,300]);

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
// パネル初期配置。左＝体ビュー(大)／中＝ネット+数値／右＝会話。
function layoutPanels(){
  const W=innerWidth,H=innerHeight,top=28,bottomH=120,avail=Math.max(200,H-top-bottomH-8);
  const chatW=Math.max(240,Math.round(W*0.22)), chatX=W-chatW-8;
  const zoneW=Math.max(200,chatX-16);          // 体＋中列に使える幅
  const bodyW=Math.round(zoneW*0.56), midX=8+bodyW+12, midW=Math.max(160,chatX-midX-12);
  const netH=Math.round(avail*0.58);
  place("panel_body",8,top,bodyW,avail);
  place("panel_net",midX,top,midW,netH);
  place("panel_metrics",midX,top+netH+8,midW,avail-netH-8);
  place("panel_chat",chatX,top,chatW,avail);
}
document.addEventListener("DOMContentLoaded",()=>{
  layoutPanels();
  ["panel_body","panel_net","panel_metrics","panel_chat"].forEach(id=>{const p=el(id);makeDraggable(p,p.querySelector(".bar"));});
  let rt; addEventListener("resize",()=>{clearTimeout(rt);rt=setTimeout(layoutPanels,150);});
});
