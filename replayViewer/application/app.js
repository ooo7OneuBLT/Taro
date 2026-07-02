// 太郎リプレイビューア v3 — 多粒度対応。
// 概観（時/日/月/年バケツ）を軽く表示し、粒度切替と生ログ(詳細)を行き来する。
// マイルストーン（節目）を旗で表示し、クリックでその時点へジャンプする。

// 解剖学的な近似配置（脳の構造は頭の中／声道=首・肺=胸・胃=腹）。
// mx,my は各部位の中心。organ:trueは形を描く臓器、falseは点マーカー。
// 【将来3D化のメモ】究極の見やすさには3Dが有利。その場合もこのデータ駆動構造は
// 流用できる：BODYに z を足して3座標にし、体ビューのSVGだけを Three.js等の
// WebGLシーンに差し替える（臓器=メッシュ、flows=3D線、ラベルはスプライト）。
// パネル/会話/ウィンドウ操作/再生の仕組みはそのまま使える。近すぎて重なる問題は
// 奥行きで解消できる。
const BODY = {
  cortex:        { label:"大脳皮質",   mx:150, my:70,  side:"L", organ:true  },  // 頭・上部
  insula:        { label:"島皮質",     mx:124, my:86,  side:"L", organ:false },  // 脳の外側（肺でなく頭内）
  basal_ganglia: { label:"基底核",     mx:142, my:90,  side:"L", organ:false },  // 脳の深部
  vocal:         { label:"声道",       mx:150, my:151, side:"L", organ:true  },  // 首・喉頭
  lungs:         { label:"肺",         mx:150, my:196, side:"L", organ:true  },  // 胸
  hippocampus:   { label:"海馬",       mx:166, my:96,  side:"R", organ:false },  // 脳の深部
  cerebellum:    { label:"小脳",       mx:171, my:113, side:"R", organ:true  },  // 頭・後下部
  critic:        { label:"クリティック", mx:158, my:104, side:"R", organ:false }, // 抽象（腹側線条体あたり）
  locus:         { label:"青斑核",     mx:150, my:130, side:"R", organ:false },  // 脳幹（頭の底）
  stomach:       { label:"胃",         mx:152, my:250, side:"R", organ:true  },  // 腹
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
// イベント種別→日本語ラベル（scene表示用）と重要度（粗い粒度で残す優先度）
const KIND_LABEL = { babble:"喃語", babble_response:"喃語に親が反応", word_request:"要求語（まんま）",
  feed:"授乳", comfort:"あやし", cry:"泣く", sleep:"睡眠" };
const KIND_SIG = { feed:6, word_request:6, babble_response:4, cry:3, comfort:2, sleep:1, babble:0 };

const GAUGES = [
  { key:"hunger", label:"空腹" }, { key:"ne", label:"探索(NE)" },
  { key:"dopamine", label:"ドーパミン分泌量" }, { key:"happiness", label:"幸福度" },
];
const GRANS = [["year","年"],["month","月"],["day","日"],["hour","時"],["raw","生"]];

// 臓器・脳部位の説明（クリックで小窓に出す。太郎の設計に即した内容）
const ORGAN_INFO = {
  cortex:        ["大脳皮質", "太郎の脳の本体。聞いた音を予測し、次に出す音を選ぶ再帰ネット（GRU）。知覚・産出・理解の統合中枢。"],
  cerebellum:    ["小脳", "発声の運動をなめらかに整える。運動の学習と協調をになう。"],
  vocal:         ["声道", "口・のど・声帯。音を実際に作る発声器官（調音）。"],
  lungs:         ["肺", "発声のための空気。息が続く範囲で声を出す（喃語の長さを左右する）。"],
  stomach:       ["胃", "食べ物を受け取り消化する器官。空腹・満腹のもと。"],
  critic:        ["クリティック", "「今どれくらい良い状態か」を評価する価値関数（強化学習のcritic）。報酬の基準線。"],
  basal_ganglia: ["基底核", "行動選択と強化学習の中枢。うまくいった発声を出やすくする（方策の更新）。"],
  insula:        ["島皮質", "内受容感覚。体の中（空腹など）を感じ取って脳に伝える。"],
  hippocampus:   ["海馬", "記憶。起きた出来事を覚え、睡眠中に反芻して定着させる（睡眠リプレイ）。"],
  locus:         ["青斑核", "ノルアドレナリンで「探索⇄集中」を調整。新しい音を試す度合い（喃語のゆらぎ）を制御する。"],
};
function showOrganInfo(id, evt){
  const info=ORGAN_INFO[id]; if(!info) return;
  el("organInfoTitle").textContent=info[0];
  el("organInfoDesc").textContent=info[1];
  const box=el("organInfo"); box.hidden=false;
  const w=box.offsetWidth||260, h=box.offsetHeight||90;
  let x=(evt?evt.clientX:innerWidth/2)+12, y=(evt?evt.clientY:120)+8;
  box.style.left=Math.max(6,Math.min(x,innerWidth-w-6))+"px";
  box.style.top=Math.max(6,Math.min(y,innerHeight-h-6))+"px";
}
function hideOrganInfo(){ const b=el("organInfo"); if(b) b.hidden=true; }

const SVGNS = "http://www.w3.org/2000/svg";
let datasets = {};      // gran -> array of items (buckets or raw moments)
let milestones = [];
let activeGran = null;
let items = [];         // 現在表示中のデータ列
let idx = 0, timer = null, speed = 1;
let TOTAL_T = 2 * 365 * 86400;
let chatMsgs = [];      // {t, who:'parent'|'taro', text, cry}
let comprehension = []; // {t, mama, maman, aua} 月イチの「語→食べ物予期」測定（理解の配線用）
let bodyBox = {x:0,y:0,w:300,h:380};   // 体ビューの現在のviewBox（ズーム/パンで変わる）
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
// テキストの実サイズ(getBBox)を測って、後ろに背景の角丸長方形を敷く。
// 矢印や臓器と文字が重なっても読めるようにするため。textより前(下)に挿入する。
function bgFor(textEl, group, cls){
  try{
    const bb=textEl.getBBox();
    const f=parseFloat(textEl.getAttribute("font-size"))||12;
    const px=f*0.3, py=f*0.16;
    const r=mk("rect",{x:bb.x-px,y:bb.y-py,width:bb.width+px*2,height:bb.height+py*2,
      rx:f*0.28,class:cls||"lblbg"});
    group.insertBefore(r, textEl);
    return r;
  }catch(e){ return null; }
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
// 引き出し線＋名前を配置する。名前は臓器の座標を基準に置く（＝モデルと一緒に
// パン/ズームで動く）。臓器から少し外側にずらし、はみ出さないよう現在の表示範囲に
// クランプする。見えている臓器だけ表示。文字は画面上ほぼ一定サイズ（実ピクセル基準）。
function layoutBodyLabels(){
  const labels=el("bodyLabels"),leaders=el("bodyLeaders");
  if(!labels||!leaders) return;
  labels.innerHTML=""; leaders.innerHTML="";
  const svg=el("panel_body")&&el("panel_body").querySelector("svg");
  const b=bodyBox, x0=b.x, x1=b.x+b.w, y0=b.y, y1=b.y+b.h, cxm=b.x+b.w/2;
  const rect=svg?svg.getBoundingClientRect():{width:b.w,height:b.h};
  const scale=Math.min(rect.width/b.w, rect.height/b.h)||1;
  const font=13/scale;                 // 画面上つねに約13px
  const lineH=font*1.5, margin=font*0.7, off=b.w*0.32;
  const inside=m=>m.mx>=x0&&m.mx<=x1&&m.my>=y0&&m.my<=y1;
  ["L","R"].forEach(side=>{
    const ids=Object.keys(BODY).filter(id=>BODY[id].side===side&&inside(BODY[id]))
              .sort((a,c)=>BODY[a].my-BODY[c].my);
    let lastY=-1e9;
    ids.forEach(id=>{
      const o=BODY[id];
      let lx = side==="L" ? o.mx-off : o.mx+off;                 // 臓器基準でずらす
      lx = side==="L" ? Math.max(x0+margin, Math.min(lx, cxm-margin))   // 表示範囲にクランプ
                      : Math.min(x1-margin, Math.max(lx, cxm+margin));
      let ly = Math.max(o.my, lastY+lineH);                      // 上から順に重なり回避
      ly = Math.min(ly, y1-margin); lastY = ly;
      mk("line",{x1:lx,y1:ly,x2:o.mx,y2:o.my,class:"leader",id:"leader_"+id},leaders);
      const t=mk("text",{x:lx, y:ly+font*0.34, "text-anchor":side==="L"?"end":"start",
        class:"blabel", id:"blabel_"+id, "font-size":font}, labels);
      t.textContent=o.label;
      t.addEventListener("click", e=>{ e.stopPropagation(); showOrganInfo(id, e); });
      const bg=bgFor(t, labels); if(bg) bg.id="blabelbg_"+id;
    });
  });
  applyActive();
}
function applyActive(){
  for(const id in BODY){ const on=activeMods.has(id);
    const o=el("organ_"+id); if(o) o.classList.toggle("on",on);
    const l=el("leader_"+id); if(l) l.classList.toggle("on",on);
    const t=el("blabel_"+id); if(t) t.classList.toggle("on",on);
    const bg=el("blabelbg_"+id); if(bg) bg.classList.toggle("on",on);
  }
}
// NN活性ビュー（かっこよさ用・汎用）：実アーキテクチャ 入力→GRU隠れ128→出口 を描き、
// イベントごとに発火した隠れノード(rec.fire)を光らせる。意味は読めないが「本物のNNが
// 動いている」感を出す。どのモデルでも fire さえ記録すれば使える汎用ビュー。
const NN_HID = 128;
let _nnHidPos = [], _nnLit = [];
function buildNNView(){
  const svg=el("nnSvg"); if(!svg) return; svg.innerHTML="";
  const edges=mk("g",{},svg), nodes=mk("g",{},svg), labels=mk("g",{},svg);
  const inX=28, inYs=[]; for(let i=0;i<6;i++){ const y=50+i*38; inYs.push(y);
    mk("circle",{cx:inX,cy:y,r:4,class:"nn-node nn-in"},nodes); }
  const cols=16, rows=8, x0=92, x1=338, y0=28, y1=274; _nnHidPos=[];
  for(let i=0;i<NN_HID;i++){ const c=i%cols, r=Math.floor(i/cols);
    const x=x0+(x1-x0)*c/(cols-1), y=y0+(y1-y0)*r/(rows-1); _nnHidPos.push([x,y]);
    mk("circle",{cx:x,cy:y,r:3.1,class:"nn-node",id:"nnh_"+i},nodes); }
  const headDefs=[["食べ物予期",70],["発声",118],["価値",166],["次の音の予測",214]];
  const hX=432, headPos=[];
  headDefs.forEach(([lab,y])=>{ headPos.push([hX,y]);
    mk("circle",{cx:hX,cy:y,r:5,class:"nn-node nn-head"},nodes);
    const t=mk("text",{x:hX-9,y:y+3,"text-anchor":"end",class:"nn-headlabel"},labels); t.textContent=lab; });
  const rnd=a=>a[Math.floor(Math.random()*a.length)];
  for(let k=0;k<80;k++){ const y=rnd(inYs), p=rnd(_nnHidPos);
    mk("line",{x1:inX,y1:y,x2:p[0],y2:p[1],class:"nn-edge"},edges); }
  for(let k=0;k<80;k++){ const p=rnd(_nnHidPos), q=rnd(headPos);
    mk("line",{x1:p[0],y1:p[1],x2:q[0],y2:q[1],class:"nn-edge"},edges); }
  svg.insertBefore(edges, nodes);   // 辺を下に
}
function updateNN(fire){
  _nnLit.forEach(i=>{ const n=el("nnh_"+i); if(n) n.classList.remove("on"); });
  _nnLit = Array.isArray(fire)?fire:[];
  _nnLit.forEach(i=>{ const n=el("nnh_"+i); if(n) n.classList.add("on"); });
}

function buildNet() {
  const edges=el("netEdges"),nodes=el("netNodes");
  NET_EDGES.forEach(([a,b])=>mk("line",{x1:cx(a),y1:cy(a),x2:cx(b),y2:cy(b),class:"net-edge","data-edge":a+"|"+b},edges));
  for (const id in NET) {
    const n=NET[id], g=mk("g",{class:"nnode",id:"nn_"+id},nodes);
    mk("circle",{cx:n.x,cy:n.y,r:15},g);
    const t=mk("text",{x:n.x,y:n.y+30,"text-anchor":"middle","font-size":13},g); t.textContent=n.label;
    bgFor(t, g);   // 矢印と重なっても読めるよう背景を敷く
  }
}
function clearFlows() {
  el("bodyFlows").innerHTML=""; el("netFlows").innerHTML="";
  document.querySelectorAll(".net-edge.on").forEach(e=>e.classList.remove("on"));
  document.querySelectorAll(".fire-badge, .badgebg").forEach(e=>e.remove());
}
function setModules(activeSet) {
  activeMods=activeSet; applyActive();
  for (const id in NET) el("nn_"+id).classList.toggle("on",activeSet.has(id));
}
// 理解の配線図：3つの語 → 「食べ物予期」ノード。線が太い/明るいほど「その語で食べ物が
// 来る」と結びついている＝理解している。まんまが育って光り、あうあは暗いままなのを見せる。
const COMP_WORDS = [["まんま","mama",34],["ままん","maman",78],["あうあ","aua",122]];
function buildComp(){
  const svg=el("compSvg"); if(!svg) return;
  svg.innerHTML="";
  const nx=228, ny=78;
  mk("rect",{x:nx,y:ny-22,width:64,height:44,rx:10,class:"comp-node"},svg);
  const nt=mk("text",{x:nx+32,y:ny+4,"text-anchor":"middle","font-size":11.5,fill:"var(--cyan)"},svg);
  nt.textContent="食べ物予期";
  COMP_WORDS.forEach(([label,key,wy])=>{
    mk("line",{x1:66,y1:wy,x2:nx,y2:ny,class:"comp-line","data-key":key,"stroke-width":1,opacity:.25},svg);
    const t=mk("text",{x:6,y:wy+4,class:"comp-wlabel"},svg); t.textContent=label;
    mk("text",{x:(66+nx)/2,y:(wy+ny)/2-3,"text-anchor":"middle",class:"comp-val","data-key":key},svg);
  });
  updateComp(items&&items[idx]?items[idx].t:0);
}
function updateComp(t){
  const svg=el("compSvg"); if(!svg) return;
  const title=el("compTitle");
  if(!comprehension.length){ if(title) title.textContent="理解の配線：この記録には理解メーターのデータがありません（12ヶ月トレースで記録）"; return; }
  let cur=comprehension[0];
  for(const c of comprehension){ if(c.t<=t+0.5) cur=c; else break; }
  const vals={mama:cur.mama, maman:cur.maman, aua:cur.aua};
  svg.querySelectorAll(".comp-line").forEach(ln=>{
    const v=vals[ln.getAttribute("data-key")]; const s=(v==null)?0:v;
    ln.setAttribute("stroke-width",(1+s*8).toFixed(1));
    ln.setAttribute("opacity",(0.12+s*0.88).toFixed(2));
  });
  svg.querySelectorAll(".comp-val").forEach(tx=>{
    const v=vals[tx.getAttribute("data-key")]; tx.textContent=(v==null)?"":v.toFixed(2);
  });
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
  if(active.size){ const f=[...active][0]; if(NET[f]){ const badge=mk("text",{x:NET[f].x,y:NET[f].y-22,"text-anchor":"middle",class:"fire-badge","font-size":13},el("netNodes")); badge.textContent="発火!"; bgFor(badge, el("netNodes"), "lblbg badgebg"); } }
  setGauges(m.gauges);
  updateNN(m.fire);
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
  // 自分ひとりの喃語(kind==="babble")は"会話"ではないので出さない。会話（親への応答・
  // 要求語・授乳/慰め時の発話）だけ表示する。喃語そのものは体ビュー側で見える。
  if(u && ev.kind!=="babble") out.push({t:ev.t, who:"taro", text:u, cry:/泣/.test(u)});
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
    d.title="クリックでこの発言の時刻へ移動";
    d.addEventListener("click", ()=>jumpToTime(m.t));   // 発言時刻へスキップ
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
  else { el("sceneLabel").textContent=KIND_LABEL[it.kind]||it.kind||""; renderMoment(it); }
  updatePlayhead();
  updateChat(it.t);
  updateComp(it.t);
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
    else if(o.type==="comprehension") comprehension.push({t:o.t,mama:o.mama,maman:o.maman,aua:o.aua});
    else if(o.type==="snap") gauges={hunger:o.hunger,ne:o.ne,dopamine:o.dopamine,happiness:o.happiness};
    else if(o.type==="event"){ const g=(o.hunger!=null)?{hunger:o.hunger,ne:o.ne,dopamine:o.dopamine,happiness:o.happiness}:Object.assign({},gauges);
      out.push({t:o.t,kind:o.kind,active:o.modules||[],flows:o.flows||[],utter:o.utter||"",say:o.say||"",fire:o.fire||null,gauges:g}); }
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

function sigOf(ev){ const s=KIND_SIG[ev.kind]; return s==null?1:s; }
// 粗い粒度＝各時間窓で「最も重要なイベント1件」を実データのまま取り出す。
// 集約して臓器を全部光らせる（矢印も出ない）のをやめ、イベント発生時の実際の動き
// （発火した部品＋流れの矢印＋その瞬間の数値）をそのまま見せる。
function downsampleRaw(raw, windowSec){
  const pick=new Map();
  raw.forEach(ev=>{ if(ev.counts) return; const k=Math.floor(ev.t/windowSec);
    const cur=pick.get(k); if(!cur||sigOf(ev)>sigOf(cur)) pick.set(k,ev); });
  return [...pick.values()].sort((a,b)=>a.t-b.t);
}
function buildGransFromRaw(){
  const raw=datasets.raw; if(!raw||!raw.length) return;
  const W={hour:3600, day:86400, month:2592000, year:31536000};
  for(const g in W) datasets[g]=downsampleRaw(raw, W[g]);
}

function afterLoad(){
  buildGransFromRaw();                 // rawがあれば粗い粒度も実イベントで作り直す
  const order=["hour","day","month","year","raw"];
  const g=order.find(x=>datasets[x]&&datasets[x].length);
  if(g) setActive(g);
}

// ---- 初期化 ----
function init(){
  buildBody(); buildNet(); buildComp(); buildNNView();
  el("nm_region").onclick=()=>{ el("netSvg").hidden=false; el("nnSvg").hidden=true;
    el("nm_region").classList.add("sel"); el("nm_nn").classList.remove("sel"); };
  el("nm_nn").onclick=()=>{ el("netSvg").hidden=true; el("nnSvg").hidden=false;
    el("nm_nn").classList.add("sel"); el("nm_region").classList.remove("sel"); };
  enableZoom("panel_body",[0,0,300,380],box=>{ bodyBox=box; layoutBodyLabels(); });
  enableZoom("panel_net",[0,0,460,300]);

  // 臓器の図形をクリックしても説明を出す（名前ラベルと同じ）。閉じる操作も配線。
  for(const id in BODY){ const o=el("organ_"+id);
    if(o) o.addEventListener("click", e=>{ e.stopPropagation(); showOrganInfo(id, e); }); }
  el("organInfoX").addEventListener("click", hideOrganInfo);
  document.addEventListener("click", e=>{
    if(!e.target.closest("#organInfo") && !e.target.closest(".blabel") && !e.target.closest(".organ")) hideOrganInfo();
  });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape") hideOrganInfo(); });

  // 既定：同じフォルダに置かれた概観/生ログ/マイルストーンを取りに行く。
  // 無ければ sample_trace.json（生の見本）を使う。
  const files=["trace_overview_year.jsonl","trace_overview_month.jsonl","trace_overview_day.jsonl",
               "trace_overview_hour.jsonl","milestones.json","trace.jsonl"];
  const bust="?_="+Date.now();   // ブラウザのHTTPキャッシュを避けて常に最新のログを読む
  Promise.allSettled(files.map(f=>fetch(f+bust).then(r=>r.ok?r.text():Promise.reject()).then(t=>routeFile(f,t))))
    .then(()=>{
      if(Object.keys(datasets).length) afterLoad();
      else fetch("sample_trace.json"+bust).then(r=>r.text()).then(t=>{ datasets.raw=parseAny(t); setActive("raw"); })
        .catch(()=>{ el("sceneLabel").textContent="ログが読めません（サーバー経由で開く／フォルダを選択）"; });
    });

  el("playBtn").onclick=()=>timer?stop():play();
  el("prevBtn").onclick=()=>{stop();go(idx-1);};
  el("nextBtn").onclick=()=>{stop();go(idx+1);};
  el("seek").oninput=e=>{stop();go(parseInt(e.target.value,10));};
  el("speedSel").onchange=e=>{speed=parseFloat(e.target.value); if(timer)play();};
  GRANS.forEach(([g])=>{ const b=el("gran_"+g); if(b) b.onclick=()=>setActive(g); });

  el("folderInput").onchange=e=>{
    datasets={}; milestones=[]; comprehension=[];
    const fs=[...e.target.files]; let pending=fs.length;
    if(!pending) return;
    fs.forEach(f=>{ const r=new FileReader(); r.onload=()=>{ routeFile(f.name,r.result); if(--pending===0) afterLoad(); }; r.readAsText(f); });
  };
}
document.addEventListener("DOMContentLoaded", init);

// レイアウトはCSSグリッドで固定（隙間なくきっちり敷き詰め）。ウィンドウ操作は廃止。
// ウィンドウのリサイズ時に、体ラベルの文字サイズ（実ピクセル基準）を再計算する。
let _rt; addEventListener("resize",()=>{ clearTimeout(_rt); _rt=setTimeout(layoutBodyLabels,150); });
