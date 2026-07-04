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
const KIND_LABEL = { babble:"喃語", babble_response:"喃語に親が反応", word_request:"要求語",
  feed:"授乳", comfort:"あやし", cry:"泣く", sleep:"睡眠", excrete:"排泄", sleep_word:"寝かしつけ" };
const KIND_SIG = { feed:6, word_request:6, excrete:5, babble_response:4, cry:3, comfort:2,
  sleep_word:2, sleep:1, babble:0 };
// 太郎の発話はすべて「自発」が起点。バッジは"その後どうなったか"を示す。
const TARO_ORIGIN = {
  babble:          ["自発", "origin-spont"],       // 自発 → 親は反応せず
  babble_response: ["自発→親反応", "origin-resp"],  // 自発 → 親が気づいて反応
  word_request:    ["自発→要求", "origin-req"],     // 自発 → 空腹等で要求として成立
};

const GAUGES = [
  { key:"hunger", label:"空腹" }, { key:"ne", label:"探索(NE)" },
  { key:"dopamine", label:"ドーパミン分泌量" }, { key:"happiness", label:"幸福度" },
  { key:"pleasure", label:"味の快(liking)" },
  { key:"sleepiness", label:"眠さ" }, { key:"discomfort", label:"不快" },
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
let comprehension = []; // {t, mama, nenne, dakko} 月イチの「語→食べ物予期」測定（3語の区別＝理解）
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
// 日付＋時刻（時:分:秒）。チャットで「いつ」を秒まで見せる用。
function fmtDateTime(t) {
  const sod = Math.floor(t % 86400);
  const p = n => String(n).padStart(2, "0");
  return `${fmtDate(t)} ${p(Math.floor(sod/3600))}:${p(Math.floor((sod%3600)/60))}:${p(sod%60)}`;
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
let organIntensity=null;   // {organ:0-1} 実測活動量。あれば強度で光らせる（無ければactiveModsで二値）
// 強度→見た目のマッピング（安定=薄い/小さい、活発=濃い/太い/大きく光る）。CSSの固定値
// (stroke-width:1.8, glow:5px 固定)だと0.45〜1.0のopacity差が暗い背景でほぼ見分けが
// つかなかったため、輪郭の太さ・グローの大きさ・塗りの不透明度の3つをインラインで
// 直接、強度に応じて連続的に変える（CSSの.onは色そのものの切替だけに使う）。
function paintIntensity(el, inten){
  if(!el) return;
  // .shape(体ビュー臓器) → circle(NN部位図ノード) → el自身(markerはel自身がshape) の順で対象を探す
  const shape=el.querySelector(".shape")||el.querySelector("circle")||el;
  if(inten<=0){ shape.style.filter=""; shape.style.strokeWidth=""; shape.style.fillOpacity=""; return; }
  shape.style.fillOpacity=(0.12+0.7*inten).toFixed(2);           // 塗り：ほぼ透明→ほぼ不透明
  shape.style.strokeWidth=(1.0+2.5*inten).toFixed(2);            // 輪郭：細い→太い
  const blur=(2+10*inten).toFixed(1);
  shape.style.filter=`drop-shadow(0 0 ${blur}px var(--fire))`;   // グロー：小さい→大きい
}
function applyActive(){
  for(const id in BODY){
    // 実測organsがあれば強度で判定（点灯とみなす最小閾値0.05）、無ければ従来の二値(modules)。
    let on, inten;
    if(organIntensity && organIntensity[id]!=null){ inten=organIntensity[id]; on=inten>=0.05; }
    else { on=activeMods.has(id); inten=on?1:0; }
    const o=el("organ_"+id);
    if(o){ o.classList.toggle("on",on); paintIntensity(o, on?inten:0); }
    const l=el("leader_"+id); if(l) l.classList.toggle("on",on);
    const t=el("blabel_"+id); if(t) t.classList.toggle("on",on);
    const bg=el("blabelbg_"+id); if(bg) bg.classList.toggle("on",on);
  }
}
// NN活性ビュー（かっこよさ用・汎用）：実アーキテクチャ 入力→GRU隠れ128→出口 を描き、
// イベントごとに発火した隠れノード(rec.fire)を光らせる。意味は読めないが「本物のNNが
// 動いている」感を出す。どのモデルでも fire さえ記録すれば使える汎用ビュー。
// 正直な図：太郎の本物のGRUは「1層・128ユニット」。128を16×8に並べる（層は1つ。並びは
// 見やすさのための配置で、層を増やしているわけではない）。発火(rec.fire)は実ノード番号に直結。
const NN_HID = 128;
const NN_COLORS = ["#46c8e0","#d86ac0","#ffd23a"];
let _nnLit=[];
let _nnEdges=[];   // {el, h:隠れ層ユニット番号} 発火時にその線を光らせる
function buildNNView(){
  const svg=el("nnSvg"); if(!svg) return; svg.innerHTML="";
  const gE=mk("g",{},svg), gN=mk("g",{},svg), gL=mk("g",{},svg);
  const inX=34, inYs=[]; for(let i=0;i<6;i++){ const y=64+i*34; inYs.push(y);
    mk("circle",{cx:inX,cy:y,r:4.5,class:"nn-node nn-in"},gN); }
  const cols=16, rows=8, x0=96, x1=330, y0=44, y1=272, hp=[];
  for(let i=0;i<NN_HID;i++){ const c=i%cols, r=Math.floor(i/cols);
    const x=x0+(x1-x0)*c/(cols-1), y=y0+(y1-y0)*r/(rows-1); hp.push([x,y]);
    mk("circle",{cx:x,cy:y,r:3.2,class:"nn-node",id:"nnh_"+i},gN); }
  const heads=[["食べ物予期",70],["発声",118],["価値",166],["次の音の予測",214]], hX=424, hpos=[];
  heads.forEach(([lab,y])=>{ hpos.push([hX,y]); mk("circle",{cx:hX,cy:y,r:6,class:"nn-node nn-head"},gN);
    const t=mk("text",{x:hX+12,y:y+3,"text-anchor":"start",class:"nn-headlabel"},gL); t.textContent=lab; });
  const rnd=a=>a[Math.floor(Math.random()*a.length)]; let ci=0;
  _nnEdges=[];
  // 実際のGRUは全結合＝どのユニットも入力・出力とつながっている。全部描くと潰れるので
  // 各隠れユニットにつき左右1本ずつだけ描く（孤立点が出ないよう全128個をカバー）。
  for(let hi=0;hi<NN_HID;hi++){ const p=hp[hi];
    const y=rnd(inYs);  // 入力→隠れ（左・薄く）
    const l1=mk("line",{x1:inX,y1:y,x2:p[0],y2:p[1],class:"nn-edge2",stroke:"#5b6b7d","stroke-width":.5,opacity:.22},gE);
    _nnEdges.push({el:l1,h:hi});
    const q=rnd(hpos);  // 隠れ→ヘッド（右・色つき）
    const l2=mk("line",{x1:p[0],y1:p[1],x2:q[0],y2:q[1],class:"nn-edge2",stroke:NN_COLORS[ci++%3],"stroke-width":.6,opacity:.4},gE);
    _nnEdges.push({el:l2,h:hi}); }
  svg.insertBefore(gE,gN);
  const t=mk("text",{x:x0,y:28,class:"nn-headlabel"},gL);
  t.textContent="GRU隠れ層：1層 × 128ユニット（発火＝活性上位32が光る）";
  updateNN(items&&items[idx]?items[idx].fire:null);
}
function updateNN(fire){
  _nnLit.forEach(i=>{ const n=el("nnh_"+i); if(n) n.classList.remove("on"); });
  _nnLit=Array.isArray(fire)?fire:[];
  _nnLit.forEach(i=>{ const n=el("nnh_"+i); if(n) n.classList.add("on"); });
  // 発火したユニットにつながる線も光らせる
  const lit=new Set(_nnLit);
  _nnEdges.forEach(e=>{ if(lit.has(e.h)) e.el.classList.add("on"); else e.el.classList.remove("on"); });
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
  // NETWORK/部位図も体ビューと同じ実測(organIntensity)を優先して光らせる（無ければ従来通り
  // modulesの二値）。2つのパネルで発火の判定・見た目がズレないようロジックを共通化する。
  for (const id in NET) {
    let on, inten;
    if(organIntensity && organIntensity[id]!=null){ inten=organIntensity[id]; on=inten>=0.05; }
    else { on=activeSet.has(id); inten=on?1:0; }
    const g=el("nn_"+id);
    if(g){ g.classList.toggle("on",on); paintIntensity(g, on?inten:0); }
  }
}
// 理解の配線図：3つの語 → 「食べ物予期」ノード。線が太い/明るいほど「その語で食べ物が
// 来る」と結びついている＝理解している。まんまが育って光り、あうあは暗いままなのを見せる。
const COMP_WORDS = [["まんま","mama",34],["ねんね","nenne",78],["だっこ","dakko",122]];
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
  const vals={mama:cur.mama, nenne:cur.nenne, dakko:cur.dakko};
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
  organIntensity = m.organs || null;   // 実測があれば強度表示、無ければmodulesで二値（後方互換）
  const active=new Set(m.active||[]); setModules(active); clearFlows();
  // 実測organsがある時：流れ(矢印)は「両端がともに点灯している」ものだけ描く。矢印そのものは
  // 実測でない模式的な経路なので、点灯していない臓器へ矢印が伸びてノードと矛盾しないようにする。
  const litByMeasure = id => organIntensity && organIntensity[id]!=null && organIntensity[id]>=0.3;
  (m.flows||[]).forEach(([a,b])=>{
    if(organIntensity && !(litByMeasure(a) && litByMeasure(b))) return;   // 片方でも消灯なら流れを描かない
    if(BODY[a]&&BODY[b]){ const c=seg(BODY[a].mx,BODY[a].my,BODY[b].mx,BODY[b].my,8,9);
      mk("line",{x1:c[0],y1:c[1],x2:c[2],y2:c[3],class:"flow","marker-end":"url(#arrowF)"},el("bodyFlows")); }
    if(NET[a]&&NET[b]){ const c=seg(cx(a),cy(a),cx(b),cy(b),17,23,6);
      mk("line",{x1:c[0],y1:c[1],x2:c[2],y2:c[3],class:"net-flow","marker-end":"url(#arrowF)"},el("netFlows")); }
    document.querySelectorAll(".net-edge").forEach(e=>{const[ea,eb]=e.getAttribute("data-edge").split("|");
      if((ea===a&&eb===b)||(ea===b&&eb===a))e.classList.add("on");});
  });
  // 「発火!」バッジ：実測がある時は最も活動が強い臓器に付ける（無い時は従来=先頭module）。
  let badgeId=null;
  if(organIntensity){
    let best=-1; for(const id in organIntensity){ if(NET[id]&&organIntensity[id]>best){ best=organIntensity[id]; badgeId=id; } }
    if(best<0.3) badgeId=null;
  } else if(active.size){ badgeId=[...active][0]; }
  if(badgeId && NET[badgeId]){ const badge=mk("text",{x:NET[badgeId].x,y:NET[badgeId].y-22,"text-anchor":"middle",class:"fire-badge","font-size":13},el("netNodes")); badge.textContent="発火!"; bgFor(badge, el("netNodes"), "lblbg badgebg"); }
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
// 発話でない「行動・出来事」も時系列に混ぜて見せる（会話とは別スタイル）。
const ACTION_LABEL = {
  feed:"🍼 授乳した", cry:"😢 泣きだした", excrete:"💩 排泄（おむつが汚れた）",
  sleep:"😴 眠った", comfort:"🤱 あやした",
};
// 1イベント→発話（親/太郎）＋行動、を時系列アイテムとして返す。
function messagesOf(ev){
  const out=[];
  const say=ev.say||ev.parent;
  if(say) out.push({t:ev.t, who:"parent", type:"speech", text:say});
  const u=ev.utter;
  // 自分ひとりの喃語(kind==="babble")は既定では"会話"でないので出さない。ただし
  // 「喃語も表示」チェック時は太郎の独り言も含める（全発話を見たいとき用）。
  const showBab = el("showBabble") && el("showBabble").checked;
  if(u && (ev.kind!=="babble" || showBab)) out.push({t:ev.t, who:"taro", type:"speech", text:u, cry:/泣/.test(u), bab:ev.kind==="babble", kind:ev.kind});
  // 行動（授乳・泣き・排泄・睡眠・あやし）を1行の出来事として追加
  const act=ACTION_LABEL[ev.kind];
  if(act) out.push({t:ev.t, type:"action", text:act});
  return out;
}
// 会話の元データは常に生イベント列（バケツには発話が無いため）。
function chatSource(){
  if(datasets.raw && datasets.raw.length) return datasets.raw;
  return items.filter(it=>!it.counts);
}
function rebuildChat(){
  const box=el("chatList"); if(!box) return;
  chatMsgs=[]; _chatIdx=-1;   // 差分更新の追跡をリセット
  chatSource().forEach(ev=>{ if(!ev.counts) messagesOf(ev).forEach(m=>chatMsgs.push(m)); });
  chatMsgs.sort((a,b)=>a.t-b.t);
  box.innerHTML="";
  if(!chatMsgs.length){ box.innerHTML='<div class="chat-empty">この期間に発話ログはありません</div>'; return; }
  // クリック→時刻ジャンプは1つの委譲リスナーで処理（1.6万件の個別リスナーを付けない）。
  if(!box._jumpDelegated){
    box._jumpDelegated=true;
    box.addEventListener("click", e=>{ const t=e.target.closest(".msg"); if(t && t.dataset.t) jumpToTime(+t.dataset.t); });
  }
  // DocumentFragmentに一括構築してから1回だけ挿入（挿入毎のリフローを避ける）。
  const frag=document.createDocumentFragment();
  chatMsgs.forEach(m=>{
    const d=document.createElement("div");
    d.dataset.t=m.t;
    if(m.type==="action"){
      d.className="msg action";
      d.innerHTML='<span class="who">'+fmtDateTime(m.t)+"</span>"+escapeHtml(m.text);
    } else {
      d.className="msg "+m.who+(m.cry?" cry":"")+(m.bab?" bab":"");
      // 太郎の発話は全て「自発」が起点。その後の結末で色分けする（親反応・要求成立）。
      let badge="";
      if(m.who==="taro" && !m.cry){
        const b=TARO_ORIGIN[m.kind];
        if(b) badge='<span class="origin '+b[1]+'">'+b[0]+"</span>";
      }
      d.innerHTML='<span class="who">'+(m.who==="parent"?"親":"太郎")+" ・ "+fmtDateTime(m.t)+"</span>"+badge+escapeHtml(m.text);
    }
    d.title="クリックでこの時刻へ移動";
    frag.appendChild(d);
  });
  box.appendChild(frag);
}
let _chatIdx=-1;   // 直近で"seen"にした末尾の位置（差分更新用）
function updateChat(curT){
  const box=el("chatList"); if(!box||!chatMsgs.length) return;
  const kids=box.children;
  const target=curT+0.5;
  // 二分探索：t<=target を満たす最大index（chatMsgsは時刻昇順）。全走査O(N)を避ける。
  let lo=0, hi=chatMsgs.length-1, nb=-1;
  while(lo<=hi){ const mid=(lo+hi)>>1; if(chatMsgs[mid].t<=target){ nb=mid; lo=mid+1; } else hi=mid-1; }
  if(nb===_chatIdx) return;                       // 変化なし＝何もしない（重さの元を断つ）
  // 前回位置との差分だけ .seen を付け外し（YouTube的に"動いた分"だけ処理）
  if(nb>_chatIdx){ for(let i=_chatIdx+1;i<=nb;i++){ const k=kids[i]; if(k) k.classList.add("seen"); } }
  else { for(let i=nb+1;i<=_chatIdx;i++){ const k=kids[i]; if(k) k.classList.remove("seen"); } }
  if(_chatIdx>=0 && kids[_chatIdx]) kids[_chatIdx].classList.remove("cur");
  _chatIdx=nb;
  if(nb>=0 && kids[nb]){ kids[nb].classList.add("cur"); kids[nb].scrollIntoView({block:"nearest"}); }
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
  if(!items.length) return;
  // items は時刻昇順 → 二分探索で最近傍を探す（O(N)→O(log N)）
  let lo=0, hi=items.length-1;
  while(lo<hi){ const mid=(lo+hi)>>1; if(items[mid].t < t) lo=mid+1; else hi=mid; }
  let best=lo;
  if(lo>0 && Math.abs(items[lo-1].t - t) <= Math.abs(items[lo].t - t)) best=lo-1;
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
    else if(o.type==="comprehension") comprehension.push({t:o.t,mama:o.mama,nenne:o.nenne,dakko:o.dakko});
    else if(o.type==="snap") gauges={hunger:o.hunger,ne:o.ne,dopamine:o.dopamine,happiness:o.happiness};
    else if(o.type==="event"){
      // ゲージ：記録にある数値キーを全部拾う（hunger等が無い行は直前のスナップを流用）
      const GK=["hunger","ne","dopamine","happiness","pleasure","sleepiness","discomfort"];
      let g;
      if(o.hunger!=null){ g={}; GK.forEach(k=>{ if(o[k]!=null) g[k]=o[k]; }); }
      else g=Object.assign({},gauges);
      out.push({t:o.t,kind:o.kind,active:o.modules||[],flows:o.flows||[],utter:o.utter||"",say:o.say||"",
                fire:o.fire||null,gauges:g,organs:o.organs||null}); }
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
  const showNet=on=>{ el("netSvg").style.display=on?"block":"none";
    el("nnSvg").style.display=on?"none":"block";
    el("nm_region").classList.toggle("sel",on); el("nm_nn").classList.toggle("sel",!on); };
  showNet(true);                       // 初期は部位図だけ（インラインstyleで確実に）
  el("nm_region").onclick=()=>showNet(true);
  el("nm_nn").onclick=()=>showNet(false);
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
  { const sb=el("showBabble"); if(sb) sb.addEventListener("change", ()=>{ rebuildChat(); render&&render(); }); }

  // 既定：同じフォルダに置かれた概観/生ログ/マイルストーンを取りに行く。
  // 無ければ sample_trace.json（生の見本）を使う。
  const files=["trace_overview_year.jsonl","trace_overview_month.jsonl","trace_overview_day.jsonl",
               "trace_overview_hour.jsonl","milestones.json","trace.jsonl"];
  const bust="?_="+Date.now();   // ブラウザのHTTPキャッシュを避けて常に最新のログを読む
  // 今どのランを表示しているか（_meta.jsonがあれば読む。無ければ「不明」と表示）。
  // ここで読んだ run_name を覚えておき、下のポーリングで「差し替わったら自動リロード」に使う。
  let _loadedRun=null;
  fetch("_meta.json"+bust).then(r=>r.ok?r.json():Promise.reject()).then(m=>{
    _loadedRun=m.run_name||"?";
    el("runBadge").textContent=`📂 表示中: ${_loadedRun}${m.copied_at?"（読み込み: "+m.copied_at+"）":""}`;
  }).catch(()=>{ _loadedRun=""; el("runBadge").textContent="📂 表示中: (このフォルダの既定ログ／_meta.jsonなし)"; });
  // Claudeが別のシミュレーションのデータに丸ごと差し替えた時、データファイル自体は
  // ページ読み込み時にしか取りに行かない（毎回全部読み直すのは無駄なため）。
  // _meta.jsonのrun_nameが変わっていたら「差し替わった」と判断し、ページ全体を
  // 自動リロードすることで、ユーザーが手動更新しなくても新しいランに切り替わる。
  setInterval(()=>{
    fetch("_meta.json?_="+Date.now()).then(r=>r.ok?r.json():null).then(m=>{
      if(_loadedRun===null) return;   // 初回読み込みがまだなら判定しない
      const name=m?(m.run_name||"?"):"";
      if(name!==_loadedRun) location.reload();
    }).catch(()=>{});
  }, 2000);
  Promise.allSettled(files.map(f=>fetch(f+bust).then(r=>r.ok?r.text():Promise.reject()).then(t=>routeFile(f,t))))
    .then(()=>{
      if(Object.keys(datasets).length) afterLoad();
      else fetch("sample_trace.json"+bust).then(r=>r.text()).then(t=>{ datasets.raw=parseAny(t); setActive("raw"); })
        .catch(()=>{ el("sceneLabel").textContent="ログが読めません。「ビューアを開く.bat」から起動するか、右上のフォルダ選択でtrace_*.jsonlを選んでください（file://直接では読めません）"; });
    });

  // 外部（Claude）から特定時点へ誘導する仕組み：_goto.json を数秒おきに確認し、
  // seq（連番）が前回と変わっていたら jumpToTime(t) する。ユーザーが普段通り
  // ブラウザで開いているこの画面が、サーバー側のファイル書き換えだけで動く
  // （拡張機能や新しいサーバーは不要）。gran指定があれば先にその粒度へ切り替える
  // （例：生ログ(raw)で1件ずつ見せたい時）。
  let _lastGotoSeq=null;
  setInterval(()=>{
    if(!items.length) return;
    fetch("_goto.json?_="+Date.now()).then(r=>r.ok?r.json():null).then(g=>{
      if(!g || g.seq===_lastGotoSeq) return;
      _lastGotoSeq=g.seq;
      if(g.gran && datasets[g.gran] && datasets[g.gran].length && g.gran!==activeGran) setActive(g.gran);
      jumpToTime(g.t);
    }).catch(()=>{});
  }, 1500);

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
