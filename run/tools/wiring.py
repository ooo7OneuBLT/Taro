"""太郎の中身の配線図を描く（SVG）。実装したもの全部を出し、使っていないものは薄く。

【なぜ要るか、2026-07-30】ユーザーの要望：

> 今のプロジェクトで実装したものはすべて表示して、実験に使ってないものは薄くするとか？
> その図でどの経路を通ってどの判断がされたのか。みたいなのが分かるといい！
> 長井研究室での発表でも使えそう！

太郎の設定は30個以上あり、「この実験は何が違うのか」が読まないと分からない。
実際そこで事故が起きた（学習は関節モード・測定は筋肉モードという**別の体**）。

【何が見えるか】
    ・実装したもの**全部**（37個）が並ぶ。この実験で効いていないものは薄い
    ・枠の色＝根拠の段階（Tier1 一次文献の実測 ／ Tier3 恣意的 ／ 逸脱 ／ 未実装）
      ⇒ **どこが人間模倣で、どこが自分の決めごとか**が一目で分かる
    ・日本語をメインに、英語を小さく添える（発表・論文で使い回せるように）
    ・通った回数（段階3）：counters を渡すと線の太さと数字に反映する

【使い方】
    .venv/Scripts/python.exe -m run.tools.wiring E/experiments/<名前>.json
    → 同じフォルダに 配線図.html を作る（単体で開ける）
  ダッシュボード（run/tools/dashboard.py）にも埋め込まれ、30秒ごとに最新になる。

注意：中身の定義（機構の一覧・繋がり・根拠のラベル）は `run/wiring_map.py`。
  ここは**描くだけ**。機構を足すときは定義側を直す。
"""
import argparse
import html
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.wiring_map import (COLUMNS, LOWER, NODES, EDGES, TIER_STYLE,  # noqa: E402
                            is_on, node_index)

# 絵の寸法
BOX_W, BOX_H = 176, 46
COL_GAP, ROW_GAP = 34, 16
PAD_X, PAD_Y = 18, 64
LOWER_GAP = 54          # 上段と下段のあいだ


def _order_in_columns(order, rounds=12):
    """列の中の並び順を「繋がる相手の平均の高さ」に寄せて決める。

    【なぜ、2026-07-30】最初は定義に書いた順に上から並べていたので、
    線が別の箱を35件横切り、線どうしの交差が51箇所あった（＝読みにくい）。
    ⇒ グラフ描画の標準的な手法（バリセンター法：Sugiyama らの層別描画の一部）で、
      **各ノードを「繋がっている相手の平均の高さ」に近づける**ように並べ替える。
    注意：列（感覚／脳／体…）の割り当ては**変えない**。意味で決めた区分なので、
      並び順だけを機械的に整える。
    """
    nb = {}          # ノード → 繋がっている相手
    for a, b, _l, _c in EDGES:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    for _ in range(rounds):
        rank = {nid: i for col in order for i, nid in enumerate(order[col])}
        for col, ids in order.items():
            def bary(nid):
                ns = [rank[o] for o in nb.get(nid, ()) if o in rank]
                return sum(ns) / len(ns) if ns else rank[nid]
            # 安定に並べる（同じ値なら元の順を保つ）＝実行ごとに絵が変わらない
            order[col] = sorted(ids, key=lambda n: (bary(n), rank[n]))
    return order


def layout():
    """機構の置き場所（x, y）を決める。上段＝感覚→体の流れ、下段＝報酬と調節。"""
    pos, col_title = {}, []
    # ---- 列ごとの中身（定義の順）------------------------------------------
    order = {}
    for key, jp, eng in list(COLUMNS) + list(LOWER):
        order[key] = [n[0] for n in NODES if n[3] == key]
    # 注意：自動での並べ替え（バリセンター法）は**試したが悪化した**：
    #   線が別の箱を横切る 35→42件、線どうしの交差 51→57箇所（2026-07-30 実測）。
    #   理由＝この図は綺麗な層構造ではない（報酬の輪が下段から上段へ長く飛ぶ、
    #   多対多の繋がりが多い）ので、層別描画の前提が成り立たない。
    #   ⇒ 機械に任せず、**人が動かして決める**（HTMLで箱をドラッグできる）。
    #     決めた位置は `run/wiring_map.py` の POSITIONS に貼れば固定される。
    if os.environ.get("TARO_WIRING_AUTOSORT") == "1":
        order = _order_in_columns(order)     # 試したいとき用に残す

    x = PAD_X
    upper_h = 0
    for key, jp, eng in COLUMNS:
        col_title.append((x, PAD_Y - 30, jp, eng))
        y = PAD_Y
        for nid in order[key]:
            pos[nid] = (x, y)
            y += BOX_H + ROW_GAP
        upper_h = max(upper_h, y)
        x += BOX_W + COL_GAP
    total_w = x - COL_GAP + PAD_X
    y0 = upper_h + LOWER_GAP
    x = PAD_X
    lower_h = y0
    for key, jp, eng in LOWER:
        col_title.append((x, y0 - 30, jp, eng))
        y = y0
        for nid in order[key]:
            pos[nid] = (x, y)
            y += BOX_H + ROW_GAP
        lower_h = max(lower_h, y)
        x += BOX_W + COL_GAP
    # 人が決めた位置があれば上書きする（HTMLの「位置を書き出す」で作った値）
    try:
        from run.wiring_map import POSITIONS
    except ImportError:
        POSITIONS = {}
    if POSITIONS:
        for nid, xy in POSITIONS.items():
            if nid in pos:
                pos[nid] = (float(xy[0]), float(xy[1]))
        total_w = max(total_w, max(p[0] for p in pos.values()) + BOX_W + PAD_X)
        lower_h = max(lower_h, max(p[1] for p in pos.values()) + BOX_H)
    return pos, col_title, total_w, lower_h + PAD_Y


def _edge_path(p1, p2):
    """箱と箱を結ぶ線。横に離れているときは曲線、縦に近いときは直角に近い形。"""
    x1, y1 = p1[0] + BOX_W, p1[1] + BOX_H / 2      # 右端から出る
    x2, y2 = p2[0], p2[1] + BOX_H / 2              # 左端に入る
    if x2 < x1:        # 戻る線（体→感覚など）は下に回す
        x1, y1 = p1[0] + BOX_W / 2, p1[1] + BOX_H
        x2, y2 = p2[0] + BOX_W / 2, p2[1] + BOX_H
        dy = 26 + abs(x2 - x1) * 0.03
        return (f"M{x1:.0f},{y1:.0f} C{x1:.0f},{y1+dy:.0f} "
                f"{x2:.0f},{y2+dy:.0f} {x2:.0f},{y2:.0f}"), (x1 + x2) / 2, max(y1, y2) + dy * 0.75
    mx = (x1 + x2) / 2
    return (f"M{x1:.0f},{y1:.0f} C{mx:.0f},{y1:.0f} {mx:.0f},{y2:.0f} "
            f"{x2:.0f},{y2:.0f}"), mx, (y1 + y2) / 2 - 5


def _fit(text, width, size):
    """字が箱から出ないよう、長ければ文字を小さくする（字が重なるのを防ぐ）。"""
    # 日本語は1文字≒size、英数字は≒size*0.55 として見積もる
    w = sum(size if ord(c) > 0x2000 else size * 0.55 for c in text)
    return size if w <= width else max(7.5, size * width / w)


def build_svg(cfg, counters=None):
    """配線図の SVG を返す。counters は {キー: 数値} で線の太さに反映（段階3）。"""
    counters = counters or {}
    pos, titles, W, H = layout()
    idx = node_index()
    on = {n[0]: is_on(n, cfg) for n in NODES} if cfg is not None else {
        n[0]: True for n in NODES}

    out = [f'<svg viewBox="0 0 {W} {H}" class="wire" role="img" '
           f'aria-label="太郎の中身の配線図" data-w="{W}" data-h="{H}">']
    out.append('<defs>'
               '<marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
               'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" '
               'fill="#8a97ab"/></marker>'
               '<marker id="arOn" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
               'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" '
               'fill="#3d4d63"/></marker></defs>')
    # ここから下は「カメラ」の中。JS が transform で拡大・移動する
    out.append('<g id="cam">')

    # ---- 列の見出し --------------------------------------------------------
    for x, y, jp, eng in titles:
        out.append(f'<text x="{x}" y="{y}" class="coljp">{html.escape(jp)}</text>')
        out.append(f'<text x="{x}" y="{y+13}" class="coleng">{html.escape(eng)}</text>')

    # ---- 線（先に描いて箱の下に置く）----------------------------------------
    for a, b, label, ckey in EDGES:
        if a not in pos or b not in pos:
            continue
        live = on.get(a, True) and on.get(b, True)
        d, lx, ly = _edge_path(pos[a], pos[b])
        n = counters.get(ckey) if ckey else None
        # 通った回数（counters）が分かっている線は太くする。
        #   基準は「その値がどれだけ大きいか」ではなく**0か0でないか**に留める
        #   ＝単位が違う量（回数・割合・平均）を同じ物差しで太さにすると嘘になる。
        if not live:
            wdt = 1.4
        elif n is None:
            wdt = 1.9
        else:
            wdt = 1.9 + (2.4 if float(n) > 0 else 0.0)
        cls = "eOn" if live else "eOff"
        out.append(f'<path id="e_{a}__{b}" d="{d}" class="{cls}" '
                   f'stroke-width="{wdt:.1f}" '
                   f'marker-end="url(#{"arOn" if live else "ar"})"/>')
        txt = label
        if live and ckey and n is not None:
            txt = (label + f" {n}").strip()
        if txt:
            out.append(f'<text id="t_{a}__{b}" x="{lx:.0f}" y="{ly:.0f}" '
                       f'class="elab {"on" if live else "off"}" '
                       f'text-anchor="middle">{html.escape(txt)}</text>')

    # ---- 箱 ----------------------------------------------------------------
    for n in NODES:
        nid, jp, eng, col, tier, desc, _fn, _axis, _dev = n
        if nid not in pos:
            continue
        x, y = pos[nid]
        live = on.get(nid, True)
        color, tier_jp, tier_en = TIER_STYLE.get(tier, TIER_STYLE[None])
        cls = "box" + ("" if live else " off")
        title = f"{jp}（{tier or '根拠未記載'}）\n{desc}"
        out.append(f'<g id="n_{nid}" class="{cls}" data-x="{x}" data-y="{y}">'
                   f'<title>{html.escape(title)}</title>')
        out.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="7" '
                   f'stroke="{color}"/>')
        # 左端に根拠の色帯（枠だけだと薄いときに見分けづらい）
        out.append(f'<rect x="{x}" y="{y}" width="5" height="{BOX_H}" rx="2.5" '
                   f'fill="{color}" class="tierbar"/>')
        fs = _fit(jp, BOX_W - 22, 12.5)
        out.append(f'<text x="{x+12}" y="{y+19}" class="njp" '
                   f'font-size="{fs:.1f}">{html.escape(jp)}</text>')
        fs2 = _fit(eng, BOX_W - 22, 9.5)
        out.append(f'<text x="{x+12}" y="{y+33}" class="neng" '
                   f'font-size="{fs2:.1f}">{html.escape(eng)}</text>')
        if not live:
            out.append(f'<text x="{x+BOX_W-8}" y="{y+14}" class="offmark" '
                       f'text-anchor="end">OFF</text>')
        out.append('</g>')

    out.append('</g>')      # /cam
    out.append("</svg>")
    data = {"nodes": {nid: [pos[nid][0], pos[nid][1]] for nid in pos},
            "edges": [[a, b] for a, b, _l, _c in EDGES if a in pos and b in pos],
            "boxW": BOX_W, "boxH": BOX_H, "w": W, "h": H}
    out.append(f'<script id="wireData" type="application/json">'
               f'{json.dumps(data, ensure_ascii=False)}</script>')
    return "".join(out), W, H


def legend_html():
    rows = []
    for tier in ("Tier1", "Tier2", "Tier3", "逸脱", "未実装", None):
        color, jp, en = TIER_STYLE[tier]
        name = tier or "未記載"
        rows.append(f'<span class="lgw"><i style="background:{color}"></i>'
                    f'<b>{html.escape(name)}</b> {html.escape(jp)}'
                    f'<em>{html.escape(en)}</em></span>')
    return ('<div class="wlegend"><div class="wlt">枠の色＝根拠の段階'
            '<span class="eng">evidence level</span></div>' + "".join(rows) +
            '<div class="wnote">注意薄い箱＝この実験では使っていない機構（実装はある）。'
            '箱にマウスを乗せると説明が出ます。'
            '根拠が「未記載」のものは確かめる宿題です。</div></div>')


CSS = """
 svg.wire { width:100%; height:auto; min-width:900px; }
 svg.wire text.coljp { font-size:12.5px; fill:var(--ink); font-weight:700; }
 svg.wire text.coleng { font-size:9.5px; fill:var(--sub); font-style:italic; }
 svg.wire g.box rect { fill:var(--card); stroke-width:1.6; }
 svg.wire g.box text.njp { fill:var(--ink); }
 svg.wire g.box text.neng { fill:var(--sub); font-style:italic; }
 svg.wire g.box.off { opacity:.30; }
 svg.wire g.box.off rect { stroke-dasharray:4 3; }
 svg.wire text.offmark { font-size:8px; fill:#a0aec0; letter-spacing:.5px; }
 svg.wire path.eOn { fill:none; stroke:#5a6b83; }
 svg.wire path.eOff { fill:none; stroke:#cfd8e3; stroke-dasharray:3 4; }
 svg.wire text.elab { font-size:9px; }
 svg.wire text.elab.on { fill:#4a5568; }
 svg.wire text.elab.off { fill:#b6c2d1; }
 .wlegend { margin-top:10px; font-size:.8rem; color:var(--sub); }
 .wlt { font-weight:700; color:var(--ink); margin-bottom:5px; font-size:.86rem; }
 .wlt .eng { font-weight:400; font-style:italic; margin-left:7px; color:var(--sub); }
 .lgw { display:inline-flex; align-items:baseline; gap:5px; margin:0 14px 5px 0; }
 .lgw i { width:11px; height:11px; border-radius:2px; flex:0 0 auto; position:relative; top:1px; }
 .lgw em { font-style:italic; opacity:.7; margin-left:4px; }
 .wnote { margin-top:6px; }
 .wctl { display:flex; gap:9px; align-items:center; flex-wrap:wrap; margin:0 0 9px; }
 .wctl button { font:inherit; font-size:.8rem; padding:4px 11px; border-radius:6px;
   border:1px solid var(--line); background:var(--card); color:var(--ink); cursor:pointer; }
 .wctl button:hover { border-color:#8fa6c4; }
 .whelp { font-size:.76rem; color:var(--sub); }
 .wstat { font-size:.76rem; color:#276749; margin-left:auto; }
 svg.wire { touch-action:none; cursor:grab; }
 svg.wire g[id^="n_"] { cursor:move; }
 @media (prefers-color-scheme: dark) {
   svg.wire path.eOff { stroke:#3b4757; }
   svg.wire text.elab.off { fill:#5b6b80; }
 }
"""


def html_page(cfg, spec_name, counters=None):
    svg, W, H = build_svg(cfg, counters)
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>太郎の配線図 ── {html.escape(spec_name)}</title>
<style>
 :root {{ --ink:#1a202c; --sub:#4a5568; --line:#e2e8f0; --bg:#f7f9fc; --card:#fff; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:18px 16px 40px; background:var(--bg); color:var(--ink);
   font-family:"Yu Gothic","Hiragino Sans","Meiryo",system-ui,sans-serif; line-height:1.6; }}
 h1 {{ font-size:1.2rem; margin:0 0 3px; }}
 .sub {{ color:var(--sub); font-size:.82rem; margin-bottom:12px; }}
 .wrap {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
   padding:14px; overflow-x:auto; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --ink:#e8eef7; --sub:#a0aec0; --line:#2d3748; --bg:#161b22; --card:#1c2230; }}
 }}
{CSS}
</style></head><body>
<h1>太郎の中身の配線図 <span class="sub">Taro's internal wiring</span></h1>
<div class="sub">{html.escape(spec_name)}</div>
<div class="wrap">{CONTROLS}{svg}{legend_html()}</div>
{SCRIPT}
</body></html>"""


# 操作のためのスクリプト。ダッシュボードに埋め込むときも同じものを使う。
#   注意：線を引く式は Python の `_edge_path` と**同じもの**をここにも書いている。
#     箱を動かしたら線も追いかける必要があるため。片方だけ直すと形が食い違う。
SCRIPT = r"""
<script>
(function(){
  const svg = document.querySelector('svg.wire');
  if (!svg) return;
  const cam = svg.querySelector('#cam');
  const D = JSON.parse(document.getElementById('wireData').textContent);
  const BW = D.boxW, BH = D.boxH;
  const P = D.nodes;                       // 箱の位置（動かすとここが変わる）
  let sc = 1, tx = 0, ty = 0;              // 拡大率と移動量
  const moved = {};                        // 人が動かした箱だけ覚える

  function applyCam(){ cam.setAttribute('transform',
      'translate('+tx.toFixed(1)+','+ty.toFixed(1)+') scale('+sc.toFixed(3)+')'); }

  // Python の _edge_path と同じ式（片方だけ直さない）
  function edgePath(p1, p2){
    let x1 = p1[0]+BW, y1 = p1[1]+BH/2, x2 = p2[0], y2 = p2[1]+BH/2;
    if (x2 < x1){
      x1 = p1[0]+BW/2; y1 = p1[1]+BH;
      x2 = p2[0]+BW/2; y2 = p2[1]+BH;
      const dy = 26 + Math.abs(x2-x1)*0.03;
      return ['M'+x1.toFixed(0)+','+y1.toFixed(0)+' C'+x1.toFixed(0)+','+(y1+dy).toFixed(0)
              +' '+x2.toFixed(0)+','+(y2+dy).toFixed(0)+' '+x2.toFixed(0)+','+y2.toFixed(0),
              (x1+x2)/2, Math.max(y1,y2)+dy*0.75];
    }
    const mx = (x1+x2)/2;
    return ['M'+x1.toFixed(0)+','+y1.toFixed(0)+' C'+mx.toFixed(0)+','+y1.toFixed(0)
            +' '+mx.toFixed(0)+','+y2.toFixed(0)+' '+x2.toFixed(0)+','+y2.toFixed(0),
            mx, (y1+y2)/2-5];
  }
  function redraw(nid){
    const g = document.getElementById('n_'+nid);
    if (g) g.setAttribute('transform',
        'translate('+(P[nid][0]-(+g.dataset.x))+','+(P[nid][1]-(+g.dataset.y))+')');
    D.edges.forEach(function(e){
      if (e[0]!==nid && e[1]!==nid) return;
      const path = document.getElementById('e_'+e[0]+'__'+e[1]);
      if (!path) return;
      const r = edgePath(P[e[0]], P[e[1]]);
      path.setAttribute('d', r[0]);
      const t = document.getElementById('t_'+e[0]+'__'+e[1]);
      if (t){ t.setAttribute('x', r[1].toFixed(0)); t.setAttribute('y', r[2].toFixed(0)); }
    });
  }

  // ---- スクロールで拡大・縮小 ------------------------------------------
  svg.addEventListener('wheel', function(ev){
    ev.preventDefault();
    const r = svg.getBoundingClientRect();
    const k = D.w / r.width;                       // 画面の1pxが図の何単位か
    const mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
    const s2 = Math.min(4, Math.max(0.3, sc * (ev.deltaY < 0 ? 1.12 : 1/1.12)));
    tx = mx - (mx - tx) * (s2/sc); ty = my - (my - ty) * (s2/sc);
    sc = s2; applyCam();
  }, {passive:false});

  // ---- ドラッグ（背景＝全体を移動 ／ 箱＝その箱だけ移動）-----------------
  let drag = null;
  svg.addEventListener('pointerdown', function(ev){
    const r = svg.getBoundingClientRect(), k = D.w / r.width;
    const g = ev.target.closest ? ev.target.closest('g[id^="n_"]') : null;
    drag = {x: ev.clientX, y: ev.clientY, k: k,
            nid: g ? g.id.slice(2) : null,
            ox: g ? P[g.id.slice(2)][0] : tx, oy: g ? P[g.id.slice(2)][1] : ty};
    svg.setPointerCapture(ev.pointerId);
    svg.style.cursor = 'grabbing';
  });
  svg.addEventListener('pointermove', function(ev){
    if (!drag) return;
    const dx = (ev.clientX - drag.x) * drag.k, dy = (ev.clientY - drag.y) * drag.k;
    if (drag.nid){
      P[drag.nid] = [drag.ox + dx/sc, drag.oy + dy/sc];
      moved[drag.nid] = P[drag.nid];
      redraw(drag.nid);
    } else { tx = drag.ox + dx; ty = drag.oy + dy; applyCam(); }
  });
  svg.addEventListener('pointerup', function(ev){
    drag = null; svg.style.cursor = '';
    const n = Object.keys(moved).length;
    const s = document.getElementById('wStat');
    if (s) s.textContent = n ? ('動かした箱: ' + n + '個') : '';
  });

  // ---- ボタン ----------------------------------------------------------
  const btnReset = document.getElementById('wReset');
  if (btnReset) btnReset.onclick = function(){
    sc = 1; tx = 0; ty = 0; applyCam();
  };
  const btnCopy = document.getElementById('wCopy');
  if (btnCopy) btnCopy.onclick = function(){
    // 人が動かした箱だけを Python に貼れる形で出す
    const lines = Object.keys(P).sort().map(function(k){
      return '    "' + k + '": (' + Math.round(P[k][0]) + ', ' + Math.round(P[k][1]) + '),';
    });
    const txt = 'POSITIONS = {\n' + lines.join('\n') + '\n}';
    navigator.clipboard.writeText(txt).then(function(){
      btnCopy.textContent = 'コピーしました（run/wiring_map.py の POSITIONS に貼る）';
      setTimeout(function(){ btnCopy.textContent = '位置を書き出す'; }, 4000);
    }, function(){
      const w = window.open('', '_blank');
      w.document.write('<pre>' + txt.replace(/</g,'&lt;') + '</pre>');
    });
  };
})();
</script>
"""

CONTROLS = """
<div class="wctl">
  <button id="wReset" type="button">表示を元に戻す</button>
  <button id="wCopy" type="button">位置を書き出す</button>
  <span class="whelp">スクロール＝拡大・縮小 ／ 背景をドラッグ＝全体を動かす ／
    箱をドラッグ＝その箱だけ動かす</span>
  <span id="wStat" class="wstat"></span>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", help="実験ファイル（省略すると全部ONの地図）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg, name = None, "実装したもの全部（どの実験にも縛られない地図）"
    if a.spec:
        from run.main import load_spec
        from run.config import Config
        spec = load_spec(a.spec)
        cfg = Config.from_spec(spec)
        name = spec.get("name") or os.path.basename(a.spec)
    out = a.out or os.path.join(
        os.path.dirname(a.spec) if a.spec else os.path.join(_ROOT, "E", "docs"),
        "配線図.html")
    out = out if os.path.isabs(out) else os.path.join(_ROOT, out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(html_page(cfg, name))
    n_on = sum(1 for n in NODES if cfg is None or is_on(n, cfg))
    print(f"機構 {len(NODES)}個（この実験で効いているもの {n_on}個）→ "
          f"{os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
