"""★学習の進み具合と結果を1枚の絵にする（HTML）。走行中も見られる。

【なぜ要るか、2026-07-30】学習は1本40分以上かかり、条件×シードで何本も並ぶ。
「どの条件のどれが、どこまで進んで、いまどんな数字か」を文字のログから読み取るのは
（ログが6本あると）現実的でない。＝**絵で一望できるようにする**。

【使い方】
    .venv/Scripts/python.exe -m run.tools.dashboard
    .venv/Scripts/python.exe -m run.tools.dashboard --watch          （30秒ごとに作り直す）
    .venv/Scripts/python.exe -m run.tools.dashboard --dir E/logs/別のフォルダ

出力: <対象フォルダ>/ダッシュボード.html
  ブラウザで開くと**30秒ごとに自分で読み直す**ので、走らせたまま置いておける。

【読むもの】対象フォルダの中の
    *.csv   チェックポイントごとの数値（step, classify, margin, corr, persist）
    *.log   画面に出た文字。ここから経過時間（real=N min）と月齢を拾う

⚠️絵の中身は**そのフォルダにあるデータだけ**から描く（1ランのデータを混ぜない
  ＝[[feedback-per-sim-graph]]）。条件の比較は「条件ごとに別の線」として重ねる。
⚠️外部のCSS・フォント・スクリプトを読み込まない（オフラインで開ける・壊れない）。
"""
import argparse
import csv
import glob
import html
import os
import re
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 条件ごとの色。★字と線がかぶらないよう、明度を離した3色＋灰
COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#4a5568"]
# ★列の「意味づけ」表。CSVに出てくる列を**自動で見つけて**全部描くが、
#   ここに載っている列は見出し・単位・基準線を人に読める形で出す。
#   ⚠️載っていない列も描く（「知らない列」として末尾に回す）＝★取りこぼさない。
#   (列名) -> (見出し, 英語, 単位, 基準線, 基準線の説明, 大きいほど良いか)
META = {
    "margin":    ("自己モデルの質", "self-model margin", "%", 0.0, None, True),
    "persist":   ("「何もしない」予測との比べ", "vs naive forecast", "%", 100.0,
                  "★100より下＝勝っている", False),
    "classify":  ("自分と他人の区別", "self/other classification", "%", 50.0,
                  "50＝偶然", True),
    "corr":      ("予測と実際の相関", "prediction correlation", "", 0.0, None, True),
    "agency":    ("行為主体感（★再現性なし）", "sense of agency", "%", 50.0, None, True),
    "age_months": ("体の月齢", "body age", "ヶ月", None, None, None),
    "noise":     ("探索の強さ", "exploration noise", "", None, None, None),
    "act_abs":   ("力の出し具合", "action magnitude", "", None,
                  "★0付近＝固まった", None),
    "d_action2": ("行動の変化量", "action change", "", None, None, None),
    "effort":    ("努力（代謝コスト）", "effort cost", "", None, None, None),
    "cereb_err": ("小脳の馴染み度", "cerebellar familiarity", "", None, None, None),
    "ne_maturation": ("探索の結晶化", "NE maturation", "", None, None, None),
    "hand_in_view": ("手が視界に入った割合", "hand in view", "%", None, None, True),
    "toy_touches": ("おもちゃに触れた回数", "toy touches", "回", None, None, True),
    "toy_per_min": ("おもちゃに触れた頻度", "touches per minute", "回/分", None, None, True),
    "real_min":   ("経過した実時間", "wall-clock time", "分", None, None, None),
    "life_min":   ("太郎が生きた時間", "simulated life", "分", None, None, None),
}
# ★上のほうに出す順番（大事な指標を先に）。ここに無い列はこの後ろに自動で並ぶ
ORDER = ["margin", "persist", "classify", "corr", "hand_in_view", "toy_touches",
         "toy_per_min", "age_months", "act_abs", "noise", "d_action2",
         "cereb_err", "effort", "ne_maturation", "agency"]
# ★描かない列（時間の経過そのものは横軸なので線にしない）
SKIP = {"step", "real_min", "life_min", "mag_ratio"}


def _read_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                try:
                    rows.append({k: float(v) for k, v in r.items() if v != ""})
                except ValueError:
                    continue       # 書き込み途中の行は飛ばす
    except (OSError, StopIteration):
        pass
    return rows


def _read_log(path):
    """log から「経過時間・いまの月齢・体を育てた回数」を拾う。"""
    info = {"real_min": None, "age": None, "regrow": 0, "steps": None, "name": None}
    if not os.path.exists(path):
        return info
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.findall(r"real=(\d+)min", txt)
    if m:
        info["real_min"] = int(m[-1])
    m = re.findall(r"age=([\d.]+)mo", txt)
    if m:
        info["age"] = float(m[-1])
    info["regrow"] = len(re.findall(r"\[body-growth\] \d+回目", txt))
    m = re.search(r"train / (\d+)ステップ", txt)
    if m:
        info["steps"] = int(m.group(1))
    for line in txt.splitlines()[:6]:
        s = line.strip()
        if s and not s.startswith(("=", "シーン", "太郎", "動かし方", "道具", "設定")):
            info["name"] = s
            break
    return info


def collect(d):
    """フォルダから「1本ずつの状態」を集める。"""
    runs = []
    for cp in sorted(glob.glob(os.path.join(d, "*.csv"))):
        base = os.path.basename(cp)[:-4]
        rows = _read_csv(cp)
        # ログの名前は短縮されている場合があるので、前方一致で探す
        cand = [p for p in glob.glob(os.path.join(d, "*.log"))
                if base.startswith(os.path.basename(p)[:-4].split("_seed")[0])
                and base.endswith(os.path.basename(p)[:-4].split("_")[-1])]
        info = _read_log(cand[0]) if cand else {"real_min": None, "age": None,
                                                "regrow": 0, "steps": None,
                                                "name": None}
        steps = info.get("steps") or (int(rows[-1]["step"]) if rows else 1)
        cur = int(rows[-1]["step"]) if rows else 0
        runs.append({"key": base, "rows": rows, "steps": steps, "cur": cur,
                     "info": info, "mtime": os.path.getmtime(cp)})
    return runs


def _cond_of(key):
    """ファイル名から条件名とシードを取る（例 A_ずっと4ヶ月_seed0）。"""
    m = re.match(r"(.+)_seed(\d+)$", key)
    return (m.group(1), int(m.group(2))) if m else (key, 0)


# ------------------------------------------------------------------ 絵を描く
def bar(pct, w=100, h=14, color="#2b6cb0"):
    p = max(0.0, min(1.0, pct))
    return (f'<svg class="bar" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'role="img" aria-label="{p*100:.0f}%">'
            f'<rect x="0" y="0" width="{w}" height="{h}" rx="3" fill="#e2e8f0"/>'
            f'<rect x="0" y="0" width="{p*w:.2f}" height="{h}" rx="3" fill="{color}"/>'
            f'</svg>')


def line_chart(series, col, base=None, base_note=None, width=760, height=260):
    """折れ線。series = [(見出し, 色, [(x, y), ...]), ...]"""
    pad_l, pad_r, pad_t, pad_b = 58, 14, 16, 34
    pts = [p for _, _, ps in series for p in ps]
    if not pts:
        return '<p class="none">まだデータがありません</p>'
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = 0, max(max(xs), 1)
    y0, y1 = min(ys), max(ys)
    if base is not None:
        y0, y1 = min(y0, base), max(y1, base)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1.0
    m = (y1 - y0) * 0.12
    y0, y1 = y0 - m, y1 + m
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * iw

    def sy(y):
        return pad_t + ih - (y - y0) / (y1 - y0) * ih

    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    # 横の目安線と目盛り（★字と線が重ならないよう左に置く）
    for i in range(5):
        y = y0 + (y1 - y0) * i / 4
        yy = sy(y)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" '
                   f'stroke="#e6ebf2" stroke-width="1"/>')
        out.append(f'<text x="{pad_l-8}" y="{yy+4:.1f}" class="tick" '
                   f'text-anchor="end">{y:.1f}</text>')
    # 基準線
    if base is not None:
        yy = sy(base)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" '
                   f'stroke="#c53030" stroke-width="1.4" stroke-dasharray="5 4"/>')
        if base_note:
            out.append(f'<text x="{width-pad_r-4}" y="{yy-6:.1f}" class="basenote" '
                       f'text-anchor="end">{html.escape(base_note)}</text>')
    # 縦の目盛り（学習回数）
    for i in range(5):
        x = x0 + (x1 - x0) * i / 4
        out.append(f'<text x="{sx(x):.1f}" y="{height-12}" class="tick" '
                   f'text-anchor="middle">{int(x):,}</text>')
    out.append(f'<text x="{width/2:.0f}" y="{height-1}" class="axis" '
               f'text-anchor="middle">学習回数（判断）</text>')
    # 線
    for label, color, ps in series:
        if not ps:
            continue
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ps)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        lx, ly = sx(ps[-1][0]), sy(ps[-1][1])
        out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3" fill="{color}"/>')
    out.append("</svg>")
    return "".join(out)


def _fmt_eta(cur, steps, real_min):
    if not real_min or cur <= 0:
        return "—"
    if cur >= steps:
        return f"おわり（{real_min}分）"
    rest = real_min * (steps - cur) / cur
    return f"あと約{rest:.0f}分"


def build_html(runs, title):
    conds = {}
    for r in runs:
        c, s = _cond_of(r["key"])
        conds.setdefault(c, []).append((s, r))
    color_of = {c: COLORS[i % len(COLORS)] for i, c in enumerate(sorted(conds))}
    done = sum(1 for r in runs if r["cur"] >= r["steps"])
    total_pct = (sum(min(1.0, r["cur"] / max(r["steps"], 1)) for r in runs)
                 / max(len(runs), 1))

    # ---- 上：全体の進み具合 -------------------------------------------------
    cards = []
    for c in sorted(conds):
        col = color_of[c]
        for s, r in sorted(conds[c]):
            pct = min(1.0, r["cur"] / max(r["steps"], 1))
            last = r["rows"][-1] if r["rows"] else {}
            age = r["info"].get("age")
            grow = r["info"].get("regrow", 0)
            fresh = (time.time() - r["mtime"]) < 300
            cards.append(f"""
      <div class="card">
        <div class="chead"><span class="dot" style="background:{col}"></span>
          <b>{html.escape(c)}</b><span class="seed">seed {s}</span>
          <span class="state {'live' if fresh and pct < 1 else ('fin' if pct>=1 else 'idle')}">
            {'走行中' if fresh and pct < 1 else ('完了' if pct >= 1 else '停止中')}</span></div>
        {bar(pct, color=col)}
        <div class="nums">
          <span class="big">{pct*100:.0f}%</span>
          <span>{r['cur']:,} / {r['steps']:,} 回</span>
          <span>{_fmt_eta(r['cur'], r['steps'], r['info'].get('real_min'))}</span>
        </div>
        <div class="mets">
          <div><em>自己モデルの質</em><b>{last.get('margin', float('nan')):+.1f}%</b></div>
          <div><em>何もしない比</em><b class="{'good' if last.get('persist',999)<100 else 'bad'}">{last.get('persist', float('nan')):.0f}%</b></div>
          <div><em>体の月齢</em><b>{'—' if age is None else f'{age:.2f}ヶ月'}</b></div>
          <div><em>体を作り直した</em><b>{grow} 回</b></div>
        </div>
      </div>""")

    # ---- 下：グラフ（★CSVにある列を自動で見つけて全部描く）------------------
    found = []
    for r in runs:
        for x in r["rows"]:
            for k in x:
                if k not in SKIP and k not in found:
                    found.append(k)
    cols = [c for c in ORDER if c in found] + [c for c in found if c not in ORDER]
    charts = []
    for col_name in cols:
        head, eng, unit, base, note, big_good = META.get(
            col_name, (col_name, "", "", None, None, None))
        series = []
        for c in sorted(conds):
            for s, r in sorted(conds[c]):
                ps = [(x["step"], x[col_name]) for x in r["rows"] if col_name in x]
                if ps:
                    series.append((f"{c} seed{s}", color_of[c], ps))
        hint = ("大きいほど良い / higher is better" if big_good is True
                else "小さいほど良い / lower is better" if big_good is False else "")
        unk = "" if col_name in META else ' <span class="unk">★表に無い列</span>'
        charts.append(f"""
      <section class="panel">
        <h3>{html.escape(head)}{f'（{unit}）' if unit else ''}
          <span class="eng">{html.escape(eng or col_name)}</span>
          <span class="hint">{hint}</span>{unk}</h3>
        {line_chart(series, col_name, base, note)}
      </section>""")

    legend = "".join(
        f'<span class="lg"><i style="background:{color_of[c]}"></i>{html.escape(c)}</span>'
        for c in sorted(conds))

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>{html.escape(title)}</title>
<style>
 :root {{ --ink:#1a202c; --sub:#4a5568; --line:#e2e8f0; --bg:#f7f9fc; --card:#fff; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; padding:18px 16px 40px; background:var(--bg); color:var(--ink);
   font-family:"Yu Gothic","Hiragino Sans","Meiryo",system-ui,sans-serif; line-height:1.6; }}
 h1 {{ font-size:1.24rem; margin:0 0 2px; }}
 .meta {{ color:var(--sub); font-size:.82rem; margin-bottom:14px; }}
 .overall {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
   padding:14px 16px; margin-bottom:16px; }}
 .overall .big {{ font-size:1.7rem; font-weight:700; }}
 .grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fill,minmax(268px,1fr)); }}
 .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
 .chead {{ display:flex; align-items:center; gap:7px; font-size:.92rem; margin-bottom:8px; flex-wrap:wrap; }}
 .dot {{ width:10px; height:10px; border-radius:50%; flex:0 0 auto; }}
 .seed {{ color:var(--sub); font-size:.78rem; }}
 .state {{ margin-left:auto; font-size:.72rem; padding:1px 7px; border-radius:9px; }}
 .state.live {{ background:#e6fffa; color:#22684f; border:1px solid #9decc8; }}
 .state.fin  {{ background:#ebf4ff; color:#2b4f81; border:1px solid #a8c7f0; }}
 .state.idle {{ background:#f7f0e8; color:#7a5a35; border:1px solid #e3c9a8; }}
 svg.bar {{ width:100%; height:14px; display:block; }}
 .nums {{ display:flex; gap:10px; align-items:baseline; margin:7px 0 4px; font-size:.8rem; color:var(--sub); }}
 .nums .big {{ font-size:1.24rem; font-weight:700; color:var(--ink); }}
 .mets {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; margin-top:8px;
   border-top:1px dashed var(--line); padding-top:8px; }}
 .mets div {{ display:flex; justify-content:space-between; font-size:.8rem; }}
 .mets em {{ font-style:normal; color:var(--sub); }}
 .good {{ color:#276749; }} .bad {{ color:#9b2c2c; }}
 .panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
   padding:12px 14px 6px; margin-top:14px; overflow-x:auto; }}
 .panel h3 {{ font-size:.96rem; margin:0 0 6px; display:flex; gap:9px; align-items:baseline; }}
 .hint {{ font-weight:400; font-size:.76rem; color:var(--sub); }}
 .eng {{ font-weight:400; font-size:.74rem; color:var(--sub); font-style:italic; }}
 .unk {{ font-weight:400; font-size:.72rem; color:#b7791f; }}
 svg.chart {{ width:100%; height:auto; min-width:420px; }}
 text.tick {{ font-size:10px; fill:#718096; }}
 text.axis {{ font-size:10.5px; fill:#4a5568; }}
 text.basenote {{ font-size:10px; fill:#c53030; }}
 .legend {{ display:flex; gap:14px; flex-wrap:wrap; margin:10px 0 0; font-size:.82rem; color:var(--sub); }}
 .lg i {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:5px; }}
 .none {{ color:var(--sub); font-size:.85rem; }}
 footer {{ margin-top:20px; color:var(--sub); font-size:.76rem; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --ink:#e8eef7; --sub:#a0aec0; --line:#2d3748; --bg:#161b22; --card:#1c2230; }}
   svg.bar rect:first-child {{ fill:#2d3748; }}
   text.tick {{ fill:#8f9bb0; }} text.axis {{ fill:#a0aec0; }}
 }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">{time.strftime('%Y-%m-%d %H:%M:%S')} 現在 ／ 30秒ごとに自分で読み直します</div>

<div class="overall">
  <div class="nums"><span class="big">{total_pct*100:.0f}%</span>
    <span>全体の進み（{done} / {len(runs)} 本 完了）</span></div>
  {bar(total_pct, color="#2b6cb0")}
  <div class="legend">{legend}</div>
</div>

<div class="grid">{''.join(cards)}</div>
{''.join(charts)}

<footer>
  自己モデルの質（margin）＝自分の行動と他人の行動で予測させたときの★予測誤差の差。
  大きいほど「自分の体を分かっている」。<br>
  「何もしない」予測との比べ（persist）＝★100より下なら「何もしない」と予測するより上手。
  100を超えていたらモデルが壊れている疑い。<br>
  ⚠️絵はこのフォルダのデータだけから描いています（別のランを混ぜていません）。
</footer>
</body></html>"""


def make(d, title=None):
    """★フォルダを見て HTML を1枚作る。プラグインからも呼ばれる（自動化の入口）。

    Returns: 作った HTML のパス
    """
    d = d if os.path.isabs(d) else os.path.join(_ROOT, d)
    os.makedirs(d, exist_ok=True)
    title = title or f"学習の様子 ── {os.path.basename(d)}"
    out = os.path.join(d, "ダッシュボード.html")
    runs = collect(d)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        fp.write(build_html(runs, title))
    os.replace(tmp, out)     # ★書き換え中の半端なHTMLを開かせない
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("E", "logs", "selfmodel_v2"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--watch", action="store_true", help="30秒ごとに作り直す")
    ap.add_argument("--open", action="store_true", help="作ったらブラウザで開く")
    a = ap.parse_args()

    d = a.dir if os.path.isabs(a.dir) else os.path.join(_ROOT, a.dir)
    if not os.path.isdir(d):
        print(f"⚠️フォルダがない: {d}")
        return 1
    title = a.title or f"学習の様子 ── {os.path.basename(d)}"
    out = os.path.join(d, "ダッシュボード.html")

    def once():
        make(d, title)
        runs = collect(d)
        n_live = sum(1 for r in runs if (time.time() - r["mtime"]) < 300
                     and r["cur"] < r["steps"])
        print(f"[{time.strftime('%H:%M:%S')}] {len(runs)}本（走行中 {n_live}）→ "
              f"{os.path.relpath(out, _ROOT)}", flush=True)
        return runs

    runs = once()
    if a.open:
        import webbrowser
        webbrowser.open("file:///" + out.replace(os.sep, "/"))
    if a.watch:
        print("★30秒ごとに作り直します（Ctrl+C で終わり）", flush=True)
        try:
            while True:
                time.sleep(30)
                runs = once()
                # ⚠️★「全部おわった」だけで止めてはいけない。条件ごとに順番に流す
                #   使い方（2並列×3回など）では、次のランが始まる前に止まってしまう。
                #   2026-07-30 に実際に起きた（条件Aが終わった瞬間に終了し、
                #   条件B/Cの絵が更新されなくなった）。
                #   ⇒ **どのCSVも一定時間書かれていない**ことまで確かめてから止める。
                if runs and all(r["cur"] >= r["steps"] for r in runs) \
                        and (time.time() - max(r["mtime"] for r in runs)) > 600:
                    print("★全部おわって10分動きがないので終了します", flush=True)
                    break
        except KeyboardInterrupt:
            print("\n止めました", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
