# -*- coding: utf-8 -*-
"""机上確認・段3（2026-09-03・分節第2案）：記録済みの親のセリフ908回を新しい切り出し
（終わり確率＋既知語の足がかり）に通し、語彙の表にどんな単位が立つかを見る。走行なし。

手順：F2-49 r8 の脳に、セリフを <EOS> 付きで1周聞かせ直す（f50_desk_boundary と同じ）→
その脳で各発話の「終わる確率」を読みながら、Lexicon.observe(mode="end_prob") に順に流す。
対照：(a) 従来（自信度1.0決め打ち＝発話まるごと1単位）(b) 谷（第1案・token_probs）(c) 第2案。
合否：8語（くつ・コップ・おさら・おわん・かばん・がおー・バス・ボール）と機能語（だよ・だね・いるね）が
別単位で立ち、名詞が真ん中で砕けないこと。

    .venv/Scripts/python.exe F/scripts/f50_desk_segment.py
出力: F/logs/F2-50_分節/机上_切り出し単位.txt / 図_机上_切り出し単位.png
"""
import os, sys, io, csv, warnings, collections
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
for p in (os.getcwd(), "run", "taro_core/src", "taro_core/src/brain", "taro_core/src/senses", "F/scripts"):
    sys.path.insert(0, os.path.abspath(p))
import numpy as np, torch
import f50_desk_boundary as D
import f_gen_f49 as G
from cerebral_cortex.temporal_lobe.lexicon import Lexicon
from hearing import expand_long_vowel

OUT = "F/logs/F2-50_分節"
NOUNS = [w for w, _, _ in G.WORDS]
FUNC = ["だよ", "だね", "いるね"]


def run_condition(label, b, pv, utts, cap, mode):
    lex = Lexicon(min_len=2)
    par = pv.char2idx[D.PARENT_NAME]
    for text in utts:
        chars = [c for c in expand_long_vowel(text) if c in pv.char2idx and pv.char2idx[c] < cap]
        ids = [pv.char2idx[c] for c in chars]
        if len(ids) < 2:
            continue
        if mode == "whole":
            lex.observe(tuple(chars), [1.0] * len(chars))
        elif mode == "valley":
            ps = b.token_probs(ids)
            lex.observe(tuple(chars), ps)
        else:
            pe = D.p_end(b, [par] + ids)[1:]        # 各トークン直後
            lex.observe(tuple(chars), [1.0] * len(chars), end_probs=[float(v) for v in pe], mode="end_prob")
    units = collections.Counter({"".join(k): v for k, v in lex.counts.items()})
    return units


def judge(units):
    ok_n = [w for w in NOUNS if units.get(expand_long_vowel(w), 0) > 0]
    ok_f = [w for w in FUNC if units.get(w, 0) > 0]
    frags = [(u, c) for u, c in units.most_common() if any(u != expand_long_vowel(w) and u in expand_long_vowel(w) for w in NOUNS) and len(u) >= 2]
    return ok_n, ok_f, frags[:6]


def main():
    os.makedirs(OUT, exist_ok=True)
    utts = []
    for r in range(1, 9):
        for row in csv.DictReader(io.open("F/logs/F2-49_実物スキャン/r%d/発話イベント.csv" % r, encoding="utf-8")):
            utts.append(row["text"])
    b, pv = D.load(); cap = b.embedding.num_embeddings
    D.relisten(b, pv, utts, cap, 1)                 # 1周＝本走行と同じ回数だけ EOS 付きで聞く
    lines = []
    results = {}
    for label, mode in (("従来（まるごと）", "whole"), ("第1案（谷）", "valley"), ("第2案（終わり確率＋足がかり）", "end_prob")):
        units = run_condition(label, b, pv, utts, cap, mode)
        ok_n, ok_f, frags = judge(units)
        results[label] = units
        lines.append("## %s：単位 %d 種" % (label, len(units)))
        lines.append("  名詞が単独で立つ %d/8: %s" % (len(ok_n), "・".join(ok_n)))
        lines.append("  機能語が単独で立つ %d/3: %s" % (len(ok_f), "・".join(ok_f)))
        lines.append("  名詞の断片: %s" % ("、".join("%s×%d" % f for f in frags) if frags else "なし"))
        lines.append("  上位20: " + "、".join("%s×%d" % (u, c) for u, c in units.most_common(20)))
        lines.append("")
    txt = "\n".join(lines)
    io.open(OUT + "/机上_切り出し単位.txt", "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    # 図：3条件の上位単位を横棒で
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    for c in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if any(c in f.name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = c; break
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (label, units) in zip(axes, results.items()):
        top = units.most_common(18)[::-1]
        cols = ["tab:green" if (u in [expand_long_vowel(w) for w in NOUNS] or u in FUNC) else "tab:gray" for u, _ in top]
        ax.barh([u for u, _ in top], [c for _, c in top], color=cols)
        ax.set_title(label, fontsize=10); ax.set_xlabel("回数")
    fig.suptitle("机上確認：親のセリフ908回から切り出された単位（緑＝名詞か機能語そのもの、灰＝それ以外）", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + "/図_机上_切り出し単位.png", dpi=110)
    print("図:", OUT + "/図_机上_切り出し単位.png")


if __name__ == "__main__":
    main()
