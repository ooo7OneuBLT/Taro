# -*- coding: utf-8 -*-
"""F2-49：どの順番で学習したかを記録する（2026-09-03・ユーザー指示）。

各ラウンドの 発話イベント.csv（親が名前を呼んだ歩・語・枠）から
  ・一巡ごとの提示順（語の並び）
  ・語ごとの呼びかけ回数
を集計し、F/logs/F2-49_実物スキャン/学習順序.md に追記する。

    .venv/Scripts/python.exe F/scripts/f49_learning_order.py
"""
import os, sys, io, csv, glob, json, re
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts"))
import f_gen_f49 as G

SLOT2WORD = {k: w for k, (w, _, _) in zip(G.SLOT_KEYS, G.WORDS)}
LOG = "F/logs/F2-49_実物スキャン"


def summarize(path):
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    seq, counts = [], {}
    for r in rows:
        w = SLOT2WORD.get(r["target"], r["target"])
        counts[w] = counts.get(w, 0) + 1
        if not seq or seq[-1][0] != w:
            seq.append([w, int(r["step"])])
    return rows, seq, counts


def main():
    out = []
    for r in range(1, 9):
        p = "%s/r%d/発話イベント.csv" % (LOG, r)
        if not os.path.exists(p):
            continue
        rows, seq, counts = summarize(p)
        words = [w for w, _ in seq]
        n = len(G.WORDS)
        cycles = [words[i:i + n] for i in range(0, len(words), n)]
        out.append("### r%d：発話%d回・提示%d回（%d巡）" % (r, len(rows), len(seq), len(cycles)))
        for c, cy in enumerate(cycles, start=1):
            out.append("- %d巡目：%s" % (c, "→".join(cy)))
        out.append("- 語ごとの呼びかけ回数：" + "、".join("%s %d" % (w, counts.get(w, 0)) for w, _, _ in G.WORDS))
        out.append("")
    md = "%s/学習順序.md" % LOG
    s = io.open(md, encoding="utf-8").read()
    marker = "## ラウンドごとの親の提示順"
    head = s.split(marker)[0]
    io.open(md, "w", encoding="utf-8").write(head + marker + "（走行後に各ラウンドの 発話イベント.csv から自動集計）\n\n" + "\n".join(out) + "\n")
    print("\n".join(out[:12]))
    print("記録:", md)


if __name__ == "__main__":
    main()
