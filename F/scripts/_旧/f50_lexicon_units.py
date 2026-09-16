# -*- coding: utf-8 -*-
"""保存モデルの語彙の表を復号し、切り出された単位の内訳を出す（2026-09-03・F2-50の判定用）。

    .venv/Scripts/python.exe F/scripts/f50_lexicon_units.py <model.pt> [...]
"""
import os, sys, io, collections
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("F/scripts")); sys.path.insert(0, os.path.abspath("taro_core/src/senses"))
import torch
import f_gen_f49 as G
from hearing import expand_long_vowel

NOUNS = [expand_long_vowel(w) for w, _, _ in G.WORDS]
FUNC = ["だよ", "だね", "いるね"]


def units_of(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    i2c = {int(k): v for k, v in blob["hearing_vocab"]["idx2char"].items()}
    lx = blob["lexicon"]
    dec = lambda t: "".join(i2c.get(int(i), "?") for i in t)
    units = collections.Counter({dec(k): v for k, v in lx["counts"].items()})
    n_utt = len(lx.get("utterance_counts", {}))
    ep = (lx.get("end_prob_sum", 0.0), lx.get("end_prob_n", 0))
    return units, n_utt, ep


def report(path):
    units, n_utt, (es, en) = units_of(path)
    ok_n = [w for w in NOUNS if units.get(w, 0) > 0]
    ok_f = [w for w in FUNC if units.get(w, 0) > 0]
    frags = [(u, c) for u, c in units.most_common() if len(u) >= 2 and any(u != w and u in w for w in NOUNS)]
    print("== %s" % os.path.basename(path))
    print("  単位 %d 種／発話まるごとの記録 %d 種／終わり確率の走行平均 %.4f（%d位置）" % (len(units), n_utt, es / max(1, en), en))
    print("  名詞が単独 %d/8: %s" % (len(ok_n), "・".join(ok_n)))
    print("  機能語が単独 %d/3: %s" % (len(ok_f), "・".join(ok_f)))
    print("  名詞の断片: %s" % ("、".join("%s×%d" % f for f in frags[:8]) if frags else "なし"))
    print("  上位16: " + "、".join("%s×%d" % (u, c) for u, c in units.most_common(16)))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
