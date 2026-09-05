# -*- coding: utf-8 -*-
"""F2-25 A/B の判定：逐次(A)と並列+混合(B)の最終モデルを静止測定で比べる。

    .venv/Scripts/python.exe F/scripts/f_compare_ab.py
"""
import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, "run")
import taro_setup                                   # noqa: F401,E402  (sys.path設定)
import torch                                        # noqa: E402
from cerebral_cortex.recurrent_core import TaroBrain  # noqa: E402
from hearing import Vocabulary, expand_long_vowel   # noqa: E402

KNOWN = ["わんわん", "にゃんにゃん", "ぶうぶう", "でんしゃ", "りんご",
         "くつ", "ばなな", "ぼうし"]
UNKNOWN = ["まんま", "ねんね", "ぱぱ", "ごはん"]


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pv = Vocabulary()
    pv.char2idx = dict(blob["brain_vocab"]["char2idx"])
    pv.idx2char = {int(i): c for c, i in pv.char2idx.items()}
    pv.size = max(pv.idx2char) + 1
    b = TaroBrain(vocab_size=3)
    b.resize_embedding(blob["brain"]["embedding.weight"].shape[0])
    b.load_state_dict({k: v for k, v in blob["brain"].items()
                       if k in b.state_dict() and b.state_dict()[k].shape == v.shape},
                      strict=False)
    return b, pv, blob


def probe(b, pv):
    def ids(w):
        return [pv.char2idx[c] for c in expand_long_vowel(w)
                if c in pv.char2idx and pv.char2idx[c] < b.embedding.num_embeddings]
    known = [b.sequence_prob(ids(w)) for w in KNOWN]
    unk = [b.sequence_prob(ids(w)) for w in UNKNOWN]
    return known, unk


def main():
    pa = "F/models/F2-25_A逐次_2周目_ぼうし.pt"
    pb = "F/models/F2-25_B並列_merged_final.pt"
    rows = {}
    for tag, p in (("A 逐次", pa), ("B 並列+混合", pb)):
        b, pv, blob = load(p)
        known, unk = probe(b, pv)
        n_lex = len(blob["lexicon"]["counts"])
        tot = sum(blob["lexicon"]["counts"].values())
        rows[tag] = (known, unk, n_lex, tot)
        print("== %s ==" % tag)
        for w, v in zip(KNOWN, known):
            print("   %-8s %.3f" % (w, v))
        print("   既知の平均 %.3f ／ 未知の平均 %.3f ／ 差 %.3f"
              % (sum(known) / len(known), sum(unk) / len(unk),
                 sum(known) / len(known) - sum(unk) / len(unk)))
        print("   語彙: %d種・%d回" % (n_lex, tot))
        print()


if __name__ == "__main__":
    main()
