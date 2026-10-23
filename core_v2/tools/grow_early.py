"""育ちの段1（耳）と段2（喃語）の写しを作る（育ちの段の設計 `太郎/2026-10-18_育ちの段_設計.md`）。どちらも体なしで、短い。

* 段1（耳）：段階5と同じ（tools/hear_parent.py）。聞き分ける係に、親の話しかけ（parent_speech.make_corpus）を N 文（既定1000）
  聞かせる。数秒。決まった種なので、いつ作っても同じ。
  話しかけの語は既定で段階9の20語（--words 20）。段階5の10語（--words 10）だと、段3の名づけの中で残りの10語を覚えるには
  「語だけ」の名づけ（1割）を2回以上聞く必要があり、目だけの場面の名づけの数（1場面 約1回、クラウドの10場面で測った）では
  ほとんど届かない。仕様書 §5 の注「耳が段3の間に新しい語を覚え続けないと分かったら、親の話しかけに20語を足して段1から」のとおり
* 段2（喃語）：段1の写しの耳に、声の試験場（tools/voice_bed.py）で 0〜24か月育てた声の係と言う語の係を読む
  （voice_bed の --export の JSON。既定は段階7で育てた `core_v2/data/段階7_声_seed0_24か月.json`）。数秒。
  注意：その JSON は声の係・言う語の係・口の表・声の試験場が今と同じコード（c42feda から変わっていない）で育ったが、
  そのときの耳は試験場の中で別に育った耳（段1の写しの耳ではない）。写しの note に書く。
  声を今のコードで育て直すなら、先に `python -m core_v2.tools.voice_bed --seed 0 --days 730 --export <JSON>`（約1.6時間）

使い方：
  python -m core_v2.tools.grow_early 1 [--seed 0] [--sentences 1000] [--corpus-seed 1] [--words 20|10]
  python -m core_v2.tools.grow_early 2 [--seed 0] [--voice core_v2/data/段階7_声_seed0_24か月.json]
写しの名前：段1_耳_種<seed>・段2_喃語_種<seed>（同じ名前がもうあれば _2, _3 … を足す）
"""

from __future__ import annotations

import argparse
import os
import time

from ..boxes import box_class
from . import growth

DEFAULT_VOICE = "core_v2/data/段階7_声_seed0_24か月.json"


free_name = growth.free_name


def corpus_words(n: str) -> list[str]:
    from ..world_eye.instances import WORDS as WORDS20
    from .parent_speech import WORDS as WORDS10
    return list(WORDS20) if str(n) == "20" else list(WORDS10)


def grow_phase1(seed: int = 0, sentences: int = 1000, corpus_seed: int = 1, words: str = "20") -> dict:
    from .parent_speech import make_corpus
    t0 = time.time()
    ear = box_class("listen")(learning=True)
    for s in make_corpus(sentences, seed=corpus_seed, words=corpus_words(words)):
        ear.step({"earin": s})
    ear.learning = False
    return growth.save({"listen": ear}, 1, free_name(f"段1_耳_種{seed}"), end=True,
                       conditions={"world": "bed", "eye": "none", "seed": seed, "sentences": sentences,
                                   "corpus_seed": corpus_seed, "corpus_words": words,
                                   "known_words": sorted(ear.known())},
                       note=f"親の話しかけ {sentences} 文（{time.time() - t0:.1f}秒）")


def grow_phase2(seed: int = 0, voice: str = DEFAULT_VOICE) -> dict:
    from .voice_bed import import_voice
    row = growth.start_point(2, seed=seed)
    boxes = {b: box_class(b)() for b in ("listen", "voice", "wordplan")}
    if row is None:
        raise LookupError(f"段1の写しが無い（種{seed}）。先に `python -m core_v2.tools.grow_early 1 --seed {seed}`")
    growth.load(boxes, row)
    path = voice if os.path.isabs(voice) else os.path.join(growth.REPO, voice)
    meta = import_voice(boxes, path)
    return growth.save(boxes, 2, free_name(f"段2_喃語_種{seed}"), end=True, month=float(meta.get("month", 24.0)),
                       parent=row["name"],
                       conditions={"world": "bed", "eye": "none", "seed": seed, "voice_json": voice,
                                   "voice_meta": meta},
                       note="耳は段1の写し。声の係と言う語の係は voice_bed の書き出し（そのときの耳は試験場の中で別に育った）")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", type=int, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sentences", type=int, default=1000)
    ap.add_argument("--corpus-seed", type=int, default=1)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--words", default="20", choices=["20", "10"], help="段1の話しかけの語：段階9の20語（既定）／段階5の10語")
    a = ap.parse_args(argv)
    row = grow_phase1(a.seed, a.sentences, a.corpus_seed, a.words) if a.phase == 1 else grow_phase2(a.seed, a.voice)
    print(f"保存：{row['name']}（段{row['phase']}、{row['file']}、{row['size_kb']}KB）")
    return row


if __name__ == "__main__":
    main()
