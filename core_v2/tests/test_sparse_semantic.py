"""段階9の試し：まばらな語の表し方（tools/sparse_semantic.py）と、時間の形を残す物の音の荷札。"""
import itertools

import torch

from core_v2.boxes.soundin import sound_tag
from core_v2.formats import DTYPE, TAG_DIM, make_tag
from core_v2.tagging import word_tag
from core_v2.tools import sparse_semantic as sps
from core_v2.world_eye import instances as ins
from core_v2.world_eye import sounds


def _learn(sem, word, a=None, n=None):
    z = torch.zeros(TAG_DIM, dtype=DTYPE)
    v = torch.cat([make_tag(a) if a is not None else z, n if n is not None else z])
    sem._learn_pending({"q": word_tag(word), "k": 0, "v": v, "w": 1.0})


def test_word_codes_are_sparse_and_barely_overlap():
    codes = [sps.word_code(word_tag(w)) for w in ins.WORDS]
    assert all(int(c.sum()) == sps.K for c in codes)
    ov = [int(a @ b) for a, b in itertools.combinations(codes, 2)]
    assert max(ov) <= 2 and sum(ov) / len(ov) < 0.3      # 10個のうち、重なっても1〜2個


def test_uneven_counts_do_not_leak():
    sem = sps.SparseSemantic(learning=True)
    g = torch.Generator().manual_seed(0)
    a1, a2 = torch.randn(TAG_DIM, generator=g), torch.randn(TAG_DIM, generator=g)
    for _ in range(40):
        _learn(sem, "があがあ", a1)
    assert sem.recall(word_tag("くま")) is None          # 聞いたことのない語は出さない
    for _ in range(2):
        _learn(sem, "くま", a2)
    assert sem.recall(word_tag("くま")) is None          # 2回では出ない（3回で出る）
    _learn(sem, "くま", a2)
    r = sem.recall(word_tag("くま"))
    assert r is not None and float(r @ make_tag(a2)) > 0.95


def test_sound_links_are_averaged_per_sound_cell():
    """すぷうんの音が同じ部屋のこっぷの窓にも入っても、音の主（よく一緒だった語）が勝つ。"""
    sem = sps.SparseSemantic(learning=True)
    n = sps.sound_tag_time(sounds.synth("spoon", 0))
    for _ in range(5):
        _learn(sem, "すぷうん", torch.ones(TAG_DIM), n)
    for _ in range(2):
        _learn(sem, "こっぷ", -torch.ones(TAG_DIM), n)
    wt = torch.stack([word_tag(w) for w in ins.WORDS])
    assert ins.WORDS[int(torch.argmax(sem.word_scores(wt, n=n)))] == "すぷうん"


def test_time_shaped_sound_tag_separates_dog_and_duck_more():
    def m(f, kind):
        return make_tag(torch.stack([f(sounds.synth(kind, k, seed=k)) for k in range(4)]).mean(0))
    old = float(m(sound_tag, "dog") @ m(sound_tag, "duck"))
    new = float(m(sps.sound_tag_time, "dog") @ m(sps.sound_tag_time, "duck"))
    t = sps.sound_tag_time(sounds.synth("cat", 1))
    assert abs(float(t.norm()) - 1.0) < 1e-5
    assert new < old - 0.08
