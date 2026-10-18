"""段階9：物の音の入口（boxes/soundin.py）と、意味の辞書の音の荷札。"""
import torch

from core_v2.boxes import box_class
import pytest

from core_v2.boxes.semantic import TRACE_TICKS
from core_v2.boxes.soundin import make_sound_tag, sound_tag, sound_tag_time
from core_v2.formats import TAG_DIM, make_tag
from core_v2.tagging import word_tag
from core_v2.world_eye import instances as ins
from core_v2.world_eye import sounds


def test_sound_tag_same_kind_is_closer():
    dog = [sound_tag(sounds.synth("dog", k, seed=k)) for k in range(4)]
    spoon = sound_tag(sounds.synth("spoon", 0))
    assert all(abs(float(t.norm()) - 1.0) < 1e-5 for t in dog)
    within = min(float(dog[0] @ d) for d in dog[1:])
    assert within > 0.8 and within > float(dog[0] @ spoon) + 0.5


def test_soundin_holds_the_tag_while_the_sound_lasts():
    s = box_class("soundin")()
    w = sounds.synth("cat", 0)
    n_ticks = int((len(w) / sounds.SR) / 0.1 + 0.999)
    outs = [s.step({"env": w})] + [s.step({}) for _ in range(n_ticks + 2)]
    on = [float(o["semantic"].norm()) > 0 for o in outs]
    assert on == [True] * n_ticks + [False] * 3
    assert torch.equal(outs[0]["semantic"], outs[0]["kind"])


def _hear_with_sound(sem, word, n, a=None):
    """物が鳴っている（見た目 a を見ていることもある）3秒のあいだに語を1回聞き、名残りが消えるまで待つ。"""
    lst = None if a is None else [(a, torch.zeros(28, 28)), (a, torch.zeros(28, 28)), (a, torch.zeros(28, 28))]
    for t in range(31 + TRACE_TICKS):
        sem.step({"soundin": n, "objfile": lst, "listen": word_tag(word) if t == 30 else torch.zeros(TAG_DIM)})


def test_default_sound_tag_keeps_time_shape():
    assert box_class("soundin")().shape == "time"
    w = sounds.synth("dog", 0)
    assert torch.equal(make_sound_tag(w), sound_tag_time(w)) and torch.equal(make_sound_tag(w, "mean"), sound_tag(w))


def test_time_shaped_sound_tag_separates_dog_and_duck_more():
    def m(f, kind):
        return make_tag(torch.stack([f(sounds.synth(kind, k, seed=k)) for k in range(4)]).mean(0))
    old = float(m(sound_tag, "dog") @ m(sound_tag, "duck"))
    new = float(m(sound_tag_time, "dog") @ m(sound_tag_time, "duck"))
    assert abs(float(sound_tag_time(sounds.synth("cat", 1)).norm()) - 1.0) < 1e-5
    assert new < old - 0.08


@pytest.mark.parametrize("mem", ["sparse", "table"])
def test_semantic_recalls_word_from_sound_alone(mem):
    sem = box_class("semantic")(learning=True, memory=mem)
    n = make_sound_tag(sounds.synth("dog", 0))
    for _ in range(3):
        _hear_with_sound(sem, "わんわん", n)
    r = sem.recall_word(None, 0.0, n)
    assert r is not None and float(r @ word_tag("わんわん")) > 0.99
    assert sem.recall_word(None, 0.0, None) is None          # 何も見ていない・鳴っていない
    assert sem._seen_vec(None, 0.0) is None                  # 音が無ければ段階8と同じ


def test_sound_links_are_averaged_per_sound_cell():
    """まばら版：すぷうんの音が同じ部屋のこっぷの窓にも入っても、音の主（よく一緒だった語）が勝つ。"""
    sem = box_class("semantic")(learning=True)
    n = make_sound_tag(sounds.synth("spoon", 0))
    g = torch.Generator().manual_seed(5)
    spoon, cup = make_tag(torch.randn(TAG_DIM, generator=g)), make_tag(torch.randn(TAG_DIM, generator=g))
    for _ in range(5):
        _hear_with_sound(sem, "すぷうん", n, spoon)
    for _ in range(2):
        _hear_with_sound(sem, "こっぷ", n, cup)
    wt = torch.stack([word_tag(w) for w in ins.WORDS])
    assert ins.WORDS[int(torch.argmax(sem.word_scores(wt, n=n)))] == "すぷうん"
