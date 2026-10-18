"""段階9：物の音の入口（boxes/soundin.py）と、意味の辞書の音の荷札。"""
import torch

from core_v2.boxes import box_class
from core_v2.boxes.soundin import sound_tag
from core_v2.tagging import word_tag
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


def test_semantic_recalls_word_from_sound_alone():
    sem = box_class("semantic")(learning=True)
    n = sound_tag(sounds.synth("dog", 0))
    for _ in range(3):
        sem.learn(n, word_tag("わんわん"))
    r = sem.recall_word(None, 0.0, n)
    assert r is not None and float(r @ word_tag("わんわん")) > 0.99
    assert sem.recall_word(None, 0.0, None) is None          # 何も見ていない・鳴っていない
    assert sem._seen_vec(None, 0.0) is None                  # 音が無ければ段階8と同じ
