"""工程1（骨組み）の確かめ。シミュレータなしで回る。

    python -m pytest core_v2/tests -q

仕様書 §1 の3点：1000回回して落ちない／118本すべてに毎回値が届く／
係を呼ぶ順番を入れ替えても出力が同じ。ほかに書式・保存・環境との口を確かめる。
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from core_v2 import save as save_mod
from core_v2 import wiring
from core_v2.boxes import ENV, Box, box_class, make_all
from core_v2.brain import Brain
from core_v2.formats import GRID, TAG_DIM, FormatError, Fmt, check, make_tag, zero


# ---------------------------------------------------------------- 配線表

def test_wiring_counts():
    assert len(wiring.BOX_IDS) == 42                # 段階4で物の名簿（objfile）を足した
    assert len(wiring.EDGES) == 123
    added = [e for e in wiring.EDGES if e.added]
    assert [(e.src, e.dst) for e in added] == [("listen", "semantic"), ("eyein", "objfile"), ("prio", "objfile"),
                                               ("orienteye", "objfile"), ("objfile", "orienteye"),
                                               ("objfile", "semantic")]


def test_every_box_has_its_own_file():
    root = Path(__file__).resolve().parents[1] / "boxes"
    for bid in wiring.BOX_IDS:
        assert (root / f"{bid}.py").exists(), bid
        assert box_class(bid).BOX_ID == bid


def test_path_formats():
    fmt = {e.key: (e.fmt, e.size) for e in wiring.EDGES}
    assert fmt[("earin", "listen")] == (Fmt.TEXT, None)
    assert fmt[("eyein", "gaze")] == (Fmt.TAG_AND_MAP, None)
    assert fmt[("orienteye", "eyeout")] == (Fmt.MOTOR, 2)
    assert fmt[("gaze", "eyeout")] == (Fmt.TAG, None)  # 追従の道は段階2では素通し
    assert fmt[("eyein", "semantic")] == (Fmt.TAG_AND_MAP, None)  # 段階3
    assert fmt[("eyein", "orienteye")] == (Fmt.TAG_AND_MAP, None)  # 段階3
    assert sum(1 for e in wiring.EDGES if e.key in wiring.PATH_FORMATS) == 21


# ---------------------------------------------------------------- 書式

@pytest.mark.parametrize("fmt", list(Fmt))
def test_zero_passes_check(fmt):
    size = 2 if fmt is Fmt.MOTOR else None
    check(fmt, zero(fmt, size), size)


def test_check_rejects_bad_values():
    with pytest.raises(FormatError):
        check(Fmt.TAG, torch.ones(TAG_DIM))  # 正規化されていない
    with pytest.raises(FormatError):
        check(Fmt.MAP, torch.full((GRID, GRID), 2.0))  # 0〜1 の外
    with pytest.raises(FormatError):
        check(Fmt.MAP, torch.zeros(GRID, GRID + 1))
    with pytest.raises(FormatError):
        check(Fmt.AMOUNT, torch.tensor(0.5, dtype=torch.float64))
    with pytest.raises(FormatError):
        check(Fmt.TAG_AND_MAP, [(make_tag(torch.ones(TAG_DIM)), torch.zeros(GRID, GRID))] * 9)
    check(Fmt.TAG, make_tag(torch.randn(TAG_DIM)))


# ---------------------------------------------------------------- 素通しで回る

def test_runs_1000_ticks_and_every_edge_delivers():
    brain = Brain()
    all_keys = {e.key for e in wiring.EDGES}
    for _ in range(1000):
        brain.tick()
        assert brain.delivered == all_keys
    assert brain.t == 1000


# ---------------------------------------------------------------- 実行順に依存しない

class _Mixer(Box):
    """テスト用：届いた値と自分の状態から、書式どおりの「ゼロでない」値を作って渡す係。

    素通しだと全部ゼロで順番の違いが出ようがないので、値が回る係で確かめる。
    """

    def __init__(self, box_id: str) -> None:
        self.BOX_ID = box_id
        super().__init__()
        g = torch.Generator().manual_seed(sum(map(ord, box_id)))
        self.base = {e.dst: torch.randn(TAG_DIM, generator=g) for e in self.outputs}
        self.base_map = {e.dst: torch.randn(GRID, GRID, generator=g) for e in self.outputs}
        self.count = 0

    def reset(self):
        self.count = 0

    def _summary(self, inbox):
        s = 0.0
        for v in inbox.values():
            if isinstance(v, torch.Tensor):
                s += float(v.sum())
            elif isinstance(v, list):
                s += sum(float(t.sum()) + float(m.sum()) for t, m in v)
            elif isinstance(v, str):
                s += len(v)
        return s

    def step(self, inbox):
        self.count += 1
        s = torch.tensor(self._summary(inbox) * 0.01 + self.count * 0.001)
        out = {}
        for e in self.outputs:
            tag = make_tag(self.base[e.dst] + s)
            m = torch.sigmoid(self.base_map[e.dst] + s)
            if e.fmt is Fmt.TAG:
                out[e.dst] = tag
            elif e.fmt is Fmt.MAP:
                out[e.dst] = m
            elif e.fmt is Fmt.AMOUNT:
                out[e.dst] = torch.sigmoid(s).to(torch.float32)
            elif e.fmt is Fmt.MOTOR:
                out[e.dst] = torch.tanh(self.base[e.dst][: e.size] + s)
            elif e.fmt is Fmt.TAG_AND_MAP:
                out[e.dst] = [(tag, m)]
            elif e.fmt is Fmt.TEXT:
                out[e.dst] = f"{float(s):.4f}"
        return out


def _mixer_brain():
    return Brain({bid: _Mixer(bid) for bid in wiring.BOX_IDS})


def _run(order_fn, ticks=50):
    brain = _mixer_brain()
    history = []
    for t in range(ticks):
        brain.tick(order=order_fn(t))
        history.append(dict(brain.buffer))
    return history


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x[0], y[0]) and _same(x[1], y[1]) for x, y in zip(a, b))
    return a == b


def test_order_does_not_matter():
    rng = random.Random(0)
    fixed = _run(lambda t: None)
    shuffled = _run(lambda t: rng.sample(wiring.BOX_IDS, len(wiring.BOX_IDS)))
    reversed_ = _run(lambda t: list(reversed(wiring.BOX_IDS)))
    for t in range(len(fixed)):
        for key in fixed[t]:
            assert _same(fixed[t][key], shuffled[t][key]), (t, key)
            assert _same(fixed[t][key], reversed_[t][key]), (t, key)
    # 値が本当に回っていること（全部同じ値のままなら上の確かめは意味がない）
    key = ("goal", "gate3")
    assert not torch.equal(fixed[0][key], fixed[-1][key])


def test_one_tick_delay_per_arrow():
    """耳の入口に入れた文字が、次の回にちょうど聞き分ける係へ届く。"""
    seen = []

    class Ear(Box):
        BOX_ID = "earin"

        def step(self, inbox):
            out = self.passthrough()
            out["listen"] = inbox.get(ENV, "")
            return out

    class Listen(Box):
        BOX_ID = "listen"

        def step(self, inbox):
            seen.append(inbox["earin"])
            return self.passthrough()

    boxes = make_all()
    boxes["earin"], boxes["listen"] = Ear(), Listen()
    brain = Brain(boxes)
    brain.tick(external={"earin": "ぼおるはどこ？"})
    brain.tick()
    brain.tick()
    assert seen == ["", "ぼおるはどこ？", ""]


# ---------------------------------------------------------------- 間違いで止まる

def test_wrong_output_stops():
    class Bad(Box):
        BOX_ID = "gaze"

        def step(self, inbox):
            out = self.passthrough()
            out["prio"] = torch.zeros(3)
            return out

    boxes = make_all()
    boxes["gaze"] = Bad()
    with pytest.raises(FormatError):
        Brain(boxes).tick()


def test_missing_output_stops():
    class Lazy(Box):
        BOX_ID = "prio"

        def step(self, inbox):
            return {}

    boxes = make_all()
    boxes["prio"] = Lazy()
    with pytest.raises(FormatError):
        Brain(boxes).tick()


def test_env_in_and_out():
    class Eyeout(Box):
        BOX_ID = "eyeout"

        def step(self, inbox):
            out = self.passthrough()
            out[ENV] = inbox["orienteye"]
            return out

    boxes = make_all()
    boxes["eyeout"] = Eyeout()
    to_env = Brain(boxes).tick()
    assert torch.equal(to_env["eyeout"], torch.zeros(2))
    with pytest.raises(KeyError):
        Brain().tick(external={"nobody": 1})


# ---------------------------------------------------------------- 保存

class _Learner(Box):
    BOX_ID = "semantic"

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(TAG_DIM, TAG_DIM))


def _brain_with_weights():
    boxes = make_all()
    boxes["semantic"] = _Learner()
    return Brain(boxes)


def test_save_and_load(tmp_path):
    a = _brain_with_weights()
    for _ in range(3):
        a.tick()
    meta = save_mod.save(a, tmp_path / "brain.pt")
    assert meta["wiring_hash"] == wiring.wiring_hash()
    assert meta["tag_dim"] == TAG_DIM
    assert "git_rev" in meta and "git_dirty" in meta

    b = _brain_with_weights()
    save_mod.load(b, tmp_path / "brain.pt")
    assert torch.equal(a.boxes["semantic"].w, b.boxes["semantic"].w)
    assert b.t == 3


def test_load_refuses_other_wiring(tmp_path, monkeypatch):
    save_mod.save(_brain_with_weights(), tmp_path / "brain.pt")
    monkeypatch.setattr(save_mod, "wiring_hash", lambda: "違う配線")
    with pytest.raises(save_mod.WiringMismatch):
        save_mod.load(_brain_with_weights(), tmp_path / "brain.pt")
