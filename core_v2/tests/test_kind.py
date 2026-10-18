"""段階9：種類の層（boxes/kind.py）。生まれた時は素通し（値が1ビットも同じ）・学ぶと同じ種類の見えが近づく・脳の配線で回る。"""
import torch

from core_v2.boxes import box_class
from core_v2.boxes.eyein import ego_map
from core_v2.boxes.objfile import bump_at
from core_v2.formats import TAG_DIM, make_tag
from core_v2.tagging import noisy, word_tag


def _objs(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = [(make_tag(torch.randn(TAG_DIM, generator=g)), bump_at(13.5, 5.0 + 8 * i, sigma=1.0)) for i in range(n)]
    return out + [(torch.zeros(TAG_DIM), ego_map(0.0, 0.0))]


def test_newborn_passes_tags_through_bit_for_bit():
    k = box_class("kind")()
    assert not k.grown()
    objs = _objs()
    out = k.step({"eyein": objs})
    assert len(out["objfile"]) == 4 and len(out["gaze"]) == 3
    for (a, ma), (b, mb) in zip(out["objfile"], objs):
        assert torch.equal(a, b) and torch.equal(ma, mb)
    for (a, _), (b, _) in zip(out["gaze"], objs[:-1]):
        assert torch.equal(a, b)


def _fill(k, kinds, colors, g):
    """2種類×2色の見え。同じ行（個体）の続きの見え・同じ語・同じ場面の別の物を、名簿を通さず置き場へ直接入れる。"""
    row = 0
    for t0 in range(0, 400, 10):           # 場面40個：どの場面も種類0と種類1を1つずつ（色は乱数）
        for kd in (0, 1):
            c = int(torch.randint(0, 2, (1,), generator=g))
            base = kinds[kd] + colors[c]
            for dt in (0, 5, 10):
                k.memory.append({"x": make_tag(base + 0.3 * torch.randn(TAG_DIM, generator=g)), "row": row, "t": t0 * 10 + dt,
                                 "att": True, "word": word_tag(["わんわん", "ぶうぶう"][kd]), "sound": None})
            row += 1
    del k.memory[:-512]


def test_learning_pulls_same_kind_together_across_colors():
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)          # 小さな行列なので1本の方が速い（並べて回すと何倍も遅くなった）
    try:
        _learning_pulls_same_kind_together()
    finally:
        torch.set_num_threads(n_threads)


def _learning_pulls_same_kind_together():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    kinds = [torch.randn(TAG_DIM, generator=g) for _ in range(2)]
    colors = [torch.randn(TAG_DIM, generator=g) for _ in range(2)]
    k = box_class("kind")(learning=True)
    _fill(k, kinds, colors, g)

    def cos(a, b):
        x = torch.stack([make_tag(a), make_tag(b)])
        y = k.transform(x)
        return float(y[0] @ y[1])
    sk0 = cos(kinds[0] + colors[0], kinds[0] + colors[1])       # 同じ種類・違う色
    sc0 = cos(kinds[0] + colors[0], kinds[1] + colors[0])       # 違う種類・同じ色
    for _ in range(100):
        k.train_step()
    assert k.grown() and k.n_train == 100
    sk1 = cos(kinds[0] + colors[0], kinds[0] + colors[1])
    sc1 = cos(kinds[0] + colors[0], kinds[1] + colors[0])
    assert sk1 > sk0 + 0.2                       # 同じ種類の違う色が近づく
    assert sk1 - sc1 > (sk0 - sc0) + 0.2         # 色より種類で近い（同じ色の別の種類との差が開く）


def test_kind_in_the_brain_wiring():
    """脳の配線：目の入口の列は kind を通って名簿と gaze へ。名簿は kind へ席の列を返す。"""
    from core_v2 import wiring
    ins = {e.src for e in wiring.incoming("kind")}
    outs = {e.dst for e in wiring.outgoing("kind")}
    assert ins == {"eyein", "objfile", "listen", "soundin"} and outs == {"objfile", "gaze"}
    assert "eyein" not in {e.src for e in wiring.incoming("objfile")}


def test_label_marks_only_recently_attended_views():
    k = box_class("kind")(learning=True)
    k.t = 100
    for t, att in ((60, True), (80, True), (90, False)):
        k.memory.append({"x": noisy(make_tag(torch.ones(TAG_DIM))), "row": 0, "t": t, "att": att, "word": None,
                         "sound": None})
    k._label("word", word_tag("こっぷ"))
    assert [m["word"] is not None for m in k.memory] == [False, True, False]
