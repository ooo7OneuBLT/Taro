"""聞き分ける係（段階5）：親の話しかけを聞いて語を切り出し、知っている語だけ荷札を出す。"""

import torch

from core_v2.boxes.listen import Listen
from core_v2.tagging import word_tag
from core_v2.tools.parent_speech import make_corpus


def _grown(n=1000):
    ear = Listen(learning=True)
    for s in make_corpus(n, seed=1):
        ear.step({"earin": s})
    ear.learning = False
    return ear


def test_newborn_knows_nothing():
    out = Listen().step({"earin": "ぼおるはどこ？"})
    assert float(torch.linalg.vector_norm(out["semantic"])) == 0.0


def test_grown_ear_finds_names_and_ignores_unheard(tmp_path):
    ear = _grown()
    assert {"ぼおる", "こっぷ"} <= ear.known()
    assert "りんごはどこ" not in ear.known()             # まるごと何度も聞いた文は語にしない
    assert ear.segment("ぼおるはどこ？")[0] == "ぼおる"
    assert torch.allclose(ear.step({"earin": "ぼおるはどこ？"})["semantic"], word_tag("ぼおる"))
    assert torch.allclose(ear.step({"earin": "こっぷだね"})["semantic"], word_tag("こっぷ"))
    assert float(torch.linalg.vector_norm(ear.step({"earin": "でんしゃはどこ？"})["semantic"])) == 0.0
    p = tmp_path / "ear.pt"
    ear.save_ear(p, {"n": 1})
    other = Listen()
    assert other.load_ear(p) == {"n": 1}
    assert other.known() == ear.known()
    assert other.segment("こっぷをみて") == ear.segment("こっぷをみて")


def test_not_learning_changes_nothing():
    ear = Listen()
    before = [p.clone() for p in ear.parameters()]
    ear.step({"earin": "ぼおるだね ぼおるだね"})
    assert all(torch.equal(a, b) for a, b in zip(before, ear.parameters()))
    assert not ear.alone


def test_new_words_heard_alone_become_words():
    """段階6：スミスとユーのように新しい語を1語だけで聞くと、2回で語になる。知っている語を含む言い回しは語にしない。"""
    ear = _grown()
    ear.learning = True
    for _ in range(2):
        for w in ["ぼさ", "これっと"]:
            ear.step({"earin": w})
    assert {"ぼさ", "これっと"} <= ear.known()
    assert ear.segment("これっと") == ["これっと"]
    assert torch.allclose(ear.step({"earin": "これっと"})["semantic"], word_tag("これっと"))
    assert not {"おおきいぼおる", "ほらこっぷ", "りんごはどこ"} & ear.known()
