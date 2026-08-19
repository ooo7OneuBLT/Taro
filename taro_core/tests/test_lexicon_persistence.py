"""F1-4a「語彙の保存」の単体テスト（F/docs/仕様_F1-4a_語彙の保存.md 新設テスト2項目）。

run/taro_setup.py の save()/_load() が使うblob形式（hearing_vocab/lexicon）を
そのままここで組み立て、torch.save/torch.load を経由して往復一致するかを見る。
taro_core側の部品（hearing.py・lexicon.py）だけを使う（run/taro_setup.py 自体は
importしない＝環境依存のmujoco等を持ち込まずに単体で走らせるため）。
"""

import os
import sys
import random

import torch

_BRIDGE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_BRIDGE, "src", "senses"))
sys.path.insert(0, os.path.join(_BRIDGE, "src", "brain"))
from hearing import Hearing
from lexicon import Lexicon


def _make_blob(hearing, lexicon):
    """run/taro_setup.py save() のhearing/lexicon部分と同じ形式を組み立てる。"""
    vocab = hearing.vocab
    return {
        "hearing_vocab": {"char2idx": vocab.char2idx, "idx2char": vocab.idx2char,
                           "size": vocab.size},
        "lexicon": {"counts": lexicon.counts, "state_sum": lexicon.state_sum,
                    "state_dim": lexicon.state_dim, "min_len": lexicon.min_len},
    }


def _restore_into(blob, hearing, lexicon):
    """run/taro_setup.py _load() のhearing/lexicon部分と同じ手順で復元する。"""
    hv = blob["hearing_vocab"]
    hearing.vocab.char2idx = hv["char2idx"]
    hearing.vocab.idx2char = hv["idx2char"]
    hearing.vocab.size = hv["size"]
    lx = blob["lexicon"]
    lexicon.counts = lx["counts"]
    lexicon.state_sum = lx["state_sum"]
    lexicon.state_dim = lx["state_dim"]
    lexicon.min_len = lx["min_len"]


def _monotonic_confidences(n, rng):
    """局所的な谷が生じない単調増加の自信度列（test_lexicon_port.pyと同じ構成）。"""
    vals = sorted(rng.uniform(0.0, 1.0) for _ in range(n))
    return [v + i * 1e-4 for i, v in enumerate(vals)]


def _make_populated(seed=0, state_dim=8):
    """数発話をobserveさせたHearing+Lexiconを作る。"""
    rng = random.Random(seed)
    hearing = Hearing()
    lexicon = Lexicon(min_len=2, state_dim=state_dim)
    words = ["わんわん", "ぶーぶー", "まんま", "わんわん", "ぶーぶー"]
    for w in words:
        tokens = hearing.hear(w)
        conf = _monotonic_confidences(len(tokens), rng)
        state = [rng.uniform(-1, 1) for _ in range(state_dim)]
        lexicon.observe(tokens, conf, state=state)
    return hearing, lexicon


def test_1_roundtrip():
    print("=" * 60)
    print("1. 往復一致テスト（save形式でtorch.save→load→復元）")
    print("=" * 60)
    hearing, lexicon = _make_populated(seed=0)
    assert len(lexicon.counts) > 0, "前提が崩れている：observeで何も登録されなかった"

    blob = _make_blob(hearing, lexicon)
    path = os.path.join(os.path.dirname(__file__), "_tmp_test_lexicon_persistence.pt")
    torch.save(blob, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    os.remove(path)

    hearing2 = Hearing()
    lexicon2 = Lexicon(min_len=1, state_dim=1)  # 復元前はわざと違う値にしておく
    _restore_into(loaded, hearing2, lexicon2)

    assert hearing2.vocab.char2idx == hearing.vocab.char2idx, "char2idxが一致しない"
    assert hearing2.vocab.idx2char == hearing.vocab.idx2char, "idx2charが一致しない"
    assert hearing2.vocab.size == hearing.vocab.size, "sizeが一致しない"
    assert lexicon2.counts == lexicon.counts, "countsが一致しない"
    assert lexicon2.state_dim == lexicon.state_dim, "state_dimが一致しない"
    assert lexicon2.min_len == lexicon.min_len, "min_lenが一致しない"

    for chunk, acc in lexicon.state_sum.items():
        acc2 = lexicon2.state_sum[chunk]
        assert acc2 == acc, f"state_sum不一致: {chunk}: {acc} vs {acc2}"

    for chunk in lexicon.counts:
        a1 = lexicon.assoc(chunk)
        a2 = lexicon2.assoc(chunk)
        assert a1 is not None and a2 is not None
        diff = max(abs(x - y) for x, y in zip(a1, a2))
        assert diff <= 1e-12, f"assoc不一致: {chunk}: diff={diff}"
        print(f"  chunk={chunk} count={lexicon.counts[chunk]} assoc一致 diff={diff:.3e}")

    print(f"vocab.size復元: {hearing2.vocab.size} (元={hearing.vocab.size})")
    print(f"lexicon.counts語数復元: {len(lexicon2.counts)} (元={len(lexicon.counts)})")
    print("往復一致テスト PASS")
    return hearing, lexicon, hearing2, lexicon2


def test_2_new_word_no_collision():
    print("=" * 60)
    print("2. 未知語の独立性テスト（復元後の新しい文字がID衝突しない）")
    print("=" * 60)
    hearing, lexicon, hearing2, lexicon2 = test_1_roundtrip()

    size_before = hearing2.vocab.size
    existing_ids = set(hearing2.vocab.idx2char.keys())

    # 元のHearingが一度も見ていない新しい文字を聞かせる
    new_text = "ぴよぴよ"
    for ch in new_text:
        assert ch not in hearing2.vocab.char2idx, f"前提が崩れている：{ch}は既知語彙"
    new_ids = hearing2.hear(new_text)

    assert hearing2.vocab.size > size_before, "新しい文字を聞いてもsizeが増えていない"
    for nid in new_ids:
        # 新規に採番されたIDが復元済みの既存ID集合と衝突していないこと
        # （sizeが正しく復元されていれば、新規IDは常にsize_before以上になるはず）
        assert nid == 2 or nid >= size_before or hearing2.vocab.idx2char[nid] in new_text, (
            f"新IDが衝突: id={nid}")
    # より直接的な検証：新規に採られたIDはすべて size_before 以上（復元前の
    # 最大既知IDより後ろに割り振られている＝sizeカウンタが正しく引き継がれた証拠）
    fresh_ids = [i for ch, i in hearing2.vocab.char2idx.items() if ch in new_text]
    for i in fresh_ids:
        assert i >= size_before, (
            f"新しい文字のID({i})が復元前のsize({size_before})より小さい＝衝突の疑い")
    print(f"復元後size={size_before} → 新規文字聞き取り後size={hearing2.vocab.size}")
    print(f"新規に採番されたID: {sorted(set(fresh_ids))}（いずれも{size_before}以上）")
    print("未知語の独立性テスト PASS")


def main():
    test_1_roundtrip()
    test_2_new_word_no_collision()
    print("=" * 60)
    print("全2項目 PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
