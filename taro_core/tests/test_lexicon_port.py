"""F2「耳の移植」の単体テスト（F/docs/仕様_F2_耳の移植.md の検証4項目）。

1. 合成データ：語A+特徴a群、語B+特徴b群を各50回observe → assoc(語A)がa平均に、
   assoc(語B)がb平均に収束（コサイン類似 > 0.9）
2. 交差：assoc(語A)とb平均の類似が、a平均との類似より十分低い
3. 除去：observeを呼ばない対照（gain=0相当）で1・2が不成立
4. B原本との等価性：3次元の同一入力列でB版Lexiconと出力が一致（移植の正しさ）

taro_core側の新設ファイルのみを使う（hearing.py・lexicon.py）。B側は読み取り専用
（比較用に原本を直接importするだけで、変更は一切しない）。
"""

import os
import sys
import math
import random
import importlib.util

_BRIDGE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_BRIDGE, "src", "senses"))
sys.path.insert(0, os.path.join(_BRIDGE, "src", "brain"))
from hearing import Hearing
from lexicon import Lexicon


def _load_b_original_lexicon_class():
    """B原本のLexiconクラスをファイルパスから直接読み込む（パッケージ__init__経由
    だとtorch等の依存が芋づる式に読み込まれるため、比較目的の最小importとして
    importlib.util.spec_from_file_locationで単一ファイルだけを読む）。B側は
    読み取り専用（このテストはimportするだけで、書き換えは一切しない）。"""
    b_path = os.path.join(_BRIDGE, "..", "B", "src", "taro", "brain", "lexicon.py")
    b_path = os.path.abspath(b_path)
    spec = importlib.util.spec_from_file_location("b_lexicon_original", b_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Lexicon


def cosine(u, v):
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def _monotonic_confidences(n, rng):
    """局所的な谷が生じない（＝端の境界だけになる）ように、単調増加の自信度列を作る。
    これで segment() は必ずtokens全体を1単位として返す（発話全体を1チャンクとして
    扱ってよい、というF2初段の単純化に対応する構成）。"""
    vals = sorted(rng.uniform(0.0, 1.0) for _ in range(n))
    # 単調"厳密"増加にして同値による谷判定のブレを避ける
    return [v + i * 1e-4 for i, v in enumerate(vals)]


def test_1_convergence():
    print("=" * 60)
    print("1. 合成データでの収束テスト")
    print("=" * 60)
    rng = random.Random(0)
    hearing = Hearing()
    lex = Lexicon(min_len=2, state_dim=64)

    word_a = "ぶーぶー"
    word_b = "わんわん"
    tokens_a = tuple(hearing.hear(word_a))
    tokens_b = tuple(hearing.hear(word_b))

    a_center = [rng.uniform(-1, 1) for _ in range(64)]
    b_center = [rng.uniform(-1, 1) for _ in range(64)]

    a_feats = []
    b_feats = []
    for _ in range(50):
        conf = _monotonic_confidences(len(tokens_a), rng)
        feat = [c + rng.uniform(-0.05, 0.05) for c in a_center]
        a_feats.append(feat)
        chunk = lex.observe(list(tokens_a), conf, state=feat)
        assert chunk == tokens_a, f"語Aの分節結果が全体と一致しない: {chunk} vs {tokens_a}"

        conf = _monotonic_confidences(len(tokens_b), rng)
        feat = [c + rng.uniform(-0.05, 0.05) for c in b_center]
        b_feats.append(feat)
        chunk = lex.observe(list(tokens_b), conf, state=feat)
        assert chunk == tokens_b, f"語Bの分節結果が全体と一致しない: {chunk} vs {tokens_b}"

    a_avg = [sum(f[i] for f in a_feats) / 50 for i in range(64)]
    b_avg = [sum(f[i] for f in b_feats) / 50 for i in range(64)]

    assoc_a = lex.assoc(tokens_a)
    assoc_b = lex.assoc(tokens_b)
    assert assoc_a is not None and assoc_b is not None

    sim_a = cosine(assoc_a, a_avg)
    sim_b = cosine(assoc_b, b_avg)
    print(f"assoc(語A) vs a平均 のコサイン類似度: {sim_a:.6f}")
    print(f"assoc(語B) vs b平均 のコサイン類似度: {sim_b:.6f}")
    assert sim_a > 0.9, f"語Aの収束が不十分: {sim_a}"
    assert sim_b > 0.9, f"語Bの収束が不十分: {sim_b}"
    print("収束テスト PASS")
    return lex, tokens_a, tokens_b, a_avg, b_avg


def test_2_cross():
    print("=" * 60)
    print("2. 交差テスト")
    print("=" * 60)
    lex, tokens_a, tokens_b, a_avg, b_avg = test_1_convergence()
    assoc_a = lex.assoc(tokens_a)

    sim_a_to_a = cosine(assoc_a, a_avg)
    sim_a_to_b = cosine(assoc_a, b_avg)
    print(f"assoc(語A) vs a平均 のコサイン類似度: {sim_a_to_a:.6f}")
    print(f"assoc(語A) vs b平均 のコサイン類似度: {sim_a_to_b:.6f}")
    margin = sim_a_to_a - sim_a_to_b
    print(f"差（十分低いことの確認）: {margin:.6f}")
    assert sim_a_to_a > sim_a_to_b + 0.3, (
        f"交差識別が不十分: a-a={sim_a_to_a}, a-b={sim_a_to_b}"
    )
    print("交差テスト PASS")


def test_3_removal():
    print("=" * 60)
    print("3. 除去テスト（observeを呼ばない対照）")
    print("=" * 60)
    rng = random.Random(1)
    hearing = Hearing()
    lex = Lexicon(min_len=2, state_dim=64)  # observeを一度も呼ばない対照

    word_a = "ぶーぶー"
    word_b = "わんわん"
    tokens_a = tuple(hearing.hear(word_a))
    tokens_b = tuple(hearing.hear(word_b))

    assoc_a = lex.assoc(tokens_a)
    assoc_b = lex.assoc(tokens_b)
    print(f"assoc(語A)（観測なし）: {assoc_a}")
    print(f"assoc(語B)（観測なし）: {assoc_b}")
    assert assoc_a is None, "observeを呼んでいないのにassocが値を返した"
    assert assoc_b is None, "observeを呼んでいないのにassocが値を返した"
    print("除去テスト PASS（1・2で成立していた収束・交差が、観測なしでは不成立＝Noneであることを確認）")


def test_4_b_equivalence():
    print("=" * 60)
    print("4. B原本との等価性テスト（3次元・同一入力列）")
    print("=" * 60)
    BLexicon = _load_b_original_lexicon_class()

    b_lex = BLexicon(min_len=2)
    p_lex = Lexicon(min_len=2, state_dim=3)  # F2版はstate_dimを3にすればB原本相当

    rng = random.Random(42)
    vocab_chars = list("あいうえおぶーわんこいぬ")

    results_match = True
    for trial in range(30):
        n = rng.randint(1, 6)
        tokens = [rng.choice(vocab_chars) for _ in range(n)]
        confidences = [rng.uniform(0.0, 1.0) for _ in range(n)]
        state = [rng.uniform(0.0, 1.0) for _ in range(3)]

        b_chunk = b_lex.observe(list(tokens), list(confidences), state=list(state))
        p_chunk = p_lex.observe(list(tokens), list(confidences), state=list(state))
        if b_chunk != p_chunk:
            results_match = False
            print(f"不一致（segment）: trial={trial} b={b_chunk} p={p_chunk}")

    b_counts = sorted(b_lex.counts.items())
    p_counts = sorted(p_lex.counts.items())
    counts_match = b_counts == p_counts
    print(f"分節結果の一致: {results_match}")
    print(f"頻度カウントの一致: {counts_match}")

    assoc_match = True
    for chunk in b_lex.counts:
        b_a = b_lex.assoc(chunk)
        p_a = p_lex.assoc(chunk)
        if b_a is None or p_a is None:
            if b_a != p_a:
                assoc_match = False
            continue
        diff = max(abs(x - y) for x, y in zip(b_a, p_a))
        if diff > 1e-9:
            assoc_match = False
            print(f"assoc不一致: chunk={chunk} b={b_a} p={p_a}")
    print(f"assoc（連合）の一致: {assoc_match}")

    assert results_match, "分節結果がB原本と一致しない"
    assert counts_match, "頻度カウントがB原本と一致しない"
    assert assoc_match, "assocの値がB原本と一致しない"
    print("等価性テスト PASS")


def main():
    test_1_convergence()
    test_2_cross()
    test_3_removal()
    test_4_b_equivalence()
    print("=" * 60)
    print("全4項目 PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
