"""F2-12「意味を感覚ごとに分けて持つ」の往復テスト
（設計：F/docs/設計_F2-12_意味を感覚ごとに分けて持つ.md 技術付録）。

run/taro_setup.py の save()/_load() が使う lexicon blob の形（channels優先・
旧キーも両方書く）を、taro_core側の部品（hearing.py・lexicon.py）だけで
再現する（run/taro_setup.py自体はmujoco等に依存するのでimportしない。
test_lexicon_persistence.py と同じ流儀）。
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


def _save_blob(hearing, lexicon):
    """run/taro_setup.py save()（1340行付近）の lexicon 部分と同じ手順。"""
    vocab = hearing.vocab
    blob = {
        "hearing_vocab": {"char2idx": vocab.char2idx, "idx2char": vocab.idx2char,
                           "size": vocab.size},
        "lexicon": {"counts": lexicon.counts, "state_sum": lexicon.state_sum,
                    "state_dim": lexicon.state_dim, "min_len": lexicon.min_len},
    }
    if lexicon.mode == "contrast":
        blob["lexicon"]["mode"] = lexicon.mode
        blob["lexicon"]["proto"] = lexicon.proto
        blob["lexicon"]["view_sum"] = lexicon.view_sum
        blob["lexicon"]["view_n"] = lexicon.view_n
        blob["lexicon"]["channels"] = lexicon.channels
    return blob


def _load_blob(blob, hearing, lexicon):
    """run/taro_setup.py _load()（942行付近）の lexicon 部分と同じ手順。"""
    hv = blob["hearing_vocab"]
    hearing.vocab.char2idx = hv["char2idx"]
    hearing.vocab.idx2char = hv["idx2char"]
    hearing.vocab.size = hv["size"]
    lx = blob["lexicon"]
    lexicon.counts = lx["counts"]
    lexicon.state_sum = lx["state_sum"]
    lexicon.min_len = lx["min_len"]
    if "mode" in lx:
        lexicon.mode = lx["mode"]
    if "channels" in lx:
        lexicon.channels = lx["channels"]
    else:
        lexicon.channels = {
            "vision": {
                "dim": lx["state_dim"],
                "proto": lx.get("proto", {}),
                "view_sum": lx.get("view_sum"),
                "view_n": lx.get("view_n", 0),
            }
        }


def _monotonic_confidences(n, rng):
    vals = sorted(rng.uniform(0.0, 1.0) for _ in range(n))
    return [v + i * 1e-4 for i, v in enumerate(vals)]


def _make_populated_contrast(seed=0, state_dim=8, observe_views=True):
    rng = random.Random(seed)
    hearing = Hearing()
    lexicon = Lexicon(min_len=2, state_dim=state_dim, mode="contrast")
    words = ["わんわん", "ぶーぶー", "まんま", "わんわん", "ぶーぶー", "ぱぱ"]
    for w in words:
        tokens = hearing.hear(w)
        conf = _monotonic_confidences(len(tokens), rng)
        state = [rng.uniform(-1, 1) for _ in range(state_dim)]
        if observe_views:
            lexicon.observe_view(state)
        lexicon.observe(tokens, conf, state=state)
    return hearing, lexicon


def _assert_channels_equal(c1, c2):
    assert set(c1.keys()) == set(c2.keys()), f"チャンネル集合が一致しない: {c1.keys()} vs {c2.keys()}"
    for name in c1:
        ch1, ch2 = c1[name], c2[name]
        assert ch1["dim"] == ch2["dim"], f"[{name}] dim不一致"
        assert ch1["view_n"] == ch2["view_n"], f"[{name}] view_n不一致"
        if ch1["view_sum"] is None:
            assert ch2["view_sum"] is None, f"[{name}] view_sum不一致(None)"
        else:
            diff = max(abs(a - b) for a, b in zip(ch1["view_sum"], ch2["view_sum"]))
            assert diff <= 1e-12, f"[{name}] view_sum不一致 diff={diff}"
        assert ch1["proto"].keys() == ch2["proto"].keys(), f"[{name}] protoキー不一致"
        for chunk, p in ch1["proto"].items():
            p2 = ch2["proto"][chunk]
            diff = max(abs(a - b) for a, b in zip(p, p2))
            assert diff <= 1e-12, f"[{name}] proto不一致: {chunk} diff={diff}"


def test_1_properties_mirror_vision_channel():
    """state_dim/proto/view_sum/view_n が channels['vision'] を指すプロパティであること。"""
    print("=" * 60)
    print("1. プロパティ = channels['vision'] の一致テスト")
    print("=" * 60)
    _, lexicon = _make_populated_contrast(seed=0)
    ch = lexicon.channels["vision"]
    assert lexicon.state_dim == ch["dim"]
    assert lexicon.proto is ch["proto"]
    assert lexicon.view_sum is ch["view_sum"]
    assert lexicon.view_n == ch["view_n"]
    assert lexicon.view_n > 0, "前提が崩れている：observe_viewが呼ばれていない"
    assert len(lexicon.proto) > 0, "前提が崩れている：contrastモードでprotoが育っていない"
    print(f"channels keys = {list(lexicon.channels.keys())}")
    print(f"vision.dim={ch['dim']} view_n={ch['view_n']} proto語数={len(ch['proto'])}")
    print("PASS")


def test_2_save_load_save_roundtrip():
    """保存 → 読み込み → 保存 でblob（channelsキー込み）が一致すること（設計・受け入れ条件2）。"""
    print("=" * 60)
    print("2. 保存→読み込み→保存 往復一致テスト")
    print("=" * 60)
    hearing, lexicon = _make_populated_contrast(seed=1)
    blob1 = _save_blob(hearing, lexicon)

    path = os.path.join(os.path.dirname(__file__), "_tmp_test_lexicon_channels_rt.pt")
    torch.save(blob1, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    os.remove(path)

    hearing2 = Hearing()
    lexicon2 = Lexicon(min_len=1, state_dim=1, mode="sum")  # わざと違う値で初期化
    _load_blob(loaded, hearing2, lexicon2)

    blob2 = _save_blob(hearing2, lexicon2)

    # 旧キー（バイト互換の要）
    assert blob1["lexicon"]["counts"] == blob2["lexicon"]["counts"]
    assert blob1["lexicon"]["state_dim"] == blob2["lexicon"]["state_dim"]
    assert blob1["lexicon"]["min_len"] == blob2["lexicon"]["min_len"]
    assert blob1["lexicon"]["mode"] == blob2["lexicon"]["mode"]
    assert blob1["lexicon"]["proto"].keys() == blob2["lexicon"]["proto"].keys()
    for chunk, p in blob1["lexicon"]["proto"].items():
        p2 = blob2["lexicon"]["proto"][chunk]
        diff = max(abs(a - b) for a, b in zip(p, p2))
        assert diff <= 1e-12, f"旧キーproto不一致: {chunk} diff={diff}"
    diff = max(abs(a - b) for a, b in
                zip(blob1["lexicon"]["view_sum"], blob2["lexicon"]["view_sum"]))
    assert diff <= 1e-12, f"旧キーview_sum不一致 diff={diff}"
    assert blob1["lexicon"]["view_n"] == blob2["lexicon"]["view_n"]

    # 新キー（channels）
    _assert_channels_equal(blob1["lexicon"]["channels"], blob2["lexicon"]["channels"])

    print(f"語数={len(blob1['lexicon']['counts'])} "
          f"proto語数={len(blob1['lexicon']['proto'])} "
          f"view_n={blob1['lexicon']['view_n']}")
    print("旧キー・新キーとも往復一致 PASS")


def test_3_old_format_model_loads_without_channels_key():
    """旧形式（channelsキー無し）のblobを読んでも警告・エラーなく復元できること
    （設計・受け入れ条件3。F2-9Cのような既存モデルのシミュレーション）。
    """
    print("=" * 60)
    print("3. 旧形式（channelsキー無し）blobの復元テスト")
    print("=" * 60)
    hearing, lexicon = _make_populated_contrast(seed=2)
    blob = _save_blob(hearing, lexicon)
    # 旧形式をシミュレート：channelsキーを持たないblobにする
    #（F2-12より前に保存されたモデルはこの形しか持たない）
    del blob["lexicon"]["channels"]
    assert "channels" not in blob["lexicon"]

    hearing2 = Hearing()
    lexicon2 = Lexicon(min_len=1, state_dim=1, mode="sum")
    _load_blob(blob, hearing2, lexicon2)

    assert lexicon2.mode == "contrast"
    assert lexicon2.state_dim == lexicon.state_dim
    assert lexicon2.proto.keys() == lexicon.proto.keys()
    for chunk, p in lexicon.proto.items():
        p2 = lexicon2.proto[chunk]
        diff = max(abs(a - b) for a, b in zip(p, p2))
        assert diff <= 1e-12, f"proto不一致: {chunk} diff={diff}"
    assert lexicon2.view_n == lexicon.view_n
    diff = max(abs(a - b) for a, b in zip(lexicon.view_sum, lexicon2.view_sum))
    assert diff <= 1e-12, f"view_sum不一致 diff={diff}"
    assert set(lexicon2.channels.keys()) == {"vision"}, (
        "旧形式からはvisionチャンネル1つだけが組み立てられるはず")

    print(f"復元後 channels keys = {list(lexicon2.channels.keys())}")
    print("旧形式blobの復元 PASS（警告・例外なし）")


def test_4_sum_mode_blob_has_no_new_keys():
    """既定sumモードでは mode/proto/view_sum/view_n/channels の5キーとも
    追加されない（旧blobとバイト互換を保つ、というF1-5以来の規約を壊していないこと）。
    """
    print("=" * 60)
    print("4. sumモードblobにchannelsキーが増えていないことの確認")
    print("=" * 60)
    rng = random.Random(3)
    hearing = Hearing()
    lexicon = Lexicon(min_len=2, state_dim=8, mode="sum")
    for w in ["わんわん", "ぶーぶー"]:
        tokens = hearing.hear(w)
        conf = _monotonic_confidences(len(tokens), rng)
        state = [rng.uniform(-1, 1) for _ in range(8)]
        lexicon.observe(tokens, conf, state=state)

    blob = _save_blob(hearing, lexicon)
    assert set(blob["lexicon"].keys()) == {"counts", "state_sum", "state_dim", "min_len"}, (
        f"sumモードで想定外のキーが増えている: {blob['lexicon'].keys()}")
    print(f"sumモードblobキー = {sorted(blob['lexicon'].keys())}")
    print("PASS")


def main():
    test_1_properties_mirror_vision_channel()
    test_2_save_load_save_roundtrip()
    test_3_old_format_model_loads_without_channels_key()
    test_4_sum_mode_blob_has_no_new_keys()
    print("=" * 60)
    print("全4項目 PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
