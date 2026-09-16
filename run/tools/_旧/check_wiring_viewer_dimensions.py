"""TaroMap（配線可視化アプリ）の scan_dimensions.py（実測モジュール）が正しく
測れているかを、Qt起動なしで検証する。

scan_dimensions.py はUIから独立した純粋関数（Qt非依存）として作られているため、
ここではQApplicationを一切構築せず、関数を直接呼んで値を照合するだけの
軽量な検証にする（既存の run/tools/check_wiring_viewer_extract.py と同じ流儀）。

使い方（既存の run/tools/check_*.py 群と同じ流儀）:
    .venv\\Scripts\\python.exe run\\tools\\check_wiring_viewer_dimensions.py
    -> pytestからも `pytest run/tools/check_wiring_viewer_dimensions.py` で呼べる。
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.viewer_tools.wiring_viewer import scan_dimensions  # noqa: E402

# 実測済みの表（仕様「実測済みの結果」節。この試作値で照合する。パラメータ数は
# 整数として完全一致するはず、許容誤差なし）。
_EXPECTED_SENSE_PARAMS = {
    "insula": 320,
    "vest": 448,
    "prop": 14528,
    "touch": 505152,
    "vision": 2128768,
}
_EXPECTED_OUT_DIM = 64


def test_measure_dimensions_sense_params_match_known_table():
    """measure_dimensions()の感覚5個のパラメータ数が既知の実測表と完全一致するか。"""
    result = scan_dimensions.measure_dimensions(use_cache=False)
    senses = result["感覚"]
    assert set(senses.keys()) == set(_EXPECTED_SENSE_PARAMS.keys()), (
        f"感覚のid集合が想定と違う: {sorted(senses.keys())}")
    for nid, expected in _EXPECTED_SENSE_PARAMS.items():
        got = senses[nid]["パラメータ数"]
        assert got == expected, f"{nid} のパラメータ数={got} != 想定{expected}"
        out_dim = senses[nid]["出力次元"]
        assert out_dim == _EXPECTED_OUT_DIM, f"{nid} の出力次元={out_dim} != 想定{_EXPECTED_OUT_DIM}"


def test_measure_dimensions_return_shape():
    """戻り値が仕様「戻り値の形」節どおりの主要キーを持つか。"""
    result = scan_dimensions.measure_dimensions(use_cache=False)
    for key in ("生成時刻", "所要秒", "キャッシュから", "指紋", "感覚", "融合層",
                "脳", "予測構造", "遠心性コピー"):
        assert key in result, f"戻り値に必須キー'{key}'が無い"

    assert result["所要秒"] > 0, "所要秒が0以下（実測していない可能性）"
    assert result["キャッシュから"] is False, "use_cache=Falseなのにキャッシュからになっている"

    fusion = result["融合層"]
    for key in ("統合する感覚id", "出力次元", "パラメータ数_学習側", "パラメータ数_凍結正解側"):
        assert key in fusion, f"融合層に必須キー'{key}'が無い"
    assert fusion["統合する感覚id"] == ["insula", "prop", "vest", "touch", "vision"]
    assert fusion["出力次元"] == 320
    assert fusion["パラメータ数_学習側"] > 0
    assert fusion["パラメータ数_凍結正解側"] > 0
    # 学習側と凍結正解側は同じ構造の別インスタンス（RND式）なので、パラメータ数は
    # 同数でなければならない（乱数初期値が違うだけで、構造は完全に同一のはず）。
    assert fusion["パラメータ数_学習側"] == fusion["パラメータ数_凍結正解側"], (
        "学習側と凍結正解側でパラメータ数が違う（構造が揃っていない可能性）")

    brain = result["脳"]
    for key in ("入力次元", "出力次元", "パラメータ数", "代表値の前提"):
        assert key in brain, f"脳に必須キー'{key}'が無い"
    assert brain["入力次元"] == 320
    assert brain["出力次元"] == 90
    assert brain["パラメータ数"] > 0

    efference = result["遠心性コピー"]
    assert "有効か" in efference
    assert isinstance(efference["有効か"], bool)


def test_measure_dimensions_efference_copy_reflects_cfg():
    """cfg.efference_copy の値だけが"遠心性コピー"."有効か"に反映されるか。

    感覚・融合層・脳の構造自体はcfgに依らず不変（仕様の仕様）。
    """
    config_mod = __import__("run.config", fromlist=["Config"])
    cfg_on = config_mod.Config(taro={"efference_copy": True})
    cfg_off = config_mod.Config(taro={"efference_copy": False})

    r_on = scan_dimensions.measure_dimensions(cfg=cfg_on, use_cache=False)
    r_off = scan_dimensions.measure_dimensions(cfg=cfg_off, use_cache=False)

    assert r_on["遠心性コピー"]["有効か"] is True
    assert r_off["遠心性コピー"]["有効か"] is False

    # 構造自体（感覚・融合層・脳）はcfgに依らず同じでなければならない。
    assert r_on["感覚"]["vision"]["パラメータ数"] == r_off["感覚"]["vision"]["パラメータ数"]
    assert r_on["脳"]["パラメータ数"] == r_off["脳"]["パラメータ数"]
    assert r_on["融合層"]["パラメータ数_学習側"] == r_off["融合層"]["パラメータ数_学習側"]


def test_measure_dimensions_cache_roundtrip_and_speedup():
    """キャッシュ無し→保存→キャッシュ有りの2回目が、指紋一致でキャッシュ命中するか。

    速度自体はマシン負荷に左右されるため厳密比較はしない
    （落とし穴チェックリスト ##3 乱数・環境依存の値を閾値で断定しない、と同じ考え方
    でここでは「キャッシュから=Trueになる」ことだけを厳密条件にする）。
    """
    r1 = scan_dimensions.measure_dimensions(use_cache=False)  # キャッシュを最新化
    assert os.path.exists(scan_dimensions._CACHE_PATH), "キャッシュファイルが保存されていない"

    t0 = time.time()
    r2 = scan_dimensions.measure_dimensions(use_cache=True)
    elapsed_cache_hit = time.time() - t0

    assert r2["キャッシュから"] is True, "指紋が一致するはずなのにキャッシュ命中しなかった"
    assert r2["指紋"] == r1["指紋"]
    assert r2["感覚"]["vision"]["パラメータ数"] == r1["感覚"]["vision"]["パラメータ数"]
    # キャッシュ命中は「ファイル読み込みだけ」で速いはず（仕様：1秒を大きく超えるなら
    #   申し送り、との目安があるので緩めに5秒で判定する）。
    assert elapsed_cache_hit < 5.0, f"キャッシュ命中のはずが{elapsed_cache_hit:.2f}秒かかった"


def test_measure_dimensions_fingerprint_changes_when_dependency_mtime_changes():
    """依存ファイル1つのmtimeを更新すると、指紋が変わりキャッシュが再計測されるか。

    【なぜ大事か、仕様より】手書きの表が2週間で腐った実例（touch_adaptation欠落）
    があり、この自動失効の仕組みが機能することが最重要の検証項目。
    """
    r1 = scan_dimensions.measure_dimensions(use_cache=True)  # まずキャッシュを作る/更新する
    fp_before = r1["指紋"]

    target_rel = scan_dimensions._DEP_FILES_REL[0]  # insula.py
    target_abs = os.path.join(scan_dimensions._ROOT, target_rel)
    st = os.stat(target_abs)
    # 元のmtimeを覚えておき、検証後に必ず復元する（他の検証・実行環境を汚さない）。
    orig_atime, orig_mtime = st.st_atime, st.st_mtime
    try:
        new_mtime = orig_mtime + 5  # 5秒進める（ファイルシステムの時刻分解能より確実に大きい差）
        os.utime(target_abs, (orig_atime, new_mtime))

        fp_after = scan_dimensions._compute_fingerprint()
        assert fp_after != fp_before, "依存ファイルのmtimeを変えたのに指紋が変わらなかった"

        r2 = scan_dimensions.measure_dimensions(use_cache=True)
        assert r2["キャッシュから"] is False, "指紋が変わったのにキャッシュ命中してしまった"
        assert r2["指紋"] == fp_after
    finally:
        os.utime(target_abs, (orig_atime, orig_mtime))
        # 後片付け：mtimeを元に戻したあと、キャッシュも元の状態に測り直しておく
        #   （このテストの副作用で他の検証のキャッシュ命中判定が狂わないように）。
        scan_dimensions.measure_dimensions(use_cache=False)


def test_measure_dimensions_fingerprint_changes_when_self_file_mtime_changes():
    """scan_dimensions.py自身のmtimeを更新しても指紋が変わるか。

    【なぜ大事か、2026-08-15 実装担当の指摘で追加】最初の依存ファイル一覧に
    scan_dimensions.py自身が含まれておらず、このファイルのロジック・定数
    （_FUSION_VISION_RES 等）を変更しても指紋が変わらず、古いキャッシュが黙って
    使われ続ける穴があった。上のtest_...dependency_mtime_changesの
    「依存ファイル」を自分自身に変えた版。
    """
    r1 = scan_dimensions.measure_dimensions(use_cache=True)  # まずキャッシュを作る/更新する
    fp_before = r1["指紋"]

    self_abs = os.path.abspath(scan_dimensions.__file__)
    assert self_abs in scan_dimensions._DEP_FILES_REL, (
        "scan_dimensions.py自身が_DEP_FILES_RELに含まれていない（今回の修正が入っていない）")

    st = os.stat(self_abs)
    orig_atime, orig_mtime = st.st_atime, st.st_mtime
    try:
        new_mtime = orig_mtime + 5
        os.utime(self_abs, (orig_atime, new_mtime))

        fp_after = scan_dimensions._compute_fingerprint()
        assert fp_after != fp_before, "scan_dimensions.py自身のmtimeを変えたのに指紋が変わらなかった"

        r2 = scan_dimensions.measure_dimensions(use_cache=True)
        assert r2["キャッシュから"] is False, "指紋が変わったのにキャッシュ命中してしまった"
        assert r2["指紋"] == fp_after
    finally:
        os.utime(self_abs, (orig_atime, orig_mtime))
        scan_dimensions.measure_dimensions(use_cache=False)


def test_measure_dimensions_fusion_params_match_sum_of_senses():
    """融合層のパラメータ数が、感覚5個を単体測定した値の合計と一致するか。

    【なぜ大事か、2026-08-15の差し戻しより】MinimalFusionを作るときの引数
    （touch_dim/vision_res/proprio_dim）が、感覚エンコーダを単体測定したときの
    引数（TouchEncoder(1908)・VisionEncoder()のデフォルトimage_size=256・
    ProprioceptionEncoder(226)）と食い違うと、「融合層は5感覚を集める」という
    説明と実際の数字が矛盾してしまう（過去に vision_res=64 という試作の便宜値を
    確認せず流用し、320+448+14528+505152+2128768=2,649,216 のはずが 683,136 に
    なっていたバグが実際にあった）。layer_norm は重みを持たないため、
    差分は0でなければならない（許容誤差なし）。
    """
    result = scan_dimensions.measure_dimensions(use_cache=False)
    senses = result["感覚"]
    expected_sum = sum(info["パラメータ数"] for info in senses.values())
    fusion = result["融合層"]
    got_train = fusion["パラメータ数_学習側"]
    got_target = fusion["パラメータ数_凍結正解側"]
    assert got_train == expected_sum, (
        f"融合層(学習側)のパラメータ数={got_train} != 感覚5個の合計{expected_sum}"
        "（layer_normは重みなしなので差は0のはず）")
    assert got_target == expected_sum, (
        f"融合層(凍結正解側)のパラメータ数={got_target} != 感覚5個の合計{expected_sum}")


def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except AssertionError as exc:
            failed.append(t.__name__)
            print(f"[ng] {t.__name__}: {exc}", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)}件失敗: {failed}", file=sys.stderr)
        return 1
    print(f"\nすべて合格（{len(tests)}件）")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
