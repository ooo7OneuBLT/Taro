"""体型補正の「実験設定」— 環境変数から係数を読み、太郎の身体定義（core）へ渡す。

【役割分担・2026-07-25】
- **体型そのもの（何が新生児の体型か）** … `taro_core/src/body/infant_body.py`（core）
- **実験でどう振るか（環境変数）** … このファイル（目標フォルダ）
方針：[[feedback-core-vs-experiment-placement]]（本実装は core、実験の設定は目標フォルダ）。

【なぜ分けたか】体型補正はもともと `e_toy_env.py`（おもちゃ環境）の中にあり、
**おもちゃ環境でしか効かなかった**。そのため学習に使う仰向け環境（`SupineMimoEnv`）では
補正なし＝**太郎は成人プロポーションのまま学習していた**（2026-07-25 に発覚）。
体型を core へ移し、環境に依存せずどの実験からも同じ体型を使えるようにした。

【使い方】
    from e_body_config import newborn_scales_from_env, body_scale_custom_from_env
    custom = body_scale_custom_from_env(age=0.0)   # None なら補正なし
    if custom:
        kwargs["custom_measurements"] = custom
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
sys.path.insert(0, os.path.join(paths.SRC, "body"))
from infant_body import (NEWBORN_SHAPE_DEFAULTS, HEAD_ELONGATION,  # noqa: E402
                         body_scale_custom)

# 部位グループ -> 環境変数名。既定値は core（NEWBORN_SHAPE_DEFAULTS）が持つ。
NEWBORN_SHAPE_ENV = {
    "leg":         "E_LEG_SCALE",
    "leg_thick":   "E_LEG_THICK",
    "arm":         "E_ARM_SCALE",
    "arm_thick":   "E_ARM_THICK",
    "trunk_len":   "E_TRUNK_LEN",
    "trunk_width": "E_TRUNK_WIDTH",
    "foot":        "E_FOOT_SCALE",
    "foot_width":  "E_FOOT_WIDTH",
    "hand":        "E_HAND_SCALE",
}


def shape_enabled():
    """E_SHAPE=0 で体型補正を完全に切る（＝素のmimoGrowth体型＝アブレーション）。"""
    return os.environ.get("E_SHAPE", "1") == "1"


def newborn_scales_from_env():
    """環境変数から体型の係数を読む。補正OFFなら全部1.0（＝触らない）。"""
    if not shape_enabled():
        return {k: 1.0 for k in NEWBORN_SHAPE_DEFAULTS}
    return {k: float(os.environ.get(NEWBORN_SHAPE_ENV[k], str(v)))
            for k, v in NEWBORN_SHAPE_DEFAULTS.items()}


def head_elongation_from_env():
    """頭の楕円化率。補正OFFなら1.0（＝球のまま）。"""
    if not shape_enabled():
        return 1.0
    return float(os.environ.get("E_HEAD_ELONG", str(HEAD_ELONGATION)))


def body_scale_custom_from_env(age, verbose=True):
    """環境変数の設定で custom_measurements を作る。補正が全て1.0なら None を返す。"""
    scales = newborn_scales_from_env()
    if not any(abs(v - 1.0) > 1e-9 for v in scales.values()):
        return None
    return body_scale_custom(float(age), scales, verbose=verbose)


def body_kwargs_from_env(age, verbose=True):
    """★環境を作るときに渡す身体の設定を**まとめて**返す唯一の入口。

    使い方:
        env = SupineMimoEnv(age=0.0, **body_kwargs_from_env(0.0))

    【なぜ1つにまとめたか、2026-07-25】体型（`custom_measurements`）と頭の楕円化
    （`head_elongation`）は別々の関数で、**呼ぶ側が両方を思い出す必要があった**。
    実際に取りこぼしが起きていた：
      - `e_ctrl_freq_probe.py` の測定側 … 体型だけ渡して**頭は球のまま**
      - 同 Viewerモード … **どちらも渡さず素の体**（Viewerで見ている太郎と、
        学習している太郎が別の身体だった）
      - `e_body_measure.py` … 頭を渡しておらず、日誌に載せた数値が後日再現しなかった
    ＝「身体の設定が散らばる」問題（体型をcoreへ集約した理由そのもの）が、
    **呼び出し側にも同じ形で残っていた**。入口を1つにすれば取りこぼしが起きない。
    ⚠️新しく環境を作るコードを書くときは、必ずこの関数を使うこと。
    """
    kw = {"head_elongation": head_elongation_from_env()}
    custom = body_scale_custom_from_env(age, verbose=verbose)
    if custom:
        kw["custom_measurements"] = custom
    return kw
