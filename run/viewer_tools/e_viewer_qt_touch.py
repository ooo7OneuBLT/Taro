"""Viewer(PySide6版)「感覚と報酬」タブの非GUIロジック（触覚の順応・口元の報酬）。

【これは何か】`run/viewer_tools/e_viewer.py`（旧版・tkinter製）312〜377行目
（監視用インスタンスの構築）・2693〜2752行目（毎tickの更新）・2889〜2910行目
（表示文字列の整形）を、GUIから独立した関数として書き直したもの。
仕様 作業記録（非公開）
2026-08-13_Viewerのモダン化_第3段階_感覚と報酬タブ.md 3節。

【なぜファイルを分けるか】`e_viewer_qt_status.py`と同じ考え方。表示は必ず実体
（TouchAdaptation・DoubleTouchDetector・_MouthTouchBonusContributor）から
その場で読む。GUI組み立て（`e_viewer_qt.py`）から切り離すことで、この
ロジックだけを単体テストできるようにする。

【このタブは駆動と無関係】旧版でも「このViewerは報酬を学習に使わないため、
動きは変わりません」と明記されている（`run/taro_setup.py`のTaro自体を経由
しないため）。`env.step(zero)`だけで駆動している今の新版でも、触覚順応・
口元報酬の監視は独立に成立する。
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TouchMonitor:
    """触覚順応・口元報酬の監視用インスタンス一式。

    `available=False`のときは`touch_map`以下は全部None（構築に失敗した状態）。
    GUI側はこの`available`だけを見て「利用できません」の注意表示に切り替える
    （旧版の`_ta_available`と同じフェイルセーフパターン、仕様3節）。
    """

    available: bool
    error: str = ""
    touch_map: object = None
    mouth_mask: Optional[np.ndarray] = None
    dt_detector: object = None
    mouth_contributor: object = None
    touch_adapt: object = None
    toucher_name: str = ""
    # 【なぜここに持たせるか】旧版はViewer関数のローカル変数
    #   （mouth_bonus_count・mouth_bonus_last_t）だったが、このオブジェクトに
    #   移すことで「毎tickの更新」と「表示の整形」の両方から同じ状態を
    #   参照できる（呼び出し側にリスト・ミュータブルなセルを別途持たせずに済む）。
    mouth_bonus_count: int = 0
    mouth_bonus_last_t: float = -1.0
    last: dict = field(default_factory=dict)

    def reset_mouth_bonus_counts(self):
        """「最初からやり直す」を押したときに呼ぶ（旧版：mouth_bonus_count[0]=0相当）。

        触覚順応の内部状態（g_peripheral・g_cortical）はここでは触らない
        （旧版でも「最初からやり直す」ではリセットされず、「触覚の順応」区画の
        「既定値に戻す」ボタン＝touch_adapt.rebuild()だけがそれを行う。
        仕様2節・旧版1253〜1279行目のコメントの通り）。
        """
        self.mouth_bonus_count = 0
        self.mouth_bonus_last_t = -1.0


def build_touch_monitor(env, m, u, dt):
    """触覚順応・口元報酬の監視用インスタンスを構築する。

    移植元：`e_viewer.py` 312〜377行目。try/exceptで失敗を吸収し、
    例外を外へ投げない（呼び出し元＝GUI側は`available`だけを見ればよい）。
    """
    from somatosensory_cortex import build_touch_map_from_env
    from double_touch import DoubleTouchDetector, mouth_point_mask
    from touch_adaptation import TouchAdaptation
    from run.config import TARO_DEFAULTS

    try:
        touch_map = build_touch_map_from_env(env)
        mouth_mask = mouth_point_mask(
            touch_map, m, u.touch,
            x_frac=TARO_DEFAULTS["mouth_touch_x_frac"][0],
            z_frac=TARO_DEFAULTS["mouth_touch_z_frac"][0])
        dt_detector = DoubleTouchDetector(
            touch_map, threshold=TARO_DEFAULTS["double_touch_threshold"][0],
            touched_names=TARO_DEFAULTS["double_touch_touched_groups"][0],
            mouth_mask=mouth_mask,
            mouth_x_frac=TARO_DEFAULTS["mouth_touch_x_frac"][0],
            mouth_z_frac=TARO_DEFAULTS["mouth_touch_z_frac"][0])
        mouth_contributor = _build_mouth_contributor(
            dt_detector, bonus=float(TARO_DEFAULTS["mouth_touch_bonus"][0]),
            mouth_threshold=TARO_DEFAULTS["mouth_touch_threshold"][0],
            toucher_threshold=TARO_DEFAULTS["double_touch_threshold"][0])
        touch_adapt = TouchAdaptation(
            touch_map.n_points, dt=dt,
            fa_enabled=TARO_DEFAULTS["touch_adapt_fa"][0],
            sa_enabled=TARO_DEFAULTS["touch_adapt_sa"][0],
            include_cortical=TARO_DEFAULTS["touch_adapt_include_cortical"][0],
            tau_peripheral_s=TARO_DEFAULTS["touch_adapt_tau_peripheral_s"][0],
            tau_cortical_s=TARO_DEFAULTS["touch_adapt_tau_cortical_s"][0],
            sa_floor=TARO_DEFAULTS["touch_adapt_sa_floor"][0],
            tau_recover_s=TARO_DEFAULTS["touch_adapt_tau_recover_s"][0],
            fa_gain=TARO_DEFAULTS["touch_adapt_fa_gain"][0])
        toucher_name = f"{TARO_DEFAULTS['reach_arm_side'][0]}_palm"
    except Exception as e:
        print(f"[touch_adapt/mouth] 注意監視用インスタンスの構築に失敗: {e}",
              flush=True)
        return TouchMonitor(available=False, error=str(e))

    return TouchMonitor(
        available=True, touch_map=touch_map, mouth_mask=mouth_mask,
        dt_detector=dt_detector, mouth_contributor=mouth_contributor,
        touch_adapt=touch_adapt, toucher_name=toucher_name)


def _build_mouth_contributor(*args, **kwargs):
    """`run.taro_setup._MouthTouchBonusContributor`を遅延importして呼ぶ薄いラッパー。

    【なぜラップするか】このファイルのトップレベルで`run.taro_setup`を
    importすると、（run/taro_setup.py 自体は「読むだけ」の対象ファイルであり
    直接依存を増やしたくないのと、e_viewer.pyの312行目コメントが示す通り
    Taro構築コード全体は重い）他の関数（`build_touch_monitor`）と同じ
    タイミング（呼ばれたとき）だけimportすることで、モジュール読み込み時の
    副作用・循環importのリスクを避ける（旧版のtry/exceptブロックの中で
    importしているのと同じ配置）。
    """
    from run.taro_setup import _MouthTouchBonusContributor
    return _MouthTouchBonusContributor(*args, **kwargs)


def advance_touch_monitor(monitor, params, u, t_sim):
    """物理が1tick進んだ直後に1回だけ呼ぶ（移植元：e_viewer.py 2693〜2752行目）。

    【なぜここで呼ぶ時機が大事か】touch_adaptation.pyのdocstringの前提は
    「新しい物理観測が生まれた瞬間にだけ呼ぶ」こと。freeze中（物理を止めて
    いるとき）は呼び出し元がそもそもこの関数を呼ばないこと（`advance()`も
    しないこと）。この制約は呼び出し元（e_viewer_qt.py）が守る責任を持つ
    （このファイル自体はfreeze状態を知らない）。

    params: 現在のスライダー・チェックボックスの値をまとめたdict。
        必須キー：fa_enabled, sa_enabled, include_cortical, master_enabled,
                  tau_peripheral_s, tau_cortical_s, sa_floor, tau_recover_s,
                  fa_gain, mouth_bonus（いずれもbool/floatへ変換して使う）

    戻り値：monitor.last と同じ内容のdict（最新の測定値）。available=False
        のときは空dictを返す。
    """
    if not monitor.available:
        return {}

    ta = monitor.touch_adapt
    # マスターOFFのときは個別チェックボックス（速順応・遅順応）とのANDを取る
    #   （仕様3-2節。旧版2706〜2712行目と同じ）。
    ta.fa_enabled = bool(params["fa_enabled"]) and bool(params["master_enabled"])
    ta.sa_enabled = bool(params["sa_enabled"]) and bool(params["master_enabled"])
    ta.include_cortical = bool(params["include_cortical"])
    ta.tau_peripheral_s = float(params["tau_peripheral_s"])
    ta.tau_cortical_s = float(params["tau_cortical_s"])
    ta.sa_floor = float(params["sa_floor"])
    ta.tau_recover_s = float(params["tau_recover_s"])
    ta.fa_gain = float(params["fa_gain"])
    monitor.mouth_contributor.bonus = float(params["mouth_bonus"])

    touch_flat = u.get_touch_obs().ravel()
    ta.advance(touch_flat)

    from run.taro_setup import to_tensor
    touch_tensor = to_tensor(touch_flat)
    hit, toucher_pres, touched_pres, hit_names = monitor.dt_detector.detect(
        touch_tensor, monitor.toucher_name)
    mouth_pres = monitor.dt_detector.mouth_presence(touch_tensor)

    # 【なぜSimpleNamespaceで十分か、e_viewer.py 2727〜2730行目と同じ】
    #   _MouthTouchBonusContributor.compute(ctx) が読むのは
    #   ctx.last_double_touch（dict）と ctx.last["obs_out"]["touch"] の2つだけ。
    ctx_stub = types.SimpleNamespace(
        last_double_touch={"toucher_presence": toucher_pres},
        last={"obs_out": {"touch": touch_flat}})
    bonus = monitor.mouth_contributor.compute(ctx_stub)
    if bonus > 0:
        monitor.mouth_bonus_count += 1
        monitor.mouth_bonus_last_t = t_sim

    raw_f = touch_flat.reshape(-1, 3)
    adapted_f = np.asarray(ta.adapted()).reshape(-1, 3)
    if monitor.mouth_mask is not None and monitor.mouth_mask.any():
        raw_mouth = float(np.max(np.linalg.norm(raw_f[monitor.mouth_mask], axis=-1)))
        adapted_mouth = float(
            np.max(np.linalg.norm(adapted_f[monitor.mouth_mask], axis=-1)))
    else:
        raw_mouth = adapted_mouth = 0.0

    last = dict(
        raw_mouth=raw_mouth, adapted_mouth=adapted_mouth,
        toucher_presence=toucher_pres, mouth_presence=mouth_pres,
        g_peripheral_mean=float(np.mean(ta.g_peripheral)),
        g_cortical_mean=float(np.mean(ta.g_cortical)))
    monitor.last = last
    return last


def format_touch_status(last, monitor, params, mouth_bonus_count, mouth_bonus_last_t):
    """表示用の2つの文字列（口元の報酬／触覚の順応）に整形する。

    移植元：e_viewer.py 2889〜2910行目。monitor.available=False のときは
    呼び出し元が「利用できません」ラベルを出す想定なので、この関数は
    available=True の場合だけ呼ばれることを期待する（呼ばれた場合の安全策
    として、Noneが混ざっていても例外にせず0.0扱いにする）。
    """
    mth = float(monitor.dt_detector.threshold)
    mmt = float(monitor.mouth_contributor.mouth_threshold)
    mouth_text = (
        f"口元の接触の強さ：生 {last.get('raw_mouth', 0.0):.3f}"
        f" / 順応後 {last.get('adapted_mouth', 0.0):.3f}\n"
        f"手のひらpresence: {last.get('toucher_presence', 0.0):.2f}"
        f"（しきい値{mth:.2f}）　"
        f"口元presence: {last.get('mouth_presence', 0.0):.2f}"
        f"（しきい値{mmt:.2f}）\n"
        f"口元ボーナス発生回数：{mouth_bonus_count}回"
        + (f"（直近 t={mouth_bonus_last_t:.1f}秒）"
           if mouth_bonus_last_t >= 0 else "（まだ発生していません）"))

    ta_text = (
        f"順応の状態：末梢g平均 "
        f"{last.get('g_peripheral_mean', 1.0):.3f}"
        f"（1.0=順応なし、floor={float(params['sa_floor']):.2f}=完全順応）")
    if bool(params.get("include_cortical", False)):
        ta_text += f"\n脳g平均 {last.get('g_cortical_mean', 1.0):.3f}"

    return mouth_text, ta_text
