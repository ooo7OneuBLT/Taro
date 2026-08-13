"""Viewer(PySide6版)の状態表示欄（ステータスバー）の計算ロジック。

【なぜこのファイルを分けたか】仕様
作業記録（非公開）
2-3節。「いま何が有効になっているか」を表示するロジックを、GUI組み立て
（`e_viewer_qt.py`）から切り離す。理由は2つ：

    1. 表示は必ず実体（env・model・actuation_model・scene・TARO_DEFAULTS）から
       その場で読む、という約束を守りやすくするため（表示専用のキャッシュ変数を
       作らない。前提.mdが繰り返し警告している「表示と実体がズレる」事故と同型を
       避ける）。関数を呼ぶたびに実体から計算し直すので、キャッシュのしようが無い。
    2. 将来「駆動」「感覚と報酬」タブが実装されたとき、このファイルだけを
       差し替えれば済むようにするため。

【今回のスコープでできること・できないこと】
    この版（第2段階まで）は「駆動」タブ・「感覚と報酬」タブがまだ無い。
    そのため、実際に駆動ループへ配線されている値（自発運動・脳・反射+共通駆動の
    選択、触覚順応やmouth_touch_bonusの実測結果）は表示できない
    （そもそも計算されていないので実体が存在しない）。この場合は「未実装」と
    正直に表示する。ただし、実体として読める範囲の情報（実際のfmax・実際の
    反射+共通駆動の対象関節数・TARO_DEFAULTSの既定値）は本物の値を出す。
"""
from __future__ import annotations

import numpy as np


def compute_status(*, env, actuation_model, actuation_mode, scene, taro_defaults,
                    fmax_base, age, hands=None, touch_state=None, drive_state=None):
    """今の状態を表す文字列群を返す。

    引数はすべて「実体」（env・model・actuation_model・scene・TARO_DEFAULTS）
    そのものか、それらから作った参照値（fmax_base=起動直後に実測した基準fmax）。
    値をどこにも保存せず、呼ばれるたびにここで計算し直す。

    touch_state: 第3段階（感覚と報酬タブ）で追加した任意引数
        （`e_viewer_qt_touch.TouchMonitor`のavailable=Trueなインスタンス、
        またはNone）。Noneのときは従来どおり「既定値・未実装」表示に
        フォールバックする（仕様3節。既存の呼び出し元・シグネチャを
        壊さないため、キーワード専用の任意引数として追加した）。

    drive_state: 第5段階（駆動タブ）で追加した任意引数
        （`e_viewer_qt_drive.py`が`win._drive_state`として保持する辞書、
        またはNone）。Noneのときは従来どおり「ゼロ入力・未実装」表示に
        フォールバックする（同上の理由。呼び出し元は
        `getattr(win, "_drive_state", None)`のように渡すこと。freeze中など
        drive_tick_actionが一度も呼ばれていない場合はまだ存在しないため）。

    戻り値：{"drive": str, "limb_strength": str, "touch_adaptation": str,
             "mouth_bonus": str, "age_scene": str} の辞書。
    """
    lines = {}

    # ---- 1. 駆動モード＋対象関節数 ----------------------------------------
    if drive_state is not None:
        # 【なぜ、仕様3節】駆動タブが実際に配線されているときは実測値を出す。
        #   n_joints・muscle_ok は e_viewer_qt_drive._ensure_state が
        #   run.taro_setup._reflex_common_joint_indices を呼んで実測した
        #   値をそのまま使う（ここで再計算しない＝二重実装しない）。
        mode = drive_state.get("mode", "white")
        active = bool(drive_state.get("active", False))
        muscle_ok = bool(drive_state.get("muscle_ok", False))
        n_joints = drive_state.get("n_joints")
        drive_line = (f"駆動（実測）: {'ON' if active else 'OFF（ゼロ入力）'}　"
                       f"モード: {mode}")
        if muscle_ok and n_joints is not None:
            drive_line += f"\n（参考）反射+共通駆動を選ぶと対象になる関節数: {n_joints}（実測）"
        elif not muscle_ok:
            drive_line += "\n（参考）反射+共通駆動はactuation=muscleのときのみ対象（今は対象外）"
        lines["drive"] = drive_line
    else:
        # 【なぜ】この版の駆動ループは env.step(zero) のみ（駆動タブがまだ
        #   一度もtickを処理していない＝freeze中に起動直後の状態を見ている等）。
        #   事実をそのまま書く。
        drive_line = "駆動: なし（ゼロ入力。駆動タブが未使用、またはfreeze中）"
        if actuation_mode == "muscle":
            try:
                # 【なぜ、仕様2-3節1】手で数えたり決め打ちの定数を書かない。
                #   今日実際に起きたバグ（28関節のはずが6関節しか動いていなかった）の
                #   再発防止として、run.taro_setup の実際の計算を毎回呼ぶ。
                from run.taro_setup import _reflex_common_joint_indices
                idx = _reflex_common_joint_indices(env)
                n_joints = sum(len(v) for v in idx.values())
                drive_line += f"\n（参考）反射+共通駆動を選ぶと対象になる関節数: {n_joints}（実測）"
            except Exception as e:
                drive_line += f"\n（参考）反射+共通駆動の対象関節数: 計算できません（{e}）"
        else:
            drive_line += f"\n（参考）反射+共通駆動はactuation=muscleのときのみ対象（今は{actuation_mode}）"
        lines["drive"] = drive_line

    # ---- 2. 四肢の筋力倍率 --------------------------------------------------
    # 【なぜ、仕様2-3節2】fmaxを毎回実測して基準値と比較する。UIのスライダー
    #   自体は次段階（駆動タブ）の担当だが、実際に効いている倍率は今も読める。
    limb_line = "四肢の筋力倍率: 計測不可（このモデルにfmaxがありません）"
    if fmax_base is not None and hasattr(actuation_model, "fmax"):
        try:
            cur = np.asarray(actuation_model.fmax, dtype=float)
            if cur.shape == fmax_base.shape:
                base_mean = float(np.mean(fmax_base))
                cur_mean = float(np.mean(cur))
                if base_mean > 1e-12:
                    ratio = cur_mean / base_mean
                    limb_line = (f"四肢の筋力倍率: {ratio:.2f}倍"
                                 "（実測。調整スライダーは次段階＝駆動タブで追加予定）")
        except Exception:
            pass
    lines["limb_strength"] = limb_line

    # ---- 3・4. 触覚の順応／口元の報酬 ---------------------------------------
    # 【なぜ、2026-08-13・第3段階】touch_state（感覚と報酬タブの監視用
    #   インスタンス）が渡されたときは実測値を表示する。渡されなかった場合
    #   （呼び出し元がまだ第2段階以前、またはタブの構築に失敗した場合）は、
    #   従来どおり「このシーンでTaroを構築した場合に使われる既定値」
    #   （run.config.TARO_DEFAULTS）にフォールバックする。
    if touch_state is not None:
        ta = touch_state.touch_adapt
        last = touch_state.last or {}
        lines["touch_adaptation"] = (
            f"触覚の順応（実測）: "
            f"{'ON' if (ta.fa_enabled or ta.sa_enabled) else 'OFF'}"
            f"（速順応{'ON' if ta.fa_enabled else 'OFF'}・"
            f"遅順応{'ON' if ta.sa_enabled else 'OFF'}）　"
            f"末梢g平均 {last.get('g_peripheral_mean', 1.0):.3f}")
        lines["mouth_bonus"] = (
            f"口元の報酬（実測）: {touch_state.mouth_contributor.bonus:g}　"
            f"発生回数 {touch_state.mouth_bonus_count}回")
    else:
        ta_on = bool(taro_defaults["touch_adaptation"][0])
        fa_on = bool(taro_defaults["touch_adapt_fa"][0])
        sa_on = bool(taro_defaults["touch_adapt_sa"][0])
        lines["touch_adaptation"] = (
            f"触覚の順応（既定値・未実装。感覚と報酬タブで次段階以降に実測表示予定）: "
            f"{'ON' if ta_on else 'OFF'}"
            f"（速順応{'ON' if fa_on else 'OFF'}・遅順応{'ON' if sa_on else 'OFF'}）")

        mouth_bonus = float(taro_defaults["mouth_touch_bonus"][0])
        lines["mouth_bonus"] = (
            f"口元の報酬（既定値・未実装。感覚と報酬タブで次段階以降に実測表示予定）: "
            f"{mouth_bonus:g}")

    # ---- 5. 月齢・シーン名 --------------------------------------------------
    # 第1段階から既にある _AGE・scene["name"] をそのまま使う（再計算しない）。
    lines["age_scene"] = f"シーン「{scene.get('name', '?')}」  体年齢 {age:g}ヶ月"

    return lines


def format_status_text(lines):
    """compute_status() の辞書を、ステータス欄にそのまま出せる複数行文字列にする。"""
    return "\n".join([
        lines["age_scene"],
        lines["drive"],
        lines["limb_strength"],
        lines["touch_adaptation"],
        lines["mouth_bonus"],
    ])
