"""writhing→fidgety移行のスケジュール（2026-07-23設計）。

【文献根拠】Frontiers系総説（研究日誌続き15-16）：
  - 皮質活動がsubplateからcortical plateへ移る＝生後2-5ヶ月 [Tier1]
  - この移行は「活動依存的な皮質脊髄再編成の感受性の高い窓」と表現され、
    ①皮質脊髄路の成熟（精密制御の復帰）②脊髄CPG自体の多筋協調の構造化、が
    **同じ窓で並行して起きる** [Tier1、両者が同一ソースに記載]
  - 移行の開始は生後6-9週（≈1.4-2.1ヶ月）[Tier1]

【設計方針】太郎の体は既に月齢(C5_AGE)で現実と較正されている（MIMoの成長モジュール）。
発達時計(DevelopmentalClock)はまだ現実週数との対応表が無い未較正のカウンタなので、
今回は文献が直接"月齢"で語れる C5_AGE の方へスケジュールを乗せる。

【恣意的な部分（要ラベル）】
  - ONSET_MONTHS/END_MONTHS の具体値は文献の範囲（1.5〜5ヶ月）を採用 [Tier1範囲だが
    厳密な変換は無し]
  - W_MEAN_CEILING / SYN_W_CEILING（移行完了時にどこまで戻すか）は具体的な数値の
    文献根拠が無い [Tier3・ARBITRARY]。5ヶ月時点で「完全な成人の精密制御」になる
    わけではない（そわそわ運動自体まだ発達途上）ため、1.0ではなく控えめな値にする。
"""

ONSET_MONTHS = 1.5   # 生後6週相当 [Tier1範囲]
END_MONTHS = 5.0      # 皮質板への移行完了目安 [Tier1範囲]
W_MEAN_CEILING = 0.5  # [Tier3・ARBITRARY] 5ヶ月時点で"半分だけ"精密制御が戻る、という仮の値
SYN_W_CEILING = 0.85  # [Tier3・ARBITRARY] 現状の既定0.6より高い、"より構造化"の仮の値


def _ramp(age_months, onset=ONSET_MONTHS, end=END_MONTHS):
    """[0, 1] の進行度。onset未満は0、end以降は1、間は線形。"""
    if age_months is None:
        return 0.0
    if age_months <= onset:
        return 0.0
    if age_months >= end:
        return 1.0
    return (age_months - onset) / (end - onset)


def schedule_w_mean(age_months):
    """皮質脊髄路の成熟スケジュール。0(writhing)→W_MEAN_CEILING(fidgety途上)。"""
    return _ramp(age_months) * W_MEAN_CEILING


def schedule_syn_w(age_months, baseline=0.6):
    """CPGの多筋協調スケジュール。baseline(今の既定)→SYN_W_CEILING。"""
    p = _ramp(age_months)
    return (1.0 - p) * baseline + p * SYN_W_CEILING
