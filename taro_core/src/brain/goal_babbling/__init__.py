"""Goal Babbling（目標指向の探索）専用の部品を集めたパッケージ。

【なぜ独立パッケージか、2026-08-02】新しい予測ヘッド（ReachGoalHead）・区分線形の
軌道生成（GoalTrajectory）は、太郎の脳の中枢 `taro_brain_motor.py` を一切変更せずに
済むよう、独立モジュールとして切り出す（設計の統合判断「決定1」）。
これにより、既定値（goal_babbling=False、または goal_space=="prop_full"）のとき
`taro_brain_motor.py` は1行も変わらず、乱数消費・パラメータ数とも現状と完全に不変になる。

設計：作業記録（非公開）
"""
