"""測定器（probe類）の既定値を1箇所にまとめたもの。副作用なし・定数のみ。

【なぜ、2026-08-06】自己モデル測定器（evaluate/agency_probe/inverse_probe/
inverse_exec_probe/closed_loop_probe/infer_goal_action）のしきい値・反復回数が
`C/scripts/run_c_metrics_ac_lr.py`と`E/scripts/e_probes.py`・`e_growth_train.py`に
同じ値で別々にハードコードされていた（監査報告
作業記録（非公開））。
今日、C側で新しい設定（C_EFFCOPY）を学習ループにだけ足して評価関数側を直し忘れる
バグが起きたのと同じ構図（`doc\検証の落とし穴チェックリスト.md`項30・項98）。
数値を1箇所にまとめ、以後の変更は必ずここだけを直せば全箇所に伝わるようにする。

注意：値は一切変えていない（各所の既存デフォルト値をそのまま転記しただけ）。
`INVERSE_PROBE_*`と`INVERSE_EXEC_PROBE_*`は現在たまたま同じ値だが、本来は
Stage1診断とStage1.5実行テストという別の概念で独立に調整されうるため、
1つの変数へ統合せず別名のまま持つ。
"""

AGENCY_PROBE_N = 30

INVERSE_PROBE_N = 60
INVERSE_PROBE_STEPS = 40
INVERSE_PROBE_RESTARTS = 3
INVERSE_PROBE_LR = 0.1

INVERSE_EXEC_PROBE_N = 40
INVERSE_EXEC_PROBE_STEPS = 40
INVERSE_EXEC_PROBE_RESTARTS = 3
INVERSE_EXEC_PROBE_LR = 0.1

CLOSED_LOOP_PROBE_N = 30
CLOSED_LOOP_PROBE_MAX_REACH = 10

GOAL_INFER_STEPS = 15
GOAL_INFER_LR = 0.1
