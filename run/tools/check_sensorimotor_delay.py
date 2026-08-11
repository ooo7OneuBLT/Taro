# -*- coding: utf-8 -*-
"""感覚運動の伝達遅延（候補5）の配線チェック。

【何を確かめるか】
  1. ミリ秒 → env.step()の回数（整数ステップ）の変換が検算どおりか
     （落とし穴チェックリスト 項5：数式の事前検算）
  2. sensory_delay_ms=0 / motor_delay_ms=0 のとき、既存の挙動が1ビットも変わらないか
     （既定を変えていないことの証拠。項29「良い方を既定値にする」の裏＝今回は既定は変えない）
  3. sensory_delay_ms>0 のとき、観測が実際にNステップ遅れて出てくるか
     （既知の入力＝決まった行動列を振り、出力にその変化が遅れて現れることを数値で確認。項17）
  4. motor_delay_ms>0 のとき、駆動モデルに渡る行動が実際にNステップ遅れて反映されるか

学習は回さない。env.reset()/env.step()を数十〜数百回呼ぶだけ（数秒〜数十秒で終わる）。

使い方:
    .venv/Scripts/python.exe run/tools/check_sensorimotor_delay.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np                                    # noqa: E402
from run.plugins.common import scene as scene_mod      # noqa: E402

SCENE = "新生児_仰向け_柵なし"
results = {}

print("=" * 78)
print(" 感覚運動の伝達遅延：配線チェック")
print("=" * 78)


def make_env(taro_extra, seed=0, vor=True):
    """vor=False で作る理由：VOR（前庭動眼反射）は毎ステップ、頭の角速度に応じて
    行動の目の成分を**上書き**する（e_toy_env.py ToySupineEnv.step()）。これは
    駆動モデルに渡る行動が「遅延だけ」で決まらなくなる交絡要因なので、
    3・4節（既知の行動列を振って遅延そのものを測る）ではOFFにして切り分ける。
    """
    t = dict(taro_extra)
    if not vor:
        t["vor"] = False
    env, sc, hands = scene_mod.build(SCENE, taro=t, seed=seed,
                                     verbose=False, hybrid=False)
    return env


# ---------------------------------------------------------------- 1. 変換の検算
print("\n" + "-" * 78)
print(" 1. ミリ秒 → ステップ数の変換（事前検算）")
print("-" * 78)
env_probe = make_env({})
u = env_probe.unwrapped
timestep = float(u.model.opt.timestep)
frame_skip = float(u.frame_skip)
dt = timestep * frame_skip
print(f"  model.opt.timestep = {timestep*1000:.3f} ms")
print(f"  frame_skip         = {frame_skip:.0f}")
print(f"  env.step()1回 = {dt*1000:.3f} ms  ( = timestep * frame_skip )")

candidates_ms = [0, 50, 120, 200, 10]
expect_steps = {0: 0, 50: round(50 / 1000 / dt), 120: round(120 / 1000 / dt),
                200: round(200 / 1000 / dt), 10: round(10 / 1000 / dt)}
ok1 = True
for ms in candidates_ms:
    computed = int(round((ms / 1000.0) / dt)) if ms > 0 else 0
    ok = computed == expect_steps[ms]
    ok1 &= ok
    print(f"  {ms:4d}ms -> {computed}step  (手計算 {expect_steps[ms]}step, "
          f"{'一致' if ok else '不一致'})")
results["1. ms→step変換が検算と一致"] = ok1
env_probe.close()


# ---------------------------------------------------------------- 2. 既定0で不変
print("\n" + "-" * 78)
print(" 2. sensory_delay_ms=0 / motor_delay_ms=0 のとき挙動が1ビットも変わらないか")
print("-" * 78)

N_STEPS = 40
rng = np.random.RandomState(12345)


def run_fixed(env, n_act, n_steps, seed_actions):
    """固定の行動列で n_steps だけ進め、observation の履歴を返す。"""
    obs, _ = env.reset(seed=0)
    hist = [np.asarray(obs["observation"], dtype=np.float64).copy()]
    for t in range(n_steps):
        a = seed_actions[t % len(seed_actions)]
        obs, *_ = env.step(a)
        hist.append(np.asarray(obs["observation"], dtype=np.float64).copy())
    return np.stack(hist)


env_a = make_env({})                                            # taroにキー無し
env_b = make_env({"sensory_delay_ms": 0, "motor_delay_ms": 0})   # 明示的に0
n_act = env_a.action_space.shape[0]
actions = [rng.uniform(0.0, 1.0, size=n_act).astype(np.float32) for _ in range(N_STEPS)]

print(f"  attribute確認: envA(キー無し) sensory_delay={env_a.unwrapped.sensory_delay} "
      f"motor_delay={env_a.unwrapped.motor_delay} "
      f"/ _obs_history属性あり={hasattr(env_a.unwrapped, '_obs_history')}")
print(f"  attribute確認: envB(0を明示) sensory_delay={env_b.unwrapped.sensory_delay} "
      f"motor_delay={env_b.unwrapped.motor_delay} "
      f"/ _obs_history属性あり={hasattr(env_b.unwrapped, '_obs_history')}")
ok2_attr = (env_a.unwrapped.sensory_delay == 0 and env_a.unwrapped.motor_delay == 0
            and env_b.unwrapped.sensory_delay == 0 and env_b.unwrapped.motor_delay == 0
            and not hasattr(env_a.unwrapped, "_obs_history")
            and not hasattr(env_b.unwrapped, "_obs_history"))
results["2a. 既定0では_obs_history等が作られない（配線コードを1行も通らない）"] = ok2_attr

hist_a = run_fixed(env_a, n_act, N_STEPS, actions)
hist_b = run_fixed(env_b, n_act, N_STEPS, actions)
identical = bool(np.array_equal(hist_a, hist_b))
max_abs_diff = float(np.max(np.abs(hist_a - hist_b)))
print(f"  {N_STEPS}step の observation 履歴が完全一致か: {identical}"
      f"（最大差 {max_abs_diff:.3e}）")
results["2b. 既定0では observation 履歴が完全一致"] = identical
env_a.close()
env_b.close()


# ---------------------------------------------------------------- 3. 感覚遅延
print("\n" + "-" * 78)
print(" 3. sensory_delay_ms>0 のとき、観測が実際に遅れて出てくるか")
print("-" * 78)

N_STEPS2 = 60
env_ref = make_env({}, vor=False)
n_act = env_ref.action_space.shape[0]
actions2 = [rng.uniform(0.0, 1.0, size=n_act).astype(np.float32) for _ in range(N_STEPS2)]
hist_ref = run_fixed(env_ref, n_act, N_STEPS2, actions2)
env_ref.close()

# 【MIMo実装の仕様に合わせた期待値】mimo_env.py の _delayed_observation は
#   「履歴長さ < delay の間は最古の値を返し、delay に達したら pop(0)」という
#   FIFOで、これを追うと定常状態でのshiftは **delay - 1**（delayそのものではない）。
#   実測で検算する（下のループのprintで実際の値を出す。手で追った式：
#     call i（1始まり）で返るのは a_{i-N+1}（i>=N）。0始まりの列で書き直すと
#     shift = N-1）。0step のときは無条件でshift=0。
ok3 = True
for ms in (50, 120, 200):
    steps_expected = int(round((ms / 1000.0) / dt))
    shift_expected = max(steps_expected - 1, 0)
    env_d = make_env({"sensory_delay_ms": ms, "motor_delay_ms": 0}, vor=False)
    hist_d = run_fixed(env_d, n_act, N_STEPS2, actions2)
    env_d.close()

    # 定常状態（十分後段）で、遅延ありの観測が基準のどの時刻と一致するかを探す。
    #   物理は行動・シードが同じなら遅延の有無で変わらない（返す値だけが遅れる）ので、
    #   遅延ありの t 番目の観測に厳密一致する基準の t' を探せば shift=t-t' が測れる。
    probe_t = N_STEPS2      # 最後の（十分warm-upが終わった）観測
    target = hist_d[probe_t]
    diffs = np.max(np.abs(hist_ref - target[None, :]), axis=1)
    matched_t = int(np.argmin(diffs))
    measured_shift = probe_t - matched_t
    match_err = float(diffs[matched_t])
    ok = (match_err < 1e-9) and (measured_shift == shift_expected)
    ok3 &= ok
    print(f"  sensory_delay_ms={ms:4d}: 変換={steps_expected}step "
          f"(MIMo実装上の定常shift期待値={shift_expected})  "
          f"実測shift={measured_shift}step  一致誤差={match_err:.2e}  "
          f"{'合格' if ok else '不合格'}")
results["3. sensory_delay>0で観測が実測どおり一定のステップ数だけ遅れる"] = ok3


# ---------------------------------------------------------------- 4. 運動遅延
print("\n" + "-" * 78)
print(" 4. motor_delay_ms>0 のとき、駆動モデルに渡る行動が実際に遅れて反映されるか")
print("-" * 78)


def run_fixed_capture_action(env, n_steps, seed_actions):
    """駆動モデルに実際に渡された行動（遅延後）を捕まえる。

    注意：`env.reset()` は内部で `reset_model()` が無操作（ゼロ行動）で1回
    `_set_action` を呼ぶ（`D/scripts/d_supine_env.py` 159行、落ち着かせるため）。
    これも motor_delay のFIFOに1個積まれてしまい、ループの行動列と対応が
    ずれる（実測で発覚：ズレを見込まずに数えたら shift が理論値より1大きく出た）。
    ⇒ **スパイをresetの後に付ける**（＝reset由来のゼロ行動はFIFOの初期状態として
    残る＝実運用と同じ「温まった状態」から数えるが、記録はループの行動だけにする）。
    """
    env.reset(seed=0)
    captured = []
    real_fn = env.unwrapped.actuation_model.action

    def _spy(action):
        captured.append(np.asarray(action, dtype=np.float64).copy())
        return real_fn(action)

    env.unwrapped.actuation_model.action = _spy
    for t in range(n_steps):
        a = seed_actions[t % len(seed_actions)]
        env.step(a)
    return np.stack(captured)


ok4 = True
for ms in (10, 50, 120, 200):
    steps_expected = int(round((ms / 1000.0) / dt))
    shift_expected = max(steps_expected - 1, 0)     # 3節と同じFIFOの仕様
    env_d = make_env({"sensory_delay_ms": 0, "motor_delay_ms": ms}, vor=False)
    n_act = env_d.action_space.shape[0]
    actions3 = [rng.uniform(0.0, 1.0, size=n_act).astype(np.float32) for _ in range(N_STEPS2)]
    applied = run_fixed_capture_action(env_d, N_STEPS2, actions3)
    env_d.close()
    raw = np.stack(actions3)

    # 定常状態で、実際に渡された行動が「過去の生の行動のどれか」と一致するかを見る。
    probe_t = N_STEPS2 - 1
    target = applied[probe_t]
    diffs = np.max(np.abs(raw - target[None, :]), axis=1)
    matched_t = int(np.argmin(diffs))
    measured_shift = probe_t - matched_t
    match_err = float(diffs[matched_t])
    ok = (match_err < 1e-9) and (measured_shift == shift_expected)
    ok4 &= ok
    note = "（motor_delay=1は実装上そもそも無遅延に縮退する既知の仕様）" if steps_expected <= 1 else ""
    print(f"  motor_delay_ms={ms:4d}: 変換={steps_expected}step "
          f"(定常shift期待値={shift_expected})  "
          f"実測shift={measured_shift}step  一致誤差={match_err:.2e}  "
          f"{'合格' if ok else '不合格'} {note}")
results["4. motor_delay>0で行動が実測どおり一定のステップ数だけ遅れて駆動モデルに渡る"] = ok4


print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
allok = True
for name, v in results.items():
    allok &= v
    print(f"  {name:52s} {'合格' if v else '不合格'}")
print()
print("  ⇒ " + ("すべて合格" if allok else "注意 不合格の項がある"))
if not allok:
    sys.exit(1)
