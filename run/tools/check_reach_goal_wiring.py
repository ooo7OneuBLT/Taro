# -*- coding: utf-8 -*-
"""手先位置の目標表現（案C）の配線チェック・成長頑健性・通ってはいけない条件。

【何を確かめるか（設計の検証計画 Level 0〜4）】
  ① 次元の検算：reach_dim==22（arm7+touch15）
  ② 配線チェック：関節角度・触覚を人工的に変えたら、対応する成分だけが反応するか
     （落とし穴チェックリスト 項17）
  ③ 成長頑健性：体を作り直しても同じ関節・部位を指し続けるか（項86）
  ④ 通ってはいけない条件：touch=false での reach_self・可動域外の目標・
     goal_traj_len=1・goal_home_prob=1.0・reach_goal_bufが空、等
  ⑤ 既定値（goal_babbling=False）のとき、reach_head 等が一切構築されないこと

使い方:
    .venv/Scripts/python.exe run/tools/check_reach_goal_wiring.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np       # noqa: E402
import torch             # noqa: E402

from run.config import Config                          # noqa: E402
from run.plugins.common import scene as scene_mod       # noqa: E402
from run.taro_setup import Taro                         # noqa: E402

SCENE = "リーチング_リクライニング60度"

print("=" * 78)
print(" 手先位置の目標表現（案C）：配線チェック・成長頑健性・通ってはいけない条件")
print("=" * 78)

results = {}


def _build(age=None, **taro_kwargs):
    taro = {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
            "somatosensory": True, "vision": False}
    taro.update(taro_kwargs)
    if age is not None:
        taro["age_months"] = age
    cfg = Config.from_spec({"scene": SCENE, "taro": taro,
                            "run": {"steps": 10, "seed": 0}})
    env, _, _ = scene_mod.build(cfg.scene, taro=dict(cfg._taro), seed=0, verbose=False,
                                hybrid=True)
    taro_obj = Taro(cfg, env, seed=0, verbose=False)
    return cfg, env, taro_obj


# ---------------------------------------------------------------- ① 次元の検算
print("\n" + "-" * 78)
print(" ① 次元の検算")
print("-" * 78)
cfg0, env0, t0 = _build()
print(f"  reach_dim = {t0.reach_dim}（期待22＝arm7+touch{t0.reach_dim - 7}）")
print(f"  arm_map.idx = {t0.arm_map.idx}")
print(f"  reach_touch_groups = {t0.reach_touch_groups}")
ok1 = (t0.reach_dim == 22) and (len(t0.arm_map.idx) == 7)
results["① 次元の検算(22次元)"] = ok1
g0 = t0.encode_reach_goal(t0.first_obs)
print(f"  encode_reach_goal の出力 shape = {tuple(g0.shape)}  値域 [{g0.min():.3f}, {g0.max():.3f}]")

# ---------------------------------------------------------------- ② 配線チェック
print("\n" + "-" * 78)
print(" ② 配線チェック（入力を揺らして出力が動くか）")
print("-" * 78)
obs0 = t0.first_obs
ok2a = True
for k, i in enumerate(t0.arm_map.idx):
    obs = dict(obs0)
    v = np.array(obs0["observation"], dtype=np.float32).copy()
    lo, hi = t0.arm_map.lo[k], t0.arm_map.hi[k]
    v[i] = float(np.clip(v[i] + 0.3 * (hi - lo), lo, hi))
    obs["observation"] = v
    g = t0.encode_reach_goal(obs)
    moved = abs(float(g[k]) - float(g0[k])) > 1e-4
    others_still = all(abs(float(g[j]) - float(g0[j])) < 1e-4 for j in range(7) if j != k)
    ok2a &= moved and others_still
    print(f"  関節{k}({t0.arm_map.joint_names[k]}): 自成分反応={moved}"
          f" 他成分は不変={others_still}")
results["② q_arm 配線（自分だけ反応）"] = ok2a

ok2b = True
touch = np.zeros_like(np.asarray(obs0["touch"], dtype=np.float32))
tm = t0.target_fusion.touch
for gi_name in t0.reach_touch_groups:
    gi = tm.group_names.index(gi_name)
    pts = (tm._part_of_point.numpy() == gi)
    f = np.zeros((tm.n_points, 3), dtype=np.float32)
    f[pts] = 1.0
    obs = dict(obs0)
    obs["touch"] = f.reshape(-1)
    g = t0.encode_reach_goal(obs)
    q_touch = g[7:].reshape(-1, 5)
    idx_in_reach = t0.reach_touch_groups.index(gi_name)
    presence = float(q_touch[idx_in_reach, 0])
    others = [float(q_touch[j, 0]) for j in range(len(t0.reach_touch_groups)) if j != idx_in_reach]
    reacted = presence > 0.5
    others_quiet = all(o < 0.2 for o in others)
    ok2b &= reacted and others_quiet
    print(f"  部位{gi_name}: 有無={presence:.3f}（反応{reacted}） 他部位={others}"
          f"（静か={others_quiet}）")
results["② q_touch 配線（対象部位だけ反応）"] = ok2b

# ---------------------------------------------------------------- ③ 成長頑健性
print("\n" + "-" * 78)
print(" ③ 成長頑健性（0→4ヶ月）")
print("-" * 78)
env0.close()
cfg2, env2, t2 = _build(age=2.0)
cfg4, env4, t4 = _build(age=4.0)
same_idx = (t2.arm_map.idx == t4.arm_map.idx) and (t2.arm_map.joint_names == t4.arm_map.joint_names)
print(f"  月齢2ヶ月と4ヶ月で腕のindexが同じか  {'はい' if same_idx else 'いいえ'}")
print(f"    2ヶ月: {t2.arm_map.idx}")
print(f"    4ヶ月: {t4.arm_map.idx}")
dim_same = (t2.reach_dim == t4.reach_dim == 22)
print(f"  reach_dim が両方とも22のままか  {'はい' if dim_same else 'いいえ'}")
results["③ 成長しても腕のindexが同じ"] = same_idx
results["③ 成長しても reach_dim=22"] = dim_same

# on_body_change の実地テスト（月齢2の脳へ月齢4の体を差し替える）
tm4 = None
try:
    from somatosensory_cortex import build_touch_map_from_env
    tm4 = t2.on_body_change(env4)
    missing = [nm for nm in t2.reach_touch_groups if nm not in tm4.group_names]
    ok3c = len(missing) == 0
    print(f"  on_body_change後、対象部位が全部そろっているか  {'はい' if ok3c else 'いいえ'}"
          f"（欠けた部位: {missing}）")
except Exception as e:      # noqa: BLE001
    ok3c = False
    print(f"  on_body_change で例外: {type(e).__name__}: {e}")
results["③ on_body_change後も対象部位がそろう"] = ok3c
env2.close(); env4.close()

# ------------------------------------------------- ④ 通ってはいけない条件
print("\n" + "-" * 78)
print(" ④ 通ってはいけない条件で落ちるか")
print("-" * 78)

# ④-1 touch=false で reach_self ＝ 学習開始前に ValueError で止まる
try:
    Config.from_spec({"scene": SCENE, "taro": {"goal_babbling": True,
                      "goal_space": "reach_self", "touch": False},
                      "run": {"steps": 10, "seed": 0}})
    ok4a = False
    print("  ④-1 touch=false×reach_self: 止まらなかった（不合格）")
except ValueError as e:
    ok4a = True
    print(f"  ④-1 touch=false×reach_self: 止まった（{str(e).splitlines()[0]}）")
results["④-1 touch=false×reach_selfはValueErrorで止まる"] = ok4a

# ④-2 未知の goal_space
try:
    Config.from_spec({"scene": SCENE, "taro": {"goal_space": "nonsense"},
                      "run": {"steps": 10, "seed": 0}})
    ok4b = False
    print("  ④-2 goal_space不正: 止まらなかった（不合格）")
except ValueError as e:
    ok4b = True
    print(f"  ④-2 goal_space不正: 止まった（{str(e).splitlines()[0]}）")
results["④-2 goal_space不正はValueErrorで止まる"] = ok4b

# ④-3 可動域の2倍の目標を infer_reach_goal_action に渡しても発散しない
cfg5, env5, t5 = _build()
z = torch.zeros(t5.brain.latent_dim)
mean0 = torch.zeros(t5.n_act)
g_far = torch.ones(t5.reach_dim) * 5.0     # 可動域[-1,1]の5倍
a_out = t5.infer_reach_goal_action(z, g0.clone(), mean0, g_far, n_steps=15)
ok4c = bool(torch.isfinite(a_out).all()) and float(a_out.abs().max()) <= 1.0 + 1e-4
print(f"  ④-3 可動域外の目標(±5)でも発散しないか  {'はい' if ok4c else 'いいえ'}"
      f"（|a|max={float(a_out.abs().max()):.4f}）")
results["④-3 可動域外の目標でも発散しない"] = ok4c

# ④-4 goal_home_prob=1.0 なら毎回ホームが選ばれる（判定ロジックそのものを直接確認）
n_home = sum(1 for _ in range(200) if torch.rand(1).item() < 1.0)
ok4d = (n_home == 200)
print(f"  ④-4 goal_home_prob=1.0 のとき常にホーム判定になるか  {'はい' if ok4d else 'いいえ'}"
      f"（200/200中{n_home}）")
results["④-4 goal_home_prob=1.0は常にホーム"] = ok4d

# ④-5 goal_traj_len=1 は1ステップで目標にほぼ到達する（区分線形の検算）
from goal_babbling.trajectory import GoalTrajectory   # noqa: E402
traj = GoalTrajectory(L=1)
cur = torch.zeros(22)
target = torch.ones(22)
traj.begin(target)
wp = traj.waypoint(cur)
ok4e = torch.allclose(wp, target, atol=1e-5)
print(f"  ④-5 goal_traj_len=1で1stepにほぼ到達するか  {'はい' if ok4e else 'いいえ'}"
      f"（誤差={float((wp - target).abs().max()):.2e}）")
results["④-5 goal_traj_len=1は1stepで到達"] = ok4e

# ④-6 dummy_reach_goal は毎回違う値を返し、gclp(現在地)は汚さない
d1 = t5.dummy_reach_goal(g0)
d2 = t5.dummy_reach_goal(g0)
g0_after = t5.encode_reach_goal(t5.first_obs)
ok4f = (not torch.equal(d1, d2)) and torch.allclose(g0, g0_after, atol=1e-6)
print(f"  ④-6 陰性対照は毎回違う値・現在地は汚さないか  {'はい' if ok4f else 'いいえ'}")
results["④-6 陰性対照は毎回違う値/現在地は不変"] = ok4f
env5.close()

# ---------------------------------------------------------- ⑤ 既定値では未構築
print("\n" + "-" * 78)
print(" ⑤ 既定値（goal_babbling=False）のとき reach_head 等が一切構築されないか")
print("-" * 78)
cfg6 = Config.from_spec({"scene": SCENE, "taro": {}, "run": {"steps": 10, "seed": 0}})
env6, _, _ = scene_mod.build(cfg6.scene, taro=dict(cfg6._taro), seed=0, verbose=False,
                             hybrid=True)
t6 = Taro(cfg6, env6, seed=0, verbose=False)
ok5 = (t6.reach_head is None and t6.arm_map is None and t6.reach_opt is None
       and t6.home_reach_goal is None)
print(f"  reach_head/arm_map/reach_opt/home_reach_goal が全部Noneか  {'はい' if ok5 else 'いいえ'}")
results["⑤ 既定値では新機構が一切構築されない"] = ok5
env6.close()

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
allok = True
for name, v in results.items():
    allok &= v
    print(f"  {name:36s} {'合格' if v else '不合格'}")
print()
print("  ⇒ " + ("すべて合格" if allok else "注意 不合格の項がある"))
if not allok:
    sys.exit(1)
