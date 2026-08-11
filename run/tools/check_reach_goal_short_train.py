# -*- coding: utf-8 -*-
"""手先位置の目標表現（案C）：短時間学習で配線・勾配を確認する（検証計画 Level 5）。

goal_space=reach_self で80ステップだけ学習を回し（reach_goal_bufが64件貯まって
goal_stepが実際に発生する回数を確保するため）、
  ① 例外なく終わるか
  ② reach_head の全パラメータに勾配が流れたか（None・全ゼロでないか）
  ③ 目標gを変えると infer_reach_goal_action の返す行動が変わるか

注意：これは「検証」であって「実験」ではない（80ステップ・数秒で終わる）。
  本番の実験（E/experiments/目標指向_段階1_*.json、6000ステップ）は回さない。

使い方:
    .venv/Scripts/python.exe run/tools/check_reach_goal_short_train.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import torch                       # noqa: E402
from run.config import Config       # noqa: E402
from run.trainer import Trainer, close_env   # noqa: E402

print("=" * 78)
print(" 手先位置の目標表現（案C）：短時間学習(80step)での配線・勾配確認")
print("=" * 78)

cfg = Config.from_spec({
    "scene": "リーチング_リクライニング60度",
    "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
             "somatosensory": True, "vision": False},
    "run": {"steps": 80, "seed": 0, "checkpoint": 1000},
})

tr = Trainer(cfg, plugins=(), verbose=False, log_row=lambda row: None)
ok_ran = True
try:
    tr.build()
    before = {n: p.detach().clone() for n, p in tr.taro.reach_head.named_parameters()}
    tr.run()
    print(f"  ① 例外なく80ステップ終わったか  はい"
          f"（reach_goal_buf={len(tr.reach_goal_buf)}件）")
except Exception as e:      # noqa: BLE001
    ok_ran = False
    print(f"  ① 例外なく80ステップ終わったか  いいえ（{type(e).__name__}: {e}）")
    raise

after = {n: p.detach().clone() for n, p in tr.taro.reach_head.named_parameters()}
ok_grad = True
for n in before:
    changed = not torch.equal(before[n], after[n])
    ok_grad &= changed
    print(f"  reach_head.{n}: 学習で値が変わったか  {'はい' if changed else 'いいえ'}")
print(f"  ② reach_head の全パラメータが学習で更新されたか  {'はい' if ok_grad else 'いいえ'}")

# ③ 目標を変えると行動が変わるか
t = tr.taro
z = torch.zeros(t.brain.latent_dim)
mean0 = torch.zeros(t.n_act)
gclp = t.encode_reach_goal(tr.state["obs"])
g_a = gclp.clone(); g_a[0] = 1.0
g_b = gclp.clone(); g_b[0] = -1.0
a1 = t.infer_reach_goal_action(z, gclp, mean0, g_a, n_steps=15)
a2 = t.infer_reach_goal_action(z, gclp, mean0, g_b, n_steps=15)
ok_diff = not torch.allclose(a1, a2, atol=1e-4)
print(f"  ③ 目標を変えると行動が変わるか  {'はい' if ok_diff else 'いいえ'}"
      f"（差の最大={float((a1 - a2).abs().max()):.4f}）")

close_env(tr.env)

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
allok = ok_ran and ok_grad and ok_diff
print(f"  短時間学習での配線・勾配確認  {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
