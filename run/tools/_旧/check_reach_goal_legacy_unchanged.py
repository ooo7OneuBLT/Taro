# -*- coding: utf-8 -*-
"""goal_babbling=True・goal_space=prop_full（旧実装）のとき、学習が再現するか。

【なぜ「別プロセス」で比較するか、2026-08-02】最初は同一プロセス内で
Trainer.build()/run() を2回呼んで比較しようとしたが、**変更を一切加えていない
現行コードそのものでも、同一プロセス内で2回学習を回すと最終的な脳の重みが
一致しない**ことが分かった（差は 1e-8 〜 1e-3 とばらつき、再現性が無い）。
一方、このリポジトリに既にある `run/tools/check_divergence.py`（**別プロセス**を
2つ立てて比較する）では、視覚のレンダリング(`obs_out.eye_left`)を除く全ての量
（重み W・Wf を含む）が一致する。

⇒ **同一プロセス内で複数回 env/Taro を作って比較する、という検証方法自体が
この環境（MuJoCoの描画コンテキスト等、プロセス内でグローバルに持たれる状態）
とは相性が悪く、偽陽性（本当は同じなのに違って見える）を生む**。これは
このタスクで新たに発見した、この検証方法に関する制約（人間模倣からの逸脱では
ないが、検証手法そのものの限界として記録する価値がある）。

⇒ 本スクリプトは `check_divergence.py` と同じ「別プロセス」方式を踏襲し、
goal_babbling=True（旧実装 prop_full）の設定で2つの独立プロセスを起動し、
学習後の脳の重みを比較する。

使い方:
    .venv/Scripts/python.exe run/tools/check_reach_goal_legacy_unchanged.py
"""
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)

_WORKER = r'''
import os, sys
sys.path.insert(0, r"{root}")
import torch
from run.config import Config
from run.trainer import train

cfg = Config.from_spec({{
    "scene": "リーチング_リクライニング60度",
    "taro": {{"goal_babbling": True, "vision": False}},
    "run": {{"steps": 40, "seed": 0, "checkpoint": 1000}},
}})


def _catch(row):
    pass


class _Cap:
    def __init__(self):
        self.taro = None


_orig_from_spec = None
import run.trainer as trainer_mod

class _CapturingTrainer(trainer_mod.Trainer):
    def run(self):
        out = super().run()
        torch.save(self.taro.brain.state_dict(), r"{out}")
        return out


tr = _CapturingTrainer(cfg, plugins=(), verbose=False, log_row=_catch)
tr.build()
tr.run()
trainer_mod.close_env(tr.env)
'''


def _run_worker(out_path):
    src = _WORKER.format(root=_R, out=out_path)
    r = subprocess.run([sys.executable, "-c", src], cwd=_R, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-4000:])
        raise RuntimeError("worker process failed")


print("=" * 78)
print(" 旧実装（goal_babbling=True, goal_space=prop_full）を")
print(" 2つの独立プロセスで学習させ、脳の重みを比較する")
print("=" * 78)

with tempfile.TemporaryDirectory() as td:
    p1 = os.path.join(td, "sd1.pt")
    p2 = os.path.join(td, "sd2.pt")
    print("\n  プロセス1（40ステップ学習）...")
    _run_worker(p1)
    print("  プロセス2（40ステップ学習）...")
    _run_worker(p2)

    import torch
    sd1 = torch.load(p1, map_location="cpu", weights_only=True)
    sd2 = torch.load(p2, map_location="cpu", weights_only=True)

same_keys = set(sd1) == set(sd2)
print(f"\n  ① 脳の層の名前が一致するか  {'はい' if same_keys else 'いいえ'}")
all_equal, n = True, 0
for k in sd1:
    if k not in sd2:
        continue
    n += 1
    eq = torch.equal(sd1[k], sd2[k])
    all_equal &= eq
    if not eq:
        print(f"     不一致: {k}  最大差={float((sd1[k]-sd2[k]).abs().max()):.6g}")
print(f"  ② 独立プロセス2本の学習後の脳の重み {n} 層すべてが bit-exact で一致するか"
      f"  {'はい' if all_equal else 'いいえ'}")

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
ok = same_keys and all_equal and n > 0
print(f"  旧実装(prop_full)が同一シードで再現するか  {'合格' if ok else '不合格'}")
if not ok:
    sys.exit(1)
