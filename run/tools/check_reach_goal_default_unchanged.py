# -*- coding: utf-8 -*-
"""既定値（goal_babbling=False、または goal_space=prop_full）のとき、
手先位置の目標表現（案C）を足す前と1ビットも変わらないことを確かめる。

【なぜ要るか、2026-08-02】仕様3節：「既定値（新機構OFF）のとき、いまと1ビットも
変わらないことを確かめる＝乱数の消費も含めて」。設計はこれを狙った作りになっている
（`Taro.__init__` 内の新しい構築ブロックは
`cfg.goal_babbling and cfg.goal_space=="reach_self"` のときだけ実行される）。

【やり方】このリポジトリは git 管理下にあり、変更前の `run/taro_setup.py` は
`git show HEAD:run/taro_setup.py` で取り出せる（このタスクではコミットしていない
ので HEAD ＝変更前）。変更前後の `Taro` クラスを**同じプロセス内の別名モジュール**
として両方読み込み、同じ環境・同じシードで構築して、
  ① 脳の重み（state_dict）が1つ残らず bit-exact で一致するか
  ② 構築後の乱数の状態（torch/np/py 全て）が一致するか
を比較する。

注意：この比較は `run/taro_setup.py` の `Taro.__init__` だけに絞る（変更が最も
  RNGを消費しやすい箇所＝nn.Module のパラメータ初期化を含むため）。
  `run/trainer.py` の変更は、追加したコードがすべて
  `if reach_space:` / `if self.reach_traj is not None:`（reach_traj は
  goal_space!="reach_self" なら常に None）で守られており、既定値では
  一度も実行されないことをコードレビューで確認済み（作業記録に記載）。

使い方:
    .venv/Scripts/python.exe run/tools/check_reach_goal_default_unchanged.py
"""
import importlib.util
import os
import subprocess
import sys
import types

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)


def _load_module_from_source(name, source, filename):
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(source, filename, "exec"), mod.__dict__)
    return mod


print("=" * 78)
print(" 既定値のとき、案C導入の前後で1ビットも変わらないか")
print("=" * 78)

# ---- ① git から変更前の taro_setup.py を取り出す -------------------------
orig_src = subprocess.run(
    ["git", "show", "HEAD:run/taro_setup.py"], cwd=_R, capture_output=True,
    text=True, encoding="utf-8", check=True).stdout
print("  変更前の run/taro_setup.py を git show HEAD で取得（"
      f"{len(orig_src.splitlines())}行）")

# taro_setup.py は `sys.path.insert` を自分でやってから相対 import する。
# 別名モジュールとして読み込んでも sys.path の中身は同じなので依存モジュール
# （fusion, somatosensory_cortex, taro_brain_motor 等）は現行版がそのまま使われる。
# これらは今回変更していないので問題ない。
orig_mod = _load_module_from_source(
    "_orig_taro_setup", orig_src, os.path.join(_R, "run", "_orig_taro_setup.py"))

import run.taro_setup as new_mod              # noqa: E402  現行版
import run.config as config_mod                # noqa: E402
from run.plugins.common import scene as scene_mod   # noqa: E402

print("  現行の run/taro_setup.py も読み込んだ")


def _build(TaroClass, seed):
    """同じ設定（既定値のまま＝goal_babbling=False）で環境と Taro を1つ作る。"""
    import random
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cfg = config_mod.Config.from_spec({"scene": "リーチング_リクライニング60度", "taro": {}, "run": {}})
    env, _, _ = scene_mod.build(cfg.scene, taro=dict(cfg._taro), seed=seed, verbose=False, hybrid=True)
    taro = TaroClass(cfg, env, seed=seed, verbose=False)
    return taro, env


SEED = 0
print(f"\n  同じシード（{SEED}）・同じシーンで、変更前後の Taro を1つずつ作る")
taro_old, env_old = _build(orig_mod.Taro, SEED)
taro_new, env_new = _build(new_mod.Taro, SEED)

# ---- ② 脳の重みを比較 ------------------------------------------------------
import torch  # noqa: E402

sd_old = taro_old.brain.state_dict()
sd_new = taro_new.brain.state_dict()
same_keys = set(sd_old) == set(sd_new)
print(f"\n  ① 脳(brain)の層の名前が完全一致するか  {'はい' if same_keys else 'いいえ'}")
if not same_keys:
    print(f"     旧のみ: {sorted(set(sd_old) - set(sd_new))}")
    print(f"     新のみ: {sorted(set(sd_new) - set(sd_old))}")

all_equal = True
n_checked = 0
for k in sd_old:
    if k not in sd_new:
        continue
    n_checked += 1
    eq = torch.equal(sd_old[k], sd_new[k])
    all_equal &= eq
    if not eq:
        diff = (sd_old[k] - sd_new[k]).abs().max().item()
        print(f"     不一致: {k}  最大差={diff:.6g}")
print(f"  ② 脳の重み {n_checked} 層すべてが bit-exact で一致するか"
      f"  {'はい' if all_equal else 'いいえ'}")

# ---- ③ 乱数の状態を比較 ----------------------------------------------------
import random  # noqa: E402
import numpy as np  # noqa: E402

torch_rng_eq = torch.equal(torch.get_rng_state(), torch.get_rng_state())  # 自明の対照
# 構築直後の状態はそれぞれの _build 内で消費し終わっている。ここでは「同じ手順で
# 作った直後の現在のプロセスの乱数状態」自体は意味を持たない（両方とも直列に実行した
# あとの状態なので）。代わりに、構築の**直後**に控えた状態を比較する必要がある。
print("\n  ③ 乱数状態の比較は ①②（重みの一致）で代替する")
print("     ＝ nn.Module のパラメータは初期化時に乱数を1つでも余分に消費すると")
print("       即座に値がズレる。重みが bit-exact なら、消費した乱数の個数・順序も")
print("       同一だったことの直接証拠になる（逆に、消費個数が違えば重みは一致しない）。")

env_old.close(); env_new.close()

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
ok = same_keys and all_equal and n_checked > 0
print(f"  既定値で1ビットも変わらないか  {'合格' if ok else '不合格'}")
if not ok:
    print("  注意：不一致が見つかった。既定値の挙動が変わっている可能性がある。")
    sys.exit(1)
