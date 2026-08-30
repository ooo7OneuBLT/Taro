# -*- coding: utf-8 -*-
"""太郎の脳のファイルを、人間の脳の部位ごとのフォルダへ整理する（2026-08-30）。

【なぜ】部位（脳のどこか）と機能（何をするか）が平置きで混ざっていた。
`taro_brain.py` が「大脳皮質」を名乗っているが、実際にやっているのは発話の実行と
運動の生成だけで、語彙の選択も発話の計画も別ファイルにあった。
ユーザーの指摘：「大脳皮質というフォルダの下にGRUのファイルがあるのが正しいのでは」。

【安全策】元の場所に**転送する1行だけのファイル**を残す。
既存の import（105箇所・32ファイル）は1つも壊れない。
古い参照は、あとから1つずつ新しい場所へ書き換えればよい。

【モデルへの影響なし】既存の .pt は state_dict（重みの数値）と素の辞書だけを
持っており、クラス名やモジュールの場所を記録していない（2026-08-30 に確認済み）。

    .venv/Scripts/python.exe taro_core/tools/reorganize_brain.py --dry-run   # 計画だけ表示
    .venv/Scripts/python.exe taro_core/tools/reorganize_brain.py             # 実行
"""
import os
import sys
import io
import shutil
import argparse

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\claude\AI\Taro"
BRAIN = os.path.join(ROOT, "taro_core", "src", "brain")

# ファイル名 → 移す先（BRAIN からの相対パス）
PLAN = {
    # ---- 大脳皮質 ----
    "taro_brain.py":              "cerebral_cortex/recurrent_core.py",
    "taro_brain_motor.py":        "cerebral_cortex/recurrent_core_motor.py",
    "lexicon.py":                 "cerebral_cortex/temporal_lobe/lexicon.py",
    "motor_cortex.py":            "cerebral_cortex/frontal_lobe/motor_cortex.py",
    "corticospinal.py":           "cerebral_cortex/frontal_lobe/corticospinal.py",
    "sensorimotor_brain.py":      "cerebral_cortex/sensorimotor_brain.py",
    # ---- 大脳辺縁系 ----
    "hippocampus.py":             "limbic/hippocampus.py",
    # ---- 大脳基底核 ----
    "basal_ganglia.py":           "basal_ganglia/basal_ganglia.py",
    # ---- 中脳 ----
    "superior_colliculus.py":     "midbrain/superior_colliculus.py",
    # ---- 小脳 ----
    "cerebellum.py":              "cerebellum/speech.py",
    "cerebellum_motor.py":        "cerebellum/motor.py",
    # ---- 脳幹 ----
    "locus_coeruleus.py":         "brainstem/locus_coeruleus.py",
    # ---- 神経修飾物質（部位をまたぐ） ----
    "dopamine.py":                "neuromodulator/dopamine.py",
    # ---- 部位をまたぐ計算の原理 ----
    "predictive_coding_latent.py": "principles/predictive_coding_latent.py",
    "precision_perception.py":    "principles/precision_perception.py",
    "homeostatic_scaling.py":     "principles/homeostatic_scaling.py",
    # ---- 動機（本能） ----
    "learning_progress.py":       "drives/learning_progress.py",
    "imitation_reward.py":        "drives/imitation_reward.py",
    "double_touch.py":            "drives/double_touch.py",
    # ---- 発達の進行 ----
    "developmental_clock.py":     "development/developmental_clock.py",
    "developmental_schedule.py":  "development/developmental_schedule.py",
    # ---- 器官（脳ではない。声道は body へ動かすべきだが今回は据え置き） ----
    # "vocal_tract.py":           は body/ に移すべきだが、影響が広いので別途
}

SHIM = '''# -*- coding: utf-8 -*-
"""【転送のみ】このファイルの中身は {new} へ移した（2026-08-30の整理）。

既存の `from {old_mod} import ...` を壊さないために、ここに転送だけを残している。
新しく書くコードは移動先を直接 import すること。
整理の全体像は `doc/脳の地図.md`。
"""
from {new_mod} import *          # noqa: F401,F403
from {new_pkg} import {new_name} as _m   # noqa: E402

globals().update({{k: v for k, v in vars(_m).items() if not k.startswith("__")}})
'''


def module_path(rel):
    """BRAIN からの相対パス → import 用のモジュール名（brain 直下を起点とする）"""
    return rel.replace("/", ".").replace(".py", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dirs = sorted({os.path.dirname(v) for v in PLAN.values() if os.path.dirname(v)})
    print("作るフォルダ（%d個）" % len(dirs))
    for d in dirs:
        print("   ", d)
    print()
    print("移すファイル（%d個）" % len(PLAN))
    for old, new in sorted(PLAN.items()):
        print("   %-30s → %s" % (old, new))
    if a.dry_run:
        print("\n--dry-run なので何もしていない")
        return

    # フォルダと __init__.py を作る
    for d in dirs:
        p = os.path.join(BRAIN, d)
        os.makedirs(p, exist_ok=True)
        # 途中の階層すべてに __init__.py を置く
        cur = BRAIN
        for part in d.split("/"):
            cur = os.path.join(cur, part)
            ini = os.path.join(cur, "__init__.py")
            if not os.path.exists(ini):
                io.open(ini, "w", encoding="utf-8").write(
                    '# -*- coding: utf-8 -*-\n"""%s（2026-08-30の整理で作成）"""\n' % part)

    moved = 0
    for old, new in PLAN.items():
        src = os.path.join(BRAIN, old)
        dst = os.path.join(BRAIN, new)
        if not os.path.exists(src):
            print("  飛ばす（元が無い）:", old)
            continue
        if os.path.exists(dst):
            print("  飛ばす（先に既にある）:", new)
            continue
        shutil.move(src, dst)
        mp = module_path(new)
        io.open(src, "w", encoding="utf-8").write(
            SHIM.format(new=new, old_mod=old.replace(".py", ""), new_mod=mp,
                        new_pkg=mp.rsplit(".", 1)[0], new_name=mp.rsplit(".", 1)[1]))
        moved += 1
        print("  移した: %-30s → %s" % (old, new))
    print("\n%d ファイルを移し、元の場所に転送を置いた" % moved)


if __name__ == "__main__":
    main()
