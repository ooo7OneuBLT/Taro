"""四肢・体幹の「筋力 vs 自重」比を月齢間で比較し、発達の向きが逆転していないか調べる。

【なぜ測るか】首(head_tilt)では実測で
    age=0 : 持ち上げ能力比 4.21倍 ／ age=18 : 3.00倍
＝**新生児のほうが相対的に強い＝発達の向きが逆転**という内部矛盾が見つかり、これを根拠に
補正した（doc/人間模倣からの逸脱リスト.md 2026-07-20）。原因は MIMo が「乳児の筋力データが
無いのでgeom体積から筋力を近似」していること。**同じ矛盾が腕・脚にもあるのか**を確認する。

【重要な違い】首は「新生児は頭を持ち上げられない(head lag)」という臨床標準所見があるので
「比<1.0」を目標にできた。**腕・脚は動かせるのが正常**（writhing movementsは腕脚の運動そのもの）
なので、「比<1.0にすべき」という根拠は無い。ここで使える判断材料は
**「age=0 の比が age=18 より大きい＝発達の向きが逆」かどうか**だけ。

【比の定義】各アクチュエータについて
    比 = gear(最大トルク) ÷ (その関節から先の質量 × g × 水平方向の腕)
＝「その関節が、自分より先の体を重力に抗して動かす余力」。

使い方: python e_limb_strength.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
from d_supine_env import SupineMimoEnv  # noqa: E402

G = 9.81
# 代表的な関節だけ見る（左右対称なので左側＋首）。名前はMIMoの実際の関節名で拾う。
INTEREST = ("head_tilt", "left_shoulder", "left_elbow", "left_hip", "left_knee",
            "hip_bend", "chest")


def descendants(model, bid):
    ids = []
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == bid:
                ids.append(b)
                break
            p = model.body_parentid[p]
    return ids


def measure(age):
    env = SupineMimoEnv(vision_params=None, age=age)
    obs, _ = env.reset(seed=0)
    m, d = env.model, env.data
    out = {}
    for i in range(m.nu):
        name = m.actuator(i).name
        jid = int(m.actuator_trnid[i, 0])
        if jid < 0:
            continue
        bid = int(m.jnt_bodyid[jid])
        ids = descendants(m, bid)
        mass = sum(m.body(k).mass[0] for k in ids)
        if mass <= 0:
            continue
        com = sum(m.body(k).mass[0] * d.xipos[k] for k in ids) / mass
        anchor = d.xanchor[jid]
        arm = float(np.linalg.norm((com - anchor)[:2]))   # 水平＝重力モーメントの腕
        tau = mass * G * arm
        gear = abs(float(m.actuator_gear[i, 0]))
        out[name] = dict(gear=gear, mass=mass, arm=arm, tau=tau,
                         ratio=gear / max(tau, 1e-9))
    total = sum(m.body(k).mass[0] for k in range(m.nbody)
                if not m.body(k).name.startswith("test_object"))
    env.close()
    return out, total


def main():
    a0, tot0 = measure(0)
    a18, tot18 = measure(18)
    print(f"body mass: age0={tot0:.3f} kg   age18={tot18:.3f} kg   (x{tot0/tot18:.3f})\n")

    keys = [k for k in a0 if k in a18 and any(s in k for s in INTEREST)]
    keys.sort()
    print(f"{'actuator':26s} {'ratio@0':>9s} {'ratio@18':>9s} {'0/18':>7s}  verdict")
    inverted = []
    for k in keys:
        r0, r18 = a0[k]["ratio"], a18[k]["ratio"]
        rel = r0 / max(r18, 1e-9)
        flag = "INVERTED (0 stronger)" if rel > 1.05 else ""
        if rel > 1.05:
            inverted.append((k, r0, r18, rel))
        print(f"{k:26s} {r0:9.2f} {r18:9.2f} {rel:7.2f}  {flag}")

    print("\n=== summary ===")
    allk = [k for k in a0 if k in a18]
    rels = np.array([a0[k]["ratio"] / max(a18[k]["ratio"], 1e-9) for k in allk])
    print(f"  actuators compared      : {len(allk)}")
    print(f"  inverted (age0 stronger): {(rels > 1.05).sum()} / {len(allk)} "
          f"({(rels>1.05).mean()*100:.0f}%)")
    print(f"  ratio(0/18) median      : {np.median(rels):.2f}")
    print("\n  NOTE: >1 means the NEWBORN is relatively stronger = developmental")
    print("        direction is inverted (newborns should be the weakest).")
    print("  NOTE: unlike the neck, there is NO literature basis for making limbs")
    print("        'unable to lift' - infants DO move their limbs (writhing).")


if __name__ == "__main__":
    main()
