"""身体の寸法・質量の測定器 — 体型補正の効果を人間の新生児と比べる。

【なぜ作ったか、2026-07-25】体型補正の効果を測ろうとしたら**同じ設定で 27.7cm と 18.9cm**
という矛盾した値が出た。原因は「体長を x方向の広がりで測っていた」こと＝**手足の位置
（姿勢）に依存する**ので再現性がない。[[feedback-bug-to-checklist]]「おかしいときはまず
計測器を疑う」に従い、測定器を作り直した。

【姿勢に依存しない測り方】
`qpos = model.qpos0`（モデルの基準姿勢）に固定してから `mj_forward` で位置を確定させる。
リセット直後の揺らぎ（jitter）や物理の落ち着き（settle）を経ないので、**同じ設定なら必ず
同じ値**になる。身長は「体軸方向の広がり＋両端のgeom半径」で測る。

【人間の新生児の基準値】
- 身長 49.9cm（WHO成長基準の中央値付近）
- 体重 3.5kg
- 頭の質量が体重に占める割合 約25%
- 上肢/下肢の長さ比 1.07（**腕の方が脚より長い**。太郎の素の体は0.77で逆転している）
  出典：Segmental Limb Length Measurements in Term Neonates From Southern India,
  Indian Pediatrics 2024 (n=950, 満期産) 上肢20.96cm・下肢19.60cm

【使い方】
    python E/scripts/e_body_measure.py            # 補正あり/なしを比較
    E_SWEEP=leg_thick python ...                   # 指定した係数を振って影響を見る
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "taro_core"))
import paths; paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)
sys.path.insert(0, os.path.join(paths.SRC, "body"))

import numpy as np
import mujoco
from d_supine_env import SupineMimoEnv
from mimoActuation.muscle import MuscleModel
from infant_body import NEWBORN_SHAPE_DEFAULTS, HEAD_ELONGATION, body_scale_custom

# 人間の新生児の基準値
HUMAN = {"height_cm": 49.9, "mass_kg": 3.5, "head_frac": 0.25, "arm_leg_ratio": 1.07}

# 環境物（太郎の体でないもの）を除くためのキーワード
_NOT_BODY = ("floor", "wall", "fence", "object", "toy", "target", "world")


def _taro_bodies(model):
    return [i for i in range(1, model.nbody)
            if not any(k in model.body(i).name for k in _NOT_BODY)]


def _segment_length(model, data, a, b):
    """2つのbodyの中心間距離（m）。基準姿勢で測るので姿勢に依存しない。"""
    try:
        return float(np.linalg.norm(data.body(a).xpos - data.body(b).xpos))
    except Exception:
        return float("nan")


def measure(scales=None, verbose=False, head_elong=None):
    """指定した体型係数で身体を作り、寸法と質量を測る。

    姿勢に依存しないよう `qpos0`（モデルの基準姿勢）に固定してから測る。

    Args:
        head_elong: 頭の楕円化率。**None なら実運用と同じ既定値**（HEAD_ELONGATION=1.16）。
            1.0 を渡すと球のまま＝アブレーション。
            注意：【2026-07-25 に発見したバグ】この引数が無かったため、測定器は
            `SupineMimoEnv` の既定（head_elongation=1.0＝球）で測っていた。
            一方 **学習で走る太郎は楕円**（`e_body_config.head_elongation_from_env`
            が 1.16 を渡す）＝**測定器と実物が違う体だった**。そのため日誌に載せた
            「脚0.70+太さ1.3+頭楕円＝身長49.8cm」を後日まったく再現できなかった。
            [[feedback-bug-to-checklist]]「動かないときはまず計測器を疑う」の実例。
    """
    kw = {}
    if scales is not None:
        custom = body_scale_custom(0.0, scales, verbose=verbose)
        if custom:
            kw["custom_measurements"] = custom
    kw["head_elongation"] = HEAD_ELONGATION if head_elong is None else float(head_elong)
    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=0.0, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data

    # 基準姿勢に固定（リセットの揺らぎ・物理の落ち着きを経ない＝再現性がある）
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    ids = _taro_bodies(m)
    mass_total = float(sum(m.body_mass[i] for i in ids))
    head_mass = float(m.body("head").mass[0])

    # 身長＝体軸方向の広がり。仰向けなので体軸はx。両端のgeom半径ぶんを足す。
    pos = np.array([d.body(i).xpos for i in ids])
    spans = pos.max(axis=0) - pos.min(axis=0)
    axis = int(np.argmax(spans))          # 最も広がっている軸＝体軸
    # 端のbodyに付くgeomの最大半径を足す（頭頂・足先のはみ出し分）
    edge_r = 0.0
    for i in ids:
        if abs(d.body(i).xpos[axis] - pos[:, axis].max()) < 1e-6 or \
           abs(d.body(i).xpos[axis] - pos[:, axis].min()) < 1e-6:
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == i]
            if gs:
                edge_r += float(max(m.geom_size[g][0] for g in gs))
    height_cm = float(spans[axis] + edge_r) * 100

    # 上肢＝肩→手、下肢＝股関節→足。基準姿勢での直線距離。
    arm = _segment_length(m, d, "right_upper_arm", "right_hand")
    leg = _segment_length(m, d, "right_upper_leg", "right_foot")

    # 頭の長さ（体軸方向）＝「何頭身か」を出すため。人間の新生児は約4頭身（頭/身長0.25）で、
    # 成人の約7.5頭身と大きく違う。「赤ちゃんらしい見た目」の主因はここなので、
    # 見た目の議論をするときは必ずこの数字を見る（印象だけで決めると頭を盛りすぎる）。
    _hg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("head").id]
    head_len_cm = 2 * max(float(np.max(m.geom_size[g])) for g in _hg) * 100 if _hg else float("nan")

    res = {
        "height_cm": height_cm,
        "head_len_cm": head_len_cm,
        "head_ratio": head_len_cm / height_cm if height_cm else float("nan"),
        "mass_kg": mass_total,
        "head_frac": head_mass / mass_total if mass_total else float("nan"),
        "arm_cm": arm * 100, "leg_cm": leg * 100,
        "arm_leg_ratio": arm / leg if leg else float("nan"),
        "head_mass_kg": head_mass,
    }
    env.close()
    return res


def _fmt(r, label):
    return (f"{label:<14}{r['height_cm']:>8.1f}{r['mass_kg']:>9.3f}"
            f"{r['head_frac']*100:>9.1f}{r['arm_cm']:>8.1f}{r['leg_cm']:>8.1f}"
            f"{r['arm_leg_ratio']:>9.3f}")


def main():
    print("身体の測定（基準姿勢 qpos0 に固定＝姿勢に依存しない）\n")
    hdr = f"{'条件':<14}{'身長cm':>8}{'体重kg':>9}{'頭の%':>9}{'上肢cm':>8}{'下肢cm':>8}{'上肢/下肢':>9}"
    print(hdr); print("-" * len(hdr.encode('utf-8')) // 2 * "-" if False else "-" * 66)

    r_off = measure(None)
    print(_fmt(r_off, "補正OFF"))
    r_on = measure(NEWBORN_SHAPE_DEFAULTS)
    print(_fmt(r_on, "補正ON"))
    print(f"{'人間の新生児':<13}{HUMAN['height_cm']:>8.1f}{HUMAN['mass_kg']:>9.3f}"
          f"{HUMAN['head_frac']*100:>9.1f}{20.96:>8.1f}{19.60:>8.1f}"
          f"{HUMAN['arm_leg_ratio']:>9.3f}")

    # 再現性の確認（同じ設定で2回測って一致するか）＝測定器の健康診断
    r2 = measure(NEWBORN_SHAPE_DEFAULTS)
    same = all(abs(r_on[k] - r2[k]) < 1e-9 for k in r_on)
    print(f"\n[測定器の健康診断] 同じ設定を2回測って一致するか: {'✅ 一致' if same else '❌ 不一致（測定器が壊れている）'}")

    sweep = os.environ.get("E_SWEEP", "")
    if sweep:
        print(f"\n── {sweep} を振る ──")
        print(hdr); print("-" * 66)
        for v in [0.45, 0.60, 0.75, 0.90, 1.00]:
            s = dict(NEWBORN_SHAPE_DEFAULTS); s[sweep] = v
            print(_fmt(measure(s), f"{sweep}={v:.2f}"))


if __name__ == "__main__":
    main()
