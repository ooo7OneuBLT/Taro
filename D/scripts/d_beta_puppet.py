"""
ベータ（見た目は太郎の複製・中身は単純な操り人形）を**単体で**動かせるか検証する。

【設計】ユーザー提案：見た目だけ太郎と同じにして、太郎本体(アルファ)は自律させるが、
もう1体(ベータ=養育者役)は操れるようにする。制御はMIMoの本体actuator(~90個)ではなく、
もっと単純なプログラム（3自由度のスライド関節）で。

【なぜ物理的に安全か】2026-07-15の教訓（`d1_carer_env.py`docstring）：MIMo2体を関節ごと
重ねて置いたら3.3m吹き飛んだ＝危険なのは「関節が独立に動くラグドールを2つ置くこと」。
そこで、ベータの**全関節をMjSpecでdelete()**する（`joint.delete()`で njnt=0 のまま
コンパイル可能なことを確認済み）。関節の無い体はMuJoCo上「親に剛結合」＝1個の硬い塊になる。
これを`d1_carer_env.py`の手と同じ3自由度スライド関節（位置サーボ）で動かす。
＝見た目はフルボディ(顔・服のテクスチャが残る)、物理的には「手」と同じくらい単純・安全。

まずアルファ（太郎本体）を置かず、**ベータ単体**で「操れるか・物理が安定するか」だけを見る。
二人を組み合わせるのは、これが確認できてから（1機構ずつ検証）。

使い方: python d_beta_puppet.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

_PUPPET_RANGE = 1.0   # スライド関節の可動域(m)


def build_puppet_only_model():
    """太郎の複製(全関節削除=1個の剛体)を、3自由度スライド関節で動かせる形でcompileする。"""
    # d_env.py(TwoMimoEnv)と同じパターン：複製元は別のMjSpecインスタンスから読む
    # （同じspecを自分自身にattachしようとするとID不整合でエラーになる＝実測して判明）。
    spec = mujoco.MjSpec.from_file(paths.SCENE)
    spec_beta = mujoco.MjSpec.from_file(paths.SCENE)

    # 複製元の体（mimo_location配下）をアタッチ。接頭辞betaで名前空間を分離。
    fr = spec.worldbody.add_frame()
    fr.pos = [0, 0, 1.0]   # 床より高い位置に置く（自由落下させず、操縦関節だけで浮かせる）
    puppet_root = fr.attach_body(spec_beta.body("mimo_location"), "beta_", "")
    # 重力補償。ベータ配下は56体（頭・腕・脚…）あり、ルート1体だけにgravcompを設定しても
    # 残り55体分の重さは補償されず全体が沈む（実測：ルートのみ設定→z 1.0→0.76に沈下、
    # 無設定と数値が完全一致＝効いていなかった）。**配下の全56体**に設定する必要がある。
    # d1_carer_env.pyの手と同じ発想（養育者の腕は大人自身が支えている＝重さを感じさせない）。
    beta_bodies = [b for b in spec.bodies if b.name.startswith("beta_")]
    for b in beta_bodies:
        b.gravcomp = 1.0
    print(f"重力補償を設定: ベータ配下{len(beta_bodies)}体すべて")

    # ★ここが肝：アタッチした複製配下の全関節を削除＝1個の剛体にする
    # ★ベータを「関節ゼロ=1個の剛体」にする。attach_body は名前空間分離のため、
    # 関節だけでなくアクチュエータ・腱・等式拘束にも同じ接頭辞"beta_"を付ける（実測確認済み）。
    # 関節を先に消すと、それを参照するこれらの要素が「存在しない関節への参照」で
    # compile失敗するので、**依存の逆順（アクチュエータ→腱→等式拘束→関節）**で消す。
    # ベータは元々の筋肉アクチュエータでは動かさない（操縦用のpuppet関節だけで動かす）ので、
    # ベータ側のアクチュエータ・腱は丸ごと不要＝全部消してよい。
    n_act_before = len(spec.actuators)
    for a in list(spec.actuators):
        if a.name.startswith("beta_"):
            a.delete()
    print(f"ベータのアクチュエータを削除: {n_act_before}本 → {len(spec.actuators)}本")

    n_ten_before = len(spec.tendons)
    for t in list(spec.tendons):
        if t.name.startswith("beta_"):
            t.delete()
    print(f"ベータの腱(muscle tendon)を削除: {n_ten_before}本 → {len(spec.tendons)}本")

    beta_joint_names = {j.name for j in spec.joints if j.name.startswith("beta_")}
    n_eq_before = len(spec.equalities)
    for eq in list(spec.equalities):
        if eq.name1 in beta_joint_names or eq.name2 in beta_joint_names:
            eq.delete()
    print(f"ベータ関節がらみの等式拘束を削除: {n_eq_before}本 → {len(spec.equalities)}本")

    n_before = len(spec.joints)
    for j in list(spec.joints):
        if j.name.startswith("beta_"):
            j.delete()
    n_after = len(spec.joints)
    print(f"ベータの関節を削除: {n_before}本 → {n_after}本"
          f"（削除した{len(beta_joint_names)}本ぶん、体は1個の剛体になったはず）")

    # 3自由度の操縦関節（d1_carer_env.pyの手と同じパターン）をpuppet_rootに追加。
    for ax, nm in (([1, 0, 0], "x"), ([0, 1, 0], "y"), ([0, 0, 1], "z")):
        j = puppet_root.add_joint()
        j.name = f"puppet_{nm}"
        j.type = mujoco.mjtJoint.mjJNT_SLIDE
        j.axis = ax
        j.range = [-_PUPPET_RANGE, _PUPPET_RANGE]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    for nm in ("x", "y", "z"):
        a = spec.add_actuator()
        a.name = f"puppet_{nm}"
        a.target = f"puppet_{nm}"
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        kp, kv = 200.0, 20.0
        gp = np.zeros(10); gp[0] = kp
        bp = np.zeros(10); bp[1] = -kp; bp[2] = -kv
        a.gainprm = gp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm = bp
        a.ctrlrange = [-_PUPPET_RANGE, _PUPPET_RANGE]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    model = spec.compile()
    return model


def main():
    model = build_puppet_only_model()
    data = mujoco.MjData(model)
    print(f"モデル全体: nbody={model.nbody} njnt={model.njnt} nu={model.nu}")

    # 静止（無操作）で安定するか＝爆発しないかの最低限チェック
    for _ in range(200):
        mujoco.mj_step(model, data)
    pos0 = data.body("beta_mimo_location").xpos.copy()
    print(f"200step静止後のベータ位置: {pos0}（爆発なら大きくズレるはず）")
    assert np.all(np.abs(pos0) < 5.0), "静止状態で位置が異常＝爆発の疑い"

    # 実際に操縦してみる：puppet_x/y/zに目標位置を与えて動くか
    puppet_act = [model.actuator(f"puppet_{a}").id for a in "xyz"]
    targets = [(0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (-0.2, -0.2, 0.1)]
    for tgt in targets:
        data.ctrl[puppet_act] = tgt
        for _ in range(100):
            mujoco.mj_step(model, data)
        pos = data.body("beta_mimo_location").xpos.copy()
        print(f"目標ctrl={tgt} -> 100step後の実位置={pos}")

    print("\nOK: ベータ単体（関節削除＋3自由度操縦）で物理は安定・操縦も可能")


if __name__ == "__main__":
    main()
