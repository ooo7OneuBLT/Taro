"""固有感覚（腕の関節角度）の中から、狙った関節だけを取り出す索引表を作る。

【何のためか、2026-08-02】手先位置の目標表現（案C）は、座標を作らず
「腕の構え（固有感覚）」と「自己接触（触覚）」だけで目標を作る
（設計：作業記録（非公開））。
そのために、`obs["observation"]`（固有感覚の生配列。関節モード621次元／筋肉モード801次元）
の中から、片腕7関節の値だけを取り出す索引が要る。

【索引を決め打ちの定数にしない理由】`run/taro_setup.py` の ARM_R/ARM_L 定数（シナジー用）は
「行動空間」のindexであり、ここで扱う「観測空間(obs["observation"])」のindexとは別物
（E/scripts/e_reach_feasibility.py が同種の取り違えを罠として記録している）。
決め打ちの数値を書くと、駆動モード・観測の構成（velocity/torque/limits/actuationの
有無）が変わったときに黙って別の関節を読む（落とし穴チェックリスト 項86・項50）。

⇒ 索引は**実際に proprioception の内部構造から機械的に求め**、さらに
**実際に関節を動かして、その索引が反応するかを確認してから**使う
（落とし穴チェックリスト 項17「配線チェック」・項50「身体の構造を確認する」）。

【可動域による正規化について（人間模倣としての位置づけ）】
q_arm は関節角度そのものを、MuJoCoの可動域（jnt_range、固定の身体定数）で
[-1,1]に線形正規化する。これは**学習を伴わない決定的な計算**であり、
案Cが避けたい「座標変換（関節角度→3次元空間位置）」には当たらないと判断した
（設計の判断、ユーザー確認済み・2026-08-02仕様2節 論点1）。
"""
import numpy as np


# 片腕7関節（肩3・肘1・手首3）。MIMo_modelv2.xml で実機確認済み
# （71-83/199-211行、肩・肘・手首の順。指の関節は無い）。
ARM_JOINT_SUFFIXES = (
    "shoulder_horizontal", "shoulder_ad_ab", "shoulder_rotation",
    "elbow", "hand1", "hand2", "hand3",
)


def _opposite_side(side):
    return "left" if side == "right" else "right"


class ArmProprioMap:
    """片腕7関節の、obs["observation"]内での位置と可動域。

    Attributes:
        idx: list[int]  obs["observation"] 内での位置（7個）
        lo, hi: np.ndarray(7,) float32  各関節の可動域（ラジアン、MuJoCoのjnt_range）
        joint_names: list[str]  対応する関節名（"robot:right_elbow" 等）
    """

    def __init__(self, idx, lo, hi, joint_names):
        self.idx = list(idx)
        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)
        self.joint_names = list(joint_names)


def build_arm_proprio_map_from_env(env, side="right"):
    """env から、片腕7関節の obs["observation"]内indexと可動域を作る。

    手順：
      ① proprioception の内部構造（sensor_outputs の並び）から、qpos ブロックが
         obs["observation"] のどこから始まるかを実際に計算する（決め打ちしない）。
      ② 各関節の可動域を MuJoCo の jnt_range から読む。
      ③ **実際に各関節を少しだけ動かし、①で決めたindexが本当に反応するか確認する**
         （静かに別の関節を読んでいないかの配線チェック。落とし穴 項17・項50・項86）。
         確認後、動かした分は必ず元に戻す（環境の乱数・状態は消費しない：
         mj_forward は qpos/qvel から派生量を作り直すだけの決定的な計算）。

    Raises:
        AssertionError: 関節が見つからない、または配線チェックに落ちたとき
            （静かに間違った索引を使わないため、必ず例外で止める）。
    """
    import mujoco

    u = env.unwrapped
    model, data = u.model, u.data
    proprio = getattr(u, "proprioception", None)
    if proprio is None:
        raise AssertionError(
            "固有感覚(proprioception)が無効な環境です。build_arm_proprio_map_from_env は"
            "proprioception が有効な環境にのみ使えます。")

    joint_names = [f"robot:{side}_{suf}" for suf in ARM_JOINT_SUFFIXES]

    # ① qpos ブロックの開始位置（obs["observation"] は sensor_outputs を
    #    sorted(key) 順に連結したもの。qpos は常に含まれる。proprio.py 参照）
    obs_before = proprio.get_proprioception_obs()
    offset = 0
    for key in sorted(proprio.sensor_outputs.keys()):
        if key == "qpos":
            break
        offset += int(proprio.sensor_outputs[key].shape[0])

    idx, lo, hi = [], [], []
    for jn in joint_names:
        if jn not in proprio.joint_names:
            raise AssertionError(
                f"関節 {jn} が proprioception.joint_names に見つからない。"
                f"MIMoの身体構成（関節の有無・命名）が想定と違う可能性がある。\n"
                f"  proprioception.joint_names の一部: {proprio.joint_names[:10]}...")
        pos_in_qpos_block = proprio.joint_names.index(jn)
        idx.append(offset + pos_in_qpos_block)
        jid = model.joint(jn).id
        r = model.jnt_range[jid]
        lo.append(float(r[0]))
        hi.append(float(r[1]))

    # ③ 実際に揺らして確認する（静かに別のindexを読まないことの保証）
    q_saved = data.qpos.copy()
    try:
        for jn, i, l, h in zip(joint_names, idx, lo, hi):
            jid = model.joint(jn).id
            addr = int(model.jnt_qposadr[jid])
            v0 = float(data.qpos[addr])
            span = h - l
            delta = span * 0.1 if span > 1e-6 else 0.05
            target = v0 + delta if (v0 + delta) <= h else v0 - delta
            data.qpos[addr] = target
            mujoco.mj_forward(model, data)
            obs_after = proprio.get_proprioception_obs()
            data.qpos[addr] = v0
            mujoco.mj_forward(model, data)
            if abs(float(obs_after[i]) - float(obs_before[i])) < 1e-6:
                raise AssertionError(
                    f"配線チェック失敗：関節 {jn} を動かしても "
                    f"obs['observation'][{i}] が反応しない（索引がズレている可能性）。"
                    "落とし穴チェックリスト 項17・項50 を参照。")
    finally:
        # 途中で例外が出ても、必ず元の姿勢に戻す
        data.qpos[:] = q_saved
        mujoco.mj_forward(model, data)

    return ArmProprioMap(idx=idx, lo=lo, hi=hi, joint_names=joint_names)
