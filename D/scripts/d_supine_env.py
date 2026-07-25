"""
仰向け(supine)の太郎の環境。触覚のON/OFFを切り替えられる。

【なぜ作るか】2026-07-15の発見
録画を見たところ、Cの太郎（margin+51＝最高成績）は**立った直後に転倒し、20秒間ずっと床で
もがいていた**。一方D0（座位・自己接触）の太郎は**腕を胸に畳んだまま10秒間ほぼ静止**し、
触覚は persist 96%（＝「何も変わらない」予測と互角＝学ぶものが無い）だった。
接触ペアを直接読むと、**何もしない太郎ですら「自己接触」が100%**（指が前腕に載りっぱなし）。

→ 仮説：**Cが成功したのは、転んで偶然「手足が自由に振れる状態」になったから**。
   自己モデルの成否を決めているのは動機でも触覚でもなく、**行動で信号がどれだけ変わるか**。

仰向けなら3つ同時に解ける：
  ①転倒しない（既に床にいる）②腕が自由（畳まれていない）③新生児の自己接触は仰向けで起きる

【交絡を避けるための設計】
- シーンは **Cと同一の benchmarkv2_scene.xml**（`MIMoV2DummyEnv`の既定）を使う。
  `roll_over_scene.xml` を借りると `<weld body1="head" body2="upper_body"/>` が付いてきて、
  「姿勢」と「首の溶接」の2つが同時に変わり比較が壊れる（落とし穴チェック項1＝交絡）。
  仰向けにする**計算式だけ**を roll_over.py から借りる。
- したがって立位Cとの差は**初期姿勢ただ1つ**。
- `MIMoBenchV2-v0` は max_episode_steps=6000・終了条件は全てFalse＝**課題が人生を打ち切らない**
  （落とし穴チェック項9はここでは起きない。汚染されていたのはD0の MIMoSelfBody-v0 だけ）。

【出典】仰向けの姿勢の作り方は MIMo 同梱の mimoEnv/envs/roll_over.py（STARTING_POSITION="supine"）
"""
import copy
import os
import numpy as np
import mujoco

from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS_V2
from mimo_lean import LeanMimoEnv


def infant_touch_params(factor=2.0):
    """全身の触覚の解像度を factor 倍だけ粗くする（＝乳児の触覚acuity）。

    MIMoの既定はsomatotopy（指先0.002 vs 下腿0.038＝19倍の密度差＝皮質拡大に相当）を
    持っている。**その比を保ったまま**全体を粗くしたいので、全部位に同じ係数を掛ける。
    """
    p = copy.deepcopy(DEFAULT_TOUCH_PARAMS_V2)
    p["scales"] = {k: v * factor for k, v in p["scales"].items()}
    return p


class SupineMimoEnv(LeanMimoEnv):
    """仰向けで始まる太郎。シーン・センサ・終了条件は MIMoBenchV2-v0 と同一。

    LeanMimoEnv を継承＝**視覚OFFのときだけ**巨大テクスチャ(976MB)を単色に置き換える
    （物理は不変・実測でqpos合計まで一致）。視覚をONにすると自動で素のテクスチャに戻る
    ので、「絵が消えたまま視覚実験をしてしまう」事故は起きない。詳細は mimo_lean.py。

    Args:
        settle_steps: 開始前に無操作で落ち着かせるstep数（roll_over.pyに倣い100）。
        jitter: リセット時に全関節へ加える一様乱数の幅。毎回わずかに違う仰向けになる
            （同じ初期姿勢を予測するだけで当たる、という汚染を防ぐ＝落とし穴チェック項9の症状）。
    """

    def __init__(self, settle_steps=100, jitter=0.01, head_elongation=1.0,
                 body_corrections=True, limb_scale=1.0, limb_fix=True,
                 distal_mass=1.0, flexion=False, flexion_stiffness=None, **kwargs):
        self._settle_steps = settle_steps
        self._jitter = jitter
        # 【2026-07-25】頭の楕円化。MIMoの頭は球で、頭囲は正しいが真上から見た長さが
        # 人間より15%短い（10.8cm、人間12.0cm）。体軸方向にだけ伸ばして人間に合わせる。
        # ⚠️この処理は従来 e_toy_env.py（おもちゃ環境）にしかなく、**学習に使うこの環境では
        # 頭が球のまま**だった。体型補正と同じ「身体の設定が環境に散らばっている」問題。
        # 処理の実体は taro_core の infant_body.elongate_head（＝太郎の身体そのもの）。
        self._head_elongation = float(head_elongation)
        self._body_corrections = bool(body_corrections)
        # 四肢の筋力補正の感度分析用の係数（1.0＝補正そのまま）。この補正の目標値には
        # 根拠が無いことが分かっているので、振って結論の頑健性を確かめる。
        self._limb_scale = float(limb_scale)
        # limb_fix=False で四肢の筋力補正を完全に切る（＝素のmimoGrowth）。
        # 文献（実測の除脂肪量・二乗三乗則）が示唆するのはこちらの姿なので、
        # アブレーションとして必ず回せるようにしてある。
        self._limb_fix = bool(limb_fix)
        # 手足の質量の倍率（感度分析用。サイズは変えない）。
        # 新生児の体節質量比は実測が存在しないので、振って頑健性を確かめる。
        self._distal_mass = float(distal_mass)
        # 【生理的屈曲・2026-07-25】新生児は放っておいても股・膝・肘が曲がっている
        # （屈曲拘縮）。関節のバネ（stiffness + 中立位置）で表現する。実体は
        # taro_core の infant_body.apply_physiological_flexion（身体は太郎そのもの）。
        # 既定OFF＝Viewerでの目視を通してから既定ONにする。
        self._flexion = bool(flexion)
        self._flexion_stiffness = flexion_stiffness
        super().__init__(**kwargs)

        # --- 仰向けにする（roll_over.py の supine と同じ式）---
        self.model.body("hip").pos = [0, 0, 0.2]
        self.model.body("hip").quat = np.array([0, -0.7071068, 0, 0.7071068])
        self.model.body("hip").quat *= np.array([1, -1, 1, 1])   # supine（これが無いとprone＝うつ伏せ）

        # 【2026-07-25】筋力の補正（首・四肢）を太郎の身体定義（core）から適用する。
        # MIMoは gear を geom の体積から計算するため、頭が大きい新生児ほど首も強くなり
        # **発達の向きが逆転**する（age=0で持ち上げ能力比4.21倍 > 18ヶ月の3.00倍）。
        # ⚠️従来この補正は ToySupineEnv（おもちゃ環境）にしかなく、**環境ごとに違う体**
        # になっていた（実測で3種類以上）。身体は環境の性質ではないので core に集約した。
        if self._body_corrections and kwargs.get("age") is not None:
            import os as _os, sys as _sys
            _b = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               _os.pardir, _os.pardir, "taro_core", "src", "body")
            if _b not in _sys.path:
                _sys.path.insert(0, _b)
            from infant_body import apply_runtime_corrections
            apply_runtime_corrections(self.model, self.data, kwargs["age"],
                                      limbs=self._limb_fix,
                                      limb_scale=self._limb_scale,
                                      distal_mass=self._distal_mass,
                                      flexion=self._flexion,
                                      flexion_stiffness=self._flexion_stiffness,
                                      # ★筋肉モデルでは筋力が fmax にあり、gear は毎ステップ
                                      #   上書きされる。補正が実効を持つよう渡す（2026-07-25）。
                                      actuation_model=getattr(self, "actuation_model", None))

        # 【2026-07-25】床のころがり摩擦（感度分析用）。既定は触らない。
        # ★MIMoの床は friction=[1.0, 0.005, 0.0001]・condim=3 ＝「すべり摩擦しか
        #   計算しない」＝**転がることへの抵抗が事実上ゼロ**。実際の新生児は
        #   服＋寝具（布と布）の上にいて、転がるとき布が引っかかる。
        #   太郎が新生児にできない寝返りをする原因の候補として振れるようにした。
        # ⚠️2026-07-23 に「床摩擦を人工的に上げる補正は入れない」と決めているが、
        #   それは**対処**への判断。ここは**原因かどうかの検証**のための仕組み。
        _roll = os.environ.get("E_FLOOR_ROLL")
        if _roll is not None:
            _cd = int(os.environ.get("E_FLOOR_CONDIM", "6"))
            _n = 0
            for _gi in range(self.model.ngeom):
                if self.model.body(self.model.geom_bodyid[_gi]).name == "world":
                    self.model.geom_friction[_gi][1] = float(_roll)
                    self.model.geom_condim[_gi] = _cd
                    _n += 1
            print(f"[floor] roll friction={_roll} condim={_cd} ({_n} geoms) "
                  f"[SENSITIVITY: 転がり抵抗が原因かの検証]", flush=True)

        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.init_position = self.data.qpos.copy()

    def _edit_spec(self, spec):
        """LeanMimoEnv のフック：モデル構築前に spec を編集する。
        頭の楕円化を taro_core の実装で行う（身体の定義は core、適用は環境側）。"""
        import os, sys
        _core = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.pardir, os.pardir, "taro_core")
        _body = os.path.join(_core, "src", "body")
        if _body not in sys.path:
            sys.path.insert(0, _body)
        from infant_body import elongate_head
        elongate_head(spec, getattr(self, "_head_elongation", 1.0))

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_position.copy()
        # 関節だけを揺らす。qpos[:7]はfreejoint（体全体の位置と向き）なので触らない。
        qpos[7:] += self.np_random.uniform(low=-self._jitter, high=self._jitter,
                                           size=len(qpos[7:]))
        self.set_state(qpos, np.zeros(self.data.qvel.shape))
        self._set_action(np.zeros(self.action_space.shape))
        mujoco.mj_step(self.model, self.data, nstep=self._settle_steps)
        return self._get_obs()
