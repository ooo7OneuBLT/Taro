"""
D1の環境：太郎アルファ（学ぶ側）＋太郎ベータ（相手）を、姿勢を変えて置く。

【なぜ2案あるか】2026-07-15
目標は介護AI＝**太郎が触る側**（ユーザー判断）。設計書は「ベータを腕の届く位置に**座位**で配置」
だが、**座らせるのが最初なのは発達順序に反する**：
  ・自己接触＝胎児14週から（Zoia et al. 2007, n=8）。**姿勢制御ゼロ**（羊水に浮いている）で成立
    ※【訂正・2026-07-16】以前ここには「19週で先読みまで完成」と書いていたが**過大主張**だった。
      文献調査の結果：口の開きが接触に先行するのは事実（Myowa-Yamakoshi & Takeshita 2006, n=27）
      だが、「予期」は**著者の解釈**であり、反射や手-口カップリング運動プログラムを排除する統制は
      存在しない（Reissland et al. 2014 自身が "it is unclear whether mouth opening anticipates
      the touch or is a reaction to touch" と明記）。＝「先読みが完成している」とは言えない。
  ・外界へのリーチ＝生後4〜6ヶ月。**ここで初めて姿勢制御が要る**（Rochat & Goubet 1995）
さらに「触る＝正確な到達が要る」も早とちりの可能性がある。**ベータを腕の振れる範囲に置けば、
今のバタバタでも接触は起きる**かもしれない。＝支えも座位も要らないかもしれない。

そこで**憶測で決めず、両方作って測る**：
  A案（supine）: 仰向けのアルファの隣に、仰向けのベータ。**支え無し**（胎児〜新生児の条件）
  B案（seated）: 座位のアルファの前にベータ。**骨盤支持あり**（Rochat & Goubetの実験条件＝
                 大人が骨盤を支えた、まだ座れない乳児。膝に乗せて支えるのと同じ）
判定＝アルファの手がベータに触れるか。Aで触れるなら支えも座位も要らない。

【継承】
  ・2体化は `d_env.TwoMimoEnv` の方式（接頭辞`beta_`でattach＝MIMoEnvが自動でアルファだけを
    「自分」と認識する）
  ・省メモリ化は `mimo_lean.strip_textures`（1本2.64GB→0.28GB。視覚ONなら自動で素に戻る）
  ・仰向けの式は `roll_over.py` の supine（シーンはCと同一のbenchmarkv2のまま＝交絡させない）
"""
import os
import sys
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

from mimoEnv.envs.mimo_env import EMOTES
from mimoEnv.envs.dummy import MIMoV2DummyEnv
from mimo_lean import strip_textures

BETA = "beta_"

# 仰向けにする四元数（roll_over.py の supine と同じ式）
_SUPINE_QUAT = np.array([0, -0.7071068, 0, 0.7071068]) * np.array([1, -1, 1, 1])


class D1Env(MIMoV2DummyEnv):
    """2体（アルファ＝自分／ベータ＝相手）。姿勢と支えを切り替えられる。

    Args:
        layout: "supine"（両者仰向け・支え無し）または "seated"（アルファ座位・骨盤支持）
        sep: ベータをどれだけ離すか[m]。腕の届く範囲かどうかを振るための主パラメータ
        settle_steps: 開始前に無操作で落ち着かせるstep数
        jitter: リセット時に関節へ加える一様乱数の幅（毎回同じ姿勢を予測するだけ、を防ぐ）
    """

    def __init__(self, layout="supine", sep=0.30, settle_steps=100, jitter=0.01, **kwargs):
        self._layout = layout
        self._sep = sep
        self._settle_steps = settle_steps
        self._jitter = jitter
        # 【修正・2026-07-15】座位なら **MIMo公式の座る姿勢** を初期姿勢に入れる。
        # これを渡さないと既定＝**立った姿勢のまま腰だけ溶接**になり、上体が前に崩れ落ちる。
        # 実際にそれをやって「太郎は骨盤を支えても上体が崩れる／頭から突っ込む」と誤診しかけた。
        # 崩れていたのは太郎ではなく私の設定。`selfbody.py:SITTING_POSITION` が正解を持っている。
        if layout == "seated" and "initial_qpos" not in kwargs:
            from mimoEnv.envs.selfbody import SITTING_POSITION
            qpos = dict(SITTING_POSITION)
            qpos.update({BETA + k: v for k, v in SITTING_POSITION.items()})  # ベータも座らせる
            kwargs["initial_qpos"] = qpos
        super().__init__(**kwargs)

        m = self.model
        if self._layout == "held":
            # 【2026-07-15】接触は「達成するもの」ではなく「与えられるもの」ではないか、を試す配置。
            # 仰向けのアルファの**胸の上にベータを乗せる**＝重力が接触を保つ。
            #   ・**到達力が要らない**：太郎は手を伸ばせない（実測 到達力~50%・縮退が原因で学習量では
            #     解けない）。今日の壁はそこだった。抱っこなら接触は最初から在るので壁が消える。
            #   ・**信号が一定にならない**：太郎が自分の腕を触る接触は太郎の姿勢で決まる（＝固有感覚の
            #     焼き直し）。抱えた相手が動く接触は**相手の行動で決まる**＝太郎自身の状態からは
            #     絶対に予測できない＝それが「他者」の定義。
            #   ・**人間にも実在する**：カンガルーケア（肌と肌を合わせて抱く）。乳児は親を探して手を
            #     伸ばさない。抱かれる。
            m.body("hip").pos = [0, 0, 0.2]
            m.body("hip").quat = _SUPINE_QUAT.copy()
            m.body(BETA + "hip").pos = [0, 0, 0.2 + self._sep]   # アルファの真上
            m.body(BETA + "hip").quat = _SUPINE_QUAT.copy()
        elif self._layout == "supine":
            # 両者を仰向けに。ベータはアルファの真横（y方向）に、同じ向きで寝かせる。
            m.body("hip").pos = [0, 0, 0.2]
            m.body("hip").quat = _SUPINE_QUAT.copy()
            m.body(BETA + "hip").pos = [0, self._sep, 0.2]
            m.body(BETA + "hip").quat = _SUPINE_QUAT.copy()
        # seated は _initialize_simulation 側で weld を入れ、既定の初期姿勢のまま座らせる

        for _ in range(self._settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.init_position = self.data.qpos.copy()

    def _initialize_simulation(self):
        # 【修正・2026-07-16】paths.SCENE(無調整の生データ)ではなく self.fullpath を読む。
        # MIMoEnv.__init__ の既定age=18によるadjust_mimo_to_age（一時シーンをself.fullpathに
        # 格納）を素通りしていたバグ（d1_carer_env.pyで発見・詳細はそちら参照）と同型。
        spec_a = mujoco.MjSpec.from_file(self.fullpath)
        spec_b = mujoco.MjSpec.from_file(self.fullpath)
        fr = spec_a.worldbody.add_frame()
        # 配置の高さは姿勢で変える（仰向けは床すれすれ、座位は少し上げる）
        fr.pos = ([0, 0, self._sep] if self._layout == "held" else
                  ([0, self._sep, 0.0] if self._layout == "supine" else [self._sep, 0, 0.0]))
        fr.attach_body(spec_b.body("mimo_location"), BETA, "")
        if self._layout == "seated":
            # 骨盤支持＝根元(mimo_location)を世界に溶接する。
            # 【重要】これは「無理やり」ではない：固定されるのは**腰の位置だけ**で関節は全部動く。
            # Rochat & Goubet (1995) の実験条件「大人が骨盤を支えた、まだ座れない乳児」そのもの
            # ＝支えた瞬間、体幹を使ってリーチできるようになる（precocious ability）。
            # MIMo公式 selfbody_scene.xml が使っているのと同じ機構。
            #
            # 【修正・2026-07-15】**ベータも支える**。最初はアルファだけ支えたため、
            # ベータが倒れてアルファに寄りかかり、**何もしないのに体の接触が15/15**になった
            # ＝「アルファが触った」ではなく「相手が倒れてきた」を数えていた（基準線の汚染）。
            # 2人の乳児をそれぞれ椅子に座らせて向かい合わせる、が正しい像。
            for nm in ("mimo_location", BETA + "mimo_location"):
                eq = spec_a.add_equality()
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                eq.name1 = nm
                eq.name2 = "world"
        # 視覚OFFのときだけ絵を落とす（物理は不変・視覚ONなら自動で素に戻る）
        self.n_textures_stripped = strip_textures(spec_a) if self.vision_params is None else 0
        self.model = spec_a.compile()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        fps = int(np.round(1 / self.dt))
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": fps}
        self._get_joints()      # `robot:` のみ → アルファの関節だけ
        self._get_actuators()   # `act:` のみ   → アルファのアクチュエータだけ
        self._get_facial_expressions(EMOTES)
        # 【注意・2026-07-15】`selfbody.SITTING_POSITION` は **MIMo v1（指1本の手）用**の
        # 関節名（`robot:left_fingers` など）で書かれている。Cのシーンは **v2（5本指）**で
        # 関節名が違う（`robot:left_ff_distal` など）ので、そのまま渡すとKeyErrorで落ちる。
        # 座る姿勢を決めるのは腰・膝・肩・肘＝**両方に在る関節**なので、在るものだけ使う。
        if self._initial_qpos:
            have = {self.model.joint(i).name for i in range(self.model.njnt)}
            drop = [k for k in self._initial_qpos if k not in have]
            self._initial_qpos = {k: v for k, v in self._initial_qpos.items() if k in have}
            if drop:
                print(f"[D1Env] v2に無い関節を除外: {len(drop)}個（例 {drop[:3]}）", flush=True)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)
        # ベータを別に動かすためのチャネル（アルファのactuation_modelからは見えない）
        self.beta_actuators = np.asarray([i for i in range(self.model.nu)
                                          if self.model.actuator(i).name.startswith(BETA)])
        return self.model, self.data

    def set_beta_ctrl(self, ctrl):
        """ベータを動かす（相手の行動）。アルファの行動空間とは完全に別。"""
        self.data.ctrl[self.beta_actuators] = ctrl

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_position.copy()
        # freejointは触らず関節だけ揺らす（アルファ7次元＋ベータ7次元が先頭にある）
        n_free = 14 if self.model.nq > len(qpos) // 2 else 7
        qpos[n_free:] += self.np_random.uniform(low=-self._jitter, high=self._jitter,
                                                size=len(qpos) - n_free)
        self.set_state(qpos, np.zeros(self.data.qvel.shape))
        self._set_action(np.zeros(self.action_space.shape))
        mujoco.mj_step(self.model, self.data, nstep=self._settle_steps)
        return self._get_obs()
