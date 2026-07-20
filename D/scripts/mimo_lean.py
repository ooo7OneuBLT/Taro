"""
MuJoCoモデルの「絵」を落として省メモリにする（物理は一切変えない）。

【なぜ必要か】2026-07-15 実測
太郎1本のメモリ2,638MBの内訳を段階ごとに測ったところ：
    import torch                +  146 MB
    import gym / mimoEnv        +   88 MB
    gym.make（MuJoCo構築）       + 2370 MB   ←90%がここ
    脳＋小脳の構築               +  2.8 MB   ←太郎の脳の実体は1.3MB
さらに mjModel の中を数えると **tex_data だけで976MB**：
    tex_head_default/happy/sad/angry/surprised/disgusted/scared  各2500x15000 = 107MB × 7
    tex_top_sleeve / tex_pants_leg                               各2480x14880 = 106MB × 2
＝**MIMoの表情7種と服の柄**。太郎は視覚OFF(vision_params=None)なのでこれを一度も見ていない。

【効果（実測・別プロセスで対照）】
    素のまま      RSS=2369.7MB  tex=976.7MB  nv=111 nbody=59 ngeom=78 nu=90  qpos合計=8.308976
    11枚を極小化  RSS=  16.3MB  tex=  5.5MB  nv=111 nbody=59 ngeom=78 nu=90  qpos合計=8.308976
→ **145分の1**。物理は50step回した後のqpos合計まで完全一致＝無傷。
→ 1本 2.64GB→0.28GB。同時実行が6本→約22本（メモリ律速からCPU律速へ）＝シードを増やせる。
   今日の最大の失敗は n=1 でノイズ(margin 46〜57)に埋もれたことなので、ここが直接効く。

【視覚を足すときの注意 ★重要】
テクスチャを消すと**太郎が見る映像から顔と服の柄が消える**。目標Dは「自分の体か相手の体か」を
見分ける課題なので、見た目の情報が消えていることに気づかないまま実験すると、原因不明の失敗を
する。そのため strip_textures は **vision_params が None のときだけ**適用する（LeanMimoEnv が
自動判定する＝人が覚えておく必要がない）。視覚ONなら素のフルテクスチャに戻る。
"""
import numpy as np
import mujoco

from mimoEnv.envs.dummy import MIMoV2DummyEnv
from mimoEnv.envs.mimo_env import EMOTES


def strip_textures(spec, rgb=(0.85, 0.65, 0.55)):
    """ファイル由来の巨大テクスチャを、同じ名前のまま極小の単色に置き換える。

    **消さずに置き換える**のが要点。MIMoEnv._get_facial_expressions(EMOTES) が
    表情テクスチャを名前で引くので、削除するとそこで落ちる。単色で残せば参照は生き、
    メモリだけが消える。

    Returns: 置き換えた枚数
    """
    n = 0
    for t in spec.textures:
        # builtin==0（mjBUILTIN_NONE）＝PNGファイル由来。顔7枚・服2枚・目3枚が該当。
        # 注意：ファイル由来のテクスチャは compile するまで width/height が 0 なので、
        # 「大きいものを選ぶ」という条件では絞れない（最初これで失敗した）。
        if t.builtin != 0:
            continue
        is_cube = int(t.type) == int(mujoco.mjtTexture.mjTEXTURE_CUBE)
        t.cubefiles = [""] * 6
        t.file = ""
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        t.rgb1 = list(rgb)
        t.width = 8
        t.height = 8 * (6 if is_cube else 1)   # キューブマップは6面が縦に積まれる形
        n += 1
    return n


class LeanMimoEnv(MIMoV2DummyEnv):
    """MIMoBenchV2-v0 と物理的に同一で、絵だけ落とした版（視覚ONなら自動で素に戻る）。

    MIMoEnv._initialize_simulation は MujocoEnv 側の from_xml_path でモデルを読むため、
    テクスチャを差し替えるにはモデル構築そのものを引き取る必要がある。d_env.py の
    TwoMimoEnv と同じ「_initialize_simulation を置き換える」形に揃えた。
    """

    def _edit_spec(self, spec):
        """モデルを compile する前にスペックを編集するフック（既定は何もしない）。

        サブクラスが**XMLファイルを新規に作らずに**物体を足せるようにするためのもの
        （例：目標E1のベビーサークルの柱）。MjSpec は compile 前なら body/geom を
        追加できるので、**MIMo同梱のXMLを一切変更せずに**シーンを拡張できる
        （MIMoはジャンクションの共有物なので汚したくない）。
        """
        pass

    def _initialize_simulation(self):
        spec = mujoco.MjSpec.from_file(self.fullpath)
        # self.vision_params は MIMoEnv.__init__ の中で super().__init__() より前に
        # 代入される（mimo_env.py:328 vs :358）ので、ここで既に参照できる。
        self.n_textures_stripped = strip_textures(spec) if self.vision_params is None else 0
        self._edit_spec(spec)          # ← サブクラスの拡張点（既定は無操作）
        self.model = spec.compile()
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.data = mujoco.MjData(self.model)

        fps = int(np.round(1 / self.dt))
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": fps}
        self._get_joints()
        self._get_actuators()
        self._get_facial_expressions(EMOTES)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)
        return self.model, self.data
