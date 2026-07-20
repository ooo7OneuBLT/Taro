"""E1環境：仰向けの太郎の「手の届く所」に、押すと動く**おもちゃ**を置く。

【設計の根拠】E/docs/全体設計_目標E.md §10
おもちゃは「報酬」ではなく「**自分の動きで予測できるようになる、学べる相手**」。
太郎の予測対象は固有感覚のまま変えない＝おもちゃに触れて押すと**その反作用で関節の感覚が
変わる**ので、既存のprogress報酬(学習進度)が自然にそこへ向く、という設計。
＝**新規実装はこの環境（おもちゃ）だけ。脳はいじらない。**

【実測に基づく配置（e_reach_space.py, age=0）】
  肩→手の距離 : 静止0.158 m / 暴れても最大0.160 m
     → 腕は初期姿勢でほぼ伸びきっており「もっと遠くへ伸ばす」余地はほぼ無い。
       **肩から16cm以内に置かないと物理的に一生届かない**。
  体のドリフト : 頭のワールドx が −0.283〜+0.185 m 動く
     → 暴れると体ごと移動するので、**ワールド固定で置くと位置関係が壊れる**。
       よって毎リセット時に**肩を基準に相対配置**し、離れすぎたら置き直す
       （＝現実の育児で親がおもちゃを拾って手元に戻すのと同じ scaffolding）。

【なぜ既存の test_object1 を使うか】
benchmarkv2_scene.xml には既に freejoint の箱(test_object1)と球(test_object2)があるが、
遠方(x=1.0, z=0.7)に置かれていて届かない。**MIMoは共有(ジャンクション)なのでXMLは編集せず**、
実行時に位置・大きさ・質量だけを変える＝C/Dの既存実験を1バイトも壊さない。
箱を使うのは、球より転がりにくく置いた場所に留まるため。

【注意】
- geom size を実行時に変えても質量・慣性は再計算されないので、**質量と慣性も明示的に設定**
  する（元は20cm角の箱＝新生児には重すぎて押せない）。
- おもちゃONは opt-in。`toy=False` なら test_object を遠方に退避＝従来の仰向け環境と同一。

使い方（環境として）:
    from e_toy_env import ToySupineEnv
    env = ToySupineEnv(age=0, toy=True)
"""
import os

import numpy as np
import mujoco

from d_supine_env import SupineMimoEnv

# --- 実測（e_reach_space.py, age=0）に基づく既定値 ---
REACH_MAX = 0.160          # 肩→手の最大距離[m]（これを超えると物理的に届かない）
REST_DIST = 0.158          # 静止時の肩→手の距離[m]
# 【2026-07-20 目視で判明した設計ミスと修正】
# 旧：おもちゃを「肩の横の床」に置き、離れたら瞬間移動で置き直していた（respawn）。
#     Viewerでの目視により2つの欠陥が判明。
#       ①仰向けの太郎は**天井を向く**ので、横の床のおもちゃは**視界に全く入らない**。
#       ②置き直しが200判断で37回も発動し、**おもちゃがワープして見える**。物理的に不自然な上、
#         「勝手に動くもの＝予測できない」ので随伴性の学習を壊す（設計の根幹に反する）。
# 新：**ベビージム（モビール）方式**。顔の上に吊るす。現実の育児用品そのもの＝人間模倣として正当。
#       ・視界に入る（仰向けの赤ちゃんは上を見る）
#       ・手を上げれば届く（接触機会が増える）
#       ・紐で吊るされている＝押せば揺れて戻る＝**ワープ不要**（随伴性は保たれる）
# 【2026-07-20 再修正】置き場所を「頭からの固定オフセット」→「**視線の正面・距離D**」へ。
#   旧 TOY_OFFSET=(-0.05,0,0.07) は手が届くかだけで決めており、**視線が通るかを確認していなかった**。
#   実測（e_gaze_geometry.py）：初期姿勢で視線から既に **20.4°** ずれ＝視野の半角30°の2/3を
#   配置だけで消費し、頭の揺れに使えるマージンが10°しか残っていなかった。
#   視線の正面に置けば初期ずれ0°＝30°丸ごとを頭の揺れに回せる。
# ⚠️ ただしこれで解決するのは配置ぶんだけ。**頭は平均50°振れる**ので、
#   空間に固定したおもちゃを見続けるには視野の半角が54°以上必要＝根本原因は別（要検討）。
TOY_DISTANCE = 0.086       # 目からおもちゃまでの距離[m]。⚠️暫定＝旧オフセットの長さを維持
                           #   （新生児の適切な注視距離は文献調査中。決まり次第ここを更新）
TOY_OFFSET = np.array([-0.05, 0.0, 0.07])   # 旧方式（アブレーション用に残す）
# 【2026-07-20 修正】吊り方を「バネ」から「紐（振り子）」へ。
# 旧：変位に比例するバネ＋重力補償 → 強く叩かれるとバネが伸びきって**柵の外へ飛んでいった**
#     （Viewerでの目視で発覚）。また重力補償のせいで宙に浮いたままで不自然だった。
# 新：現実のベビージムと同じ**紐で吊るした振り子**。
#     ・紐の長さL以内 … 自由（重力で支点の真下に垂れる）→押せば振り子のように揺れる＝随伴性維持
#     ・L を超える    … 強い張力で引き戻す（紐は伸びない）→**どんなに叩かれても外に出ない**
TETHER_LENGTH = 0.15       # 紐の長さ[m]（支点からおもちゃまで）
TETHER_K_TAUT = 150.0      # 紐が張ったときのばね定数[N/m]（大きい＝実質伸びない）
TETHER_C = 0.05            # 減衰[N/(m/s)]（揺れが自然に収まる）

# --- ベビーサークルの柵（柱）---
# 【なぜ要るか】太郎は運動性喃語で**体ごと20〜47cm移動する**（実測）。すると吊るした
# おもちゃとの位置関係が崩れ、届かなくなる。現実の新生児もベッド/クーファン/サークルの
# 中にいるので、**環境として囲う**のは人間模倣として正当（太郎の中身はいじらない）。
# 【なぜ壁でなく柱か】壁だと視界を塞ぐ。柱なら隙間から外が見えて視界の邪魔になりにくい
# （ユーザーの図＝ベビーサークルの柵の発想）。
# 【寸法の根拠】実測：太郎の体は 44.3cm(頭〜足) × 34.0cm(腕幅)。
#   半径0.32m＝直径64cm ≒ 体長の1.5倍（現実のクーファン/サークルと同程度）。
#   柱12本＝間隔約17cm＝体幅34cmは通り抜けられない。
#   高さ0.15m＝体の高さ(床から約10cm)より高く越えられない。細さ2.4cmで視界を遮りにくい。
# 【形状の根拠】円でなく**長方形**にする。太郎の体は 44.3cm(頭〜足)×34.0cm(腕幅)＝
# 比率1.30なので、同じ比率の長方形が体の形に沿い、無駄なく囲える（円だと角が余る）。
#   長辺 0.62m / 短辺 0.48m ＝比率1.29（体とほぼ同じ）。体長44.3cmに対し余裕18cm。
# 【柱の形の根拠】円柱でなく**薄い板**。厚さは維持（2.4cm）したまま幅を広げて
# 遮蔽面積を増やす＝**隙間が減って抜け出しにくくなる**が、板の間は素通しなので
# 視界は保たれる（壁で囲うと視界を塞ぐ、という問題への答え）。
#   長辺5本・短辺3本＝計16本。隙間は長辺で約9.5cm・短辺で約6cm＝頭(約10cm)も通りにくい。
# 【半径スイープの実測（円形時・120判断・age=0）】
#   なし 移動0.135m/ズレ0.088m/接触0%   0.30 移動0.121/ズレ0.084/接触20.4%
#   0.26 移動0.163/ズレ0.087/接触41.3%  0.23 移動0.084/ズレ0.163/接触70.0%
#   0.20 移動0.062/ズレ0.154/接触95.6%
# → 小さくすると並進移動は減るが、柵に押されて姿勢が変わり**おもちゃとのズレはむしろ悪化**。
#   よって柵の目的は「移動制限」でなく**長時間の学習で遠くへ行かない保険**と位置づける。
#   位置関係を崩す主因は並進でなく**頭そのものの大きな動き**（＝運動の非人間性・別途対処）。
FENCE_HALF_X = 0.31        # 長辺方向（頭〜足）の半分[m]
# 目視の結果「横幅に余裕がありすぎる」→ 短辺を2/3に（0.48m→0.32m）。
# 太郎の腕を広げた幅は実測34.0cmなので、32cmは腕幅とほぼ同じ＝かなり詰まった配置。
# 窮屈すぎないかはViewerで要確認（柵への接触率が上がりすぎたら戻す）。
FENCE_HALF_Y = 0.16        # 短辺方向（左右）の半分[m]
FENCE_POST_W = 0.06        # 柱の幅[m]（柵の辺に沿う向き＝遮蔽面積）
FENCE_POST_T = 0.024       # 柱の厚さ[m]（柵の法線方向＝従来の直径と同じ）
# 本数は目視の結果2倍に増やした（隙間 長辺9.5cm/短辺6cm → 約3.4cm/2.4cm＝「密度が欲しい」）
FENCE_N_LONG = 9           # 長辺1本あたりの柱の本数
FENCE_N_SHORT = 6          # 短辺1本あたりの柱の本数（角は長辺側が担うので端を除く）
FENCE_HEIGHT = 0.225       # 柱の高さ[m]（目視の結果15cm→1.5倍に。低いと越えられそうに見えた）
# 視覚的な「豊かさ」の切替。E_PLAIN=1(既定)＝床の市松模様を消し柵を床と同色に＝**見えないnest**。
# 根拠と意図は _make_visually_plain() のdocstring参照（White 1966 と Ferrari 2007 の両立）。
# 【色の統一】床・柵・空を**同じ色**にする＝どこを向いても同じ＝最も「貧しい」視界。
# 目視で判明した2点：①柵が茶色になっていた（MjSpecのデフォルト材質 matgeom の継承。下記で修正）
# ②見上げると空が水色（skyboxのグラデーション rgb1="0.3 0.5 0.7"）で、灰色の柵とコントラストが出る。
# → 空の色に合わせて全部を淡い青灰色にする。⚠️色の選択自体に文献の根拠はない（**恣意的**）が、
#   「面と面の境界が見えないほどコントラストが小さい」ことが目的なので、値そのものは重要でない。
PLAIN_RGBA = np.array([0.55, 0.62, 0.70, 1.0])       # 床・柵・空に共通で使う淡い青灰色
FENCE_RGBA_RICH = np.array([0.35, 0.45, 0.85, 1.0])  # 豊かな条件での柵＝青（従来の色）
TOY_RADIUS = 0.025         # 箱の half-size[m]＝5cm角。新生児が握れる大きさ
TOY_MASS = 0.03            # 30g＝新生児が押して動かせる軽さ
# 摩擦[slide, spin, roll]。**転がり続けを止めるために roll/spin を既定より上げる**。
# 理由＝実測で「手が遠いのにおもちゃが動く(0.342mm/tick)」＝一度押されると転がり続け、
# 「今の自分の運動」と無関係に動いて**随伴性(自分の行為→結果)が濁る**ため。
# 押した分だけ動いてすぐ止まる＝カーペット上のおもちゃに相当し、随伴が明確になる。
TOY_FRICTION = np.array([1.0, 0.05, 0.02])

# --- 光るおもちゃ（随伴性を"薄まらないチャネル"に出すため）---
# 【なぜ光らせるか】実測で、接触の反力は**固有感覚621次元に薄まって消える**ことが判明した
#   （接触あり/なしで予測誤差 pe の効果量 d=-0.005＝差なし。腕の次元だけ見ると d=-0.22 で
#    ようやく現れる＝希釈。D0で踏んだ「触覚が次元数に薄められる」罠と同じ構造）。
#   おもちゃを重くする案は、3kg必要＝太郎の体重2.9kgと同等で非現実的＝対症療法。
#   予測対象を腕に絞る案は、人間の脳が全身を予測している以上、人間模倣でない。
# 【なぜ光か】Rochatの随伴性実験（生後2ヶ月）は「おしゃぶりを噛む→**音**が鳴る」で、
#   モビール実験は「足を蹴る→モビールが**動く**（視覚）」＝**人間の随伴性学習の古典は
#   触覚でなく音・視覚**。現実のベビージムも光る/鳴る/鏡がつく＝育児用品として実在する。
#   MIMoに聴覚モジュールは無い（proprio/touch/vision/vestibular/actuationの5つのみ）ので、
#   実現できるのは**光**。視覚は独立チャネルなので**他の次元に薄まらない**。
# 【光り方】触れたら **GLOW_HOLD_S 秒のあいだ光り続ける**（触れている間だけ、ではない）。
#   第1版は「接触中だけ」にしたが、接触は一瞬（数十ms）で終わるのに対し視覚が読まれるのは
#   制御周期（1秒に1回）なので、**光った瞬間を視覚が一度も見ないまま消える**。
#   現実の光る/鳴るおもちゃも、叩いた後しばらく光り続ける＝余韻がある方が普通。
#   ⚠️ 2.0秒という長さに文献の裏付けはない＝**恣意的**（制御周期1秒を確実にまたぐ長さとして選択）。
TOY_RGBA_OFF = np.array([0.9, 0.2, 0.15, 1.0])   # 通常＝赤
TOY_RGBA_ON = np.array([1.0, 1.0, 0.45, 1.0])    # 点灯中＝明るい黄（視界で目立つ）
GLOW_HOLD_S = 2.0                                 # 点灯の持続[sim秒]（⚠️恣意的）

# --- 視覚 ---
# 【重要・MIMoのバグ回避】MIMoの mimoVision は `env.camera_name` を設定して `env.render()`
# を呼ぶ方式だが、gymnasium 1.2.3 の MujocoEnv.render() は camera_name を無視するため、
# **全ての視覚obsが「外から太郎を見た第三者視点の映像」になる**（D側で実測・確定。
# renderer.camera_id=-1, cam.type=FREE）。上流mainも未修正。
# → `get_vision_obs` を差し替えて**生APIで眼球カメラを直接描画**する（D側と同じ方式）。
#    これを知らずに視覚を使うと、一人称のつもりで第三者視点を学習させることになる。
# 【視力】`acuity` に月齢を渡すと Mayer et al.(1995) の実測テーブルからMTFを作り高周波を落とす。
#   新生児は 0 を渡す（MIMo側のバグ回避：内部テーブルの値に完全一致するとクラッシュするため、
#   0 はテーブル最小値1.0より小さく安全）。
# 【解像度】視力フィルタが効く下限として128。fovy=60はMIMo本家のまま（新生児の視野は成人より
#   狭く、成人単眼120°に対し乳児6-7ヶ月で74%＝約89°、新生児はさらに狭いので的外れではない）。
VISION_RES = 128
VISION_FOVY = 60
# 【視覚の更新周期】眼のレンダリングは**物理1ステップ(10ms)ごとに毎回**行われていたが、
#   方策が視覚を読むのは制御周期＝1秒に1回だけなので、99%が捨てられていた。
#   実測でこれが 2.15 ms/step（全体6.46msの1/3）を占め、ビューアの倍速が頭打ちになる原因の
#   一つだった。0.1秒(10Hz)に間引いても制御周期(1Hz)より10倍細かいので**情報の損失はない**。
#   ⚠️これは計算の最適化であって発達的な主張ではない（乳児の視覚時間分解能を模したものではない）。
VISION_MIN_DT = 0.1


ACUITY_AGE = 0.5   # 視力テーブルに渡す月齢。★下記のとおり 0.0 は**無効**になるので使えない


def infant_vision_params(size=VISION_RES, fovy=VISION_FOVY, acuity_age=ACUITY_AGE):
    """新生児の視覚パラメータ。acuityに月齢を渡す（＝解像度を恣意的に決めない）。

    ⚠️【2026-07-20 修正・重大】以前は `acuity_age=0.0` を渡しており、**視力フィルタが
    まったく効いていなかった**（太郎はフル解像度で見ていた＝新生児の視力ではない）。
    目視で「acuityあり/なしの画像がほぼ同じ」ことに気づき、MIMo本体のコードを読んで判明した。

    原因＝MIMo `mimoVision/vision.py` の2箇所：
      L95: `if camera_parameters[camera]["acuity"]:` … **0.0 は Falsy なので関数が作られない**
           （`is not None` ではなく truthy 判定になっている）
      L200-201: `if acuity_age in ages: acuity = acuities[ages.index(self.env.age)]`
           … **検索キーが acuity_age ではなく self.env.age**。テーブル値
           `[1.0, 1.169, 1.366, ...]` に完全一致する月齢を渡すと env.age で引き直され
           ValueError になりうる。以前「0.0 なら安全」と書いたのはこのクラッシュ回避が理由だった
           が、**回避と引き換えにフィルタ自体を無効にしていた**。

    正しい設定＝**ages[0]=1.0ヶ月より小さい"正の"値**を渡す。L203-204 が
    `acuity = acuities[0]`（＝テーブル最年少 1ヶ月の 0.852 cycles/deg）にクランプするので、
    クラッシュせず・フィルタも有効になる。0.5 はその条件を満たす任意の値。
    ⚠️**残る逸脱**：Mayer et al.(1995) のテーブルは**1ヶ月から**しかない。新生児(0ヶ月)の実測値は
    このテーブルに存在しないため、太郎は「**1ヶ月児の視力**」で代用している。0.852 cycles/deg は
    スネレン換算でおよそ 20/700 相当＝成人(20/20)の約1/35。
    """
    eye = {"width": size, "height": size, "fovy": fovy,
           "acuity": acuity_age, "foveation": False}
    return {"eye_left": dict(eye), "eye_right": dict(eye)}
FAR_AWAY = np.array([3.0, 3.0, 0.05])   # 使わない物体の退避先


def _box_inertia(mass, half):
    """一様な立方体(half-size=half)の慣性モーメント。size変更時に手で入れ直すため。"""
    i = mass * (2 * half) ** 2 / 6.0
    return np.array([i, i, i])


class ToySupineEnv(SupineMimoEnv):
    """仰向け＋手の届く所におもちゃ（押すと動く対象）。

    Args:
        toy: Falseなら物体を遠方に退避＝従来の仰向け環境と同一（アブレーション用）。
        toy_side: "right" / "left"。どちらの肩を基準に置くか。
        toy_offset: 肩から見た配置オフセット[m]（x,y,z）。zは接地させるので実質x,yのみ。
        toy_radius: 箱の half-size[m]。
        toy_mass: 質量[kg]。
        respawn_dist: 肩からこの距離[m]より遠ざかったら手元に置き直す（＝親が渡す）。
            Noneで無効。既定はREACH_MAX（届かなくなったら戻す）。
    """

    def __init__(self, toy=True, toy_side="right", toy_offset=None,
                 toy_dist=TOY_DISTANCE,
                 toy_radius=TOY_RADIUS, toy_mass=TOY_MASS,
                 tether_length=TETHER_LENGTH, tether_k=TETHER_K_TAUT,
                 tether_c=TETHER_C,
                 fence=True, fence_half_x=FENCE_HALF_X, fence_half_y=FENCE_HALF_Y,
                 fence_post_w=FENCE_POST_W, fence_post_t=FENCE_POST_T,
                 fence_n_long=FENCE_N_LONG, fence_n_short=FENCE_N_SHORT,
                 fence_height=FENCE_HEIGHT, newborn_neck=None, newborn_limbs=None,
                 vor=None, **kwargs):
        # VOR（前庭動眼反射）。眼球を方策から切り離し、頭の動きを打ち消して視線を安定させる。
        # E_VOR=0 でOFF（アブレーション）。根拠と簡略化は e_vor.py 参照。
        if vor is None:
            vor = os.environ.get("E_VOR", "1") == "1"
        self._use_vor = vor
        self._vor = None
        # 既定は環境変数から（E_NECK=0 / E_LIMBS=0 でアブレーション）。引数指定が優先。
        if newborn_neck is None:
            newborn_neck = os.environ.get("E_NECK", "1") == "1"
        if newborn_limbs is None:
            newborn_limbs = os.environ.get("E_LIMBS", "1") == "1"
        # 【2026-07-20】おもちゃ・柵を個別に消せるようにする（E_TOY_OBJ=0 / E_FENCE=0）。
        # 理由＝hand regard の一次文献（White 1966）が示す最重要の実験条件：
        #   何もない環境の乳児は生後**46日**で手を見はじめ、視覚的に豊かな環境では**66日**と
        #   **遅れた**。＝手は「他に見るものがないときに見られる対象」。太郎でも、視界に
        #   おもちゃや模様のある柵があると progress報酬がそちらへ向かい hand regard が出ない。
        # → 「貧しい環境で出る／豊かな環境で遅れる」の**二条件比較**が最強の実験デザインなので、
        #   両方を独立に切れる必要がある。（[参考文献リスト §目標E-15](../../doc/参考文献リスト.md)）
        if os.environ.get("E_TOY_OBJ", "1") != "1":
            toy = False
        if os.environ.get("E_FENCE", "1") != "1":
            fence = False
        # E_PLAIN=1（既定）＝視覚的に貧しくする。E_PLAIN=0 で従来の見た目（市松床・青い柵）。
        # ⚠️_edit_spec は super().__init__() の中で呼ばれるので super() より前に代入する。
        self._plain = os.environ.get("E_PLAIN", "1") == "1"
        # 身体の補正。**必ずON/OFFできるようにする**＝E1で「dampingが創発したのか、
        # 身体を弱めただけか」を切り分けるアブレーションに使う（これが無いと結果を解釈できない）。
        #   newborn_neck  : 首がすわっていない（head lag）の再現。⚠️恣意的（e_infant_neck.py）
        #   newborn_limbs : 四肢の発達の向きの逆転を解消。測定値ベース（e_infant_body.py）
        self._newborn_neck = newborn_neck
        self._newborn_limbs = newborn_limbs
        self._neck_age = kwargs.get("age", None)
        # 柵（柱）の設定。fence=False で従来どおり囲いなし＝アブレーション用。
        # ⚠️ _edit_spec は super().__init__() の中（モデル構築時）に呼ばれるので、
        #    これらの属性は super() より**前**に代入しておく必要がある。
        self._fence = fence
        self._fence_half_x = float(fence_half_x)
        self._fence_half_y = float(fence_half_y)
        self._fence_post_w = float(fence_post_w)
        self._fence_post_t = float(fence_post_t)
        self._fence_n_long = int(fence_n_long)
        self._fence_n_short = int(fence_n_short)
        self._fence_height = float(fence_height)
        self._toy = toy
        self._toy_side = toy_side
        # toy_offset=None（既定）＝**視線の正面**に置く（_set_anchor 参照）。
        # 明示的にベクトルを渡すと旧方式（頭からの固定オフセット）になる＝アブレーション用。
        self._toy_offset = None if toy_offset is None else np.array(toy_offset, dtype=float)
        self._toy_dist = float(toy_dist)
        self._toy_radius = float(toy_radius)
        self._toy_mass = float(toy_mass)
        self._tether_len = float(tether_length)
        self._tether_k = float(tether_k)
        self._tether_c = float(tether_c)
        self._anchor = None        # 吊り下げの基準点（リセット時に頭の位置から決める）
        # 置き直し(親が渡す)の回数と、そのstepで置き直したかのフラグ。
        # ＝「おもちゃが動いた」を随伴性の証拠として数えるとき、瞬間移動を除くために要る
        #   （これを見ずに移動量だけ見ると置き直しを"触れて動いた"と誤認する）。
        self.n_respawn = 0
        self.respawned_this_step = False
        super().__init__(**kwargs)

        self._arm_body = f"{toy_side}_upper_arm"
        self._hand_body = f"{toy_side}_hand"

        # --- おもちゃ(箱)の大きさ・質量・慣性を新生児向けに作り替える ---
        self._toy_bid = self.model.body("test_object1").id
        gadr = self.model.body("test_object1").geomadr[0]
        self.model.geom_size[gadr] = [self._toy_radius] * 3
        self.model.body_mass[self._toy_bid] = self._toy_mass
        self.model.body_inertia[self._toy_bid] = _box_inertia(self._toy_mass,
                                                              self._toy_radius)
        self.model.geom_friction[gadr] = TOY_FRICTION   # 転がり続けを止める（上のコメント）
        # 目視用に目立つ色（赤）。太郎の体・床と区別がつかないと動画で確認できないため。
        # 接触中は TOY_RGBA_ON（明るい黄）に切り替わる＝「触れている間だけ光る」。
        self._toy_gadr = gadr
        self.model.geom_rgba[gadr] = TOY_RGBA_OFF
        self.toy_lit = False          # 今光っているか（測定・記録用）
        # freejoint の qpos 先頭アドレス（位置3＋姿勢4）
        jadr = self.model.body("test_object1").jntadr[0]
        self._toy_qadr = self.model.jnt_qposadr[jadr]
        self._toy_dadr = self.model.jnt_dofadr[jadr]

        # 使わない球は遠方へ退避（見えても届かない所に置くと視覚の交絡になるため）
        self._obj2_bid = self.model.body("test_object2").id
        jadr2 = self.model.body("test_object2").jntadr[0]
        self._obj2_qadr = self.model.jnt_qposadr[jadr2]

        # 首の補正は「落ち着いた初期姿勢」で重力モーメントを測ってから適用する
        # （姿勢で腕の長さが変わるため）。SupineMimoEnvのsettle後＝ここが適切な位置。
        if self._newborn_neck and self._neck_age is not None:
            from e_infant_neck import apply_newborn_neck
            apply_newborn_neck(self.model, self.data, float(self._neck_age))
        if self._newborn_limbs and self._neck_age is not None:
            from e_infant_body import apply_limb_inversion_fix
            apply_limb_inversion_fix(self.model, self.data, float(self._neck_age))
        if self._use_vor:
            from e_vor import VOR
            self._vor = VOR(self.model, self.data)
            print(f"[vor] enabled: gain={self._vor.gain} on {len(self._vor.units)} eye actuators "
                  f"(policy output to eyes is ignored)")

    # ------------------------------------------------------------------
    def _make_visually_plain(self, spec):
        """★視界を「貧しく」する：床の市松模様を消し、柵を床と同じ色にする。

        【なぜ＝White 1966（[参考文献リスト §目標E-15](../../doc/参考文献リスト.md)）】
        hand regard は「他に見るものがないとき」に最も早く出る（何もない環境の乳児は生後46日、
        視覚的に豊かな環境では66日と**遅れた**）。太郎の視界に模様があると、progress報酬は
        手ではなくそちらへ向かう。
        【何が豊かだったか（実測）】MIMoの標準シーンの床は
          `<texture name="texplane" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2"
                    mark="cross" markrgb=".8 .8 .8">`
        ＝**高コントラストの市松模様＋白い十字マーク**。仰向けの太郎の視界に入る主要な面がこれ。
        柵も青(0.35,0.45,0.85)で床と明確に区別できる色だった。
        【何をするか】床の材質を外して無地の灰色に、柵を床と同じ灰色にする。
        ⚠️skybox（空のグラデーション）は残す＝仰向けで上を向いたときの背景。消すと真っ暗になり
          「視覚が無い」条件になってしまうため。
        ⚠️**柵を消すのではなく色だけ変える**のは、Ferrari et al.(2007)が「nest（囲い）を与えると
          肩内転・肘屈曲・**正中方向への運動が有意に増える**」と実測しているため
          ＝**視覚的な豊かさ（除くべき）と物理的な支持（与えるべき）は別物**。
        """
        for g in spec.geoms:                      # worldbody直下のgeom（床はここ）
            if g.name == "floor":
                g.material = ""                   # 市松テクスチャを外す
                g.rgba = list(PLAIN_RGBA)
                break
        # 空（skybox）も同じ色の単色に。仰向けの太郎が最も長く見ているのは空なので、
        # ここが水色のグラデーションのままだと「床＝灰／空＝水色／柵＝その境界」で
        # コントラストが残る（目視で指摘された）。builtinをflatにして単色化する。
        try:
            for t in spec.textures:
                if int(t.type) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
                    t.builtin = int(mujoco.mjtBuiltin.mjBUILTIN_FLAT)
                    t.rgb1 = list(PLAIN_RGBA[:3])
                    t.rgb2 = list(PLAIN_RGBA[:3])
        except Exception as e:      # MjSpecのtexture APIはバージョン差があるので落とさない
            print(f"[E1] skyboxの単色化をスキップ（{type(e).__name__}: {e}）")

    def _edit_spec(self, spec):
        """モデル構築前に、ベビーサークルの柱をワールドへ追加する（LeanMimoEnvのフック）。

        MjSpec は compile 前なら geom を足せるので、**XMLファイルを一切作らず**に
        シーンを拡張できる＝MIMo同梱のXML（共有物）を汚さない。
        柱は静的（freejointなし）なので、太郎が当たっても動かない＝壁として働く。
        """
        if self._plain:
            self._make_visually_plain(spec)
        if not self._fence:
            return
        a, b = self._fence_half_x, self._fence_half_y
        w, t, h = self._fence_post_w, self._fence_post_t, self._fence_height
        posts = []
        # 長辺（±y側）：柱の幅は x 方向＝辺に沿う。角は端の柱が担う。
        for x in np.linspace(-a, a, self._fence_n_long):
            posts.append((float(x), +b, "long"))
            posts.append((float(x), -b, "long"))
        # 短辺（±x側）：柱の幅は y 方向。角の重複を避けるため両端を除く。
        for y in np.linspace(-b, b, self._fence_n_short + 2)[1:-1]:
            posts.append((+a, float(y), "short"))
            posts.append((-a, float(y), "short"))
        for i, (x, y, kind) in enumerate(posts):
            g = spec.worldbody.add_geom()
            g.name = f"fence_post_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            # 厚さは一定のまま、辺に沿う向きの幅を広げて遮蔽面積を稼ぐ（隙間を減らす）
            g.size = ([w / 2, t / 2, h / 2] if kind == "long"
                      else [t / 2, w / 2, h / 2])
            g.pos = [x, y, h / 2.0]
            # 貧しい条件では背景と同色＝**見えないnest**（支持は残し、視覚的な刺激だけ消す）
            # ⚠️バグ修正：MjSpecで足したgeomは**デフォルトクラスの material="matgeom"（茶色の
            #   テクスチャ）を継承する**ため、rgbaを指定しても茶色に描かれていた（目視で発覚）。
            #   material を空にしないと rgba が効かない。
            g.material = ""
            g.rgba = list(PLAIN_RGBA if self._plain else FENCE_RGBA_RICH)
            g.condim = 3

    def _place(self, qadr, pos):
        """freejoint物体をワールド座標posへ置き直す（速度も0に戻す）。"""
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _gaze_dir(self, cam="eye_left"):
        """今の視線方向（ワールド）。MuJoCoのカメラは**-z方向**を見るので符号を反転する。"""
        cid = int(self.model.camera(cam).id)
        return -np.array(self.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]

    def _set_anchor(self):
        """吊り下げの基準点（ベビージムの支点）を決める。

        ★**リセット時の視線の正面**に置く（第1版は「頭の上へ固定オフセット」だった）。
        第1版は手が届くかだけを見て置き場所を決め、**視線が通るかを確認していなかった**。
        実測すると初期姿勢で既に視線から20.4°ずれており、視野の半角30°の2/3を
        配置だけで食い潰していた（残りマージン10°／頭は平均50°振れる）。
        視線の正面に置けば初期ずれ0°＝マージンを丸ごと頭の揺れに使える。
        ⚠️これは「頭が振れると視界から外れる」問題そのものは解決しない（後述の課題）。

        リセット時に一度だけ決めてエピソード中は固定＝現実のベビージムも動かない。
        （太郎に追従させると「おもちゃが赤ちゃんを追いかける」不自然さになる）
        """
        head = self.data.body("head").xpos.copy()
        # おもちゃがぶら下がる位置。支点はその**真上に紐の長さぶん**取る＝
        # 重力で自然に垂れると、ちょうどこの位置に来る（振り子の静止点）。
        if self._toy_offset is None:
            g = self._gaze_dir()
            self._rest_pos = head + g * self._toy_dist     # 視線の正面・距離 _toy_dist
        else:
            self._rest_pos = head + self._toy_offset       # 旧方式（アブレーション用に残す）
        self._anchor = self._rest_pos + np.array([0.0, 0.0, self._tether_len])

    def _apply_tether(self):
        """おもちゃを基準点に吊るす力（ベビージムの紐/ゴム）。

        重力を打ち消し、変位に比例した復元力＋減衰を加える＝**押せば動き、離せば戻る**。
        ★瞬間移動(respawn)を使わずに手元へ留めるための機構。ワープは物理的に不自然な上、
        「勝手に動く＝予測できない」ので**随伴性の学習を壊す**（Viewerでの目視で判明）。
        """
        if not self._toy or self._anchor is None:
            return
        pos = self.data.body("test_object1").xpos
        vel = self.data.qvel[self._toy_dadr:self._toy_dadr + 3]   # freejointの線速度
        d = pos - self._anchor                    # 支点からおもちゃへ
        dist = float(np.linalg.norm(d))
        if dist > self._tether_len:
            # 紐が張った：伸びた分だけ支点方向へ強く引く（＝紐は伸びない）
            over = dist - self._tether_len
            f = -self._tether_k * over * (d / max(dist, 1e-9)) - self._tether_c * vel
        else:
            # 紐がたるんでいる：自由（重力で落ちる）。減衰だけ与えて暴れを抑える
            f = -self._tether_c * vel
        self.data.xfrc_applied[self._toy_bid, :3] = f

    def _spawn_toy(self):
        if self._toy:
            self._set_anchor()
            self._place(self._toy_qadr, self._rest_pos)   # 支点の真下＝紐が垂れた位置
        else:
            self._place(self._toy_qadr, FAR_AWAY)
        self._place(self._obj2_qadr, FAR_AWAY + np.array([0.5, 0.0, 0.0]))
        self.data.qvel[self._toy_dadr:self._toy_dadr + 6] = 0.0
        self.n_respawn += 1
        self.respawned_this_step = True

    def toy_contacts(self):
        """今おもちゃに触れている体の部位名のリスト。

        距離ではなく**MuJoCoの接触**で見る。距離が近い＝触れている、ではないため
        （半径2.5cmの箱＋手の厚みがあるので「3.6cm」は接触かどうか判定できない）。
        どの部位が触れるか（手/脚/胴）は随伴性の質を左右するので部位名まで返す。
        """
        gadr = self.model.body("test_object1").geomadr[0]
        parts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gadr or c.geom2 == gadr:
                other = c.geom2 if c.geom1 == gadr else c.geom1
                bid = self.model.geom_bodyid[other]
                parts.append(self.model.body(bid).name)
        return parts

    def toy_distance(self):
        """肩からおもちゃまでの距離[m]（届く範囲=REACH_MAX と比べるため）。"""
        return float(np.linalg.norm(self.data.body("test_object1").xpos
                                    - self.data.body(self._arm_body).xpos))

    def hand_toy_distance(self):
        """手からおもちゃまでの距離[m]（リーチの成否を測る素材）。"""
        return float(np.linalg.norm(self.data.body("test_object1").xpos
                                    - self.data.body(self._hand_body).xpos))

    # ------------------------------------------------------------------
    def reset_model(self):
        obs = super().reset_model()        # 仰向け＋jitter＋settle
        self._spawn_toy()                  # 落ち着いた後の肩位置を見て配置
        self._glow_until = -1e9            # 点灯の余韻を持ち越さない
        self._vision_cache = None          # data.time が巻き戻るのでキャッシュを捨てる
        if self._toy:
            self.model.geom_rgba[self._toy_gadr] = TOY_RGBA_OFF
            self.toy_lit = False
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        # 吊り力は物理を進める前に設定する（xfrc_appliedはframe_skip回ぶん効く）。
        # 旧実装の「離れたら瞬間移動で置き直す(respawn)」は**廃止**。目視でワープが
        # 見えたうえ、随伴性（自分の行為→結果）を壊すため。代わりに吊り紐で留める。
        self.respawned_this_step = False
        self._apply_tether()
        self._update_glow()
        if self._vor is not None:
            # 方策の眼球出力を捨て、VORの指令に差し替える（皮質は反射弓に介入しない）
            action = self._vor.override(action, self.model, self.data, self.dt)
        return super().step(action)

    def get_vision_obs(self):
        """★MIMoの壊れたgym描画を迂回し、眼球カメラを生APIで直接描画する（D側と同じ）。

        MIMo本家は `env.camera_name` を設定して `env.render()` を呼ぶが、gymnasium 1.2.3 は
        camera_name を無視するため、**視覚obsが第三者視点になる**（D側で実測・確定）。
        この差し替えは MIMoEnv.get_vision_obs のdocstringが明示的に許可している差し込み口。
        acuity（視力）は本家のヘルパを再利用して同じ後処理をかける。
        """
        import mujoco
        cache = getattr(self, "_eye_renderers", None)
        if cache is None or getattr(self, "_eye_renderers_model", None) is not self.model:
            for r in (cache or {}).values():
                r.close()
            self._eye_renderers = {}
            self._eye_renderers_model = self.model
            cache = self._eye_renderers

        # 前回の描画から VISION_MIN_DT(sim秒) 経っていなければ使い回す（上の定数コメント参照）。
        # ⚠️ data.time はリセットで巻き戻るので、時間が戻ったら必ず描き直す。
        now = float(self.data.time)
        last = getattr(self, "_vision_t", None)
        cached = getattr(self, "_vision_cache", None)
        if cached is not None and last is not None and 0.0 <= now - last < VISION_MIN_DT:
            self.vision.sensor_outputs = cached
            return cached

        imgs = {}
        for cam_name, p in self.vision_params.items():
            wh = (p["width"], p["height"])
            if wh not in cache:
                cache[wh] = mujoco.Renderer(self.model, height=p["height"], width=p["width"])
            ren = cache[wh]
            cid = self.model.camera(cam_name).id
            if "fovy" in p:      # 視野角をモデルへ反映（vision_paramsのfovyは描画に未反映＝バグ）
                self.model.cam_fovy[cid] = p["fovy"]
            mjcam = mujoco.MjvCamera()
            mjcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            mjcam.fixedcamid = cid
            ren.update_scene(self.data, camera=mjcam)
            # ★重大なバグ修正（2026-07-20、配線チェックで発覚）：
            #   `Renderer.render()` は**内部バッファへの参照**を返す。コピーしないと
            #   左目・右目で同じレンダラを使い回している以上、**右目を描いた瞬間に左目の
            #   配列も右目の画像に書き換わる**＝太郎は両目とも右目の画を見ていた。
            #   さらに、以前に取り出しておいた画像も後の描画で書き換わるため、
            #   「前フレームと比べる」たぐいの処理がすべて壊れる（実際、配線チェックで
            #   「頭を回しても視覚が変わらない」という不可解な結果が出て発覚した）。
            img = ren.render().copy()
            if self.vision is not None:      # 視力(acuity)・中心窩(foveation)の後処理は本家を再利用
                af = getattr(self.vision, "_acuity_functions", {}).get(cam_name)
                if af is not None:
                    img = self.vision._apply_acuity(img, cam_name)
                fov = getattr(self.vision, "_foveation", {}).get(cam_name)
                if fov:
                    img = self.vision._apply_foveation(img, fov)
            imgs[cam_name] = img
        self.vision.sensor_outputs = imgs
        self._vision_cache = imgs
        self._vision_t = now
        return imgs

    def _update_glow(self):
        """体が触れたら GLOW_HOLD_S 秒のあいだ、おもちゃの色を明るく変える（＝光る）。

        随伴性を**視覚という独立チャネル**に出すための仕掛け。固有感覚だと621次元に
        薄まって消えることが実測で分かったため（上の定数のコメント参照）。
        床(world)との接触は除く＝**自分の体が触れたときだけ**光る＝随伴性が明確。
        接触は一瞬で終わるので、余韻を持たせないと視覚(10Hz)に一度も映らずに消える。
        """
        if not self._toy:
            return
        if any(c != "world" for c in self.toy_contacts()):
            self._glow_until = float(self.data.time) + GLOW_HOLD_S
        lit = float(self.data.time) < getattr(self, "_glow_until", -1e9)
        if lit != self.toy_lit:
            self.model.geom_rgba[self._toy_gadr] = TOY_RGBA_ON if lit else TOY_RGBA_OFF
            self.toy_lit = lit

    def anchor_distance(self):
        """おもちゃが吊り下げ基準点からどれだけずれているか[m]（押された量の目安）。"""
        if self._anchor is None:
            return 0.0
        return float(np.linalg.norm(self.data.body("test_object1").xpos - self._anchor))
