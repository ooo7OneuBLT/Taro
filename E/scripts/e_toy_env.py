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
import sys

import numpy as np
import mujoco

from d_supine_env import SupineMimoEnv

# 【なぜ、2026-08-23】development.py（taro_core側・月齢→発達パラメータの一本化）を
#   import するための sys.path 追加。呼び出し元（e_scene.py等）は既に
#   taro_core/src/body を sys.path へ足しているが、e_toy_env.py が直接単体で
#   import されるスクリプトも多い（require_px 等を直接呼ぶ14ファイル）ため、
#   呼び出し側に依存せずこのファイル自身でも通す。
_HERE_DEV = os.path.dirname(os.path.abspath(__file__))
_ROOT_DEV = os.path.abspath(os.path.join(_HERE_DEV, os.pardir, os.pardir))
_DEV_PATH = os.path.join(_ROOT_DEV, "taro_core", "src", "body")
if _DEV_PATH not in sys.path:
    sys.path.insert(0, _DEV_PATH)
import development  # noqa: E402

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
# 注意：ただしこれで解決するのは配置ぶんだけ。**頭は平均50°振れる**ので、
#   空間に固定したおもちゃを見続けるには視野の半角が54°以上必要＝根本原因は別（要検討）。
# 目からおもちゃまでの距離[m]。E_TOY_DIST で上書きできる。
# 注意：旧値 0.086 は「旧オフセットの長さを維持した暫定値」で、根拠がなかった。
#   2026-07-26の実測（基準点を両目の中点に直したあと）：
#     距離[cm]   視線のズレ   めり込み
#        8.6        7.9度     13.65mm  ← 柵に当たる
#       12.0        5.8度      5.23mm
#       13.0        5.4度      1.37mm
#       15.0        4.7度      0.00mm  ← めり込まず、腕(18.6cm)で届く
#       18.0        3.9度      0.00mm  ← 腕の長さぎりぎり
TOY_DISTANCE = float(os.environ.get("E_TOY_DIST", "0.086"))
                           #   （新生児の適切な注視距離は文献調査中。決まり次第ここを更新）

# 【2026-08-03 追加】おもちゃの配置が、屈曲した腕（手・前腕）と重なる問題への対処。
#   仕様：作業記録（非公開）
#
# 【なぜ要るか】肘のlimb_tone目標角度を屈曲側（-130度）にする姿勢バイアス版シーンで、
#   肘が顔・胸の近くまで曲がった状態のまま「おもちゃを視線の正面へ再配置する」処理
#   （_set_anchor）を通すと、その位置がちょうど屈曲した手の位置と重なり、次の
#   mj_forward／stepで接触の反力によって腕が弾かれることが実測で判明した
#   （2秒沈めた時点 右肘-125.22度・左肘-123.96度 → 再配置0.3秒後 右肘-63.84度・
#   左肘-53.74度＝狙いと逆の伸展側。報告：作業記録（非公開）
#   報告\2026-08-03_姿勢バイアス版の実装.md 3節）。
#
# 【対処方針】狙った腕の角度（limb_toneの目標）を優先し、腕の角度そのものは動かさない。
#   代わりに、置こうとした位置が手・前腕と重なるときだけ、重ならなくなるまで
#   おもちゃ側を最小限ずらす（＝案B、ユーザー確認済み）。重ならない通常のシーンでは
#   この判定は常にFalseのまま何もせず、既存の配置と1ビットも変わらない。
#
# 【判定の仕方】geom単位の正確な衝突検出（mj_step相当）は行わず、MuJoCoが自動計算する
#   bounding sphere半径（geom_rbound、各geomを覆う最小の球の半径）を使った簡易判定。
#   「おもちゃの中心と、手・前腕の各geomの中心の距離」が「両者の半径の和＋マージン」
#   より小さければ重なっていると見なす。厳密な形状ではなく球で近似するぶん、
#   実際にはまだ余裕がある位置でもずらすことがあるが、安全側（もし重ならないと判定を
#   誤っても、それは「めり込んだまま」より安全）。
#
# 注意：[Tier3・工学的判断] マージン(5mm)・対象部位(手・前腕のみ、指の詳細geomは含まない
#   代わりに手bodyの全geomを見る)に文献根拠はない。doc/人間模倣からの逸脱リスト.md 参照。
TOY_COLLIDE_BODIES = ("right_hand", "left_hand", "right_lower_arm", "left_lower_arm")
TOY_COLLIDE_MARGIN = 0.005   # 5mm。geom表面のさらに外側に残す隙間[m]（恣意的）
TOY_COLLIDE_ITERS = 8        # ずらす処理の反復回数（1回のずらしで別のgeomに近づく場合があるため）
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

# ============================================================================
# リクライニング（体を起こす）— 2026-07-28 新設
# ----------------------------------------------------------------------------
# 【なぜ要るか】リーチングに進むにあたり、仰向けでは「見える位置」と「手が届く位置」が
# 両立しないことが実測で分かった：
#     おもちゃを視線の正面（目から8.6cm）に置くと、肩からは21.7cm。
#     4ヶ月の腕は16.8cmなので**届かない**。
#   仰向けではおもちゃが顔の真上に来るため、肩から遠くなるのが原因。
#
# 【人間の実験ではどうしているか（2026-07-28 調査）】
#   **体を起こして解決している**。完全な仰向け(0度)でリーチを取る研究は少数派。
#     Carvalho, Tudella & Savelsbergh (2007) Infant Behav Dev 30(1):26-35
#       4〜6ヶ月児。座位「ベビーチェア、水平から70度」と仰向け(0度)を比較し、
#       **座位の方がリーチの頻度・質ともに優れる**（特に4ヶ月児で顕著）。
#       5〜6ヶ月で差が消える＝姿勢制御が育つと仰向けの不利をカバーできる。
#     Savelsbergh & van der Kamp (1994) J Exp Child Psychol 58(3):510-528
#       垂直90度／傾斜60度／仰向け0度。**姿勢を起こすことがリーチ成立の決定的要因**。
#       12〜19週児を座位にすると20〜27週児が仰向けで出す頻度に匹敵する。
#
# 【なぜ有利か（物理）】仰向けで手を伸ばすのは腕を**重力に逆らって持ち上げる**動作。
#   リクライニングなら前方へ出す動きになり、重力は伸ばす方向と直角に近くなる。
#
# 傾きの角度[度]。0=仰向け、90=直立。70度は Carvalho et al. 2007 のベビーチェアの値。
RECLINE_DEG = float(os.environ.get("E_RECLINE", "0"))
# 背もたれの寸法[m]。注意[Tier3] 乳児用バウンサーの寸法規格は調査で見つからなかった。
# 太郎の体（4ヶ月で身長約68cm）が乗る大きさとして決めた。
SEAT_HALF_LEN = 0.40       # 背もたれの長さの半分（体軸方向）
SEAT_HALF_WID = 0.22       # 背もたれの幅の半分（左右）
SEAT_THICK = 0.02          # 板の厚み
SEAT_RGBA = np.array([0.62, 0.55, 0.50, 1.0])
# 背もたれ・座面の摩擦。ずり落ちを防ぐ。
#
# 【なぜ要るか】実測（`e_recline_check.py`）で45度・3秒間に **1.56cm ずり落ちた**。
# 実験は15秒〜数分なので、そのままでは姿勢が保てない（ユーザーの指摘、2026-07-28）。
#
# 【人間ではどうか】乳児用のバウンサー・ベビーチェアは
#   ・布／ウレタンの表面（滑りにくい）
#   ・**股ベルトで固定する**（安全基準で義務づけられている）
# の2つでずり落ちを防いでいる。摩擦だけで足りなければベルトに相当する拘束を足す。
#
# 注意：[Tier3] 布と皮膚の摩擦係数の文献値は持っていない。MuJoCo の既定は 1.0。
#   まず 2.0（＝滑りにくい布）で試し、足りなければ上げる／ベルトを足す。
SEAT_FRICTION = float(os.environ.get("E_SEAT_FRICTION", "2.0"))
# 視覚的な「豊かさ」の切替。E_PLAIN=1(既定)＝床の市松模様を消し柵を床と同色に＝**見えないnest**。
# 根拠と意図は _make_visually_plain() のdocstring参照（White 1966 と Ferrari 2007 の両立）。
# 【色の統一】床・柵・空を**同じ色**にする＝どこを向いても同じ＝最も「貧しい」視界。
# 目視で判明した2点：①柵が茶色になっていた（MjSpecのデフォルト材質 matgeom の継承。下記で修正）
# ②見上げると空が水色（skyboxのグラデーション rgb1="0.3 0.5 0.7"）で、灰色の柵とコントラストが出る。
# → 空の色に合わせて全部を淡い青灰色にする。注意色の選択自体に文献の根拠はない（**恣意的**）が、
#   「面と面の境界が見えないほどコントラストが小さい」ことが目的なので、値そのものは重要でない。
PLAIN_RGBA = np.array([0.55, 0.62, 0.70, 1.0])       # 床・柵・空に共通で使う淡い青灰色
# 2026-07-27：床だけ明るさを落とせるようにした（E_FLOOR_DIM）。
# 【なぜ】床・柵・空を同じ色にしても、**床には照明が当たり skybox には当たらない**ため、
#   目に映る像では床が明るく背景が暗い＝画面を横切る強いコントラストの境界が残る。
#   その境界が眼球運動で上下に動くと、画面の全幅にわたる強い動き信号になり、
#   小さな対象を総量で圧倒して定位を壊していた（実測：対象の実際のずれ +0.44 に対し
#   反射の出す向きが +0.03）。
# 【逸脱ではない理由】乳児の視野測定は無地・低コントラストの背景で行う
#   （Mohn & van Hof-van Duin は黒い弧に白い球）。境界を消すことは
#   **実験条件を乳児研究の標準に合わせる**ことであり、太郎の中身をいじる話ではない。
#   注意：ただし現実の乳児の視野にも床と壁の境界はあるので、環境としては簡略化。
FLOOR_DIM = float(os.environ.get("E_FLOOR_DIM", "1.0"))
# 2026-08-21：床の「横縞」対策（F1-4h・語彙テストシーン調査）。
# 【原因】上のFLOOR_DIMコメントの通り床には照明が当たりskyboxには当たらない。
#   床はmaterial無し（geom.rgba直指定）でレンダリングされるため、Lambertian陰影
#   （拡散反射＝法線と光源方向のなす角のcos）がそのままかかる。床は無限平面
#   （benchmarkv2_scene.xml:40 の <geom type="plane" size="0 0 .25">）なので、
#   地平線に近い（＝光源から見て入射角が直角に近い）画素ほどcosがゼロに近づき
#   黒に落ちる。実測（scratchpad probe, 2026-08-21）：128x128視界の行78〜90が
#   平均輝度0.075〜0.10（他行0.6前後）まで落ちる暗帯として現れた。
# 【対処】床にmaterialを与え、emission（自己発光）を上げて陰影の影響を弱める。
#   emission=0（既定）なら旧来どおり material="" のままでバイト単位不変。
FLOOR_EMISSION = float(os.environ.get("E_FLOOR_EMISSION", "0.0"))
FENCE_RGBA_RICH = np.array([0.35, 0.45, 0.85, 1.0])  # 豊かな条件での柵＝青（従来の色）
# 2026-07-26：5cm角 → 4cm角へ。ユーザーが姿勢編集パネル（`e_pose_editor.py`）で
#   Viewer を見ながら調整した値（`E/docs/pose_editor_saved.json`）。
#   注意：[Tier3・ARBITRARY] 新生児のおもちゃの適切な大きさの文献値は未調査。
#   「握れる／視界で見える／体に当たりすぎない」を目視で満たす値として選んだ。
TOY_RADIUS = float(os.environ.get("E_TOY_RADIUS", "0.020"))
# 箱の half-size[m]＝4cm角。新生児が握れる大きさ
TOY_MASS = 0.0154          # 15.4g。密度を保ったまま4cm角にした質量（元は5cm角で30g）
TOY_DENSITY = TOY_MASS / (2 * 0.020) ** 3    # 240.6 kg/m³。形を変えても密度は保つ

# 2026-07-27：おもちゃの形を選べるようにした（既定は従来どおり箱）。
# 【なぜ】定位反射の測定で、**立方体だと動き検出の重心が中心へ寄る**ことが分かった。
#   おもちゃを視野の端に置くと、透視投影のせいで「中心を向いた側面」が見える。
#   側面は暗く正面は赤いので、その境目の明暗差が大きい。一方、外側の縁は
#   灰色の背景との境目で明暗差が小さい。結果、揺らしたときの動き信号が
#   **中心側の縁に集中**する（実測：視野の端に置くと 96:4 まで偏った）。
#   → 重心が中心へ引っ張られ、定位の向きが正しく出ない。
#   球ならどの向きから見ても見え方が同じで、この非対称が原理的に生じない。
# 【新生児の実験との対応】文献の定位実験で使う視標は平らなカードや小さな図形で、
#   こうした側面は生じない（Aslin & Salapatek 1975、Hunter & Richards 2003 など）。
#   注意：視標の形・大きさの文献値そのものは未確認 [Tier3・ARBITRARY]。
# E_TOY_SHAPE=sphere で球、E_TOY_RADIUS で大きさ[m]を変えられる。
TOY_SHAPE = os.environ.get("E_TOY_SHAPE", "box")   # box / sphere / cylinder / ellipsoid

# 【2026-08-24追加・目標F・F2】語彙4語化のため、見分けられる形をbox/sphereの2種類から
#   4種類（box/sphere/cylinder/ellipsoid）へ増やす。
#   大きさの決め方 [Tier3・ARBITRARY]：文献値は無く、box/sphere と見た目の大きさが
#   揃うように目視の代わりに寸法だけを合わせた。
#   cylinder: 半径=TOY_RADIUS相当、半分の長さも同じ値（＝全長が直径と同じ）にして、
#     box（一辺=2*TOY_RADIUS）・sphere（直径=2*TOY_RADIUS）と全体の見かけの大きさを揃えた。
#   ellipsoid: 半径(a,b,c)のうち1軸だけ TOY_ELLIPSOID_ELONGATION 倍に伸ばし「たまご型」にする。
#     倍率1.4は「卵型と分かる程度に非対称だが、他の2形状と全体サイズが極端に違わない」を
#     目視の代わりに数値で決めただけの恣意値（根拠文献なし）。
TOY_CYLINDER_HALF_LEN_RATIO = 1.0
TOY_ELLIPSOID_ELONGATION = 1.4
# 摩擦[slide, spin, roll]。**転がり続けを止めるために roll/spin を既定より上げる**。
# 理由＝実測で「手が遠いのにおもちゃが動く(0.342mm/tick)」＝一度押されると転がり続け、
# 「今の自分の運動」と無関係に動いて**随伴性(自分の行為→結果)が濁る**ため。
# 押した分だけ動いてすぐ止まる＝カーペット上のおもちゃに相当し、随伴が明確になる。
TOY_FRICTION = np.array([1.0, 0.05, 0.02])

# 太郎の体を空間に置いている body の名前（＝自由関節がぶら下がっている body）。
#
# 注意：【2026-07-29 に踏んだ】このモデルには自由関節が **3つ** ある：
#     test_object1   おもちゃ
#     test_object2   使っていない予備の物体（遠く (3.5, 3.0, 0.05) に置いてある）
#     mimo_location  太郎の体（関節名は "mimo_orientation" で body 名と違う）
#   「おもちゃ以外の自由関節」という選び方をすると **test_object2 を掴む**。
#   実際、体を留めるつもりで test_object2 を留め、その間ずっと太郎は椅子から
#   転がり落ちていた。Viewer の「仰向けに戻す」も同じ選び方をしていた。
#   ＝落とし穴チェックリスト 項67「体を動かす自由関節は、思っている body に無い」。
#   ⇒ 名指しで取る。見つからなければ**別のもので代用しない**。
ROOT_BODY = "mimo_location"

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
#   注意：2.0秒という長さに文献の裏付けはない＝**恣意的**（制御周期1秒を確実にまたぐ長さとして選択）。
TOY_RGBA_OFF = np.array([0.9, 0.2, 0.15, 1.0])   # 通常＝赤
TOY_RGBA_ON = np.array([1.0, 1.0, 0.45, 1.0])    # 点灯中＝明るい黄（視界で目立つ）
GLOW_HOLD_S = 2.0                                 # 点灯の持続[sim秒]（注意恣意的）

# --- 2個目のおもちゃ（目標F・F1：語↔物の対応づけ判定）------------------------
# 【なぜ】前に左右2個（例：赤い箱・白い球）を同時に置き、語を聞いてどちらを見るかを
#   測る実験に使う。既存の test_object1（箱）はそのまま toy1 として使い、これまで
#   ずっと遠方(3.5,3.0,0.05)へ退避するだけだった test_object2（球）を toy2 に転用する
#   （MIMo共有XMLは一切汚さない、既存方針の踏襲）。
# 【新しい環境変数は作らない】2026-08-18の依頼どおり、配線はステージA方式
#   （ToySupineEnvのkwargsで直接渡す）に統一する。toy2=None（既定）なら
#   test_object2は従来どおり退避されるだけ＝1ビットも挙動が変わらない。
TOY2_RGBA_DEFAULT = np.array([0.9, 0.9, 0.9, 1.0])   # toy2の既定色＝白系（rgba未指定時）
# 視線の正面から toy1/toy2 を振り分ける角度[度]（片側あたり）。
# 【なぜ12度か】視野の半角30度に対し余裕を持って両方を収める値として選んだ
# （±12度なら視野中心から12度、半角30度の40%）。文献値ではない[Tier3・ARBITRARY]。
TOY_ANGLE_DEG_DEFAULT = 12.0
# 【2026-08-25新設・目標F・4語テスト】toy3/toy4の既定色・既定角度。
#   4語テストは「1個ずつ提示して見比べる」（f_wordreadout_nway.pyがrgbaのアルファを
#   0にして透明化する）ので、既定角度は0度＝toy1と同じ方向・同じ距離で構わない
#   （4つが重なっていても、同時に見せることは無いため実害が無い。仕様で明示的に許容）。
TOY3_RGBA_DEFAULT = np.array([0.2, 0.4, 0.9, 1.0])   # toy3の既定色＝青系
TOY4_RGBA_DEFAULT = np.array([0.2, 0.8, 0.3, 1.0])   # toy4の既定色＝緑系
TOY34_ANGLE_DEG_DEFAULT = 0.0

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
#   注意：これは計算の最適化であって発達的な主張ではない（乳児の視覚時間分解能を模したものではない）。
VISION_MIN_DT = 0.1


ACUITY_AGE = 0.5   # 視力テーブルに渡す月齢。下記のとおり 0.0 は**無効**になるので使えない

# 【F1-7・2026-08-22】中心窩カメラの視野[度]。中心窩は既定OFF（body.fovea_camera=False）。
FOVEA_FOVY = 15.0


def _acuity_cpd(age_months):
    """月齢から視力[cycles/度]を求める（Mayer et al. 1995 の実測テーブル・線形補間）。

    【2026-08-23・二重実装の解消】以前はここに表と補間ロジックを直接複製していた
    （F1-7・2026-08-22時点のコメント参照）。今回、月齢→発達パラメータを一本化する
    `taro_core/src/body/development.py` を新設したのに合わせ、canonicalな置き場所を
    そちらへ移し、ここは委譲するだけにした（表の値・補間ロジックは1文字も変えていない。
    出典・複製の経緯は development.py 側のコメントに引き継いだ）。
    """
    return development.vision_acuity_cpd(age_months)


def required_px(age_months, fovy_deg):
    """視力フィルタが効く上限まで画素を用意するのに必要な解像度[px]（F1-7新設）。

    仕様：`F/docs/仕様_F1-7_中心窩カメラと月齢からの解像度自動化.md` 第2部(2)節。
    【2026-08-23】実体は `development.vision_required_px` へ委譲（二重実装の解消）。
    検算：6.0ヶ月・15度 → acuity=5.642cpd → px=194.6 → 208px（仕様書と一致・確認済み）。
    """
    return development.vision_required_px(age_months, fovy_deg)


def _check_resolution_acuity_mismatch(age_months, fovy_deg, px):
    """解像度が月齢の視力より粗くないかを機械的に検査する（F1-7・2026-08-22追加、ユーザー指摘）。

    【なぜ、2026-08-22】`fovea_camera` は既定OFF。「別の実験でONにし忘れて、気づかない
    まま粗い目で走らせる」事故は、このプロジェクトで環境変数スイッチのつけ忘れとして
    既に複数回起きている（CLAUDE.md「同じミスが2回起きたら文書ではなく機械で防ぐ」）。
    月齢が要求するacuity[cpd]と、実際にエンコーダへ渡る画像のナイキスト限界[cpd]
    （=画素数/視野角/2）を比べ、不足していれば警告を出す。**エラーにはしない**
    （過去実験の再現・意図的に粗い目で走らせる実験を壊さないため。仕様書の指定）。
    """
    acuity_needed = _acuity_cpd(float(age_months))
    px_per_deg = float(px) / float(fovy_deg)
    nyquist = px_per_deg / 2.0
    if nyquist < acuity_needed * 0.9:
        print("=" * 70)
        print("[警告][F1-7] 解像度が視力(acuity)より粗い可能性があります")
        print(f"  月齢 {float(age_months):g}ヶ月 が要求する視力 = {acuity_needed:.3f} cycles/度")
        print(f"  実際の解像度 = {px}px / 視野{float(fovy_deg):g}度 = "
              f"{px_per_deg:.2f}画素/度（ナイキスト限界 {nyquist:.3f} cycles/度）")
        print("  対処：シーンの body.fovea_camera を true にすると解消します"
              "（視野15度・月齢から自動計算した解像度の中心窩カメラが追加され、"
              "そちらがエンコーダへ渡ります）")
        print("=" * 70)


def infant_vision_params(size=VISION_RES, fovy=VISION_FOVY, acuity_age=ACUITY_AGE,
                          fovea_camera=False, develop_from_age=False,
                          fovea_fovy=None, acuity_filter=True):
    """新生児の視覚パラメータ。acuityに月齢を渡す（＝解像度を恣意的に決めない）。

    注意：【2026-07-20 修正・重大】以前は `acuity_age=0.0` を渡しており、**視力フィルタが
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
    注意：**残る逸脱**：Mayer et al.(1995) のテーブルは**1ヶ月から**しかない。新生児(0ヶ月)の実測値は
    このテーブルに存在しないため、太郎は「**1ヶ月児の視力**」で代用している。0.852 cycles/deg は
    スネレン換算でおよそ 20/700 相当＝成人(20/20)の約1/35。

    fovea_camera（F1-7新設・既定False）：
        False（既定・キー無しシーンも常にこちら）のときは**従来と1ビットも変わらない**
        （`eye_left`/`eye_right` の2台のみを返す）。
        True のときだけ `eye_left_fovea`/`eye_right_fovea` を追加で返す。視野は
        `FOVEA_FOVY`(=15度)固定、解像度は `required_px(acuity_age, FOVEA_FOVY)` で
        月齢から自動計算（`VISION_RES`のような固定値は使わない）。acuityは周辺と
        同じ `acuity_age` をそのまま使う（同じ月齢の目である以上、周辺と中心窩で
        視力テーブルを変える理由が無いため）。

    develop_from_age（2026-08-23新設・既定False）：
        False（既定・キー無しシーンも常にこちら）のときは**従来と1ビットも変わらない**
        （周辺カメラの解像度は `size`（既定 `VISION_RES`=128固定）のまま）。
        True のときだけ周辺カメラ(`eye_left`/`eye_right`)の解像度も
        `development.vision_required_px(acuity_age, fovy)` で月齢から自動計算する
        （引数 `size` は無視される）。
        注意：周辺カメラは視野`fovy`（既定60度、中心窩の15度より広い）全体に
        acuityと同じナイキスト限界を要求する式を適用するため、月齢が上がるほど
        画素数が急増する（例：6ヶ月・60度で784px。中心窩(15度)の208pxの約3.8倍）。
        これは計算コストに直結するので、常時ONにする前に必ずコストを実測すること
        （このタスクでは配線のみ・コスト実測は行っていない）。
    """
    if develop_from_age:
        size = development.vision_required_px(acuity_age, fovy)
    # 【F2-18・2026-08-30】視力フィルタ（FFTで高周波を落とす処理）を外せるようにした。
    #   MIMo は camera_parameters[camera]["acuity"] が Falsy だとフィルタを作らない
    #   （vision.py L103 の if 判定）。0.0 を渡すことで無効にする。
    #   【なぜ外すか・実測 2026-08-30】基本図形20個体で、フィルタあり/なし・
    #   解像度336/208/112px の6条件すべてでカテゴリの分離が 0.521〜0.542 に収まり、
    #   差は誤差の範囲だった。中心窩では成分の98.4%を捨てるのに 7.19ms かけていた。
    #   人間の乳児では低視力に学習上の意味がある可能性があるが（粗いものから学ぶ）、
    #   太郎の視覚は DINOv2 で固定＝視覚の回路が育たないため、その効果は生じない。
    #   将来、実物スキャンなど細部のある素材に移るときは測り直すこと。
    _acuity = acuity_age if acuity_filter else 0.0
    eye = {"width": size, "height": size, "fovy": fovy,
           "acuity": _acuity, "foveation": False}
    params = {"eye_left": dict(eye), "eye_right": dict(eye)}
    if fovea_camera:
        # 【F2-18・2026-08-30】中心窩の視野をシーンから指定できるようにした
        #   （既定 None は従来の FOVEA_FOVY=15度＝1ビットも変わらない）。
        #   人間の中心窩(fovea)は視角およそ5度で、15度は perifovea（周辺窩）相当。
        #   「人間の視覚に近づける」方針（2026-08-30・ユーザー判断）のための入口。
        _ffovy = float(FOVEA_FOVY if fovea_fovy is None else fovea_fovy)
        fovea_size = required_px(acuity_age, _ffovy)   # 解像度は視力から決めたまま
        # 【F1-7フォロー・2026-08-22】中心窩カメラは208px全体がそのままエンコーダへ
        #   渡る（fovea_crop をかけない）ため、視力フィルタ(FFT)の周期境界による
        #   折り返し（画像の片端の物体が反対端に実体のない滲みとして出る現象。
        #   F/logs/F1-7_中心窩_2026-08-22/検証B_FFT折り返し確認.png で実測）が
        #   そのまま学習入力に混入する。周辺カメラ(eye_left/eye_right)は128px中央
        #   32pxしか使わず端から遠いため実害が無く、視線系（顕著性・サッケード・
        #   remapping）が凍結中で変更禁止のため pad_acuity は付けない。
        fovea_eye = {"width": fovea_size, "height": fovea_size, "fovy": _ffovy,
                     "acuity": _acuity, "foveation": False, "pad_acuity": True}
        params["eye_left_fovea"] = dict(fovea_eye)
        params["eye_right_fovea"] = dict(fovea_eye)
    # 【F1-7・2026-08-22】エンコーダが実際に使う画像（中心窩があればそちら、
    #   無ければ周辺）の解像度が、月齢の視力より粗くないかを機械的に検査する。
    #   run/trainer.py._vision_backend_encode() と同じ「fovea優先」の選び方。
    _active = params["eye_left_fovea"] if fovea_camera else params["eye_left"]
    _check_resolution_acuity_mismatch(acuity_age, _active["fovy"], _active["width"])
    return params
FAR_AWAY = np.array([3.0, 3.0, 0.05])   # 使わない物体の退避先

# おもちゃの登場を遅らせる（ユーザーの提案 2026-07-26）
#
# 【なぜ】リセット直後の太郎は落ち着いていない。実測：
#   ・おもちゃが規定位置で頭に 15.6mm・右目に 13.4mm めり込み、
#     拘束反力 1461+502 Nm が発生する（首の筋力 0.066Nm の3万倍）
#   ・その反力で 0.4秒のうちに首が 64度回り、視線がおもちゃから 100度ずれる
#   ・関節角も屈筋トーンの目標と最大56度ずれた状態から始まり、バネで引かれて動く
#   ・首が落ち着くのは 2.4秒あたり
#   → 視線に関わる実験が何ひとつ成立しなかった（研究日誌 2026-07-26 続き7）
#
# 【対処】おもちゃを最初は遠くに置き、太郎が落ち着いてから**親が運んでくる**。
#   瞬間移動させないのは既存の方針と同じ（ワープは随伴性の学習を壊す。_apply_tether 参照）。
#   人間の場面としても、親がおもちゃを見せるのは赤ちゃんが落ち着いてからで自然。
#
# 注意：[Tier3・ARBITRARY] 秒数に文献の裏付けはない。実測（落ち着くまで2.4秒）と
#   「人が手を動かす速さ」から置いている。E_TOY_DELAY / E_TOY_APPROACH で変えられる。
TOY_APPEAR_DELAY = float(os.environ.get("E_TOY_DELAY", "1.0"))    # 何秒待つか
TOY_APPROACH_SEC = float(os.environ.get("E_TOY_APPROACH", "0.5"))  # 運ぶのにかける時間
TOY_APPROACH_DIST = 0.15   # どこから運び始めるか＝規定位置からの距離[m]
#   0.15m を 0.5秒 ＝ 0.3 m/s。人が手でおもちゃを差し出す速さとして妥当な範囲。
#
# どの向きから運んでくるか（ユーザーの目視 2026-07-26「おもちゃが柵に引っかかって」）
#   "above" 規定位置の**真上**から降ろす ← 既定。柵は側面にあるので上からなら通れる。
#           親が柵（ベビーベッドの柵）越しに手を入れるのも上からで、場面として自然。
#   "gaze"  視線方向の先から近づける（最初の実装）。注意柵を突き抜ける経路になる。
# 注意：私の測定はおもちゃが**到着した後**しか集計しておらず、運搬中に柵へ
#   引っかかるのを見落としていた。ユーザーの目視で発覚（落とし穴 項1・項6）。
TOY_APPROACH_FROM = os.environ.get("E_TOY_FROM", "above")

# おもちゃの持たせ方（ユーザーの提案 2026-07-26「一回紐みたいなのやめたら」）
#   "hold"   親が手に持っている＝位置を固定する。掴んでも動かない。
#            注意：新生児におもちゃを見せるのは親が手に持ってが普通なので、場面として自然。
#            力の要因が減るので**切り分けに向く**。視線誘導反射のテストはこれで行う。
#   "tether" ベビージムに吊るす（従来）。押せば動き、離せば戻る。
#            掴んで動かす段階（リーチング）で随伴性を学ばせるための仕組み。
#   "free"   何もしない＝重力で落ちる。仰向けの太郎からは見えなくなる（比較用）。
TOY_MODE = os.environ.get("E_TOY_MODE", "hold")

# --- 親の介入（見失ったら差し出し直す）------------------------------------
# E_PARENT=1 でON。定位の実験では、標的は実験者が乳児の視野へ導入する
# （Aslin & Salapatek 1975 の introduced target）。太郎が対象を視野の外へ
# 追いやったまま何も起きなくなるのを防ぐ。
# 注意：[ARBITRARY] 秒数・角度に文献値は無い。
PARENT_INTERVENE = os.environ.get("E_PARENT", "0") == "1"
PARENT_WAIT_SEC = float(os.environ.get("E_PARENT_WAIT", "2.0"))
PARENT_LOST_DEG = float(os.environ.get("E_PARENT_LOST", "25.0"))


def _box_inertia(mass, half):
    """一様な立方体(half-size=half)の慣性モーメント。size変更時に手で入れ直すため。"""
    i = mass * (2 * half) ** 2 / 6.0
    return np.array([i, i, i])


def _sphere_inertia(mass, radius):
    """一様な球(半径=radius)の慣性モーメント I = (2/5)mr²。"""
    i = 0.4 * mass * radius ** 2
    return np.array([i, i, i])


def _cylinder_inertia(mass, radius, half_length):
    """一様な円柱（半径=radius, 半分の長さ=half_length, 軸はローカルz）の慣性モーメント。

    【2026-08-24追加】MuJoCoのmjGEOM_CYLINDERはローカルz軸まわりに対称。
    I_axial(z) = (1/2) m r²、I_perp(x,y) = (1/12) m (3r² + L²)（L=全長=2*half_length）。
    標準の剛体力学公式（一様密度の直円柱）。
    """
    length = 2.0 * half_length
    i_axial = 0.5 * mass * radius ** 2
    i_perp = mass * (3.0 * radius ** 2 + length ** 2) / 12.0
    return np.array([i_perp, i_perp, i_axial])


def _ellipsoid_inertia(mass, a, b, c):
    """一様な楕円体（半径a,b,c）の慣性モーメント I_xx=(1/5)m(b²+c²) 等。

    【2026-08-24追加】標準の剛体力学公式（一様密度の楕円体）。a=b=c なら球の式に一致する
    （_sphere_inertiaの検算に使える）。
    """
    ix = 0.2 * mass * (b ** 2 + c ** 2)
    iy = 0.2 * mass * (a ** 2 + c ** 2)
    iz = 0.2 * mass * (a ** 2 + b ** 2)
    return np.array([ix, iy, iz])


# ============================================================================
# 新生児体型 v2（2026-07-21 確定）
# ----------------------------------------------------------------------------
# mimoGrowth の age=0 は「大きさは新生児だが四肢の比率が成人」だった（逸脱リスト
# 2026-07-20 その4）。加えて頭が球なので、頭囲34cmでも真上から見た頭が15%小さい。
# → ①頭を体軸方向に楕円化（頭囲を保ったまま見かけを人間に）②四肢・手足を縮小。
# 【確定の根拠】主にViewerでの目視（ユーザー＋育児経験者）。四肢を縮める向きと大きさは、
#   新生児は頭でっかちで四肢が短いという観察に一致。注意各係数そのものは目視で決めた恣意値
#   （文献で裏取りしたのは足長7.58cm・下肢19.6cm・上肢20.96cmだが、後2つは測定定義が
#    成人的で新生児に不整合と判明＝当てにしない）。詳細は逸脱リスト 2026-07-21。
# 【確定寸法】身長37.0cm 頭囲34.0cm（不変） 頭12.55 胴14.65 腕12.16 脚13.02 足7.5 手2.4cm
#   頭/身長 0.339（人間0.25より頭でっかち＝見た目重視の選択）。
#   注意：身長37cmは新生児49.9cmの74%＝**絶対サイズは小さい**。比率（見た目）を優先した結果。
# 各値: グループ名 -> (環境変数, 既定係数)
# 【2026-07-25 core へ移設】体型そのものの定義（係数・変換ロジック）は
# `taro_core/src/body/infant_body.py` へ移した。ここは「実験でどう振るか」だけを扱う
# `e_body_config.py` を経由して使う。
#   なぜ：体型は**太郎そのもの**であって環境の性質ではない。この環境クラスの中にあったため、
#   おもちゃ環境でしか補正が効かず、学習に使う仰向け環境では**成人プロポーションのまま**
#   学習していた（2026-07-25 に発覚）。方針は [[feedback-core-vs-experiment-placement]]。
from e_body_config import (NEWBORN_SHAPE_ENV, body_scale_custom_from_env,  # noqa: E402
                           head_elongation_from_env)


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

    def __init__(self, toy=None, toy_side="right", toy_offset=None,
                 toy_dist=TOY_DISTANCE,
                 toy_radius=TOY_RADIUS, toy_mass=TOY_MASS,
                 tether_length=TETHER_LENGTH, tether_k=TETHER_K_TAUT,
                 tether_c=TETHER_C,
                 fence=None, fence_half_x=FENCE_HALF_X, fence_half_y=FENCE_HALF_Y,
                 fence_post_w=FENCE_POST_W, fence_post_t=FENCE_POST_T,
                 fence_n_long=FENCE_N_LONG, fence_n_short=FENCE_N_SHORT,
                 fence_height=FENCE_HEIGHT, newborn_neck=None, newborn_limbs=None,
                 vor=None, orient=None, recline_deg=None,
                 # 【2026-08-17・配線改修ステージA】以前はここに引数が無く、
                 #   モジュール定数（SEAT_FRICTION等）をimport後に直接上書きするしか
                 #   設定を伝える方法が無かった（run/scene_tools/e_scene.py 参照）。
                 #   None なら従来どおりモジュール定数（＝環境変数の既定値）にフォールバック
                 #   するので、これらを渡さない既存の呼び出し元は1ビットも挙動が変わらない。
                 seat_friction=None, floor_dim=None, floor_emission=None,
                 toy_shape=None, toy_mode=None,
                 toy_rgba=None,
                 # 【2026-08-18・目標F・F1】2個目のおもちゃ。toy2=None（既定）なら
                 #   test_object2は従来どおり遠方退避のみ＝1ビットも挙動が変わらない。
                 toy2=None, toy2_shape=None, toy2_radius=None, toy2_rgba=None,
                 toy2_dist=None, toy_angle_deg=None,
                 # 【2026-08-25新設・目標F・4語テスト】3個目・4個目のおもちゃ。
                 #   toy2と同じ流儀（kwargs直渡し、既定None→False、1ビットも
                 #   既存の挙動を変えない）。ただし test_object3/test_object4 は
                 #   共有XML（benchmarkv2_scene.xml）には存在しない（F専用の新XML
                 #   にだけある）。toy3/toy4=True なのに body が無いシーンでは
                 #   __init__ 内で例外にする（黙って無視しない。落とし穴チェック
                 #   リスト「設定が静かに無視される」対策）。
                 toy3=None, toy3_shape=None, toy3_radius=None, toy3_rgba=None,
                 toy3_dist=None, toy3_angle_deg=None, toy3_elev_deg=None,
                 toy4=None, toy4_shape=None, toy4_radius=None, toy4_rgba=None,
                 toy4_dist=None, toy4_angle_deg=None, toy4_elev_deg=None,
                 # 【2026-08-21新設・F1-4h】おもちゃの垂直方向の角度[度]。既定None→0.0。
                 #   壁の無地部分を背にする高さへ上げるためのテスト専用パラメータ。
                 #   _set_anchor参照。既定0.0はsin(0)=0で従来位置と完全一致する。
                 toy_elev_deg=None, toy2_elev_deg=None,
                 # 【10択・2026-08-31・設計_視覚とGRUの統合.md V1b】差し出し専用の
                 #   軽量スロット。値はXML内のbody名のリスト（例：["test_object5",...]）。
                 #   保持・アンカー・形状設定は持たない（常に退避・親が見せる瞬間だけ
                 #   位置と向きを書かれる）。None（既定）なら1ビットも挙動が変わらない。
                 present_slots=None,
                 toy_appear_delay=None, toy_approach_sec=None, toy_approach_from=None,
                 parent_intervene=None, parent_wait_sec=None, parent_lost_deg=None,
                 plain=None, static_tex=None, orient_v=None, orienting_hold=None,
                 orienting_static=None, orienting_ior=None, orienting_habituation=None,
                 # 【2026-08-18新設・F1-3】親のfollow-in labeling。既定None→
                 #   ParentLabeling(enabled=False)相当になり1ビットも挙動が変わらない。
                 #   `world.parent_labeling`（辞書）をkwargs直渡しする（環境変数は新設しない）。
                 parent_labeling=None,
                 # 【2026-08-21新設・F1-4h】語の再生装置。既定None→
                 #   WordSchedule(schedule=None)相当になり1ビットも挙動が変わらない。
                 #   `world.word_test`（辞書）をkwargs直渡しする。
                 word_test=None,
                 eye_rest_vertical_deg=None, eye_centering=None,
                 eye_muscle_scale=None,
                 # F2-15新設・輻輳反射。既定None→False相当（E_VERGENCE=0）で
                 #   1ビットも既存の挙動を変えない（e_vergence.py参照）。
                 vergence=None, **kwargs):
        # VOR（前庭動眼反射）。眼球を方策から切り離し、頭の動きを打ち消して視線を安定させる。
        # E_VOR=0 でOFF（アブレーション）。根拠と簡略化は e_vor.py 参照。
        if vor is None:
            vor = os.environ.get("E_VOR", "1") == "1"
        self._use_vor = vor
        self._vor = None
        # 視線誘導反射（動きの大きい方向に首・目が向く）。設計確定・未検証のため既定OFF。
        # E_ORIENT=1 でON。根拠と簡略化は e_orienting.py 参照。
        if orient is None:
            orient = os.environ.get("E_ORIENT", "0") == "1"
        self._use_orient = orient
        self._orienting = None
        # F2-15新設：輻輳反射（両眼視差から目をどれだけ寄せるかを決める）。
        # E_VERGENCE=1 でON。既定OFF＝インスタンスも作らない（視差計算のコストも払わない）。
        # 根拠と簡略化は e_vergence.py 参照。
        if vergence is None:
            vergence = os.environ.get("E_VERGENCE", "0") == "1"
        self._use_vergence = vergence
        self._vergence = None
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
        # 【2026-08-17・ステージB2】以前は toy/fence の既定値が True で、環境変数が
        #   設定されていれば**明示引数より常に環境変数が勝つ**構造だった（他の設定は
        #   逆に「引数が渡されたら必ずそれが勝つ」構造なので、ここだけ食い違っていた）。
        #   全数調査：`E/scripts/e_smoke_all.py` が全診断スクリプトへ E_TOY_OBJ=0 を
        #   一律注入しており、その中の `e_toy_check.py`（おもちゃとの接触・随伴性を
        #   確認する専用スクリプト）が `toy=True` を明示していたのに、旧ロジックでは
        #   環境変数がそれを踏みつぶし、**おもちゃ無しで「おもちゃの検査」を空振り
        #   していた**（実際に確認した。toyがFAR_AWAYへ退避され、接触・随伴性の
        #   チェックが常に「対象なし」で終わる）。他の入口（run/scene_tools/e_scene.py
        #   のbuild()など）と同じ「Noneのときだけ環境変数を見る」形に統一する。
        #   toy/fence を明示しない既存の呼び出し元は、既定値がTrueからNoneに変わっても
        #   最終的に同じ式（環境変数が"1"ならON）に帰着するため1ビットも変わらない。
        if toy is None:
            toy = os.environ.get("E_TOY_OBJ", "1") == "1"
        if fence is None:
            fence = os.environ.get("E_FENCE", "1") == "1"
        # E_PLAIN=1（既定）＝視覚的に貧しくする。E_PLAIN=0 で従来の見た目（市松床・青い柵）。
        # 注意：_edit_spec は super().__init__() の中で呼ばれるので super() より前に代入する。
        # 【2026-08-17】plain引数を新設。None なら従来どおり環境変数（既定は貧しい＝1）。
        if plain is None:
            plain = os.environ.get("E_PLAIN", "1") == "1"
        self._plain = bool(plain)
        # 【目標E フェーズ1】E_STATIC_TEX=1 で「動かない模様のあるもの」を視界に足す：
        #   ①模様(市松)付きの天井を太郎の真上に追加、②柵にも同じ市松テクスチャを貼る。
        # 狙い＝「自分の首を動かす→見え方が変わる」の学習教材を作る（おもちゃ=動くもの と違い、
        # これは動かないので"自分の動きだけで視覚が変わる"という条件を壊さない）。既定OFF。
        # 2026-07-21：フェーズ1で視界がのっぺりしすぎ(58%が変化なし・空か柵のみ)と目視で判明した対処。
        # 【2026-08-17】static_tex引数を新設。None なら従来どおり環境変数（既定OFF）。
        if static_tex is None:
            static_tex = os.environ.get("E_STATIC_TEX", "0") == "1"
        self._static_tex = bool(static_tex)
        # 身体の補正。**必ずON/OFFできるようにする**＝E1で「dampingが創発したのか、
        # 身体を弱めただけか」を切り分けるアブレーションに使う（これが無いと結果を解釈できない）。
        #   newborn_neck  : 首がすわっていない（head lag）の再現。注意恣意的（e_infant_neck.py）
        #   newborn_limbs : 四肢の発達の向きの逆転を解消。測定値ベース（e_infant_body.py）
        self._newborn_neck = newborn_neck
        self._newborn_limbs = newborn_limbs
        self._neck_age = kwargs.get("age", None)
        # 柵（柱）の設定。fence=False で従来どおり囲いなし＝アブレーション用。
        # 注意：_edit_spec は super().__init__() の中（モデル構築時）に呼ばれるので、
        #    これらの属性は super() より**前**に代入しておく必要がある。
        self._fence = fence
        # リクライニングの角度[度]。0なら従来どおり仰向け（背もたれも作らない）
        self._recline_deg = float(RECLINE_DEG if recline_deg is None else recline_deg)
        # 【2026-08-17・配線改修ステージA】以前はモジュール定数（SEAT_FRICTION等）を
        #   メソッド内で直接参照していた（run/scene_tools/e_scene.py がimport後に
        #   TE.SEAT_FRICTION=... と直接上書きする以外に伝える手段が無かった）。
        #   ここでインスタンス属性へ写し、以降のメソッドは self._xxx を見る形にする。
        #   None ならモジュール定数（＝環境変数の既定値）にフォールバックするので、
        #   これらを渡さない既存の呼び出し元は1ビットも挙動が変わらない。
        self._seat_friction = float(SEAT_FRICTION if seat_friction is None
                                    else seat_friction)
        self._floor_dim = float(FLOOR_DIM if floor_dim is None else floor_dim)
        self._floor_emission = float(FLOOR_EMISSION if floor_emission is None
                                      else floor_emission)
        self._toy_shape = str(TOY_SHAPE if toy_shape is None else toy_shape)
        self._toy_mode = str(TOY_MODE if toy_mode is None else toy_mode)
        self._toy_appear_delay = float(TOY_APPEAR_DELAY if toy_appear_delay is None
                                       else toy_appear_delay)
        self._toy_approach_sec = float(TOY_APPROACH_SEC if toy_approach_sec is None
                                       else toy_approach_sec)
        self._toy_approach_from = str(TOY_APPROACH_FROM if toy_approach_from is None
                                      else toy_approach_from)
        self._parent_intervene_flag = (PARENT_INTERVENE if parent_intervene is None
                                       else bool(parent_intervene))
        self._parent_wait_sec = float(PARENT_WAIT_SEC if parent_wait_sec is None
                                      else parent_wait_sec)
        self._parent_lost_deg = float(PARENT_LOST_DEG if parent_lost_deg is None
                                      else parent_lost_deg)
        # 眼球の基準角・中央復帰バネ（2026-08-17新設）。reset_model() から
        #   infant_body.center_eyes / apply_eye_centering_spring へ明示的に渡す
        #   （infant_bodyのモジュール定数EYE_REST_VERTICAL_DEG/EYE_CENTERINGは
        #   Noneのときのフォールバック既定値に降格）。
        self._eye_rest_vertical_deg = (None if eye_rest_vertical_deg is None
                                       else float(eye_rest_vertical_deg))
        self._eye_centering = eye_centering    # None ならinfant_body側の既定に従う
        # 【2026-08-18新設・F1-3a】眼球の筋力補正（既定1.0＝無補正）。
        #   実体は infant_body.apply_eye_muscle_scale。四肢のlimb_scaleと同じ
        #   流儀（kwargs直渡し、環境変数は新設しない）。詳細はinfant_body.py参照。
        self._eye_muscle_scale = (1.0 if eye_muscle_scale is None
                                  else float(eye_muscle_scale))
        # 視線誘導反射の実装バージョン。None なら従来どおり環境変数（既定"1"）。
        self._orient_v = (os.environ.get("E_ORIENT_V", "1") if orient_v is None
                          else str(orient_v))
        # 【2026-08-19新設・F1-3a続き】保持成分（ステップ）のON/OFF。既定False＝
        #   v2は従来どおりパルスのみ（1ビット不変）。根拠は e_orienting_v2.py の
        #   HOLD_FB_GAIN 直前のコメント（Robinson 1975 pulse-step model）参照。
        self._orienting_hold = (os.environ.get("E_ORIENT_HOLD", "0") == "1"
                                if orienting_hold is None else bool(orienting_hold))
        # 【2026-08-19新設・F1-4c】静的顕著性チャンネル（動かないものにも視線が
        #   向く）のON/OFF。既定None → e_orienting_v2側の環境変数既定（E_STATIC_SAL、
        #   既定"0"=OFF）に従うので1ビットも挙動が変わらない。
        #   設計：F/docs/設計_F1-4c_静的顕著性.md。orienting_holdと同じ配線パターン。
        self._orienting_static = (None if orienting_static is None
                                  else bool(orienting_static))
        # 【2026-08-20新設・F1-4d】IOR（戻りの抑制）と馴化（見飽きて次を見る）。
        #   既定None → e_orienting_v2側の環境変数既定（E_IOR/E_HABITUATION、
        #   既定"0"=OFF）に従うので1ビットも挙動が変わらない。
        #   設計：F/docs/設計_F1-4d_馴化とIOR.md。orienting_hold/orienting_staticと
        #   同じ配線パターン。
        self._orienting_ior = (None if orienting_ior is None
                               else bool(orienting_ior))
        self._orienting_habituation = (None if orienting_habituation is None
                                       else bool(orienting_habituation))
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
        # おもちゃ(toy1)の色。未指定(None)なら従来どおりTOY_RGBA_OFF（赤）。
        self._toy_rgba_off = (TOY_RGBA_OFF.copy() if toy_rgba is None
                              else np.array(toy_rgba, dtype=float))
        # 【2026-08-18・目標F・F1】2個目のおもちゃ(toy2)。既定は無効(toy2=None→False)。
        self._toy2 = bool(toy2)
        self._toy2_shape = str("sphere" if toy2_shape is None else toy2_shape)
        self._toy2_radius = float(TOY_RADIUS if toy2_radius is None else toy2_radius)
        self._toy2_rgba = (TOY2_RGBA_DEFAULT.copy() if toy2_rgba is None
                           else np.array(toy2_rgba, dtype=float))
        self._toy2_dist = float(TOY_DISTANCE if toy2_dist is None else toy2_dist)
        # 視線正面からの振り分け角度[度]（片側あたり）。toy2=Falseなら使われない。
        self._toy_angle_deg = float(TOY_ANGLE_DEG_DEFAULT if toy_angle_deg is None
                                    else toy_angle_deg)
        # 【2026-08-21新設・F1-4h】垂直方向の角度[度]。既定0.0（_set_anchor参照）。
        self._toy_elev_deg = float(0.0 if toy_elev_deg is None else toy_elev_deg)
        self._toy2_elev_deg = float(0.0 if toy2_elev_deg is None else toy2_elev_deg)
        # 【2026-08-25新設・目標F・4語テスト】3個目・4個目のおもちゃ。toy2と全く
        #   同じ流儀（既定None→False。既存の呼び出し元は誰も渡さないので1ビットも
        #   挙動が変わらない）。body("test_object3"/"4")が実際に存在するかは
        #   super().__init__()後（モデル構築後）でないと分からないため、ここでは
        #   設定値の記録だけ行う（存在チェックは下のtest_object2ブロック直後）。
        #
        # 【2026-08-25改修】以前は run/scene_tools/e_scene.py の build() が
        #   「共通ファイル扱いのため変更不可」としてtoy2までしかシーンJSONから
        #   読まず、環境変数(F_TOY3_*/F_TOY4_*)を新設して迂回していた。今回の
        #   指示でe_scene.pyの変更が許可されたため、toy2と同じ「シーンJSON→
        #   build()がkwargs直渡し」の経路に統一し、環境変数の読み取りは廃止した
        #   （このファイル内・E_SCENE_XMLも含め全数grepでF_TOY3/F_TOY4/E_SCENE_XML
        #   の他の使用箇所が無いことを確認済み）。
        self._toy3 = bool(toy3)
        self._toy3_shape = str("box" if toy3_shape is None else toy3_shape)
        self._toy3_radius = float(TOY_RADIUS if toy3_radius is None else toy3_radius)
        self._toy3_rgba = (TOY3_RGBA_DEFAULT.copy() if toy3_rgba is None
                           else np.array(toy3_rgba, dtype=float))
        self._toy3_dist = float(TOY_DISTANCE if toy3_dist is None else toy3_dist)
        self._toy3_angle_deg = float(
            TOY34_ANGLE_DEG_DEFAULT if toy3_angle_deg is None else toy3_angle_deg)
        self._toy3_elev_deg = float(0.0 if toy3_elev_deg is None else toy3_elev_deg)
        self._toy4 = bool(toy4)
        self._toy4_shape = str("sphere" if toy4_shape is None else toy4_shape)
        self._toy4_radius = float(TOY_RADIUS if toy4_radius is None else toy4_radius)
        self._toy4_rgba = (TOY4_RGBA_DEFAULT.copy() if toy4_rgba is None
                           else np.array(toy4_rgba, dtype=float))
        self._toy4_dist = float(TOY_DISTANCE if toy4_dist is None else toy4_dist)
        self._toy4_angle_deg = float(
            TOY34_ANGLE_DEG_DEFAULT if toy4_angle_deg is None else toy4_angle_deg)
        self._toy4_elev_deg = float(0.0 if toy4_elev_deg is None else toy4_elev_deg)
        # 【2026-08-18新設・F1-3】toy1/toy2の左右符号。+1.0が既定（従来どおりtoy1=左・
        #   toy2=右）。ParentLabelingが5発話ごとに反転させ、_set_anchorを呼び直す
        #   （場所の丸暗記を防ぐ仕掛け。F/docs/仕様_F1-3_....md参照）。
        self._toy_swap_sign = 1.0
        self._anchor = None        # 吊り下げの基準点（リセット時に頭の位置から決める）
        # 置き直し(親が渡す)の回数と、そのstepで置き直したかのフラグ。
        # ＝「おもちゃが動いた」を随伴性の証拠として数えるとき、瞬間移動を除くために要る
        #   （これを見ずに移動量だけ見ると置き直しを"触れて動いた"と誤認する）。
        self.n_respawn = 0
        self.respawned_this_step = False
        # --- 体型の補正（2026-07-20）------------------------------------------
        # 【なぜ要るか】Viewerを見た第三者（育児経験者）の「手足が長すぎる、赤ちゃんは
        # もっと頭でっかち」という指摘を実測したところ、**下肢が新生児の127%**だった：
        #     太郎 age=0 : 上肢 19.1cm / 下肢 24.9cm （上肢/下肢 = 0.77）
        #     人間の新生児: 上肢 20.96cm / 下肢 19.60cm（上肢/下肢 = 1.07）
        #     ＝ 人間の新生児は**腕のほうが脚より長い**。太郎は逆転している。
        #     出典: Segmental Limb Length Measurements in Term Neonates From
        #           Southern India, Indian Pediatrics 2024（n=950, 満期産）
        # mimoGrowth は年齢で「大きさ」は変えるが、新生児特有の短い脚を再現していない
        # （上肢/下肢比は age=0 で 0.77、age=24 で 0.71 とほぼ動かない）。
        # → geom の**長さだけ**を係数で縮める。半径は変えない＝相対的に太くなり、
        #   結果として新生児らしい「ずんぐり」に近づく。
        # 注意：これは**逸脱の解消**であって、創発を作るための細工ではない。根拠は上の実測値。
        #   検証用に必ず 1.0（無補正）へ戻せること。
        # 既定値＝2026-07-21 に Viewer で目視確定した「新生児体型v2」（NEWBORN_SHAPE）。
        # 環境変数を渡さなければこの体型になる。1.0 に戻したいときは各変数へ "1.0" を渡す。
        # E_SHAPE=0 で補正を完全に切る（＝素のmimoGrowth体型に戻す＝アブレーション）。
        self._head_elong = head_elongation_from_env()
        if kwargs.get("age") is not None:
            _custom = body_scale_custom_from_env(kwargs["age"])
            if _custom:
                kwargs["custom_measurements"] = _custom
        # 【2026-08-25新設・目標F・4語テスト】ベースのXML（model_path）の差し替え。
        #   kwargsに明示的にmodel_pathが渡されなければ何もしない＝MIMoV2DummyEnv
        #   既定のbenchmarkv2_scene.xmlのまま（1ビットも変わらない）。
        #   【2026-08-25改修】以前はここでE_SCENE_XML環境変数を読んでいた
        #   （run/scene_tools/e_scene.pyのbuild()がmodel_pathを渡す手段を
        #   持たなかったため）。今回e_scene.pyの変更が許可され、シーンJSON
        #   （world.xml）からmodel_pathをkwargs直渡しできるようになったので、
        #   環境変数の読み取りは廃止した（他に使用箇所が無いことを確認済み）。
        super().__init__(**kwargs)

        # 【2026-08-18新設・F1-3a】眼球の筋力補正。super().__init__()の後＝
        #   self.model / self.actuation_model が構築済みになってから適用する
        #   （親クラスSupineMimoEnvがapply_runtime_correctionsで首・四肢の筋力を
        #   補正するのと同じ位置関係。眼球はrun_time_correctionsの対象外なので
        #   ここで別途適用する）。scale=1.0（既定）なら何もしない＝1ビットも
        #   従来と変わらない。
        if abs(self._eye_muscle_scale - 1.0) > 1e-9:
            from infant_body import apply_eye_muscle_scale
            apply_eye_muscle_scale(self.model, scale=self._eye_muscle_scale,
                                   actuation_model=getattr(self, "actuation_model", None))

        # リクライニング：体の向きを背もたれの角度に合わせる（2026-07-28）。
        #   親クラス（SupineMimoEnv）が仰向け（水平）に置いたあと、y軸まわりに起こす。
        if self._recline_deg > 0.0:
            self._apply_recline()

        self._arm_body = f"{toy_side}_upper_arm"
        self._hand_body = f"{toy_side}_hand"

        # --- おもちゃ(箱)の大きさ・質量・慣性を新生児向けに作り替える ---
        self._toy_bid = self.model.body("test_object1").id
        gadr = self.model.body("test_object1").geomadr[0]
        if self._toy_shape == "asis":
            # 【F2-17・2026-08-30】シーンXMLに書かれた形をそのまま使う。
            #   ここまでの分岐は「test_object1 の geom を1つだけ書き換える」作りなので、
            #   基本図形を何個も組み合わせた物体（犬・車・くつ等）を置けなかった。
            #   asis のときは形・大きさ・色を一切書き換えず、XMLで定義した複数geomを
            #   そのまま残す。質量・慣性は MuJoCo が geom から自動計算した値を使う。
            #   なぜ必要か：般化テストのために「同じカテゴリの別個体」を大量に用意する
            #   必要があり、既製の画像・3Dモデルは権利か統制で使えなかった
            #   （現在地.md「素材の選択肢」）。基本図形で自作するのが唯一の道だった。
            self._toy_mass = float(self.model.body_mass[self._toy_bid])
        elif self._toy_shape == "sphere":
            # 球（定位の測定用）。どの向きから見ても見え方が同じなので、
            #   立方体で起きる「中心側の側面だけが強く光る」非対称が生じない。
            import mujoco as _mj
            self.model.geom_type[gadr] = int(_mj.mjtGeom.mjGEOM_SPHERE)
            self.model.geom_size[gadr] = [self._toy_radius, 0.0, 0.0]
            # 質量は密度を保って体積から出し直す（大きさを変えても手触りが変わらない）
            self._toy_mass = TOY_DENSITY * (4.0 / 3.0) * np.pi * self._toy_radius ** 3
            self.model.body_mass[self._toy_bid] = self._toy_mass
            self.model.body_inertia[self._toy_bid] = _sphere_inertia(
                self._toy_mass, self._toy_radius)
        elif self._toy_shape == "box":
            # 【2026-08-24】以前はここが素の else（sphere以外は全部box扱い）だった。
            #   未知の形名が静かにboxへフォールバックする事故を防ぐため、box を明示し、
            #   その他はcylinder/ellipsoidの分岐か、どれにも当たらなければ例外にした
            #   （「設定が静かに無視される」事故が通算5件＝落とし穴チェックリスト参照）。
            #   box/sphere を指定する既存の呼び出しは、この分岐変更で1ビットも計算結果が
            #   変わらない（式は元のelse節と同一）。
            self.model.geom_size[gadr] = [self._toy_radius] * 3
            if abs(self._toy_radius - 0.020) > 1e-9:
                self._toy_mass = TOY_DENSITY * (2 * self._toy_radius) ** 3
            self.model.body_mass[self._toy_bid] = self._toy_mass
            self.model.body_inertia[self._toy_bid] = _box_inertia(self._toy_mass,
                                                                  self._toy_radius)
        elif self._toy_shape == "cylinder":
            # 【2026-08-24追加・目標F・F2】語彙4語化用の3種類目の形。
            #   大きさの決め方はTOY_CYLINDER_HALF_LEN_RATIOのコメント参照。
            import mujoco as _mj
            half_len = self._toy_radius * TOY_CYLINDER_HALF_LEN_RATIO
            self.model.geom_type[gadr] = int(_mj.mjtGeom.mjGEOM_CYLINDER)
            self.model.geom_size[gadr] = [self._toy_radius, half_len, 0.0]
            self._toy_mass = TOY_DENSITY * np.pi * self._toy_radius ** 2 * (2.0 * half_len)
            self.model.body_mass[self._toy_bid] = self._toy_mass
            self.model.body_inertia[self._toy_bid] = _cylinder_inertia(
                self._toy_mass, self._toy_radius, half_len)
        elif self._toy_shape == "ellipsoid":
            # 【2026-08-24追加・目標F・F2】語彙4語化用の4種類目の形（たまご型）。
            #   大きさの決め方はTOY_ELLIPSOID_ELONGATIONのコメント参照。
            import mujoco as _mj
            a = self._toy_radius
            b = self._toy_radius
            c = self._toy_radius * TOY_ELLIPSOID_ELONGATION
            self.model.geom_type[gadr] = int(_mj.mjtGeom.mjGEOM_ELLIPSOID)
            self.model.geom_size[gadr] = [a, b, c]
            self._toy_mass = TOY_DENSITY * (4.0 / 3.0) * np.pi * a * b * c
            self.model.body_mass[self._toy_bid] = self._toy_mass
            self.model.body_inertia[self._toy_bid] = _ellipsoid_inertia(
                self._toy_mass, a, b, c)
        elif self._toy_shape.startswith("plate:"):
            # 【F2-9C・2026-08-26】イラストの板。"plate:材質名" の形で指定する
            #   （例 "plate:mat_f_wanwan"。材質は benchmarkv2_scene_fillust.xml の
            #   assetで定義。シーンJSONの world.xml でそのXMLを指定すること）。
            #   寸法は8cm角・厚さ1cm（half 0.04,0.005,0.04）＝絵本のページに相当。
            #   薄い軸はy（法線y）＝アンカーの視線正面配置で絵が太郎を向く。
            #   材質名の埋め込み形式にしたのは、既存の toy_shape 1本の配線
            #   （シーンJSON→e_scene→ここ）を変えずに済ませるため。
            import mujoco as _mj
            mat = self._toy_shape.split(":", 1)[1]
            half = (self._toy_radius, 0.005, self._toy_radius)
            self.model.geom_type[gadr] = int(_mj.mjtGeom.mjGEOM_BOX)
            self.model.geom_size[gadr] = list(half)
            self.model.geom_matid[gadr] = int(self.model.material(mat).id)
            self._toy_mass = TOY_DENSITY * (2*half[0]) * (2*half[1]) * (2*half[2])
            self.model.body_mass[self._toy_bid] = self._toy_mass
            self.model.body_inertia[self._toy_bid] = [
                self._toy_mass / 3.0 * (half[1]**2 + half[2]**2),
                self._toy_mass / 3.0 * (half[0]**2 + half[2]**2),
                self._toy_mass / 3.0 * (half[0]**2 + half[1]**2)]
        else:
            # 【2026-08-24追加】知らない形の名前が来たら黙ってboxへフォールバックせず、
            #   はっきりしたエラーで止める（落とし穴チェックリスト「設定が静かに無視される」対策）。
            raise ValueError(
                f"未知の toy_shape={self._toy_shape!r}。対応する値: "
                f"box / sphere / cylinder / ellipsoid / plate:材質名 / asis")
        self.model.geom_friction[gadr] = TOY_FRICTION   # 転がり続けを止める（上のコメント）
        # 目視用に目立つ色（赤）。太郎の体・床と区別がつかないと動画で確認できないため。
        # 接触中は TOY_RGBA_ON（明るい黄）に切り替わる＝「触れている間だけ光る」。
        self._toy_gadr = gadr
        if self._toy_shape.startswith("plate:"):
            # 【F2-9C】イラスト板は色を塗らない（rgbaはテクスチャに乗算されるため
            #   白=素通し。赤を掛けると絵が赤茶けて潰れる）。接触発光も同じ理由でなし。
            self.model.geom_rgba[gadr] = [1.0, 1.0, 1.0, 1.0]
        elif self._toy_shape == "asis":
            # 【F2-17】XMLで個体ごとに色を決めているので上書きしない（接触発光もなし）
            pass
        else:
            self.model.geom_rgba[gadr] = self._toy_rgba_off
        self.toy_lit = False          # 今光っているか（測定・記録用）
        # freejoint の qpos 先頭アドレス（位置3＋姿勢4）
        jadr = self.model.body("test_object1").jntadr[0]
        self._toy_qadr = self.model.jnt_qposadr[jadr]
        self._toy_dadr = self.model.jnt_dofadr[jadr]

        # 【2026-08-18・目標F・F1】test_object2（使っていない予備の物体＝球）。
        #   toy2=False（既定）ならこれまでどおり遠方へ退避するだけ＝1ビットも変わらない。
        #   toy2=True なら2個目のおもちゃとして形状・質量・慣性・色を作り替える
        #   （test_object1と同じやり方。_configure_toy_geom参照）。
        self._obj2_bid = self.model.body("test_object2").id
        jadr2 = self.model.body("test_object2").jntadr[0]
        self._obj2_qadr = self.model.jnt_qposadr[jadr2]
        self._obj2_dadr = self.model.jnt_dofadr[jadr2]
        if self._toy2:
            obj2_gadr = self.model.body("test_object2").geomadr[0]
            self._configure_toy_geom(self._obj2_bid, obj2_gadr,
                                     self._toy2_shape, self._toy2_radius,
                                     self._toy2_rgba)

        # 【2026-08-25新設・目標F・4語テスト】test_object3/test_object4。
        #   共有XML（benchmarkv2_scene.xml）には存在しないので、無ければ
        #   静かに諦める（既存シーンは1ビットも挙動が変わらない）。
        #   toy3/toy4=True なのに body が無いシーンだけ例外にする
        #   （落とし穴チェックリスト「設定が静かに無視される」対策）。
        def _find_body_id(name):
            try:
                return int(self.model.body(name).id)
            except Exception:
                return None

        self._obj3_bid = _find_body_id("test_object3")
        self._obj3_qadr = None
        self._obj3_dadr = None
        if self._obj3_bid is not None:
            jadr3 = self.model.body("test_object3").jntadr[0]
            self._obj3_qadr = self.model.jnt_qposadr[jadr3]
            self._obj3_dadr = self.model.jnt_dofadr[jadr3]
            if self._toy3:
                obj3_gadr = self.model.body("test_object3").geomadr[0]
                self._configure_toy_geom(self._obj3_bid, obj3_gadr,
                                         self._toy3_shape, self._toy3_radius,
                                         self._toy3_rgba)
        elif self._toy3:
            raise ValueError(
                "toy3=True ですが、このシーンのXMLに test_object3 がありません。"
                "test_object3/test_object4 を含むF専用シーン（4物体テスト用）を"
                "使ってください（共有XMLのbenchmarkv2_scene.xmlには無い）。")

        self._obj4_bid = _find_body_id("test_object4")
        self._obj4_qadr = None
        self._obj4_dadr = None
        if self._obj4_bid is not None:
            jadr4 = self.model.body("test_object4").jntadr[0]
            self._obj4_qadr = self.model.jnt_qposadr[jadr4]
            self._obj4_dadr = self.model.jnt_dofadr[jadr4]
            if self._toy4:
                obj4_gadr = self.model.body("test_object4").geomadr[0]
                self._configure_toy_geom(self._obj4_bid, obj4_gadr,
                                         self._toy4_shape, self._toy4_radius,
                                         self._toy4_rgba)
        elif self._toy4:
            raise ValueError(
                "toy4=True ですが、このシーンのXMLに test_object4 がありません。"
                "test_object3/test_object4 を含むF専用シーン（4物体テスト用）を"
                "使ってください（共有XMLのbenchmarkv2_scene.xmlには無い）。")

        # 【10択・2026-08-31】差し出し専用スロットの登録（既定None＝空dict）。
        self._present_slots = {}
        for _k, _bname in enumerate(list(present_slots or [])):
            _bid = _find_body_id(_bname)
            if _bid is None:
                raise ValueError(
                    "present_slots に %r が指定されましたが、このシーンのXMLに"
                    "そのbodyがありません（10択用XMLを使うこと）" % _bname)
            _jadr = self.model.body(_bname).jntadr[0]
            self._present_slots["toy%d" % (5 + _k)] = {
                "bid": int(_bid), "body": _bname,
                "qadr": int(self.model.jnt_qposadr[_jadr]),
                "dadr": int(self.model.jnt_dofadr[_jadr])}

        # 【2026-08-18新設・F1-3】親のfollow-in labeling。parent_labeling未指定(None)なら
        #   ParentLabeling(enabled=False)になり、update()は毎stepNoneを返すだけ＝
        #   1ビットも既存の挙動を変えない。E/scripts/parent_labeling.py の本体を参照。
        from parent_labeling import ParentLabeling, WordSchedule
        self._parent_labeling = ParentLabeling(**dict(parent_labeling or {}))

        # 【2026-08-21新設・F1-4h】語の再生装置。word_test未指定(None)なら
        #   WordSchedule(schedule=None)になり、update()は毎stepNoneを返すだけ＝
        #   1ビットも既存の挙動を変えない。E/scripts/parent_labeling.py参照。
        self._word_schedule = WordSchedule(**dict(word_test or {}))

        # 首の補正は「落ち着いた初期姿勢」で重力モーメントを測ってから適用する
        # （姿勢で腕の長さが変わるため）。SupineMimoEnvのsettle後＝ここが適切な位置。
        if self._newborn_neck and self._neck_age is not None:
            pass   # 【2026-07-25】親クラス(SupineMimoEnv)が core の
                   # apply_runtime_corrections で適用済み＝ここでの重複を削除
        if self._newborn_limbs and self._neck_age is not None:
            pass   # 【2026-07-25】同上（親クラスが core の実装で適用済み）
        if self._use_vor:
            from e_vor import VOR
            # 2026-07-26：月齢を渡す。三半規管の時定数が月齢で変わるため
            #   （新生児10秒 → 成人20秒）。渡さないと成人の値になる。
            self._vor = VOR(self.model, self.data, age_months=float(getattr(self, "age", 0.0)))
            print(f"[vor] enabled: gain={self._vor.gain} on {len(self._vor.units)} eye actuators "
                  f"(policy output to eyes is ignored)")
        if self._use_orient:
            # 【2026-07-26】視線誘導反射に新版(v2)を追加。E_ORIENT_V=2 で切り替える。
            #   v1（既定）: e_orienting.py。6マス分割＋残差法。構造的な穴が6つ見つかっている
            #   v2       : e_orienting_v2.py。設計図（E/docs/視線誘導反射_設計図.md）に基づく
            #              複数フレーム動き検出＋中心バイアス＋側方抑制＋階段状サッケード
            # 注意：v1 は比較・アブレーション用に残す（撤回した実装を消さない方針）。
            ver = self._orient_v
            if ver == "2":
                from e_orienting_v2 import OrientingReflexV2
                # data も渡す：サッケードは「力を一定時間かける」のではなく
                #   **今の眼球角度と目標角度の差を見ながら**動かす（位置の内部
                #   フィードバック＝Robinson 1975 の local feedback model）。
                self._orienting = OrientingReflexV2(self.model, data=self.data,
                                                    dt=self.dt,
                                                    hold=self._orienting_hold,
                                                    static_salience=self._orienting_static,
                                                    ior=self._orienting_ior,
                                                    habituation=self._orienting_habituation)
            else:
                from e_orienting import OrientingReflex
                self._orienting = OrientingReflex(self.model)
            print(f"[orient] enabled: v{ver} ({type(self._orienting).__name__}) "
                  f"neck={list(self._orienting.neck_idx.keys())} "
                  f"eye_h={len(self._orienting.eye_idx['h'])} eye_v={len(self._orienting.eye_idx['v'])} "
                  f"hold={self._orienting_hold} static_salience={getattr(self._orienting, 'static_salience', None)} "
                  f"ior={getattr(self._orienting, 'ior', None)} "
                  f"habituation={getattr(self._orienting, 'habituation', None)}")
        if self._use_vergence:
            from e_vergence import VergenceReflex
            self._vergence = VergenceReflex(self.model, self.data)
            print(f"[vergence] enabled: gain={self._vergence.gain} "
                  f"max_speed={self._vergence.max_speed_deg}deg/s range=+/-{self._vergence.range_deg}deg")

    # ------------------------------------------------------------------
    def _make_visually_plain(self, spec):
        """視界を「貧しく」する：床の市松模様を消し、柵を床と同じ色にする。

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
        注意：skybox（空のグラデーション）は残す＝仰向けで上を向いたときの背景。消すと真っ暗になり
          「視覚が無い」条件になってしまうため。
        注意：**柵を消すのではなく色だけ変える**のは、Ferrari et al.(2007)が「nest（囲い）を与えると
          肩内転・肘屈曲・**正中方向への運動が有意に増える**」と実測しているため
          ＝**視覚的な豊かさ（除くべき）と物理的な支持（与えるべき）は別物**。
        """
        for g in spec.geoms:                      # worldbody直下のgeom（床はここ）
            if g.name == "floor":
                _fr = PLAIN_RGBA.copy()
                _fr[:3] *= self._floor_dim        # 照明で明るくなるぶんを相殺する
                if self._floor_emission > 0.0:
                    # 2026-08-21：床の横縞対策。material無し（旧来）だと geom.rgba に
                    #   陰影（cos(法線, 光源方向)）がそのままかかり、無限平面の床は
                    #   地平線付近（入射角が直角に近い）で黒に落ちる（実測は
                    #   FLOOR_EMISSION定義直前のコメント参照）。専用material を新設し、
                    #   emission（自己発光）を上げる。
                    #   注意：MuJoCoのシェーディングは emission と拡散反射(diffuse)が
                    #     **同じ material.rgba** を係数倍する（emission_color =
                    #     emission_scalar × rgba、diffuse_color = cos(N,L) × rgba）。
                    #     このため単に emission=1.0 だけ上げると、光源に近い画素
                    #     （cos成分が大きい）で emission と diffuse が足し合わさって
                    #     1.0を超え白飛びする（実測：emission=1.0でrow108-122が
                    #     mean 0.99〜1.00に飽和）。
                    #   対処：floor_emission を「emission_scalar と同じ値」として使い、
                    #     rgba 側を 1/floor_emission に縮小する
                    #     （emission_scalar × rgba = 元の目標色のまま、
                    #     diffuse側の寄与だけ 1/floor_emission に縮む）。
                    #     floor_emissionを大きくするほど陰影の影響（横縞）が小さくなる。
                    k = float(self._floor_emission)
                    mat = spec.add_material()
                    mat.name = "e_floor_plain"
                    mat.rgba = list(_fr[:3] / k) + [float(_fr[3])]
                    mat.emission = k
                    mat.specular = 0.0
                    mat.shininess = 0.0
                    mat.reflectance = 0.0
                    g.material = "e_floor_plain"
                else:
                    g.material = ""                   # 市松テクスチャを外す（従来どおり）
                    g.rgba = list(_fr)
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

    def _add_static_texture(self, spec):
        """【目標E フェーズ1】動かない模様のあるもの（天井＋柵の市松）を足す。

        フェーズ1で「自分の首を動かす→視覚が変わる」を学ぶには、視界に空間的な模様が要る。
        2026-07-21の目視で、E_PLAIN=1の環境は視界の58%が変化なし（空か柵の単色のみ）と判明。
        対処として、太郎の真上に市松模様の天井を置き、柵にも同じ市松を貼る。
        注意：これは"動かないもの"なので、自己運動と視覚の関係を素直に学べる（おもちゃ=動くものと違う）。
        """
        # 市松テクスチャと、それを参照する材質を1つずつ定義する（MjSpecなのでXML不要）。
        tex = spec.add_texture()
        tex.name = "e_static_checker_tex"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.width = 128
        tex.height = 128
        tex.rgb1 = [0.15, 0.20, 0.35]
        tex.rgb2 = [0.75, 0.80, 0.55]   # 高コントラストの2色（学習しやすい明確なエッジ）
        mat = spec.add_material()
        mat.name = "e_static_checker"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "e_static_checker_tex"
        mat.texrepeat = [4, 4]          # 面内で模様を4×4回繰り返す＝細かいエッジを増やす
        # 太郎の真上に天井（薄い箱）を置く。仰向けの太郎が最も長く見ている方向。
        # 高さは柵より少し高い位置。視界(fovy≈上方向)に必ず入る大きさにする。
        ceil = spec.worldbody.add_geom()
        ceil.name = "e_static_ceiling"
        ceil.type = mujoco.mjtGeom.mjGEOM_BOX
        ceil.size = [self._fence_half_x + 0.1, self._fence_half_y + 0.1, 0.005]
        ceil.pos = [0.0, 0.0, self._fence_height + 0.15]
        ceil.material = "e_static_checker"
        ceil.contype = 0     # 物理的な衝突はさせない（見えるだけ）＝太郎が触れても動かない・当たらない
        ceil.conaffinity = 0

    def _recline_lift(self):
        """リクライニング時に体（と背もたれ）を持ち上げる量[m]。

        注意：【2026-07-28 に踏んだ】この計算を `_apply_recline`（体）と `_add_seat`（板）の
        **2箇所に別々に書いていた**ため、体だけが角度に応じて持ち上がり、板は固定のままで、
        角度が大きいほど体が板から浮いた。45度では偶然乗ったが、70度では滑り落ちて
        x=-1.04m まで転がった（真横からの画像で確認）。→ 落とし穴チェックリスト項64
        （同じ処理を2箇所に書かない）。1つにまとめる。
        """
        return SEAT_HALF_LEN * np.sin(np.radians(self._recline_deg)) * 0.5

    def _apply_recline(self):
        """体を背もたれの角度まで起こす。2026-07-28 新設。

        【2026-07-28 に2回間違えた場所。総当たり（`e_recline_rot_probe.py`）で確定した】
        誤り1：`model.body("hip").quat` を書き換えた
            → **hip には自由関節が無い**。体全体を動かす free joint は `mimo_location`
              にある。hip の pos/quat を書いても姿勢は変わらなかった。
              （親クラス SupineMimoEnv も hip を書いているが、そちらは reset 前なので
                qpos0 の計算に反映され、結果として効いている）
        誤り2：回転の符号と軸を推測で決めた
            → 実測：y軸まわり **-70度** で体幹 +76度（目標70度）。
              +70度だと -63.9度（頭が下＝逆立ち）になる。

        注意：合成は**左右どちらでもほぼ同じ**（76.1度 / 76.0度）。ここでは world 基準
          （左から）を使う。
        """
        import mujoco as _mj
        # 体全体を動かす自由関節を探す（おもちゃの自由関節は除く）
        qadr = None
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) != int(_mj.mjtJoint.mjJNT_FREE):
                continue
            bn = self.model.body(int(self.model.jnt_bodyid[j])).name
            if "object" in bn or "toy" in bn:
                continue
            qadr = int(self.model.jnt_qposadr[j])
            break
        if qadr is None:
            print("[recline] 注意体の自由関節が見つからないので起こせない")
            return
        self._recline_qadr = qadr
        # qpos0（リセット時の姿勢）を書き換える。data.qpos だけ変えても
        #   reset のたびに元へ戻ってしまう。
        th = -np.radians(self._recline_deg)          # 符号は実測で確定（上記）
        q_tilt = np.zeros(4)
        _mj.mju_axisAngle2Quat(q_tilt, np.array([0.0, 1.0, 0.0]), th)
        q_now = np.array(self.model.qpos0[qadr + 3:qadr + 7], dtype=float)
        q_new = np.zeros(4)
        _mj.mju_mulQuat(q_new, q_tilt, q_now)        # world基準で起こす
        self.model.qpos0[qadr + 3:qadr + 7] = q_new
        # 位置も背もたれの上へ（起こすと体が伸びる方向が変わるので持ち上げる）
        self.model.qpos0[qadr + 2] += self._recline_lift()
        print(f"[recline] 体を {self._recline_deg:.0f}度 起こした"
              f"（0=仰向け／90=直立）[Tier2: Carvalho et al. 2007 のベビーチェアは70度]")

    def _add_seat(self, spec):
        """リクライニング用の背もたれ（傾いた板）をワールドに置く。2026-07-28 新設。

        人間の実験で使われるベビーチェア／バウンサーに相当する。

        注意：【2026-07-28 に踏んだ】最初、板を体と**反対側**（-x）に置いてしまい、
        体が板にぶつかって姿勢が崩れた（体幹 -35.7度）。太郎の体は**頭が +x 方向**に
        あるので、板も +x 側へ、体軸に沿って置く必要がある。

        【向きの決め方】体は y軸まわり -recline_deg 回転している（`_apply_recline`）。
            体軸（骨盤→頭）  = ( cos, 0,  sin)
            背中の向き        = ( sin, 0, -cos)     ※体軸に垂直で、元の -z 側
        板の中心を「骨盤から体軸方向へ半分」＋「背中側へ少し」の位置に置く。

        注意：[Tier3] 実物のバウンサーは連続した曲面で、股ベルトで固定されている。
          寸法の規格も調査で見つからなかった（2026-07-28）。太郎の体が乗る大きさとして決めた。
        """
        th = np.radians(self._recline_deg)
        axis = np.array([np.cos(th), 0.0, np.sin(th)])       # 骨盤→頭
        back_dir = np.array([np.sin(th), 0.0, -np.cos(th)])  # 背中側
        # 板は**低い位置に固定**する。体だけを持ち上げて、**上から落として乗せる**。
        #
        # 注意：【2026-07-28 に戻した】「体と同じだけ板も持ち上げる」ようにしたら
        #   45度まで壊れた（それまで45度は板に背中を預けて成立していた）。
        #   初期状態で体と板が同じ高さになると**干渉して弾かれる**。
        #   低い板に上から体を落とす関係が正しい。真横からの画像で確認。
        #   ＝物体を物体に乗せるときは「重ねない・上から落とす」。
        hip0 = np.array([0.0, 0.0, 0.05])

        back = spec.worldbody.add_geom()
        back.name = "recline_back"
        back.type = mujoco.mjtGeom.mjGEOM_BOX
        back.size = [SEAT_HALF_LEN, SEAT_HALF_WID, SEAT_THICK]
        pos = hip0 + axis * (SEAT_HALF_LEN * 0.55) + back_dir * 0.055
        back.pos = [float(pos[0]), 0.0, float(pos[2])]
        # 板の長軸（ローカルx）を体軸に向ける＝y軸まわりに -recline_deg
        back.quat = [float(np.cos(-th / 2)), 0.0, float(np.sin(-th / 2)), 0.0]
        back.material = ""
        back.rgba = list(SEAT_RGBA)
        back.condim = 3
        # 滑り摩擦を上げてずり落ちを防ぐ（横回転・転がりの摩擦もわずかに上げる）
        back.friction = [self._seat_friction, 0.02, 0.001]

        # 座面：水平な板。骨盤を受けてずり落ちを止める
        seat = spec.worldbody.add_geom()
        seat.name = "recline_seat"
        seat.type = mujoco.mjtGeom.mjGEOM_BOX
        seat.size = [SEAT_HALF_WID, SEAT_HALF_WID, SEAT_THICK]
        seat.pos = [float(hip0[0]), 0.0, float(hip0[2] - 0.045)]
        seat.material = ""
        seat.rgba = list(SEAT_RGBA)
        seat.condim = 3
        seat.friction = [self._seat_friction, 0.02, 0.001]

    def _elongate_head(self, spec):
        """頭のgeomを球→楕円体にして、**体軸方向にだけ**伸ばす。

        【なぜ要るか】MIMoの頭は球で、直径は頭囲(34cm)から計算される＝10.8cm。
        しかし**人間の頭は楕円**で、頭囲34cmでも頭頂〜顎は約12.5cmある。
        ＝ MIMoは頭囲が正しいのに、**真上から見た頭は15%小さい**。
        この差を四肢を縮めて埋めようとすると身長が犠牲になり（右案で36.5cm＝
        新生児の73%）、身長を優先すると頭囲が壊れる。**球のままでは
        頭囲・身長・見かけの3つを同時に満たす解が無い**。

        【何をするか】左右方向（＝頭囲を決める軸）は変えず、
        MIMoローカルの z（頭頂方向＝仰向けでは体軸方向）だけを ratio 倍する。
          球   [r, 0, 0]        頭囲 2πr        真上から見た長さ 2r
          楕円 [r, r, r*ratio]  頭囲 2πr（不変） 真上から見た長さ 2r*ratio
        ratio = 12.5/10.8 ≒ 1.16 で人間の新生児に一致する。

        注意：目のカメラ位置は head の geom size から計算されるが、その計算は
        成長モジュール（このフックより前）で終わっている。z方向にだけ伸ばすので
        目が頭に埋もれることは無いはずだが、**視界の画像で必ず確認すること**。
        """
        ratio = getattr(self, "_head_elong", 1.0)
        if abs(ratio - 1.0) < 1e-9:
            return
        for g in spec.geoms:
            if g.name != "head":
                continue
            r = float(np.asarray(g.size).ravel()[0])
            g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
            g.size = [r, r, r * ratio]
            print(f"[body] 頭を楕円化: 半径{r*100:.2f}cm → "
                  f"[{r*100:.2f}, {r*100:.2f}, {r*ratio*100:.2f}]cm "
                  f"（頭囲は不変、真上から見た長さ {2*r*ratio*100:.2f}cm）")
            return
        print("[body] 注意頭のgeomが見つからず、楕円化をスキップした")

    def _edit_spec(self, spec):
        """モデル構築前に、ベビーサークルの柱をワールドへ追加する（LeanMimoEnvのフック）。

        MjSpec は compile 前なら geom を足せるので、**XMLファイルを一切作らず**に
        シーンを拡張できる＝MIMo同梱のXML（共有物）を汚さない。
        柱は静的（freejointなし）なので、太郎が当たっても動かない＝壁として働く。
        """
        self._elongate_head(spec)
        if self._recline_deg > 0.0:
            self._add_seat(spec)
        if self._plain:
            self._make_visually_plain(spec)
        if self._static_tex:
            self._add_static_texture(spec)
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
            # 注意：バグ修正：MjSpecで足したgeomは**デフォルトクラスの material="matgeom"（茶色の
            #   テクスチャ）を継承する**ため、rgbaを指定しても茶色に描かれていた（目視で発覚）。
            #   material を空にしないと rgba が効かない。
            if self._static_tex:
                # フェーズ1：柵にも市松テクスチャを貼る（首を振ると縦棒＋模様が動いて見える）
                g.material = "e_static_checker"
            else:
                g.material = ""
                g.rgba = list(PLAIN_RGBA if self._plain else FENCE_RGBA_RICH)
            g.condim = 3

    def _place(self, qadr, pos):
        """freejoint物体をワールド座標posへ置き直す（速度も0に戻す）。"""
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _configure_toy_geom(self, bid, gadr, shape, radius, rgba):
        """おもちゃ用bodyの形状・質量・慣性・摩擦・色を設定する。

        【2026-08-18・目標F・F1】test_object1の初期化（__init__、このクラスの上の方）
        と同じ計算をtoy2向けに切り出したもの。密度(TOY_DENSITY)を保って質量を
        体積から出し直す点も同じ。toy1と違い、質量を個別指定する引数(toy_mass相当)は
        持たない＝「並べて見せる」用途で今のところ重さを効かせる実験が無いため。
        """
        if shape == "asis":
            # 【V1b・2026-08-31】XMLの形をそのまま使う（toy1のasisと同じ趣旨・
            #   F/docs/設計_視覚とGRUの統合（表の卒業）.md V1b）。
            #   形・大きさ・色・質量・慣性を一切書き換えない。
            return
        if shape == "sphere":
            self.model.geom_type[gadr] = int(mujoco.mjtGeom.mjGEOM_SPHERE)
            self.model.geom_size[gadr] = [radius, 0.0, 0.0]
            mass = TOY_DENSITY * (4.0 / 3.0) * np.pi * radius ** 3
            self.model.body_inertia[bid] = _sphere_inertia(mass, radius)
        elif shape == "box":
            # 【2026-08-24】以前はここが素の else（sphere以外は全部box扱い）だった。
            #   box を明示し、未知の形名はエラーで止める形にした（toy1側と同じ理由）。
            self.model.geom_type[gadr] = int(mujoco.mjtGeom.mjGEOM_BOX)
            self.model.geom_size[gadr] = [radius] * 3
            mass = TOY_DENSITY * (2 * radius) ** 3
            self.model.body_inertia[bid] = _box_inertia(mass, radius)
        elif shape == "cylinder":
            # 【2026-08-24追加・目標F・F2】toy1側と同じ大きさの決め方（TOY_CYLINDER_HALF_LEN_RATIO）。
            half_len = radius * TOY_CYLINDER_HALF_LEN_RATIO
            self.model.geom_type[gadr] = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            self.model.geom_size[gadr] = [radius, half_len, 0.0]
            mass = TOY_DENSITY * np.pi * radius ** 2 * (2.0 * half_len)
            self.model.body_inertia[bid] = _cylinder_inertia(mass, radius, half_len)
        elif shape == "ellipsoid":
            # 【2026-08-24追加・目標F・F2】toy1側と同じ大きさの決め方（TOY_ELLIPSOID_ELONGATION）。
            a = radius
            b = radius
            c = radius * TOY_ELLIPSOID_ELONGATION
            self.model.geom_type[gadr] = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
            self.model.geom_size[gadr] = [a, b, c]
            mass = TOY_DENSITY * (4.0 / 3.0) * np.pi * a * b * c
            self.model.body_inertia[bid] = _ellipsoid_inertia(mass, a, b, c)
        elif shape.startswith("plate:"):
            # 【F2-9C・2026-08-26】イラストの板（toy1側の分岐と同じ。あちらのコメント参照）。
            mat = shape.split(":", 1)[1]
            half = (radius, 0.005, radius)
            self.model.geom_type[gadr] = int(mujoco.mjtGeom.mjGEOM_BOX)
            self.model.geom_size[gadr] = list(half)
            self.model.geom_matid[gadr] = int(self.model.material(mat).id)
            mass = TOY_DENSITY * (2*half[0]) * (2*half[1]) * (2*half[2])
            self.model.body_inertia[bid] = [
                mass / 3.0 * (half[1]**2 + half[2]**2),
                mass / 3.0 * (half[0]**2 + half[2]**2),
                mass / 3.0 * (half[0]**2 + half[1]**2)]
            rgba = [1.0, 1.0, 1.0, 1.0]     # テクスチャ素通し（toy1側のコメント参照）
        else:
            # 【2026-08-24追加】知らない形の名前が来たら黙ってboxへフォールバックせず、
            #   はっきりしたエラーで止める。
            raise ValueError(
                f"未知の toy2_shape={shape!r}。対応する値: box / sphere / cylinder / "
                f"ellipsoid / plate:材質名")
        self.model.body_mass[bid] = mass
        self.model.geom_friction[gadr] = TOY_FRICTION
        # 【2026-08-18】test_object2はXML側でmaterialが設定済みのため、rgbaを
        #   書くだけでは反映されない（materialが優先される・923行の罠と同じ型）。
        #   material参照を外してからrgbaを書く必要がある（実測で発覚：色が
        #   指定した白ではなくXML既定の茶系のまま描画されていた）。
        # 【F2-9C・2026-08-26】ただしイラスト板（plate:）は材質＝絵そのものなので
        #   剥がさない（実測で発覚：ここが分岐内で設定したmatidを-1に戻していた）。
        if not shape.startswith("plate:"):
            self.model.geom_matid[gadr] = -1
        self.model.geom_rgba[gadr] = rgba

    def _gaze_dir(self, cam="eye_left"):
        """今の視線方向（ワールド）。MuJoCoのカメラは**-z方向**を見るので符号を反転する。"""
        cid = int(self.model.camera(cam).id)
        return -np.array(self.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]

    def _gaze_angle_to(self, body_name):
        """視線とbody_nameの中心のなす角度[度]。求まらなければNone。

        【2026-08-18新設・F1-3】_parent_intervene（下方）の内積計算を一般化したもの。
        toy1・toy2の両方に使う（ParentLabelingの注視判定）。
        """
        cid = int(self.model.camera("eye_left").id)
        eye = np.array(self.data.cam_xpos[cid], dtype=float)
        fwd = -np.array(self.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        try:
            pos = np.array(self.data.body(body_name).xpos, dtype=float)
        except Exception:
            return None
        vec = pos - eye
        n = float(np.linalg.norm(vec))
        if n < 1e-9:
            return None
        return float(np.degrees(np.arccos(np.clip(np.dot(fwd, vec / n), -1.0, 1.0))))

    def _set_anchor(self):
        """吊り下げの基準点（ベビージムの支点）を決める。

        **リセット時の視線の正面**に置く（第1版は「頭の上へ固定オフセット」だった）。
        第1版は手が届くかだけを見て置き場所を決め、**視線が通るかを確認していなかった**。
        実測すると初期姿勢で既に視線から20.4°ずれており、視野の半角30°の2/3を
        配置だけで食い潰していた（残りマージン10°／頭は平均50°振れる）。
        視線の正面に置けば初期ずれ0°＝マージンを丸ごと頭の揺れに使える。
        注意：これは「頭が振れると視界から外れる」問題そのものは解決しない（後述の課題）。

        リセット時に一度だけ決めてエピソード中は固定＝現実のベビージムも動かない。
        （太郎に追従させると「おもちゃが赤ちゃんを追いかける」不自然さになる）
        """
        # 2026-07-26修正：基準点を「頭の中心」から「両目の中点」に変えた。
        #   目は頭の中心より前方にあるので、頭の中心から視線方向へ進んだ点は
        #   **目から見ると大きく横にずれる**。近いほどずれが大きい。
        #   実測（修正前）：「初期ずれ0°」と書いてあるのに実際は
        #     距離 8.6cm → ずれ 69.9度 ／ 12cm → 41.9度 ／ 18cm → 22.3度
        #   ＝距離を変えるとずれが変わる（視線の正面ならどの距離でも0のはず）。
        #   → 落とし穴チェックリスト 項17（設定した値が効いていると思い込まない）
        eyes = []
        for nm in ("eye_left", "eye_right"):
            try:
                eyes.append(np.array(self.data.cam_xpos[int(self.model.camera(nm).id)],
                                     dtype=float))
            except Exception:
                pass
        origin = (np.mean(eyes, axis=0) if eyes
                  else self.data.body("head").xpos.copy())
        # おもちゃがぶら下がる位置。支点はその**真上に紐の長さぶん**取る＝
        # 重力で自然に垂れると、ちょうどこの位置に来る（振り子の静止点）。
        #
        # 【2026-08-18・目標F・F1】toy2があるときは、視線の正面から左右に
        #   toy_angle_deg 度ずつ振り分ける。片目カメラのローカルx軸（画像でいう
        #   「右」方向、cam_xmatの列0）を「右」として使う＝体の向き（仰向け／座位）
        #   によらず視野の中で正しく左右になる。toy2=False（既定）のときは角度0
        #   ＝従来のgとまったく同じ値になるので1ビットも変わらない。
        self._rest_pos2 = None
        self._rest_pos3 = None
        self._rest_pos4 = None
        if self._toy_offset is None:
            g = self._gaze_dir()
            # 【2026-08-21新設・F1-4h】up（カメラローカルの上方向）。垂直角
            #   （elev_deg）の適用に使う。right・up・g（=-cam_xmatのz列）は
            #   カメラの正規直交基底なので、right/gの線形結合であるg1・g2は
            #   常にupと直交する＝下のcos/sin合成は単位ベクトルのまま回転できる。
            #   toy2の有無に関わらず毎回計算するが、cam_xmatの参照だけで軽い。
            cid = int(self.model.camera("eye_left").id)
            cmat = np.array(self.data.cam_xmat[cid], dtype=float).reshape(3, 3)
            right = cmat[:, 0]
            up = cmat[:, 1]
            if self._toy2:
                ang = np.radians(self._toy_angle_deg)
                # 【2026-08-18新設・F1-3】左右入れ替え。既定+1.0なら従来どおり
                #   toy1=左・toy2=右（ParentLabelingが5発話ごとに-1.0へ反転させ、
                #   この_set_anchorを呼び直す＝場所の丸暗記を防ぐ仕掛け）。
                sign = getattr(self, "_toy_swap_sign", 1.0)
                g1 = g * np.cos(ang) - sign * right * np.sin(ang)     # 左（toy1、既定）
                g2 = g * np.cos(ang) + sign * right * np.sin(ang)     # 右（toy2、既定）
                # 【2026-08-21新設・F1-4h】toy2の垂直角。既定0.0ならsin(0)=0で
                #   g2は不変（1ビットも既存の挙動と変わらない）。
                elev2 = np.radians(self._toy2_elev_deg)
                g2 = g2 * np.cos(elev2) + up * np.sin(elev2)
                self._rest_pos2 = origin + g2 * self._toy2_dist
            else:
                g1 = g
            # 【2026-08-21新設・F1-4h】toy1の垂直角。既定0.0ならsin(0)=0でg1は不変。
            elev1 = np.radians(self._toy_elev_deg)
            g1 = g1 * np.cos(elev1) + up * np.sin(elev1)
            self._rest_pos = origin + g1 * self._toy_dist   # 視線の正面・距離 _toy_dist
            # 【2026-08-25新設・目標F・4語テスト】toy3/toy4。toy1/toy2が「右」軸
            #   （right）で左右に振り分けるのに対し、toy3/toy4は「上」軸（up）で
            #   振り分ける（4方向を2軸に分けて衝突を減らすため）。既定角度0度
            #   （TOY34_ANGLE_DEG_DEFAULT）なら sin(0)=0 でtoy1と同じ方向・同じ距離
            #   になる＝4つ重なる。4語テストは1個ずつ透明化して見せるので実害は無い
            #   （仕様で明示的に許容）。toy3/toy4=False（既定）ならこのブロックは
            #   実行されるが self._rest_pos3/4 は使われないので無害。
            if self._toy3:
                ang3 = np.radians(self._toy3_angle_deg)
                g3 = g * np.cos(ang3) + up * np.sin(ang3)
                elev3 = np.radians(self._toy3_elev_deg)
                g3 = g3 * np.cos(elev3) + up * np.sin(elev3)
                self._rest_pos3 = origin + g3 * self._toy3_dist
            if self._toy4:
                ang4 = np.radians(self._toy4_angle_deg)
                g4 = g * np.cos(ang4) - up * np.sin(ang4)
                elev4 = np.radians(self._toy4_elev_deg)
                g4 = g4 * np.cos(elev4) + up * np.sin(elev4)
                self._rest_pos4 = origin + g4 * self._toy4_dist
        else:
            # 旧方式（アブレーション用に残す）。こちらは頭の中心からのオフセット。
            self._rest_pos = self.data.body("head").xpos.copy() + self._toy_offset
            if self._toy2:
                # toy_offset方式(旧アブレーション)との組み合わせは未検証。
                #   y方向へ一定量ずらすだけの簡易対応（視野内に収まる保証はない）。
                self._rest_pos2 = self._rest_pos + np.array([0.0, 0.08, 0.0])
            if self._toy3:
                self._rest_pos3 = self._rest_pos + np.array([0.0, 0.0, 0.08])
            if self._toy4:
                self._rest_pos4 = self._rest_pos + np.array([0.0, 0.0, -0.08])
        # 2026-07-27：柵の内側にとどめる。人間の親は柵の外に手を出さない。
        #   新生児は仰向けで顔を横に向けているのが普通（頭位選好・右65%／Michel 1981）で、
        #   そのとき「視線の正面」は柵の外になる。実際 Viewer の初期姿勢で
        #   おもちゃが柵の向こう側に置かれ、遮られて見えなかった（目視 2026-07-27）。
        #   注意：そのぶん視線とのずれは0にならない。柵という物理的な制約が優先される
        #     ＝人間の場面としてはこちらが正しい。
        #
        # 【2026-07-28 修正】このクランプは**柵が無いときにも効いていた**。
        #   柵は新生児の体に合わせた寸法（±31cm × ±16cm）なので、体年齢を4ヶ月に
        #   上げると顔の前がこの枠の外になり、おもちゃが胸元へ落ちる。
        #   ユーザーの目視「顔の前にもっていくを押しても視界に来ないで、ちょっと
        #   胸元ぐらいに来ちゃう（多分新生児の時の相対座標に動いてるのかな？）」で発覚。
        #   → **柵があるときだけ**クランプする。柵が無ければ制約する理由が無い。
        if self._fence:
            _mgn = 0.03
            self._rest_pos[0] = float(np.clip(self._rest_pos[0],
                                              -FENCE_HALF_X + _mgn, FENCE_HALF_X - _mgn))
            self._rest_pos[1] = float(np.clip(self._rest_pos[1],
                                              -FENCE_HALF_Y + _mgn, FENCE_HALF_Y - _mgn))
            if self._rest_pos2 is not None:
                self._rest_pos2[0] = float(np.clip(self._rest_pos2[0],
                                                   -FENCE_HALF_X + _mgn, FENCE_HALF_X - _mgn))
                self._rest_pos2[1] = float(np.clip(self._rest_pos2[1],
                                                   -FENCE_HALF_Y + _mgn, FENCE_HALF_Y - _mgn))
            if self._rest_pos3 is not None:
                self._rest_pos3[0] = float(np.clip(self._rest_pos3[0],
                                                   -FENCE_HALF_X + _mgn, FENCE_HALF_X - _mgn))
                self._rest_pos3[1] = float(np.clip(self._rest_pos3[1],
                                                   -FENCE_HALF_Y + _mgn, FENCE_HALF_Y - _mgn))
            if self._rest_pos4 is not None:
                self._rest_pos4[0] = float(np.clip(self._rest_pos4[0],
                                                   -FENCE_HALF_X + _mgn, FENCE_HALF_X - _mgn))
                self._rest_pos4[1] = float(np.clip(self._rest_pos4[1],
                                                   -FENCE_HALF_Y + _mgn, FENCE_HALF_Y - _mgn))
        self._rest_pos[2] = float(max(self._rest_pos[2], 0.04))   # 床にめり込ませない
        if self._rest_pos2 is not None:
            self._rest_pos2[2] = float(max(self._rest_pos2[2], 0.04))
        if self._rest_pos3 is not None:
            self._rest_pos3[2] = float(max(self._rest_pos3[2], 0.04))
        if self._rest_pos4 is not None:
            self._rest_pos4[2] = float(max(self._rest_pos4[2], 0.04))
        # 【2026-08-18】toy1・toy2が近すぎて重なるとき、互いに引き離す
        #   （重なったままだと誤接触になり、toy1の「触れたら光る」演出が誤発火する）。
        #   toy3/toy4は角度0度なら重なることが仕様で明示的に許容されているため、
        #   _separate_toysの対象には含めない（拡張しない）。
        self._separate_toys()
        # 【2026-08-03】屈曲した手・前腕と重なるときだけ最小限ずらす（上のコメント参照）。
        #   重ならなければ無変化＝既存シーンの配置は1ビットも変わらない。
        self._rest_pos = self._avoid_arm_collision(self._rest_pos)
        if self._rest_pos2 is not None:
            self._rest_pos2 = self._avoid_arm_collision(self._rest_pos2,
                                                         toy_radius=self._toy2_radius)
        if self._rest_pos3 is not None:
            self._rest_pos3 = self._avoid_arm_collision(self._rest_pos3,
                                                         toy_radius=self._toy3_radius)
        if self._rest_pos4 is not None:
            self._rest_pos4 = self._avoid_arm_collision(self._rest_pos4,
                                                         toy_radius=self._toy4_radius)
        self._anchor = self._rest_pos + np.array([0.0, 0.0, self._tether_len])

    def _separate_toys(self):
        """toy1・toy2の中心が近すぎるとき、左右に押し離す（重なり・誤接触の防止）。

        【2026-08-18・目標F・F1】視線正面からの振り分け角度が小さいと、距離次第では
        2個のバウンディング球が重なりうる（実測で角度12度・距離8.6cm・半径2cmの
        組み合わせは重なる）。toy2=Falseなら self._rest_pos2 が None なので無効。
        """
        if self._rest_pos2 is None:
            return
        need = self._toy_radius + self._toy2_radius + TOY_COLLIDE_MARGIN
        d = self._rest_pos2 - self._rest_pos
        d[2] = 0.0                       # 高さは変えない・左右にだけ引き離す
        dist = float(np.linalg.norm(d))
        if dist >= need or dist < 1e-9:
            return
        direction = d / dist
        push = (need - dist) / 2.0
        self._rest_pos = self._rest_pos - direction * push
        self._rest_pos2 = self._rest_pos2 + direction * push

    def _avoid_arm_collision(self, pos, toy_radius=None):
        """おもちゃの候補位置 pos が、屈曲した手・前腕(TOY_COLLIDE_BODIES)と重ならない
        よう最小限ずらす。重ならなければ pos をそのまま返す。

        定数・判定方法の根拠は TOY_COLLIDE_* のコメント参照（2026-08-03 追加）。
        toy_radius: 押し離す対象の半径。省略時は self._toy_radius（toy1）を使う
            （2026-08-18・toy2向けにNoneなら従来どおりtoy1の半径＝1ビットも変わらない）。
        """
        r = self._toy_radius if toy_radius is None else float(toy_radius)
        p = np.array(pos, dtype=float)
        for _ in range(TOY_COLLIDE_ITERS):
            moved = False
            for name in TOY_COLLIDE_BODIES:
                try:
                    bid = int(self.model.body(name).id)
                except Exception:
                    continue           # このモデルに無い名前は無視（体型により変わりうる）
                # そのbodyが持つ全geomのうち、最大のbounding球半径を使う（大まかな近似）。
                radius = 0.0
                n = int(self.model.body_geomnum[bid])
                adr = int(self.model.body_geomadr[bid])
                for gi in range(adr, adr + n):
                    radius = max(radius, float(self.model.geom_rbound[gi]))
                if radius <= 0.0:
                    continue
                bpos = np.array(self.data.xpos[bid], dtype=float)
                d = p - bpos
                dist = float(np.linalg.norm(d))
                need = radius + r + TOY_COLLIDE_MARGIN
                if dist < need:
                    moved = True
                    direction = (d / dist) if dist > 1e-9 else np.array([0.0, 0.0, 1.0])
                    p = bpos + direction * need
            if not moved:
                break
        return p

    def _carry_toy(self):
        """親がおもちゃを運んでくる。TOY_APPEAR_DELAY 秒待ってから TOY_APPROACH_SEC 秒かけて動かす。

        瞬間移動させない。ワープは物理的に不自然なうえ、「勝手に動く＝予測できない」ので
        随伴性の学習を壊す（_apply_tether の注記と同じ理由）。
        運んでいる間おもちゃは動いて見えるので、視線誘導反射にとっては
        「親が注意を引く」場面そのものになる。

        行き先（アンカー）は**到着する時点の視線の正面**に決める。
        リセット時点で決めると、待っている1秒のあいだに首が回って視界の外になる
        （実測：0.4秒で首が64度回る）。親は赤ちゃんの顔の向きに合わせて差し出すので、
        到着時に決める方が人間の場面としても自然。
        """
        if not self._toy or not getattr(self, "_toy_pending", False):
            return
        self._t_since_reset += self.dt
        if self._t_since_reset < self._toy_appear_delay:
            return                                  # まだ遠くで待っている

        if not self._toy_arriving:
            # 運び始め：今の視線の正面を行き先に決め、その手前から動かし始める
            self._set_anchor()                      # _rest_pos と _anchor が決まる
            if self._toy_approach_from == "above":
                # 真上から降ろす＝柵（側面にある）に引っかからない
                away = np.array([0.0, 0.0, 1.0])
            else:
                head = self.data.body("head").xpos.copy()
                away = self._rest_pos - head
                n = float(np.linalg.norm(away))
                away = away / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            self._carry_from = self._rest_pos + away * TOY_APPROACH_DIST
            self._toy_arriving = True
            self._toy_carry_t = 0.0
            self._anchor = None                     # 運んでいる間は吊らない

        self._toy_carry_t += self.dt
        frac = min(1.0, self._toy_carry_t / max(self._toy_approach_sec, 1e-9))
        pos = self._carry_from + (self._rest_pos - self._carry_from) * frac
        self._place(self._toy_qadr, pos)
        self.data.qvel[self._toy_dadr:self._toy_dadr + 6] = 0.0
        if frac >= 1.0:
            # 到着：ここから紐に引き継ぐ（＝親が手を離してベビージムに預ける）
            self._anchor = self._rest_pos + np.array([0.0, 0.0, self._tether_len])
            self._toy_pending = False
            self._toy_arriving = False

    def place_toy_now(self):
        """登場演出を飛ばして、いますぐ視線の正面へ置く。

        【なぜ要るか】おもちゃを運ぶ `_carry_toy()` は `step()` の中にしかない。
        Viewer は物理を止めた「編集モード」（E_FREEZE=1 が既定）だと `step()` を
        一度も呼ばないため、**おもちゃが退避位置 [3,3,0.05] に取り残されたまま**に
        なっていた（ユーザーの目視「おもちゃが出てこない」2026-07-27）。
        物理を止めているあいだは登場演出そのものが意味を持たないので、
        待たずに定位置へ置く。
        """
        if not (self._toy or self._toy2):
            return
        self._set_anchor()
        if self._toy:
            self._place(self._toy_qadr, self._rest_pos)
            self.data.qvel[self._toy_dadr:self._toy_dadr + 6] = 0.0
            self._toy_pending = False
            self._toy_arriving = False
        if self._toy2 and self._rest_pos2 is not None:
            self._place(self._obj2_qadr, self._rest_pos2)
            self.data.qvel[self._obj2_dadr:self._obj2_dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _parent_intervene(self):
        """おもちゃを見失った状態が続いたら、親が視線の正面へ差し出し直す。

        【なぜ必要か】太郎が対象を視野の外へ追いやってしまうと、そこから先は
        何も起きない（動きが無いので反射も働かない）。人間の赤ちゃんは自力で
        探し当てるわけではなく、**親が顔の前に持ってくる**。既存の「登場時に
        親が運んでくる」処理と同じ場面で、それを繰り返す形にした。

        【人間の場面として】親は赤ちゃんの視線に合わせておもちゃを差し出し、
        注意を引く。定位実験でも、標的は実験者が乳児の視野へ導入する
        （Aslin & Salapatek 1975 の introduced target）。

        注意：[ARBITRARY] 何秒見失ったら差し出すか（PARENT_WAIT_SEC）と、
          何度外れたら「見えていない」とするか（PARENT_LOST_DEG）は文献値が無い。
          前者は「親が気づいて動かすまでの間」、後者は視野の半角より内側に置いた。
        """
        if not (self._parent_intervene_flag and self._toy):
            return
        if getattr(self, "_toy_pending", False):
            return                       # いま運んでいる最中
        if getattr(self, "_rest_pos", None) is None:
            return
        cid = int(self.model.camera("eye_left").id)
        eye = np.array(self.data.cam_xpos[cid], dtype=float)
        fwd = -np.array(self.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        toy = np.array(self.data.body("test_object1").xpos, dtype=float) - eye
        n = float(np.linalg.norm(toy))
        ang = 180.0 if n < 1e-9 else float(np.degrees(np.arccos(
            np.clip(np.dot(fwd, toy / n), -1.0, 1.0))))
        if ang > self._parent_lost_deg:
            self._toy_lost_t = getattr(self, "_toy_lost_t", 0.0) + self.dt
            if self._toy_lost_t >= self._parent_wait_sec:
                # 親がもう一度運んでくる（登場のときと同じ仕組みを再利用する）
                self._toy_pending = True
                self._toy_arriving = False
                self._t_since_reset = self._toy_appear_delay
                self._toy_lost_t = 0.0
                self.n_parent_help = getattr(self, "n_parent_help", 0) + 1
        else:
            self._toy_lost_t = 0.0

    def _apply_tether(self):
        """おもちゃを基準点に吊るす力（ベビージムの紐/ゴム）。

        重力を打ち消し、変位に比例した復元力＋減衰を加える＝**押せば動き、離せば戻る**。
        瞬間移動(respawn)を使わずに手元へ留めるための機構。ワープは物理的に不自然な上、
        「勝手に動く＝予測できない」ので**随伴性の学習を壊す**（Viewerでの目視で判明）。
        """
        if not self._toy:
            return
        # 持たせ方の切り替え（self._toy_mode 参照）
        if self._toy_mode == "hold":
            # 親が手に持っている＝位置を固定する。力を加えるのでなく位置を書き込む。
            if getattr(self, "_toy_pending", False):
                return                      # まだ運んでいる最中は _carry_toy が動かす
            if getattr(self, "_rest_pos", None) is None:
                return
            self._place(self._toy_qadr, self._rest_pos)
            self.data.qvel[self._toy_dadr:self._toy_dadr + 6] = 0.0
            self.data.xfrc_applied[self._toy_bid, :3] = 0.0
            return
        if self._toy_mode == "free":
            self.data.xfrc_applied[self._toy_bid, :3] = 0.0
            return
        if self._anchor is None:
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

    def _hold_toy2(self):
        """2個目のおもちゃ(toy2)を定位置に固定する（常にhold系＝押しても動かない）。

        【2026-08-18・目標F・F1】toy1のtoy_mode（hold/tether/free）とは独立させた。
        目標Fは「並べて置かれた2つを見比べる」場面で、toy2側を押す・随伴性を学ぶ
        実験は今のところ無いため、常に固定でよい。toy2=False（既定）なら
        self._rest_pos2 がNoneのままなので何もしない＝従来と不変。
        """
        if not self._toy2 or self._rest_pos2 is None:
            return
        self._place(self._obj2_qadr, self._rest_pos2)
        self.data.qvel[self._obj2_dadr:self._obj2_dadr + 6] = 0.0
        self.data.xfrc_applied[self._obj2_bid, :3] = 0.0

    def _hold_present_slots(self):
        """差し出しスロットを毎step退避位置へ置く（10択・2026-08-31）。

        _hold_toy2 と同じ「毎step置き、親のupdate()が見せている1個だけを
        上書きする」流儀。スロットが無ければ何もしない＝既存実験は不変。
        """
        for _k, _sl in enumerate(getattr(self, "_present_slots", {}).values()):
            self._place(_sl["qadr"], FAR_AWAY + np.array([2.0 + 0.5 * _k, 0.0, -2.0]))
            self.data.qvel[_sl["dadr"]:_sl["dadr"] + 6] = 0.0

    def _hold_toy3(self):
        """3個目のおもちゃ(toy3)を定位置に固定する（_hold_toy2と同じ流儀）。

        【2026-08-25新設・目標F・4語テスト】test_object3が存在しないシーン
        （共有XML）では self._obj3_qadr が None のまま＝何もしない。
        """
        if not self._toy3 or self._rest_pos3 is None or self._obj3_qadr is None:
            return
        self._place(self._obj3_qadr, self._rest_pos3)
        self.data.qvel[self._obj3_dadr:self._obj3_dadr + 6] = 0.0
        self.data.xfrc_applied[self._obj3_bid, :3] = 0.0

    def _hold_toy4(self):
        """4個目のおもちゃ(toy4)を定位置に固定する（_hold_toy2と同じ流儀）。"""
        if not self._toy4 or self._rest_pos4 is None or self._obj4_qadr is None:
            return
        self._place(self._obj4_qadr, self._rest_pos4)
        self.data.qvel[self._obj4_dadr:self._obj4_dadr + 6] = 0.0
        self.data.xfrc_applied[self._obj4_bid, :3] = 0.0

    def _spawn_toy(self):
        # おもちゃの登場を遅らせる（TOY_APPEAR_DELAY 参照）。
        #   最初は遠くに置き、太郎が落ち着いてから親が運んでくる。
        self._toy_arriving = False
        self._toy_carry_t = 0.0
        self._t_since_reset = 0.0
        self._rest_pos = None          # 到着するまで置き場所は決まっていない
        self._rest_pos2 = None
        self._rest_pos3 = None
        self._rest_pos4 = None
        # 【2026-08-18・目標F・F1】toy1・toy2のどちらかが有効なら位置を決めておく
        #   （toy2は登場演出(carry-in)を持たず常に即座に定位置＝ここで確定させる）。
        #   toy1がdelay>0のときは、この直後に FAR_AWAY へ置き直し・anchor=None に
        #   戻すので、toy1自身の見た目は従来と変わらない。
        # 【2026-08-25追記・目標F・4語テスト】toy3/toy4も同じ条件に足す
        #   （既存の呼び出し元はtoy3/toy4を渡さないので判定結果は変わらない）。
        if self._toy or self._toy2 or self._toy3 or self._toy4:
            self._set_anchor()
        if self._toy and self._toy_appear_delay > 0.0:
            self._place(self._toy_qadr, FAR_AWAY)
            self._anchor = None            # 到着するまで吊らない
            self._toy_pending = True
        elif self._toy:
            self._toy_pending = False
            self._place(self._toy_qadr, self._rest_pos)   # 支点の真下＝紐が垂れた位置
        else:
            self._toy_pending = False
            self._place(self._toy_qadr, FAR_AWAY)
        if self._toy2 and self._rest_pos2 is not None:
            self._place(self._obj2_qadr, self._rest_pos2)
            self.data.qvel[self._obj2_dadr:self._obj2_dadr + 6] = 0.0
        else:
            self._place(self._obj2_qadr, FAR_AWAY + np.array([0.5, 0.0, 0.0]))
        # 【2026-08-25新設・目標F・4語テスト】test_object3/4はこのシーンに存在
        #   しないことがある（共有XML）。存在しない(_obj3_qadr is None)ときは
        #   _place自体を呼ばない（呼ぶとmodel.jnt_qposadrがNoneでエラーになる）。
        #   既存シーン（test_object3/4なし）はここに来た時点でobj3_qadrがNoneの
        #   ままなので、以下の分岐はどちらも実行されず1ビットも挙動が変わらない。
        if self._obj3_qadr is not None:
            if self._toy3 and self._rest_pos3 is not None:
                self._place(self._obj3_qadr, self._rest_pos3)
                self.data.qvel[self._obj3_dadr:self._obj3_dadr + 6] = 0.0
            else:
                self._place(self._obj3_qadr, FAR_AWAY + np.array([1.0, 0.0, 0.0]))
        if self._obj4_qadr is not None:
            if self._toy4 and self._rest_pos4 is not None:
                self._place(self._obj4_qadr, self._rest_pos4)
                self.data.qvel[self._obj4_dadr:self._obj4_dadr + 6] = 0.0
            else:
                self._place(self._obj4_qadr, FAR_AWAY + np.array([1.5, 0.0, 0.0]))
        # 【10択・2026-08-31】差し出しスロットは常に退避から始める
        for _k, _sl in enumerate(getattr(self, "_present_slots", {}).values()):
            self._place(_sl["qadr"], FAR_AWAY + np.array([2.0 + 0.5 * _k, 0.0, -2.0]))
            self.data.qvel[_sl["dadr"]:_sl["dadr"] + 6] = 0.0
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
        # 眼球を正中位に戻してからおもちゃを置く。順序が重要：
        #   おもちゃは「視線の正面」に置くので、眼球がずれたままだと
        #   あさっての方向（柵の外）に置かれる（研究日誌 2026-07-26 続き7）。
        from infant_body import center_eyes, apply_eye_centering_spring
        # 【2026-08-17・配線改修ステージA】self._eye_rest_vertical_deg/self._eye_centering は
        #   コンストラクタ引数（未指定ならNone）。Noneを渡すとcenter_eyes/
        #   apply_eye_centering_spring側がinfant_body.EYE_REST_VERTICAL_DEG/EYE_CENTERING
        #   （後方互換の既定値＝環境変数E_EYE_REST_V/E_EYE_CENTERING）にフォールバックするので、
        #   これらを渡さない既存の呼び出し元は1ビットも挙動が変わらない。
        center_eyes(self.model, self.data, vertical_deg=self._eye_rest_vertical_deg)
        # 【2026-08-17新設】眼球の受動的な中央復帰バネ（既定OFF＝EYE_CENTERING）。
        #   center_eyesはqposを戻すだけで以降を留める力が無いため、既定では従来通り
        #   筋の受動力で可動域の壁まで転がり落ちる。ONのときだけjnt_stiffness等を
        #   入れる（詳細はinfant_body.pyのEYE_CENTERING直前のコメント参照）。
        apply_eye_centering_spring(self.model, self.data,
                                   vertical_deg=self._eye_rest_vertical_deg,
                                   enabled=self._eye_centering)
        mujoco.mj_forward(self.model, self.data)   # 眼のカメラ姿勢を更新してから配置
        self._spawn_toy()                  # 落ち着いた後の肩位置を見て配置
        self._glow_until = -1e9            # 点灯の余韻を持ち越さない
        self._vision_cache = None          # data.time が巻き戻るのでキャッシュを捨てる
        if self._orienting is not None:
            self._orienting.reset()        # 前エピソードの画像を持ち越さない
        if self._vergence is not None:
            self._vergence.reset()         # F2-15：前エピソードの目標輻輳角を持ち越さない
        if getattr(self, "_parent_labeling", None) is not None:
            self._parent_labeling.reset()  # 【2026-08-18新設・F1-3】前エピソードの状態を持ち越さない
        if getattr(self, "_word_schedule", None) is not None:
            self._word_schedule.reset()    # 【2026-08-21新設・F1-4h】同上（試行の頭出し）
        if self._toy and not self._toy_shape.startswith("plate:") and self._toy_shape != "asis":
            # 【F2-9C】イラスト板は色を塗らない（テクスチャに赤が乗算されて絵が潰れる）
            # 【2026-09-03・F2-49で発覚】asis（XMLで色・テクスチャを持つ物）も塗らない。
            #   起動時（上）は除外していたが reset では除外が抜けており、枠toy1の実物
            #   スキャン（靴）が毎エピソード赤に染まっていた（F2-19以降の全走行に影響）。
            self.model.geom_rgba[self._toy_gadr] = self._toy_rgba_off
            self.toy_lit = False
        mujoco.mj_forward(self.model, self.data)
        # 【2026-08-15・座位保持の学習】層2（立ち直り反射）の三半規管フィルタが
        #   前エピソードの「最近の角速度の平均」(baseline)を持ち越さないようにする。
        #   既定OFF（righting_reflex=False）では self._righting_canals 自体が
        #   一度も作られない＝この行は何もしない（既存実験の挙動は不変）。
        if getattr(self, "_righting_canals", None) is not None:
            self._righting_canals.reset()
        # 座らせ直し（ユーザーの確定事項：env.reset()を呼ばずエピソード継続で座位に
        #   戻す）の基準姿勢。scene復元直後＝「座位開始時」のqposを記録する。
        #   常に記録する（cfg.posture_reflex等がFalseでも計算だけは行うが、
        #   _check_posture_fallはself.taroが無ければ一度も呼ばれないので無害）。
        #
        # 【2026-08-16 復帰先の修正】ここ（reset_model内）で記録した値は、まだ
        #   「仰向け＋jitter＋settle」の既定姿勢でしかない。シーンの本当の姿勢は
        #   run/scene_tools/e_scene.py の reset_to_scene()/build() が
        #   **env.reset()の呼び出しのあとに** apply_state() で qpos を書き込む
        #   ため、ここで記録した値は座位ではなく仰向けになる（2026-08-16に実際に
        #   再現・実測済み。別の実装担当が録画作成時に発見し実行時の属性上書きで
        #   回避した。触ってよいファイルに e_scene.py が含まれないためここでは
        #   直接直せない）。
        #   → ここでは**暫定値**として記録しつつ、_posture_baseline_pending を立てる。
        #   step() の先頭（このstepでまだ何も物理を進めていない時点）で、
        #   pending中なら qpos を取り直して確定させる。reset_to_scene()は
        #   env.reset()の後、step()が呼ばれる前に apply_state()等で最終姿勢まで
        #   仕上げるので、最初のstep()の時点では qpos は既にシーンの最終姿勢に
        #   なっている。シーンを使わない既存実験（env.reset()の直後すぐstep()を
        #   呼ぶだけ）は、reset()〜最初のstep()の間にqposを動かす者がいないので
        #   この値は変わらず、挙動は今までと1ビットも変わらない。
        self._posture_seated_qpos = self.data.qpos.copy()
        # 頭の高さ報酬（reward="posture_height"）の基準値。座位開始時からの
        #   相対差を使う（絶対値ではなく相対値。設計3節(5)：月齢で体格が
        #   変わるため）。常に記録する（reward!="posture_height"のときは
        #   posture_height_reward()自体が呼ばれないだけで、記録自体は無害・軽量）。
        #   上と同じ理由で暫定値。_posture_baseline_pending で最初のstep()時に確定させる。
        self._posture_head_height_ref = float(self.data.body("head").xpos[2])
        self._posture_baseline_pending = True
        # 座り直しの遅延（posture_fall_delay_sec）用タイマー。前エピソードの
        #   カウントを持ち越さない。
        self._posture_fall_since = None
        return self._get_obs()

    def step(self, action):
        # 【2026-08-16】座り直しの復帰先(_posture_seated_qpos)を、このstep()が
        #   物理を1歩も進める前・qposを誰も書き換えていない時点で確定させる。
        #   reset_model()の直後にreset_to_scene()/build()がapply_state()等で
        #   qposを座位へ書き換えるのは「env.reset()の呼び出しが終わったあと・
        #   最初のstep()が呼ばれる前」なので、ここが「シーンの最終姿勢」を
        #   確実に捉えられる最初のタイミングになる（reset_model()のdocstring参照）。
        if getattr(self, "_posture_baseline_pending", False):
            self._posture_seated_qpos = self.data.qpos.copy()
            self._posture_head_height_ref = float(self.data.body("head").xpos[2])
            self._posture_baseline_pending = False
        # 吊り力は物理を進める前に設定する（xfrc_appliedはframe_skip回ぶん効く）。
        # 旧実装の「離れたら瞬間移動で置き直す(respawn)」は**廃止**。目視でワープが
        # 見えたうえ、随伴性（自分の行為→結果）を壊すため。代わりに吊り紐で留める。
        self.respawned_this_step = False
        self._parent_intervene()   # 見失ったら親が差し出し直す
        self._carry_toy()          # 親がおもちゃを運んでくる（登場を遅らせる仕組み）
        self._apply_tether()
        self._hold_toy2()          # 2個目のおもちゃ（toy2=False なら何もしない）
        self._hold_present_slots()  # 【10択】差し出しスロット（無ければ何もしない）
        self._hold_toy3()          # 3個目のおもちゃ（2026-08-25新設・toy3=False/無しなら何もしない）
        self._hold_toy4()          # 4個目のおもちゃ（同上）
        # 【2026-08-18新設・F1-3】親のfollow-in labeling。_hold_toy2の直後に呼ぶ
        #   ＝振っている間はここでtoy1/toy2の位置を上乗せで揺らす（_apply_shake）。
        #   parent_labeling未指定（enabled=False）なら常にNoneを返すだけ＝
        #   1ビットも既存の挙動を変えない。
        self._parent_utterance = self._parent_labeling.update(self)
        # 【2026-08-21新設・F1-4h】語の再生装置。親のfollow-in labelingが同stepで
        #   発話済み（parent_utterance is not None）でなければ呼ぶ＝両方が同時に
        #   耳へ入って発話イベントが潰れ合う事態を避ける（word_testはparent_labeling
        #   を使わないテスト場面での使用を想定しており、通常は競合しない）。
        if self._parent_utterance is None:
            self._parent_utterance = self._word_schedule.update(self)
        self._update_glow()
        if self._vor is not None:
            # 方策の眼球出力を捨て、VORの指令に差し替える（皮質は反射弓に介入しない）
            action = self._vor.override(action, self.model, self.data, self.dt)
        if self._orienting is not None:
            # 前回描画された画像から計算済みの方向を、首・（VOR後の）目に加算する
            action = self._orienting.apply(action)
        if self._vergence is not None:
            # F2-15：輻輳反射（設計「1. 全体の構成」）。定位反射の共同運動成分を
            #   上書きせず、左右差だけを加算する（additive）。
            action = self._vergence.apply(action)
        # 【2026-08-15・座位保持の学習】層1（姿勢制御反射）・層2（立ち直り反射）。
        #   env.taro は run/taro_setup.py の _setup_postural_gate/_setup_righting_damper
        #   がposture_reflex/righting_reflexのどちらかTrueのときだけ配線する
        #   （仕様C節・taro_setup.py側の設計判断、作業記録「想定外」参照）。
        #   既定（両方False）ではself.taro自体が存在せず、getattrがNoneを返して
        #   このブロックは1行も実行されない＝既存実験の挙動は1ビットも変わらない。
        taro = getattr(self, "taro", None)
        if taro is not None and getattr(taro, "postural_gate", None) is not None:
            action = taro.postural_gate.apply(action, self.data.qpos, touch=self.touch)
        if taro is not None and getattr(taro, "righting_damper", None) is not None:
            action = self._apply_righting_damper(action, taro)
        out = super().step(action)
        self._pin_root()           # 椅子とベルトが体を留める（実験の測定条件）
        # 倒れ判定と座り直し（ユーザーの確定事項：エピソード継続。env.reset()も
        #   reset_model()も呼ばない。terminated/truncatedもいじらない）。
        if taro is not None:
            self._check_posture_fall(taro)
        # 【2026-08-18新設・F1-3】親の発話をinfo経由で受け渡す（trainerが耳→連合器へ渡す）。
        #   発話が無かったstep（ほとんど）はNoneのままなのでinfoは1キーも増えず、
        #   既存のinfo読み取り側（あれば）に影響しない。
        if self._parent_utterance is not None:
            obs, rew, term, trunc, info = out
            info = dict(info)
            info["parent_utterance"] = self._parent_utterance
            out = (obs, rew, term, trunc, info)
        return out

    # -------------------------------------------------- 座位保持の学習（2026-08-15）
    def _posture_trunk_tilt_deg(self):
        """体幹の傾き[度]。run/scene_tools/e_scene.py の fingerprint() と
        まったく同じ式（骨盤→胸のベクトルが水平から何度上がっているか）。
        """
        axis = (np.array(self.data.body("upper_body").xpos, dtype=float)
                - np.array(self.data.body("hip").xpos, dtype=float))
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return 0.0
        return float(np.degrees(np.arcsin(np.clip(axis[2] / n, -1.0, 1.0))))

    def _apply_righting_damper(self, action, taro):
        """立ち直り反射（層2）を適用する。

        頭角速度は三半規管（高域通過フィルタ）を経由させる
        （taro_core/src/brain/spinal_cord/righting_damper.py の
        【前庭覚由来の頭角速度の取得方法】docstring参照。三半規管を通さないと
        「ゆっくりした傾きに反応する」というその10の問題を再現してしまう）。
        support_fraction は run/scene_tools/e_scene.py の build() が
        env.unwrapped._pinned_groups として配線した pinned_groups から作る。
        """
        if getattr(self, "_righting_canals", None) is None:
            from semicircular_canals import SemicircularCanals
            self._righting_canals = SemicircularCanals(
                age_months=float(getattr(self, "age", 0.0)))
        head_bid = int(self.model.body("head").id)
        w_world = np.array(self.data.cvel[head_bid][:3], dtype=float)
        sensed = self._righting_canals.update(w_world, float(self.dt))
        # head_tilt（前後・Y軸）に対応する成分を取り出す。太郎は常に world +X を
        #   向く姿勢を基準にする（postural_gate.pyの docstring【関節構造の実測】と
        #   同じ前提）ため、ワールドY軸まわりの角速度がほぼそのまま前後(pitch)に
        #   対応する、という近似[Tier3・簡略化。head bodyのローカル軸への厳密な
        #   投影はしていない。e_vor.pyのVORは眼球ごとにローカル軸へ投影しているが、
        #   本モジュールの対象(act:head_tilt)は1本のみで軸もモデル全体で一貫して
        #   Y軸のため、簡略化の影響は小さいと判断した]。
        head_angular_velocity_dps = float(np.degrees(sensed[1]))
        from spinal_cord.righting_damper import support_fraction_from_pinned_groups
        pinned_groups = getattr(self, "_pinned_groups", None) or ()
        support_fraction = support_fraction_from_pinned_groups(pinned_groups)
        return taro.righting_damper.apply(
            action, taro.righting_neg_indices, taro.righting_pos_indices,
            head_angular_velocity_dps, support_fraction)

    def _check_posture_fall(self, taro):
        """倒れたら座らせ直す（ユーザーの確定事項：エピソード継続。env.reset()を
        呼ばない。座位開始時に記録したqposへ書き戻すだけの軽量な処理）。

        cfg.posture_fall_deg を trunk_tilt_deg が**下回ったら**「倒れた」と
        判定する（仕様C節3の指示どおり）。posture_fall_deg<=0なら判定自体を
        無効化できる（仕様の必須要件「既定でこの判定が有効になって既存実験に
        影響してはいけない」）。既定値は run/config.py 参照（Tier3、作業記録に理由）。

        【2026-08-16 遅延を追加】以前は閾値を割った**瞬間に**qposを書き戻していた。
        実測（15秒間・posture_fall_deg=20.0）で0.19秒に1回も発動し、映像が
        「ゆっくり倒れて起こされる」ではなく痙攣のように見えた（倒れきる前に
        戻すので、太郎は「倒れる」という経験そのものができていなかった）。
        cfg.posture_fall_delay_sec だけ待ってから書き戻す。待っている間に
        閾値より上へ戻ったらカウントを取り消す（`_posture_fall_since = None`）。
        0なら従来どおり即座に戻す（後方互換）。
        """
        fall_deg = float(getattr(taro.cfg, "posture_fall_deg", 0.0) or 0.0)
        if fall_deg <= 0.0:
            self._posture_fall_since = None
            return
        tilt = self._posture_trunk_tilt_deg()
        if tilt >= fall_deg:
            self._posture_fall_since = None    # 途中で持ち直したのでカウント取り消し
            return
        delay = float(getattr(taro.cfg, "posture_fall_delay_sec", 0.0) or 0.0)
        if delay > 0.0:
            if self._posture_fall_since is None:
                self._posture_fall_since = float(self.data.time)   # 割った時刻を記録
            if (float(self.data.time) - self._posture_fall_since) < delay:
                return    # まだ待っている（戻さない＝倒れる経験をさせる）
        q = getattr(self, "_posture_seated_qpos", None)
        if q is None:
            return
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._posture_fall_since = None    # 戻したのでカウントをリセット

    def posture_height_reward(self):
        """頭の高さ報酬（reward="posture_height"）。座位開始時（reset_model直後、
        scene復元後）の頭の高さからの相対差。絶対値ではなく相対値を使う
        （設計3節(5)：月齢で体格が変わるため、絶対値だと月齢間で比較できない）。

        【なぜ座らせ直しの直後に基準を更新しないか】実装しやすい方（更新しない）を
        選んだ。座り直すたびに基準をリセットすると、エピソード内で何度倒れても
        「毎回ゼロから採点し直す」ことになり、繰り返し倒れることへの罰が薄まる
        （倒れて座り直した直後は差がほぼ0＝高評価に見えてしまう）。座位開始時の
        高さを固定基準にしておけば、エピソード全体を通じて「どれだけ崩れずに
        座り続けられたか」を一貫して評価できる。

        筋肉ペナルティ（体幹の対象筋のみ）は依頼書の指示どおり今回のスコープに
        含めていない（将来リーチングと同時学習させる拡張を見越し、体幹以外の
        筋を罰さないという要件があるため、対象筋の絞り込みが必要になる。
        頭高さの相対差そのものの計算を優先した）。
        """
        ref = getattr(self, "_posture_head_height_ref", None)
        if ref is None:
            return 0.0
        cur = float(self.data.body("head").xpos[2])
        return cur - ref

    # ------------------------------------------------------------------
    # 体そのものを空間に留める（2026-07-29 追加）
    # ------------------------------------------------------------------
    def pin_root(self, on=True, qpos=None):
        """体全体（自由関節）を今の位置・向きに留める／解除する。

        【なぜ要るか、2026-07-29】リクライニング姿勢で関節をバネで固定しても、
        **体そのものが椅子から横へ転がり落ちた**（実測：頭が中心から23cm ずれ、
        視線が真後ろを向いた）。関節のバネは `jnt_stiffness` を使うが、
        体を空間に置いている自由関節（`mimo_location`）にはバネが効かないため。

        【人間ではどうか】乳児用チェアは**股ベルトの装着が安全基準で義務**。
        リーチの実験も乳児を椅子に固定して行う（von Hofsten 1982）。
        ＝体を留めること自体は人間の実験条件をそのまま写したもの。

        注意：ただし実装は工学的な固定（毎ステップ位置を書き戻す）であって、
          ベルトの物理を再現したものではない。ユーザーの判断（2026-07-29）
          「リーチングが発動するかの実験だから、関節の固定も工学的にやっていい」。
          逸脱リストに登録すること。

        Args:
            qpos: 留める位置・向き（7要素）。None なら今の値
        """
        if not on:
            self._pin_qpos = None
            return None
        if getattr(self, "_root_qadr", None) is None:
            # 注意：【2026-07-29 に踏んだ】最初の版は関節名に "mimo_location" を含むかで
            #   探し、見つからなければ「おもちゃ以外の自由関節」を拾う保険を付けていた。
            #   ところが**保険がおもちゃの関節を拾い**、体ではなくおもちゃを留めていた
            #   （実測：留めた位置が (3.5, 3.0, 0.05) ＝ おもちゃの退避位置 FAR_AWAY）。
            #   その間、体は椅子から転がり落ち続けていた。
            #   ＝落とし穴チェックリスト 項67「体を動かす自由関節は、思っている
            #     body に無い」の再発。名前で当てにいかず、**体の中身で判定する**。
            #   ⇒ 太郎の体は **body 名が "mimo_location"**（関節名は "mimo_orientation"）。
            #     このモデルには自由関節が3つある：
            #       test_object1  … おもちゃ
            #       test_object2  … 使っていない予備の物体（遠くに置いてある）
            #       mimo_location … 太郎の体
            #     「おもちゃ以外」で選ぶと **test_object2** を掴む。名指しで取る。
            adr = None
            for j in range(self.model.njnt):
                if int(self.model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    continue
                bid = int(self.model.jnt_bodyid[j])
                if (self.model.body(bid).name or "") != ROOT_BODY:
                    continue
                adr = int(self.model.jnt_qposadr[j])
                self._root_dadr = int(self.model.jnt_dofadr[j])
                self._root_bid = bid
                break
            self._root_qadr = adr
            if adr is None:
                print(f"[pin] 注意体（body='{ROOT_BODY}'）の自由関節が見つからない")
                return None
            if adr is not None:
                p = np.array(self.data.qpos[adr:adr + 3], dtype=float)
                bn = self.model.body(self._root_bid).name or "?"
                # 注意：拾った関節が本当に体か確かめる。遠くにあるなら別物を掴んでいる
                if float(np.max(np.abs(p[:2]))) > 1.0:
                    print(f"[pin] 注意体のつもりで掴んだ関節が遠くにある "
                          f"body={bn} 位置={np.round(p, 3)}。留めるのを中止する")
                    self._root_qadr = None
                    return None
                print(f"[pin] 体の自由関節: body={bn} qpos[{adr}:{adr+7}]")
        if self._root_qadr is None:
            print("[pin] 注意体の自由関節が見つからないので留められない")
            return None
        q = (self.data.qpos[self._root_qadr:self._root_qadr + 7].copy()
             if qpos is None else np.asarray(qpos, dtype=float))
        self._pin_qpos = q
        return q

    def pin_joints(self, names=None, on=True):
        """指定した関節を「今の角度」に完全固定する（毎ステップ書き戻す）。

        【なぜバネでなく書き戻しか、2026-07-29】椅子の支え（バネ 200N·m/rad）でも
        体幹がゆっくり動き続け、リクライニング姿勢で
          ・目が3秒で2cm動く
          ・**VOR が頭の動きを打ち消そうとして眼球が可動域の上限（+33度）に張り付く**
        という問題が残った。眼が上を向いたままでは視線の実験が成立しない。
        ⇒ ユーザーの判断（2026-07-29）「体幹も固定しちゃっていいよ」により、
          バネではなく完全固定にする。

        注意：【逸脱】これは工学的な固定であって人間の体の性質ではない。
          人間のリーチ実験も乳児を椅子とベルトで支えるが、体幹はわずかに動く。
          **リーチングが発動するかを見る実験のための測定条件**であり、
          太郎の身体の性質を主張するものではない。逸脱リストに登録すること。

        Args:
            names: 固定する関節の名前（":" の後ろ）。None なら解除
        """
        if not on or not names:
            self._pin_j = None
            return 0
        want = set(names)
        pins = []
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            nm = (self.model.joint(j).name or "").split(":")[-1]
            if nm not in want:
                continue
            qadr = int(self.model.jnt_qposadr[j])
            dof = int(self.model.jnt_dofadr[j])
            pins.append((qadr, dof, float(self.data.qpos[qadr])))
        self._pin_j = pins
        return len(pins)

    def _pin_root(self):
        """留める設定があれば、毎ステップ体と指定関節を元の状態へ戻す。"""
        q = getattr(self, "_pin_qpos", None)
        if q is not None:
            a = self._root_qadr
            self.data.qpos[a:a + 7] = q
            self.data.qvel[self._root_dadr:self._root_dadr + 6] = 0.0
        for qadr, dof, val in (getattr(self, "_pin_j", None) or ()):
            self.data.qpos[qadr] = val
            self.data.qvel[dof] = 0.0

    def get_vision_obs(self):
        """MIMoの壊れたgym描画を迂回し、眼球カメラを生APIで直接描画する（D側と同じ）。

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
        # 注意：data.time はリセットで巻き戻るので、時間が戻ったら必ず描き直す。
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
            # 重大なバグ修正（2026-07-20、配線チェックで発覚）：
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
        if self._orienting is not None and "eye_left" in imgs:
            # 新しく描画された画像でだけ方向を更新する（キャッシュ流用時は呼ばれない＝重複計算を防ぐ）
            # 注意：2026-07-27：一度「両目の平均」を渡す実装にしたが**撤回した**。
            #   理由：おもちゃは顔から約9cmと近く、両眼視差が大きいため、
            #   単純に平均すると像が二重になってぼやける。実測で右10度の条件が
            #   後半ずれ 0.148 → 0.629 と悪化した。
            #   人間の新生児も両眼の統合（立体視）は生後3〜4ヶ月まで成立しないので、
            #   **単眼で処理するのが新生児として正確**。
            #   注意：そのぶん左右非対称が残る（左20度は 0.095 で成功、右20度は 0.64 で失敗）。
            #     左目の視野の中心は顔の正中より左にあるため、右側が構造的に不利。
            #     単眼処理の帰結であり、逸脱リストに記録する。
            self._orienting.update(imgs["eye_left"])
        if self._vergence is not None and "eye_left" in imgs and "eye_right" in imgs:
            # F2-15：定位反射と同じ場所で、両目の画像が新しく描画された時だけ呼ぶ
            #   （設計「5. 環境への配線」）。
            self._vergence.update(imgs["eye_left"], imgs["eye_right"])
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
        if self._toy_shape.startswith("plate:") or self._toy_shape == "asis":
            return      # 【F2-9C】イラスト板は接触発光なし（絵が赤/黄に潰れるため）。asis も同じ（2026-09-03）
        if any(c != "world" for c in self.toy_contacts()):
            self._glow_until = float(self.data.time) + GLOW_HOLD_S
        lit = float(self.data.time) < getattr(self, "_glow_until", -1e9)
        if lit != self.toy_lit:
            self.model.geom_rgba[self._toy_gadr] = TOY_RGBA_ON if lit else self._toy_rgba_off
            self.toy_lit = lit

    def anchor_distance(self):
        """おもちゃが吊り下げ基準点からどれだけずれているか[m]（押された量の目安）。"""
        if self._anchor is None:
            return 0.0
        return float(np.linalg.norm(self.data.body("test_object1").xpos - self._anchor))
