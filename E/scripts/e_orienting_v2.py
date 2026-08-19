"""視線誘導反射（新版）＝設計図（`E/docs/視線誘導反射_設計図.md`）に基づく実装。

【現行 e_orienting.py からの変更点】
  ステップ1（動き検出）    ： 単フレーム差分 → 複数フレームスケール比較
  ステップ2（中心バイアス）： 6マス分割 → 画素ごとの中心重み
  ステップ3（方向の決定）  ： 全マス合計 → 側方抑制で1箇所を選び、その周辺で重心
  ステップ4（出力）        ： 毎tick連続加算 → 階段状サッケード（未実装）
  ステップ5（IOR）         ： なし → 一度向いた場所を数百ms 抑制（未実装）
  保留                    ： egomotion 補正（残差法は取り除く。遠心性コピーが要る
                            本物の補正は保留、簡易近似はしない）

【ステップ3の設計変更・2026-07-26】
当初は「閾値以上の画素で全体の重心」だったが、これは**離れた2箇所を平均してしまい、
何もない中間を指す**という欠陥があった（人工画像テストで発覚）。
文献（調査B）が言っていたのは population vector 単独ではなく
**側方相互作用によるソフトな winner-take-all** だった：
  ・上丘に側方抑制がある＝実測された事実（Munoz & Istvan 1998、サル上丘で直接記録）
  ・その競合が1つの山に収束する＝数理モデル（Kopecz & Schöner 1995、
    実際のサッケードデータに合う工学的モデル）
  ・注意：「必ず1つが勝つ」わけではない。標的が近いと人間も中間へサッケードする
    （global effect＝大域効果、実際に観察される現象）。
    メキシカンハット型のカーネルは距離に応じてこの両方を自然に出す。

注意：実装中。ステップ4・5は未実装。
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# 関節への指令を、身体の駆動方式（筋肉2本／モーター1つ）に合った形で書き込む共通の写像。
# 反射がこれを飛ばして直接書くと、筋肉モデルでは負の指令が消えて片方向にしか動けなくなる。
import os as _os, sys as _sys
_CORE_BRAIN = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, _os.pardir,
    "taro_core", "src", "brain"))
if _CORE_BRAIN not in _sys.path:
    _sys.path.insert(0, _CORE_BRAIN)
from spinal_cord.cpg import write_joint_command as _write_joint_command

# 網膜の中心-周辺抑制（背景の流れと対象の動きを分ける）
_CORE_SENSES = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, _os.pardir,
    "taro_core", "src", "senses"))
if _CORE_SENSES not in _sys.path:
    _sys.path.insert(0, _CORE_SENSES)
from retina import object_motion as _object_motion


def _dofadr_of(model, act_i):
    """そのアクチュエータが動かす関節の dof アドレス（角速度を読むため）。"""
    try:
        jid = int(model.actuator_trnid[act_i, 0])
        if jid < 0:
            return None
        return int(model.jnt_dofadr[jid])
    except Exception:
        return None


def _qposadr_of(model, act_i):
    """そのアクチュエータが動かす関節の qpos アドレス（今の角度を読むため）。"""
    try:
        jid = int(model.actuator_trnid[act_i, 0])
        if jid < 0:
            return None
        return int(model.jnt_qposadr[jid])
    except Exception:
        return None

# ---- ステップ1（動き検出）------------------------------------------------
# frame index の 1・5・20 前と現フレームを比べる（10Hzなら 0.1・0.5・2秒 前に対応）
# [Tier3・ARBITRARY] 人間はフレーム記憶でなく連続的な時間フィルタ。効果のみ近似。
# 2026-07-27：(1,5,20) から (1,3) へ。制御周期 0.02秒なので **20ms と 60ms** 前。
#
# 【なぜ変えたか】20フレーム前＝**0.4秒前**との比較は、その間に眼球が動けば
#   必ず視野全体の流れを拾う。実測：対象が視野の端（ずれ -0.60）にあるのに
#   反射の出力が -0.15 まで潰れ、定位が振動した。
#   時間の幅を振ると、後半の平均ずれが 0.249（1,5,20）→ 0.076（1のみ）と改善。
#
# 【文献】網膜の一過性チャネルのインパルス応答は
#   ・ゼロ交差まで ON型 62±1ms / OFF型 71±1ms
#     （Chichilnisky EJ & Kalmar RS 2002 J Neurosci 22:2737、マカク・白色雑音法）
#   ・パラソル細胞の時定数：立ち上がり 48.6ms・減衰 22.2ms
#     （Manookin MB et al. 2018 Neuron、DOI 10.1016/j.neuron.2018.02.006）
#   ＝意味のある「過去のフレーム」はせいぜい数十ms。0.4秒は**5〜8倍長すぎた**。
TIME_SCALES = tuple(int(x) for x in
                    _os.environ.get("E_TIME_SCALES", "1,3").split(","))

# ---- サッケード周辺の視覚抑制 --------------------------------------------
# 【根拠】上丘中間層のサッケード直前ニューロンが、GABA作動性ニューロンを介して
#   **上丘浅層の視覚応答ニューロンを抑制する**回路が実証されている
#   （Phongphanphanee P et al. 2011 J Neurosci [PMID 21307233]、IPSC潜時 6.08±1.22ms）。
#   知覚レベルの時間窓はサッケード開始の 100ms前 〜 150〜200ms後。
#   抑制の強さは**若いほど強い**（成人で約3倍、8〜18歳で約10倍。
#     Bruno P et al. 2006 J Neurophysiol [PMID 16407425]）。新生児のデータは無い。
# 注意：[簡略化] 本物は「感度の低下」で完全遮断ではないが、ここでは
#   **その間の画像を動き検出に使わない**（＝サッケードをまたぐ比較をしない）
#   という強い形にしている。感度を下げるだけでは、流れが対象より強いままになるため。
# 注意：[ARBITRARY] 0.15秒という長さは、知覚の時間窓（150〜200ms後）から取った暫定値。
SACC_SUPPRESS_SEC = float(_os.environ.get("E_SACC_SUPPRESS", "0.15"))
# 眼球が動いているあいだも視覚を使わない（時間ではなく実際の速さで判定）。
#   時間窓だけだと、サッケードが終わったあとも眼球は慣性と粘りで動き続け、
#   その流れを動き検出が拾って**対象が無くても撃つ**（実測：対象を遠方へ
#   退避させた条件でも撃つ率109%）。
#   注意：[ARBITRARY] 閾値に文献値は無い。眼球の可動域45度・サッケード1発が数度
#     であることから、静止とみなせる速さとして置いた暫定値。
#   E_EYE_STILL=0 で切れる（アブレーション用）。
EYE_STILL_DEG_S = float(_os.environ.get("E_EYE_STILL", "1.5"))
# 実測（撃つ閾値の探索）：対象が無い条件で撃つ率が
#   切る 109% ／ 1.5度/秒 82% ／ 3.0度/秒 100% ／ 6.0度/秒 105%
# ＝1.5 が最良。ただし**根本的には消えない**（眼球が動けば必ず視野が流れる）。
#
# 2026-07-27 の気づき：「対象が無いのに撃つ」を悪と決めつけるのは誤りかもしれない。
#   ユーザーの目視で「おもちゃを揺らさなくても中心へ寄せようとする」ことが分かった。
#   太郎自身の微小な動き（首が重力で倒れる・眼球が動く）で視野が流れ、
#   **静止している対象の縁も画面上で動く**ため検出できている。
#   人間も固視微動（マイクロサッケード・ドリフト・トレモア）を止められず、
#   網膜像を人工的に完全静止させると像は数秒で消える（静止網膜像の消失）。
#   ＝「自己運動を消しきる」と、静止した対象が見えなくなる。
#   今日入れた対策（時間の幅を短く・サッケード後の抑制・網膜の中心-周辺抑制）は
#   どれも流れを消す方向なので、効かせすぎないこと。

# ---- 網膜の中心-周辺抑制（object motion sensitivity）----------------------
# 【なぜ要るか・2026-07-27】眼球が動くと視野全体が流れる。とくに**床と背景の
#   水平な境界線**が強い動き信号を出し、線は画面の全幅にわたるので総量で対象を
#   圧倒する。左右方向は線が対称なので害がないが、**上下方向は重心を完全に
#   支配され、定位が失敗していた**（対象の実際のずれ +0.44 に対し出力 +0.03）。
# 【人間はどうか】網膜と上丘に「中心と周辺の動きが一致していたら抑える」仕組みが
#   実在する（詳細は taro_core/src/senses/retina.py）。皮質を使わないので
#   新生児でも働きうる。＝環境から境界を消して回避するのではなく、
#   **太郎に足りない器官を足す**のが人間模倣として正しい。
# E_OMS=0 で切れる（アブレーション用）。
USE_OMS = _os.environ.get("E_OMS", "1") == "1"
OMS_SURROUND_DEG = float(_os.environ.get("E_OMS_SURROUND", "20.0"))
OMS_UNIFORM_GAIN = float(_os.environ.get("E_OMS_GAIN", "0.49"))

# ---- ステップ2（中心視野の優位）------------------------------------------
# 2026-07-27：ガウス窓（σ=0.32）による近似をやめ、上丘の実測に基づく
#   対数極座標マッピングに置き換えた（`taro_core/src/brain/superior_colliculus.py`）。
#
#   旧：exp(-R²/2σ²) を掛ける      … 形も強さも [Tier3・ARBITRARY]
#   新：Ottes et al. 1986 の変換で上丘座標へ写し、そこで重心を取る [Tier2]
#       A=3° / Bu=1.4mm / Bv=1.8mm/rad
#
#   注意：なぜ形を変えたか：ガウス窓は裾が軽く（数σ先でほぼゼロ）、周辺の刺激を捨てる。
#     上丘のマグニフィケーションは 1/(R+A)ⁿ ＝ べき乗則で裾が重い。**別種の関数**。
#     実測でσを大きくするほど対象の位置を正しく出せたのは、裾が重い形に
#     近づいていたため。σをどう調整しても形が違う以上たどり着けない。
#   注意：さらに「平らな画像で重心を取る」のと「歪んだ地図で重心を取ってから逆変換」は
#     数学的に別物（Jensen の不等式）。実際の上丘は後者
#     （Goossens & Van Opstal 2012 J Neurophysiol）。
#
# 2026-07-27（第2版）：「重みを掛ける」のもやめ、**上丘の格子へ写してから
#   競合させる**方式にした（E_ORIENT_BIAS=grid、既定）。
#
#   【なぜ】重みを掛ける方式には2つの欠陥が残っていた。
#     ①受容野が無い。上丘のニューロンは1画素でなく視野の広い範囲を担当する
#       （浅層で2〜20度）。この段が無いと、大きな対象の左右の縁が鋭く分離した
#       まま競合に入り、**片方の縁だけが勝って定位の向きが逆になる**
#       （実測：見かけ26度の対象で符号が4/6、7度なら6/6）。
#       人間は単一の対象なら縁を統合して正確に中心へ向く
#       （Kilpeläinen & Georgeson 2018）。逆方向へ飛ぶ報告は無い。
#     ②競合が画像の座標で起きていた。上丘の側方抑制は**組織の上の配線**で
#       起きる（Munoz & Istvan 1998「局所抑制性介在ニューロンのネットワーク」）
#       ので、届く範囲は上丘の mm で決まり、視野角では決まらない。
#       上丘の地図は中心が引き伸ばされているので、上丘で一定の mm は
#       視野角では「中心で狭く、周辺で広い」になる。
#
#   【格子方式で何が要らなくなるか】
#     ・中心視野の重み（ガウス窓も、マグニフィケーションも）＝**掛けない**。
#       中心視野の1画素が上丘の多数の升に写ること自体が中心の優位を表す。
#       重みを掛けると二重になる。
#     ・受容野の広さは上丘の mm で一定にできる（RF_DIAMETER_MM=0.8）。
#       視野角で見たときの偏心度依存は座標変換から自動的に出る＝調整不要。
#
# E_ORIENT_BIAS=collicular で重み方式、gauss で旧ガウス窓に戻せる（アブレーション用）。
CENTER_BIAS_MODE = _os.environ.get("E_ORIENT_BIAS", "grid")
CENTER_BIAS_SIGMA_FRAC = 0.32   # 旧方式のときだけ使う [Tier3・ARBITRARY]
# 画像1辺に対する視野角[度]。太郎の眼球カメラは fovy=60。
VISION_FOVY_DEG = 60.0

# ---- ステップ3（側方抑制＋重心）------------------------------------------
# メキシカンハット：近く（sigma_exc）は助け合い、遠く（sigma_inh）は邪魔し合う。
# [Tier3・ARBITRARY] 機構の存在は Munoz & Istvan 1998 の実測だが、
# 具体的な sigma・重み・反復回数は文献に値がなく、目視で調整する暫定値。
LI_SIGMA_EXC_FRAC = 0.025   # 局所興奮の広がり（画像サイズ比）旧方式でのみ使う
LI_SIGMA_INH_FRAC = 0.12    # 近距離抑制の広がり（画像サイズ比）旧方式でのみ使う
# 格子方式では上丘の組織の上の長さ[mm]で持つ。地図の歪みを打ち消す配線は
#   知られていないので、「上丘で一定」が配線の実態に近い。
#   注意：[Tier3・ARBITRARY] 具体値は文献に無い（2026-07-27 の調査でも
#   Munoz & Istvan 1998 / Kopecz & Schöner 1995 / Trappenberg 2001 の
#   カーネル幅の数値までは取れなかった）。受容野 0.34mm を基準に置いた暫定値。
LI_SIGMA_EXC_MM = 0.20      # 局所興奮の広がり [mm]
LI_SIGMA_INH_MM = 1.00      # 近距離抑制の広がり [mm]
LI_W_EXC = 1.0
LI_W_INH = 0.9
LI_RATE = 0.5               # 1反復あたりの更新率
LI_N_ITER = int(_os.environ.get("E_LI_ITER", "8"))   # 反復回数
# 2026-07-27：12 → 8。上丘の競合は**実時間で動かすには重すぎた**。
#   実測：格子96x128・反復12 で 1回 24.7ms。制御周期は10msなので、
#   Viewer で見るとシミュレーションが実時間の4倍遅くなっていた（ユーザーの目視で発覚）。
#   格子48x64・反復8 にすると 2.8ms で、出す向きの差は 0.002（無視できる）。
#   注意：反復回数は数値解法の都合であり、人間の上丘の性質ではない。
#     実際の競合は連続時間で進む。速く収束させるための工学的な選択。

# 【全体抑制・2026-07-26 追加】視野全体に届く抑制。
# 上記のガウス型抑制は届く範囲が約 46画素（sigma 15.4 の3倍）しかなく、
# 30度（64画素）以上離れた2標的が**互いを抑制できず、ただ並んで平均される**
# ことが実測で判明した。
# 【根拠】Munoz & Istvan (1998) J Neurophysiol 79:1193 が上丘で実測したのは
#   ・固視ニューロンとサッケードニューロンの相互抑制
#   ・左右の上丘の間の抑制性結合（＝視野の反対側どうしが抑制し合う）
#   ＝抑制は局所だけでなく**視野全体に届く**。
# ダイナミック神経場モデルでも「1つの山だけが残る」ことを保証するために
# 全体抑制（global inhibition）を置くのが標準形。
# 注意：[ARBITRARY] 重みの具体値は文献になく、実測で選ぶ。
LI_W_GLOBAL = 1.2

# 重心を取るときの閾値（勝った山の裾を切る）。最大値に対する比。
CENTROID_THRESH_FRAC = 0.35

# ---- ステップ4（階段状サッケード）----------------------------------------
# 【なぜ間欠出力か】新生児の平滑追従（smooth pursuit）は未成熟で、
# 追跡時間の15%未満・19度/秒で頭打ち（Kremenitzer et al. 1979, PubMed 487885）。
# 実際の定位は跳躍運動（サッケード）であり、しかも
# **第一サッケードは著しく低振幅で目標に届かず、同振幅の追加サッケードが
# 階段状に連続する**（Aslin & Salapatek 1975）。
# ＝毎tick連続的に目を動かす旧実装は smooth pursuit 型で、新生児には合わない。
#
# 【文献で決まっている値】サッケード潜時は多くが 500ms 未満（Aslin & Salapatek 1975）
# 注意：[ARBITRARY] 以下の具体値は文献に直接の記載がなく、目視で調整する暫定値。
# 次のサッケードまでの最小間隔[秒]。E_SACC_LATENCY で振れる。
# 注意：文献：新生児のサッケードは 500〜900ms 間隔（Aslin & Salapatek 1975）。
#   200ms は短すぎる＝**前のサッケードの結果を見る前に次を撃つ**。
SACCADE_LATENCY = float(_os.environ.get("E_SACC_LATENCY", "0.90"))
SACCADE_DURATION = 0.05     # 1発のサッケードが続く時間[秒]（上限。届いたら早く終わる）
SACCADE_MIN_STRENGTH = float(_os.environ.get("E_SACC_MIN_STRENGTH", "0.015"))
# これ未満の動きでは撃たない。
# 2026-07-27：0.02 → 0.015。実測（e_orient_threshold_probe.py）で
#   固視の境界 0.083 と組み合わせたとき、誤発射を半分に減らしつつ
#   対象を中心に保てる値（ずれ 0.052）。0.030 まで上げると撃つ回数が減りすぎて
#   追えなくなる（ずれ 0.801）。

# 2026-07-27追加：中心に十分近ければ撃たない（固視）。
#
# 【なぜ要るか】「動きの強さ」だけを条件にすると、**対象が無くても撃つ**。
#   実測：おもちゃを遠方へ退避させた条件でも撃つ率が109%で、strength の平均は
#   対象がある条件（0.032）と変わらなかった。太郎自身の頭・眼が動くことで
#   視野が流れ、それが対象と同じ強さの信号になるため、閾値では分けられない。
#   ただし**向きの大きさ**は違う：自己運動由来の向きは小さく散らばり、
#   対象があるときは大きく偏る。
#
# 【人間ではどうか】対象が中心窩に載っていればサッケードは要らない（固視）。
#   上丘の吻側部には固視ニューロンがあり、サッケードニューロンと相互に抑制し合う
#   （Munoz DP & Istvan PJ 1998 J Neurophysiol 79:1193）。
#   ＝「中心にあるものへは撃たない」は実在の機構。
#
# 【値の根拠】中心窩の直径は約5度（無桿体領域は約1.25度）。視野の半角30度に対し
#   2.5度/30度 ≒ 0.083。注意[Tier2に近い／境界の選び方は判断]
SACCADE_MIN_DIR = float(_os.environ.get("E_SACC_MIN_DIR", "0.083"))

# 2026-07-27：1発の大きさを「力を一定時間かける」から**位置の指令**に変えた。
#
# 【なぜ】それまでは「右向きの力をゲイン倍して 0.05秒かける」だけで、
#   **何度動くかが誰にも決まっていなかった**。設計図が定める SACCADE_FRAC
#   （ずれの何割を1発で詰めるか）が実装に存在せず、書いても意味を持たなかった。
#
# 【人間はどうか】上丘が出しているのは力ではなく**変位の指令**（地図上の
#   発火位置がサッケードの振幅を決める）。脳幹のバースト生成器が、眼球位置の
#   内部コピーと目標を比べ、差がゼロになるまで発射し続ける
#   ＝位置の内部フィードバック（Robinson DA 1975 の local feedback model。
#     以後のサッケード生成モデルの標準形）。
#
# 【新生児らしさ】1発では届かない（低振幅＝hypometric）。届かなければ次の
#   サッケードで詰める＝階段状になる（Aslin & Salapatek 1975）。
SACCADE_FRAC = 0.30         # 1発でずれの何割を詰めるか 注意[Tier3・ARBITRARY]
                            #   文献は「著しく低振幅」としか言わない
EYE_FB_GAIN = 0.25          # 位置の誤差[度] → 筋の活性化。4度で飽和 注意[ARBITRARY]
SACCADE_DONE_DEG = 0.5      # 目標にこれだけ近づいたら1発を終える[度] 注意[ARBITRARY]
# 首と目の分担。実測（2026-07-27）で**首を動かさないのが最良**と確定した。
#
#   首の分担   左20度   右10度   右20度
#      0.0     0.049    0.129    0.896
#      0.2     0.193    見失う   見失う
#      0.4     0.955    見失う   見失う
#
#   【なぜ悪化するか】首を回すと VOR が眼球を逆向きに回して打ち消し、さらに
#   頭が動くことで視野全体が流れて動き検出を汚す。二重に効く。
#   【人間ではどうか】新生児の定位は主に眼球運動による。仰向けでは首が特に弱く、
#   頭部運動を伴う定位（eye-head coordination）は月齢とともに育つ。
#   注意：[簡略化] 新生児もまったく頭を動かさないわけではない（頭位選好など）。
#     「定位のために首を使わない」だけで、首が動かないわけではない。
EYE_SHARE = float(_os.environ.get("E_EYE_SHARE", "1.0"))
NECK_SHARE = float(_os.environ.get("E_NECK_SHARE", "0.0"))
# 関節の正方向と「視野のどちら側か」の対応。MIMo の眼球の水平関節は、
#   角度を上げると視線が**左**を向く（＝正面の物体は画像の右へ動く）。
#   よって物体が画像の右（h_dir>0）にあるときは、眼球角度を**下げる**。
#   実測（2026-07-27）：符号を付けずに動かしたら、ずれが
#     -0.130 → -0.429、+0.682 → +0.830 と**遠ざかった**。
#   垂直は水平と**逆**（+1.0）。関節のプラス方向が水平は「左」、垂直は「上」で
#     食い違っていた。可動域が -47〜+33度 と非対称なことから読めたはずで、
#     -1.0 のままでは全条件で対象を見失っていた。→ 落とし穴チェックリスト 項57
EYE_SIGN_H = float(_os.environ.get("E_EYE_SIGN_H", "-1.0"))
EYE_SIGN_V = float(_os.environ.get("E_EYE_SIGN_V", "1.0"))
NECK_FB_GAIN = 0.15         # 首の位置の誤差[度] → 筋の活性化 注意[ARBITRARY]
# 旧方式（data を渡さずに使う場合）のゲイン。互換のため残す。
NECK_GAIN = 0.3
EYE_GAIN = 0.15
# apply() 1回あたりの経過時間[秒]。環境の dt に合わせて上書きする。
DEFAULT_DT = 0.01

# ---- 神経ノイズ（対称性を破って決定を生む）--------------------------------
# 【なぜ入れるか・2026-07-26】完全に対称な2標的では、側方抑制だけでは
# 引き分けたまま決着しない（数学的に対称解が安定）。実装テストで
# 45度・55度分離の対称2標的が中央を指し続けることを確認した。
#
# 【学術的根拠】上丘ニューロンの試行間変動が、実際の選択結果と直接結びつく：
#   Kim B, Basso MA (2010) "A Probabilistic Strategy for Understanding Action
#   Selection" J Neurosci 30(6):2340（PMC2841973）
#     ・上丘ニューロンの Fano factor = 1.44（同じ刺激でも発火がばらつく）
#     ・4ニューロンの活動から選択を予測：ベイズ推定 84.76% >
#       勝者総取り 71.11% > 集団ベクトル平均 55.71〜69.47%
#     ＝ばらつきが選択結果を予測する＝ノイズが決定に関与している直接証拠
#   加えて、決定モデル一般（drift-diffusion / race model）でも、
#   対称条件ではノイズ項がないと理論上決着しないことが知られている。
#
# 注意：[ARBITRARY] ノイズの大きさに文献の直接の推奨値は無い（調査で確認）。
#   Fano factor 1.44 は発火のばらつきの指標であって、
#   この実装の活動量スケールへの変換式は存在しない。
#   「単一標的の精度を崩さず、対称2標的で決着する最小値」を実測で選ぶ。
#
# 注意：人間の対称2択は純粋な 50:50 ランダムではない（中心窩寄り選好・個体差・
#   コスト依存の系統的偏りが報告されている：Van Heusen 2023、PES研究）。
#   ここではノイズのみで、偏り項は入れない＝人間の近似。
LI_NOISE = 0.05

# ---- ステップ4b（保持成分＝ステップ）・2026-08-19 追加 ---------------------
# 【診断】サッケードの位置フィードバック（EYE_FB_GAIN）が有効なのは
#   `_sacc_remaining > 0` の間（＝撃ってから SACCADE_DURATION 経過 or
#   目標に SACCADE_DONE_DEG 以内まで届くまでの、たかだか数十ms）だけだった。
#   目標に届いて `_sacc_remaining` が0になった瞬間、apply() は以後
#   `return action`（何も足さない）に落ち、次にまた撃てるのは
#   SACCADE_LATENCY=0.90秒後（かつ新たな動き信号が SACCADE_MIN_DIR 以上のとき）
#   だけ＝その間ずっと筋活性ゼロの空白ができる。中央復帰バネ（k=0.03、
#   0.3秒で中央へ戻る）に対抗する力がこの空白では出ないため、目標へ着いた
#   直後から眼球は中央へ戻り始める＝「パルス（サッケード）はあるがステップ
#   （保持）が無い」状態だった。
#
# 【人間はどうか】Robinson DA (1975) の pulse-step model（脳幹の
#   burst generator + 神経積分器）が示すのは、サッケードは
#     パルス：バースト神経支配で素早く動かす
#     ステップ：目標位置に達したあとも、その位置に比例した持続的な
#       神経支配（トニック発火）を出し続け、眼窩の弾性復元力（本実装の
#       中央復帰バネに相当）と釣り合わせて位置を保つ
#   の2つで構成される。以後のサッケード生成モデルの標準形。
#
# 【実装の形】新しい力学を書き起こすのではなく、サッケード実行中に使っている
#   位置フィードバック（今の関節角と self._tgt の差 → ゲイン倍）と同じ考え方を、
#   サッケードが終わった（_sacc_remaining=0 になった）あとも回し続ける
#   （HOLD_FB_GAIN。ただしバースト用の EYE_FB_GAIN をそのまま使うと保持が
#   振動して収束しなかったため、角速度の制動項とセットで別途実測で選び直した。
#   詳細は HOLD_FB_GAIN 定義直前のコメントと _hold_command 参照）。
#   これは Robinson の local feedback model（バースト生成器は「目標までの
#   残差がゼロになるまで」発射し続ける）を、バースト区間の外にも延長した形にあたる。
#   釣り合いの式（参考。実装はこれを解いて開ループで u を出すのではなく、
#   フィードバックが収束した先が結果的にこの釣り合い点になる＝暗黙の逆算）：
#     定常状態で 筋の出す力 ＝ バネの復元力
#       spring torque ≈ EYE_CENTERING_STIFFNESS(=0.03) * θ_err [N・m/rad]
#         （infant_body.apply_eye_centering_spring、jnt_stiffness による）
#       muscle torque(u) ≈ f(u, eye_muscle_scale) （MuscleModelの非線形写像。
#         解析的な逆関数を持たないため、開ループで u=k*θ/… と逆算する式は書けない）
#   E_ORIENT_HOLD=1 でON（既定OFF＝v2は従来どおり1ビット不変）。
USE_HOLD = _os.environ.get("E_ORIENT_HOLD", "0") == "1"
# 保持中のゲイン。当初はバーストと同じ EYE_FB_GAIN(0.25) を暫定採用したが、
#   保持を数百ms〜秒オーダーで続けると P制御だけでは筋の活性化ダイナミクスと
#   共振して振動し収束しなかった（実測：診断で目標-1.71度に対し眼球が
#   -1.06〜-2.52度を往復し続けた。_hold_command のコメント参照）。
#   角速度に比例する制動項（HOLD_DAMP_GAIN、PDのD項）を足したうえで、
#   gain×damp を実測で振って選んだ（f13_hold_gain_sweep.py、揺らし条件・
#   後半2秒の残差）：
#     0.25×0.00（元案）  平均0.70度 最大4.27度（振動）
#     0.10×0.00          平均1.45度 最大3.66度（振動）
#     0.25×0.02          平均0.19度 最大0.98度
#     0.25×0.05          平均1.59度 最大3.92度（強すぎて悪化）
#     0.10×0.02          平均0.14度 最大0.82度 ← 採用（最小）
HOLD_FB_GAIN = float(_os.environ.get("E_HOLD_GAIN", "0.10"))
# 角速度に比例する制動項（PDのD項）。[Tier3・ARBITRARY] 工学的な安定化であり、
#   神経系のdamping機構そのものの模倣ではない。値の根拠は上のHOLD_FB_GAINと同じ実測。
HOLD_DAMP_GAIN = float(_os.environ.get("E_HOLD_DAMP", "0.02"))
# 「目標が視野内にある限り」の近似：動き検出が最近 SACCADE_MIN_STRENGTH 以上の
#   信号を出したか（＝対象が実際に動いて見えているか）で判定する。
#   本物の「視野内にあるか」（対象位置と眼球可動域・カメラ画角からの幾何判定）は
#   行っていない。注意：[Tier3・ARBITRARY] 待ち時間そのものに文献値は無い。
#   固視微動さえ止められない人間の网膜像消失（本ファイル冒頭の注釈）を踏まえ、
#   「揺れが止まって数百ms検出が途切れたら見失ったとみなす」という暫定値。
HOLD_LOSE_SEC = float(_os.environ.get("E_HOLD_LOSE", "0.5"))

# 【F1-4b・2026-08-19】語から注意への読み出し回路の受け口。設計：
#   F/docs/設計_F1-4b_語から注意への読み出し回路.md 後半「部品1」。
#   「思い浮かべているものと似たものを見ている間も、離れない」という1行の追加。
#   set_recognition() が一度も呼ばれなければ self._recognition は0.0のままで、
#   REC_THRESHOLD（>0）を下回るため既存の hold 判定に一切影響しない
#   （既定 word_attention=null では trainer.py 側が呼ばないので既存経路と完全一致）。
# REC_THRESHOLDは決め打ちにせず較正する予定だったが、F1-3cの保存済み画像
#   （solo確認=正解おもちゃ注視時／full3000=2個並び・背景優位）でDINOv2の
#   sim分布を測ったところ、正例(0.65〜0.94)と負例(0.58〜0.92)が広く重なり、
#   分離できる値が無かった（実装・較正スクリプトの実測、2026-08-19）。
#   [Tier3・ARBITRARY・未解決] 分離不能のまま機構だけ通す必要があったため、
#   暫定的に正例分布の中央値付近（較正ログの目安）を仮置きした。
#   この値の妥当性は上（ユーザー）判断待ち＝作業記録「上に上げること」参照。
REC_THRESHOLD = float(_os.environ.get("E_REC_THRESHOLD", "0.80"))

# ---- 静的顕著性（動かないものにも視線が向く）・2026-08-19 追加 -----------------
# 設計：F/docs/設計_F1-4c_静的顕著性.md 後半「実装」節。
#
# 【なぜ】文献（Frank 2009・原文精読）：6ヶ月児の視線を一番よく当てる予測モデルは
#   「動き＋静的な明暗コントラストを**同じ重み**で足したもの」。今までの太郎の
#   視線誘導反射は動きチャンネルしか持たず、静止した目立つものには吸い付かない
#   ＝人間模倣として不十分だった。
#
# 【機構】中心-周辺差分（DoG＝ガウスぼかし(σ小)とガウスぼかし(σ大)の差の絶対値）。
#   Itti-Koch系の luminance contrast の最小実装。σは fovy 60度に対し
#   中心約1〜2度／周辺約5〜10度相当から開始し、手順0の絵（境界線問題の確認）で
#   調整した（`F/logs/F1-4c_顕著性地図_2026-08-19/` 参照）。
#   [Tier2]（機構の存在は実在するが、具体的なσの値は文献に無く目視で調整した
#   暫定値。DoGの中心-周辺拮抗そのものは網膜神経節細胞の受容野構造として実測されている）。
STATIC_SIGMA_CENTER_DEG = float(_os.environ.get("E_STATIC_SIGMA_C", "1.5"))
STATIC_SIGMA_SURROUND_DEG = float(_os.environ.get("E_STATIC_SIGMA_S", "7.0"))
# 【根拠：Frank 2009】動きと輝度コントラストを**チャンネル正規化後に等重み**で
#   足したものが乳児の視線を最良予測。STATIC_W=1.0（等重み）。
STATIC_W = float(_os.environ.get("E_STATIC_W", "1.0"))
# スイッチ：既定OFF（過去の実験・既定設定は1ビットも変わらない）。
USE_STATIC_SALIENCE = _os.environ.get("E_STATIC_SAL", "0") == "1"


class OrientingReflexV2:
    """視線誘導反射・新版。段階的に組み立てる。"""

    def __init__(self, model, data=None, time_scales=TIME_SCALES, noise=LI_NOISE,
                 seed=None, dt=DEFAULT_DT, hold=None, static_salience=None):
        # 2026-07-27：seed が None だと神経ノイズが毎回変わり、**同じ設定でも
        #   結果がばらつく**（同一条件3回で 0.319 / 0.215 / 0.189）。
        #   高速化の前後を1回ずつ比べて「悪化した」と誤判定した。
        #   E_ORIENT_SEED で固定できるようにする（既定 0）。
        if seed is None:
            seed = int(_os.environ.get("E_ORIENT_SEED", "0"))
        self.data = data             # 今の関節角を読むため（位置フィードバック）
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        # ステップ4b（保持成分）。既定OFF＝v2は従来のまま1ビット不変。
        self.hold = bool(USE_HOLD if hold is None else hold)
        # 静的顕著性チャンネル。既定OFF＝v2は従来のまま1ビット不変
        #   （OFF時は _static_contrast を呼びもしない。update() 参照）。
        self.static_salience = bool(USE_STATIC_SALIENCE if static_salience is None
                                    else static_salience)
        self.raw_static_map = None    # 診断用：DoG後・正規化前の静的地図
        # ステップ4（階段状サッケード）の状態
        self._t = 0.0                # apply() が刻む内部時刻[秒]
        self._last_saccade_t = -1e9  # 前回サッケードを撃った時刻
        self._sacc_remaining = 0.0   # 今のサッケードの残り時間[秒]
        self._sacc_end_t = -1e9      # サッケードが終わった時刻（抑制の起点）
        self._sacc_h = 0.0           # 今のサッケードの方向（撃った瞬間に固定）
        self._sacc_v = 0.0
        self._hold_last_seen_t = -1e9   # 動きを最後に検出した時刻（保持の解除判定）
        # F1-4b：語から読み出した認識信号（0〜1）。set_recognition() でのみ更新される。
        #   呼ばれなければ0.0のまま＝既存挙動と完全一致（下記 apply() 参照）。
        self._recognition = 0.0
        # 撃った瞬間に決める目標角度[度]（位置フィードバックの目標）
        self._tgt = {"eye_h": 0.0, "eye_v": 0.0, "neck_h": 0.0, "neck_v": 0.0}
        self.n_saccades = 0          # 撃った回数（テスト・観察用）
        self.time_scales = tuple(time_scales)
        self._max_scale = max(self.time_scales)
        self._frame_buffer = []          # 直近フレームのリング（最大 _max_scale+1 枚）
        self.motion_map = None           # ステップ1の出力（網膜の抑制ずみ）
        self.raw_motion_map = None       # 抑制をかける前（診断用）
        self.biased_map = None           # ステップ2の出力
        self.competed_map = None         # ステップ3の競合後の活動
        self.sc_input = None             # 格子方式：受容野でまとめた直後の上丘の活動
        self._center_weight = None       # ステップ2の重み（画像サイズが決まってから作る）
        self._smap = None                # 上丘の地図（画像サイズが決まってから作る）
        self.h_dir = 0.0
        self.v_dir = 0.0
        self.strength = 0.0              # 反応の強さ（0〜1相当）
        self.n_actuator = int(model.nu)
        # 目・首アクチュエータのインデックス（現行 e_orienting.py と同じ）
        self.neck_idx = {}
        self.eye_idx = {"h": [], "v": []}
        for i in range(model.nu):
            name = model.actuator(i).name
            if name == "act:head_swivel":
                self.neck_idx["h"] = i
            elif name == "act:head_tilt":
                self.neck_idx["v"] = i
            elif "eye" in name and "horizontal" in name:
                self.eye_idx["h"].append(i)
            elif "eye" in name and "vertical" in name:
                self.eye_idx["v"].append(i)
        # 各アクチュエータが動かす関節の qpos アドレス（今の角度を読むため）
        self.eye_qadr = {k: [a for a in (_qposadr_of(model, i) for i in v)
                             if a is not None]
                         for k, v in self.eye_idx.items()}
        self.eye_dadr = {k: [a for a in (_dofadr_of(model, i) for i in v)
                             if a is not None]
                         for k, v in self.eye_idx.items()}
        self.neck_qadr = {k: a for k, a in
                          ((k, _qposadr_of(model, i)) for k, i in self.neck_idx.items())
                          if a is not None}

    def reset(self):
        self._frame_buffer = []
        self.motion_map = None
        self.raw_static_map = None
        self.biased_map = None
        self.competed_map = None
        self.sc_input = None
        # 中心視野バイアスの重みもクリアする。キャッシュしたままだと
        #   CENTER_BIAS_SIGMA_FRAC を変えても**古い重みが使われ続ける**。
        #   実際にσを 0.20〜2.00 で振った測定が丸ごと無効になった（2026-07-27）。
        #   → 落とし穴チェックリスト 項53（時間で切れないキャッシュ）と同型
        self._center_weight = None
        self.h_dir = 0.0
        self.v_dir = 0.0
        self.strength = 0.0
        self._t = 0.0
        self._last_saccade_t = -1e9
        self._sacc_remaining = 0.0
        self._sacc_end_t = -1e9
        self._sacc_h = 0.0
        self._sacc_v = 0.0
        self._hold_last_seen_t = -1e9
        self.n_saccades = 0
        # F1-4b：認識信号もreset()でクリアする（前の走行・エピソードの値を持ち越さない）。
        self._recognition = 0.0

    # ------------------------------------------------------------
    # 公開インターフェース
    # ------------------------------------------------------------
    def set_recognition(self, sim):
        """F1-4b：語から読み出した「思い浮かべているものとの一致度」を渡す。

        sim: 0〜1（cos類似度をmax(0, sim)した値を想定。呼び出し側＝trainer.pyが
             クリップ済みのものを渡す前提だが、念のためここでも[0,1]へclipする）。
        毎判断ステップで上書きする想定（trainer.py側が期限切れ・OFF時は0.0を渡すか
        呼ばない）。hold=False のときは apply() 側で参照しないので無害。
        """
        self._recognition = float(np.clip(sim, 0.0, 1.0))

    def update(self, eye_image):
        """新しい画像で内部状態を更新する。両眼平均を想定するが単眼画像でも動く。"""
        # サッケードを撃った直後は、その間の画像を動き検出に使わない。
        #   自分が目を動かしたことで視野全体が流れ、それを「動き」として拾って
        #   しまうため（実測：対象が視野の端にあるのに出力が1/4に潰れた）。
        #   フレームの記憶も捨てる＝**サッケードをまたぐ比較をしない**。
        # 2026-07-27修正：抑制の起点を「撃った時刻」から**サッケードが終わった時刻**へ。
        #   位置フィードバックにしたのでサッケードの持続時間は可変で、
        #   撃った時刻から150ms数えると**まだ眼球が動いている最中に抑制が明ける**。
        #   実測：おもちゃが無い条件でも撃つ率が109%＝自己運動由来の反応が
        #   対象と同じ強さで残っていた。
        moving = False
        if EYE_STILL_DEG_S > 0 and self.data is not None:
            try:
                w = max(abs(float(self.data.qvel[a]))
                        for v in self.eye_dadr.values() for a in v)
                moving = np.degrees(w) > EYE_STILL_DEG_S
            except Exception:
                moving = False
        if moving or (SACC_SUPPRESS_SEC > 0 and (
                self._sacc_remaining > 0.0
                or (self._t - self._sacc_end_t) < SACC_SUPPRESS_SEC)):
            self._frame_buffer = []
            self.strength = 0.0      # 古い向きのまま撃たないように
            return

        motion = self._detect_motion(eye_image)          # ステップ1
        self.raw_motion_map = motion
        # ステップ1.5：網膜の中心-周辺抑制。まわりも一緒に動いていれば
        #   「自分が動いたせい」として割り引く（object motion sensitivity）。
        if USE_OMS:
            motion = _object_motion(motion, VISION_FOVY_DEG,
                                    surround_deg=OMS_SURROUND_DEG,
                                    uniform_gain=OMS_UNIFORM_GAIN)
        # ステップ1.6（静的顕著性・2026-08-19）：OMS適用後の動き地図に、静止した
        #   明暗コントラストの地図をチャンネル正規化後の等重みで足す。
        #   OMSは自己運動の割引＝動きチャンネル専用なので、静的チャンネルには
        #   適用しない（OMSより後ろで合成する）。
        #   OFF時は _static_contrast を一切呼ばない＝既存経路と完全一致。
        if self.static_salience:
            static_map = self._static_contrast(eye_image)
            self.raw_static_map = static_map
            mmax = float(motion.max()) if motion.size else 0.0
            motion_norm = motion / mmax if mmax > 1e-12 else motion
            smax = float(static_map.max()) if static_map.size else 0.0
            static_norm = static_map / smax if smax > 1e-12 else static_map
            motion = motion_norm + STATIC_W * static_norm
        self.motion_map = motion
        biased = self._apply_center_bias(motion)         # ステップ2
        self.biased_map = biased
        h, v, s = self._select_and_centroid(biased)      # ステップ3
        self.h_dir, self.v_dir, self.strength = h, v, s
        # ステップ4b（保持）：対象が実際に動いて見えた最後の時刻を記録する。
        #   hold=False のときも記録だけはしておく（副作用は無い＝ hold=False の
        #   出力には影響しない。apply() 側で self.hold を見て使うかどうかを決める）。
        if s >= SACCADE_MIN_STRENGTH:
            self._hold_last_seen_t = self._t
        # ステップ4（階段状サッケード）・5（IOR）は未実装

    def _angle_deg(self, adrs):
        """今の関節角[度]。複数あれば平均（両目は同じだけ動く前提）。"""
        if self.data is None:
            return 0.0
        if isinstance(adrs, (list, tuple)):
            if not adrs:
                return 0.0
            return float(np.degrees(np.mean([self.data.qpos[a] for a in adrs])))
        return float(np.degrees(self.data.qpos[adrs]))

    def apply(self, action, dt=None):
        """action に階段状サッケードを加算して返す（ステップ4）。

        2026-07-27：1発の大きさを「位置の指令」にした。
          ・前回から SACCADE_LATENCY 経過し、かつ動きが十分なら1発撃つ
          ・撃つ瞬間に**目標角度**を決める（今の角度 ＋ ずれの SACCADE_FRAC）
          ・そのあとは目標との差を見ながら筋を活性化し、届いたら止める
            （＝脳幹の位置フィードバック。Robinson 1975）
          ・1発で届かないので、次の潜時のあとにまた撃つ → 階段状
        """
        step = self.dt if dt is None else float(dt)
        self._t += step

        # F1-4b：「思い浮かべているものと似たものを見ている間も、離れない」。
        #   _should_hold() 自体は無変更のまま、動き検出と同じ「最後に見た時刻」を
        #   認識でも更新する形にする（設計の指示どおり最小実装）。
        #   hold=False のときは既存OFF経路を一切変えない（self.hold を先頭で見る）。
        #   self._recognition は set_recognition() が呼ばれない限り0.0のままなので、
        #   REC_THRESHOLD>0 である限りここは常にFalse＝既存挙動と完全一致。
        if self.hold and self._recognition >= REC_THRESHOLD:
            self._hold_last_seen_t = self._t

        if self._sacc_remaining <= 0.0:
            ready = (self._t - self._last_saccade_t) >= SACCADE_LATENCY
            # 中心に十分近ければ撃たない（固視）。自己運動由来の向きは小さいので、
            #   ここで落ちる。対象があるときは向きが大きく偏るので通る。
            far_enough = (abs(self.h_dir) >= SACCADE_MIN_DIR
                          or abs(self.v_dir) >= SACCADE_MIN_DIR)
            if ready and self.strength >= SACCADE_MIN_STRENGTH and far_enough:
                self._sacc_h = self.h_dir
                self._sacc_v = self.v_dir
                self._sacc_remaining = SACCADE_DURATION
                self._last_saccade_t = self._t
                self.n_saccades += 1
                # 網膜上のずれ[度] → 1発で詰める分だけ目標角度をずらす
                half = VISION_FOVY_DEG / 2.0
                dh = EYE_SIGN_H * SACCADE_FRAC * self.h_dir * half
                dv = EYE_SIGN_V * SACCADE_FRAC * self.v_dir * half
                self._tgt["eye_h"] = self._angle_deg(self.eye_qadr["h"]) + EYE_SHARE * dh
                self._tgt["eye_v"] = self._angle_deg(self.eye_qadr["v"]) + EYE_SHARE * dv
                if "h" in self.neck_qadr:
                    self._tgt["neck_h"] = self._angle_deg(self.neck_qadr["h"]) + NECK_SHARE * dh
                if "v" in self.neck_qadr:
                    self._tgt["neck_v"] = self._angle_deg(self.neck_qadr["v"]) + NECK_SHARE * dv
            elif self.hold and self._should_hold():
                # ステップ4b（保持）：サッケードは撃たないが、既に定めた目標
                #   （self._tgt）へ向けた位置フィードバックだけは続ける。
                #   本体のサッケード実行中ブロック（この if の外）とは独立の
                #   経路なので、hold=False の既存挙動には一切触れない。
                return self._hold_command(action)
            else:
                return action     # サッケード中でなければ何も足さない

        # サッケード実行中：目標角度との差を見ながら動かす
        self._sacc_remaining -= step
        out = np.array(action, dtype=float).copy()

        if self.data is not None:
            errs = []
            eh = self._tgt["eye_h"] - self._angle_deg(self.eye_qadr["h"])
            ev = self._tgt["eye_v"] - self._angle_deg(self.eye_qadr["v"])
            errs += [eh, ev]
            self.last_eye_cmd = float(np.clip(EYE_FB_GAIN * eh, -1, 1))  # 観察用
            for i in self.eye_idx["h"]:
                _write_joint_command(out, i, float(np.clip(EYE_FB_GAIN * eh, -1, 1)),
                                     self.n_actuator, co_activation=0.0, additive=True)
            for i in self.eye_idx["v"]:
                _write_joint_command(out, i, float(np.clip(EYE_FB_GAIN * ev, -1, 1)),
                                     self.n_actuator, co_activation=0.0, additive=True)
            # NECK_SHARE=0 のときは首に一切触らない。
            #   それまでは分担0でも「撃った瞬間の首の角度」を目標にして
            #   毎ステップ誤差を打ち消していた＝**首を固定しようとしていた**。
            #   首が重力で動く → 反射が戻そうとする → 視野が流れる →
            #   動き検出が反応する → また撃つ、という正のフィードバックになり、
            #   ユーザーの目視で「揺らさなくてもサッケードが撃たれる」
            #   「反射をONにすると顔が左に向く」として現れた（2026-07-27）。
            if NECK_SHARE != 0.0:
                for key in ("h", "v"):
                    if key in self.neck_idx and key in self.neck_qadr:
                        e = self._tgt[f"neck_{key}"] - self._angle_deg(self.neck_qadr[key])
                        errs.append(e)
                        _write_joint_command(out, self.neck_idx[key],
                                             float(np.clip(NECK_FB_GAIN * e, -1, 1)),
                                             self.n_actuator, co_activation=0.0,
                                             additive=True)
            # 届いたら1発を早く終える（時間切れを待たない＝本物のサッケードも
            # 振幅で持続時間が変わる）
            if max(abs(e) for e in errs) <= SACCADE_DONE_DEG:
                self._sacc_remaining = 0.0
            if self._sacc_remaining <= 0.0:
                self._sacc_end_t = self._t      # 終わった時刻を記録
            return out

        # data が無い場合の従来動作（力を一定時間かける）。互換のため残す。
        h, v = self._sacc_h, self._sacc_v
        # 2026-07-26：ここで `out[i] = clip(out[i] + gain*h, -1, 1)` と直接書いていたのが誤り。
        #   MuscleModel では1関節が2本の筋（前半＝負方向筋・後半＝正方向筋）で駆動され、
        #   各要素は [0, 1] に切り捨てられる。負の指令は消えるので、**首も目も片方向にしか
        #   動けなかった**（目標が反対側にあると永久に追えない）。VOR で同じ誤りが実測で
        #   確認され（眼が可動域の下限に張り付いて戻らない）、こちらも同型と判明した。
        #   共通の写像 write_joint_command を通す。眼球は相反神経支配なので共収縮は0。
        for key, gain in (("h", NECK_GAIN), ("v", NECK_GAIN)):
            if key in self.neck_idx:
                d = h if key == "h" else v
                _write_joint_command(out, self.neck_idx[key], gain * d,
                                     self.n_actuator, co_activation=0.0, additive=True)
        for i in self.eye_idx["h"]:
            _write_joint_command(out, i, EYE_GAIN * h, self.n_actuator,
                                 co_activation=0.0, additive=True)
        for i in self.eye_idx["v"]:
            _write_joint_command(out, i, EYE_GAIN * v, self.n_actuator,
                                 co_activation=0.0, additive=True)
        return out

    # ------------------------------------------------------------
    # ステップ4b：保持（サッケードが終わったあとも目標角度を維持する）
    # ------------------------------------------------------------
    def _should_hold(self):
        """保持を続けてよいか（＝目標がまだ視野内にありそうか、の近似）。

        既に1発もサッケードを撃っていなければ保持する目標が無い。
        直近 HOLD_LOSE_SEC 以内に動き検出が SACCADE_MIN_STRENGTH 以上の
        信号を出していれば「まだ見えている」とみなす。
        """
        if self.n_saccades == 0 or self.data is None:
            return False
        return (self._t - self._hold_last_seen_t) < HOLD_LOSE_SEC

    def _angle_vel_deg(self, adrs):
        """今の関節角速度[度/秒]。複数あれば平均。保持のダンピング項に使う。"""
        if self.data is None:
            return 0.0
        if isinstance(adrs, (list, tuple)):
            if not adrs:
                return 0.0
            return float(np.degrees(np.mean([self.data.qvel[a] for a in adrs])))
        return float(np.degrees(self.data.qvel[adrs]))

    def _hold_command(self, action):
        """サッケード実行中のブロックと同じ位置フィードバック（の考え方）を、
        _sacc_remaining の外（＝サッケードを撃っていない間）でも回す。
        HOLD_FB_GAIN で目標角度との差を打ち消す持続的な筋活性＝ステップ成分
        （Robinson 1975・脳幹の神経積分器の近似。本ファイル冒頭の
        HOLD_FB_GAIN 直前のコメント参照。値はバースト用EYE_FB_GAINとは別に
        実測で選んだ、後述の振動対策とセットで）。
        サッケードの状態（_sacc_remaining・_sacc_end_t・n_saccades）は
        一切変更しない＝サッケードの状態機械そのものには手を触れない。

        【2026-08-19追記・振動対策】位置だけの比例フィードバック（P制御）を
        バーストの外（数百ms〜秒オーダー）まで延ばすと、筋の活性化ダイナミクス
        （力が立ち上がるまでの遅れ）と噛み合って**減衰しない振動**になった
        （実測：診断スクリプトで目標-1.71度に対し眼球が-1.06〜-2.52度を往復し
        続け、収束しない）。バースト中（0.05秒以内）は遅れが問題にならないが、
        保持はもっと長く続くため顕在化する。
        対策として角速度に比例する制動項を引く（比例・微分＝PD制御。工学的な
        安定化であり、神経系の damping 機構そのものの模倣ではない
        [Tier3・ARBITRARY]）：
            cmd = HOLD_FB_GAIN * 位置誤差[度] − HOLD_DAMP_GAIN * 角速度[度/秒]
        """
        out = np.array(action, dtype=float).copy()
        eh = self._tgt["eye_h"] - self._angle_deg(self.eye_qadr["h"])
        ev = self._tgt["eye_v"] - self._angle_deg(self.eye_qadr["v"])
        vh = self._angle_vel_deg(self.eye_dadr["h"])
        vv = self._angle_vel_deg(self.eye_dadr["v"])
        cmd_h = HOLD_FB_GAIN * eh - HOLD_DAMP_GAIN * vh
        cmd_v = HOLD_FB_GAIN * ev - HOLD_DAMP_GAIN * vv
        for i in self.eye_idx["h"]:
            _write_joint_command(out, i, float(np.clip(cmd_h, -1, 1)),
                                 self.n_actuator, co_activation=0.0, additive=True)
        for i in self.eye_idx["v"]:
            _write_joint_command(out, i, float(np.clip(cmd_v, -1, 1)),
                                 self.n_actuator, co_activation=0.0, additive=True)
        if NECK_SHARE != 0.0:
            for key in ("h", "v"):
                if key in self.neck_idx and key in self.neck_qadr:
                    e = self._tgt[f"neck_{key}"] - self._angle_deg(self.neck_qadr[key])
                    _write_joint_command(out, self.neck_idx[key],
                                         float(np.clip(NECK_FB_GAIN * e, -1, 1)),
                                         self.n_actuator, co_activation=0.0,
                                         additive=True)
        return out

    # ------------------------------------------------------------
    # ステップ1：複数フレーム比較による動き検出
    # ------------------------------------------------------------
    def _detect_motion(self, image):
        """現フレームと 1・5・20 前のフレームとの差を混ぜて動きマップを返す。

        フレームが揃わないうちは、揃っているスケールだけ使う（穴埋めしない）。
        RGB画像はチャネル平均で単一マップにする。
        注意：ON（明転）とOFF（暗転）を区別せず絶対値を取る。文献調査（2026-07-26）で
        「ON優位」を支持する定量値は無く、V1はむしろOFF優位（Jansen 2019, 65:35）、
        行動レベルの非対称もパラダイム依存でバラバラと判明したため、
        **対称（1:1）が根拠のない前提を最小にする**という判断。
        """
        # 2026-07-26修正：画像を [0,1] にそろえてから差を取る。
        #   それまで uint8（0〜255）のまま差分していたため、動きマップの値が
        #   輝度スケールに乗ってしまい、`strength` が 50〜140 になっていた。
        #   発火の閾値 SACCADE_MIN_STRENGTH=0.02 は 0〜1 を想定した値なので、
        #   **常に2500倍の値が来て必ず発火**していた（6.7秒で32発＝撃ちっぱなし。
        #   おもちゃが存在しない条件でも同じ数だけ撃っていた）。
        #   → 落とし穴チェックリスト 項36（指標を作るとき向きと尺度も書く）
        arr = np.asarray(image)
        cur = (arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8
               else arr.astype(np.float32))
        cur_gray = cur.mean(axis=-1) if cur.ndim == 3 else cur

        self._frame_buffer.append(cur_gray)
        while len(self._frame_buffer) > self._max_scale + 1:
            self._frame_buffer.pop(0)

        motion = np.zeros_like(cur_gray, dtype=np.float32)
        used = 0
        for scale in self.time_scales:
            if len(self._frame_buffer) > scale:
                past = self._frame_buffer[-1 - scale]
                motion += np.abs(cur_gray - past)
                used += 1
        if used == 0:
            return np.zeros_like(cur_gray)
        return motion / used

    # ------------------------------------------------------------
    # ステップ1.6：静的顕著性（中心-周辺差分＝DoG）
    # ------------------------------------------------------------
    def _static_contrast(self, image):
        """輝度画像の局所コントラスト地図を返す（中心-周辺差分の絶対値＝DoG）。

        Itti-Koch系 luminance contrast の最小実装。網膜神経節細胞の
        中心-周辺拮抗型受容野に対応する（機構の存在は[Tier2]、σの具体値は
        文献に無く手順0の絵で調整した暫定値[Tier3・ARBITRARY]）。
        """
        arr = np.asarray(image)
        cur = (arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8
               else arr.astype(np.float32))
        gray = cur.mean(axis=-1) if cur.ndim == 3 else cur
        h, w = gray.shape
        # 視野角 → 画素（retina.object_motion と同じ変換：画像1辺=fovy_deg）。
        scale = max(h, w) / VISION_FOVY_DEG
        sigma_c = STATIC_SIGMA_CENTER_DEG * scale
        sigma_s = STATIC_SIGMA_SURROUND_DEG * scale
        center = gaussian_filter(gray, sigma_c)
        surround = gaussian_filter(gray, sigma_s)
        return np.abs(center - surround)

    # ------------------------------------------------------------
    # ステップ2：中心視野バイアス
    # ------------------------------------------------------------
    def _apply_center_bias(self, motion):
        """中心視野の優位を反映する。

        既定（collicular）：上丘のマグニフィケーション（面積）を重みにする。
          M_area(R) ∝ (Bu·Bv)/(R+A)² ＝「視野の1度が上丘で何mm²を占めるか」。
          ガウス窓と違い**べき乗則で裾が重い**ので、周辺の刺激も捨てない。
        旧（gauss）：ガウス窓。形も強さも根拠がなかった。比較用に残す。
        """
        if CENTER_BIAS_MODE == "grid":
            # 格子方式では**何も掛けない**。中心視野の優位は、次の段で
            #   上丘の格子へ写すときに座標変換そのものから出る。
            #   ここで重みを掛けると二重になる。
            return motion

        if CENTER_BIAS_MODE == "gauss":
            if self._center_weight is None or self._center_weight.shape != motion.shape:
                h, w = motion.shape
                cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
                sigma = CENTER_BIAS_SIGMA_FRAC * max(h, w)
                ys = np.arange(h)[:, None] - cy
                xs = np.arange(w)[None, :] - cx
                self._center_weight = np.exp(
                    -(ys ** 2 + xs ** 2) / (2 * sigma ** 2)).astype(np.float32)
            return motion * self._center_weight

        smap = self._collicular_map(motion.shape)
        if self._center_weight is None or self._center_weight.shape != motion.shape:
            w = smap.mag_area.astype(np.float32)
            self._center_weight = w / max(float(w.max()), 1e-12)   # 中心を1に正規化
        return motion * self._center_weight

    def _collicular_map(self, shape):
        """画像の形に対応する上丘の地図を作る（一度だけ計算してキャッシュ）。"""
        if self._smap is None or self._smap.h != shape[0] or self._smap.w != shape[1]:
            import sys as _s, os as _o
            _b = _o.path.abspath(_o.path.join(
                _o.path.dirname(_o.path.abspath(__file__)), _o.pardir, _o.pardir,
                "taro_core", "src", "brain"))
            if _b not in _s.path:
                _s.path.insert(0, _b)
            from superior_colliculus import CollicularMap
            self._smap = CollicularMap(width=shape[1], height=shape[0],
                                       fovy_deg=VISION_FOVY_DEG)
        return self._smap

    # ------------------------------------------------------------
    # ステップ2＋3（格子方式）：上丘の地図へ写し、受容野でまとめ、そこで競合する
    # ------------------------------------------------------------
    def _select_on_collicular_grid(self, activity):
        """画像を上丘の格子へ写し、受容野でぼかしてから競合させ、重心を返す。

        実際の上丘で起きている順序をそのままなぞる：
            網膜の像 → 上丘の地図へ（中心視野が引き伸ばされる）
                     → 受容野でまとめる（1ニューロンが視野の広い範囲を担当）
                     → 上丘の組織の上で側方抑制の競合
                     → 集団の重心（population vector）→ 視野の向きへ逆変換
        """
        smap = self._collicular_map(activity.shape)
        g = smap.to_grid(activity)                       # [2, nv, nu]

        # 受容野：上丘の上では一様な広がり（RF_SIGMA_MM）。視野角で見ると
        # 中心では狭く周辺では広くなる＝偏心度依存が座標変換から自動的に出る。
        sv, su = smap.rf_sigma_cells
        g = gaussian_filter(g, (0.0, sv, su))
        self.sc_input = g
        if g.max() < 1e-9:
            self.competed_map = g
            return 0.0, 0.0, 0.0

        inp = g / g.max()
        u = inp.copy()
        se = (0.0, LI_SIGMA_EXC_MM / smap.dv, LI_SIGMA_EXC_MM / smap.du)
        si = (0.0, LI_SIGMA_INH_MM / smap.dv, LI_SIGMA_INH_MM / smap.du)
        noise_scale = self.noise * np.sqrt(LI_RATE)
        for _ in range(LI_N_ITER):
            f = np.clip(u, 0, None)
            exc = gaussian_filter(f, se) * LI_W_EXC
            inh = gaussian_filter(f, si) * LI_W_INH
            # 全体抑制＝左右の上丘のあいだの抑制も含む（Munoz & Istvan 1998）
            inh_global = LI_W_GLOBAL * float(f.mean())
            u = u + LI_RATE * (-u + inp + exc - inh - inh_global)
            if noise_scale > 0:
                u = u + noise_scale * self.rng.standard_normal(u.shape)
        u = np.clip(u, 0, None)
        self.competed_map = u
        h_dir, v_dir = smap.grid_direction(u, thresh_frac=CENTROID_THRESH_FRAC)
        strength = float(min(activity.max(), 1.0))
        return float(h_dir), float(v_dir), strength

    # ------------------------------------------------------------
    # ステップ3：側方抑制で1箇所を選び、その周辺で重心を取る
    # ------------------------------------------------------------
    def _select_and_centroid(self, activity):
        """メキシカンハット型の側方相互作用を数回まわして競合させ、
        残った活動の重心を (h_dir, v_dir, strength) として返す。

        h_dir: 右が正、左が負（[-1, 1] 相当）
        v_dir: 上が正、下が負（画像座標の y は下向きなので符号を反転）
        strength: 反応の強さ（動きが無ければ 0）
        """
        if activity.max() < 1e-6:
            self.competed_map = np.zeros_like(activity)
            self.sc_input = None
            return 0.0, 0.0, 0.0

        if CENTER_BIAS_MODE == "grid":
            return self._select_on_collicular_grid(activity)

        h, w = activity.shape
        sigma_exc = LI_SIGMA_EXC_FRAC * max(h, w)
        sigma_inh = LI_SIGMA_INH_FRAC * max(h, w)

        inp = activity / activity.max()      # 入力を [0,1] に正規化
        u = inp.copy()
        # ノイズは確率微分方程式の慣例に従い sqrt(dt) でスケールする
        noise_scale = self.noise * np.sqrt(LI_RATE)
        for _ in range(LI_N_ITER):
            f = np.clip(u, 0, None)          # 発火率（負は出力しない）
            exc = gaussian_filter(f, sigma_exc) * LI_W_EXC
            inh = gaussian_filter(f, sigma_inh) * LI_W_INH
            inh_global = LI_W_GLOBAL * float(f.mean())   # 視野全体に届く抑制
            # du/dt = -u + input + 近傍興奮 - 近距離抑制 - 全体抑制 + ノイズ
            u = u + LI_RATE * (-u + inp + exc - inh - inh_global)
            if noise_scale > 0:
                u = u + noise_scale * self.rng.standard_normal(u.shape)
        u = np.clip(u, 0, None)
        self.competed_map = u

        if u.max() < 1e-6:
            return 0.0, 0.0, 0.0

        # 勝った山の裾を切ってから、上丘の座標で重心（population vector）を取る。
        #   Lee, Rohrer & Sparks (1988) Nature 332:357-360 ＝ 上丘のサッケードは
        #   活動している集団の重心で決まる（一部を薬理的に止める直接実験）。
        #
        #   注意：「平らな画像で重心を取る」のと「歪んだ地図で重心を取ってから逆変換」は
        #     数学的に別物（Jensen の不等式）。実際の上丘は後者
        #     （Goossens & Van Opstal 2012 J Neurophysiol）。
        #     対象が視野の端にあるほど出力が中心寄りに潰れるのは、
        #       バグではなく**この写像の性質そのもの**。
        if u.max() <= 1e-12:
            return 0.0, 0.0, 0.0
        if CENTER_BIAS_MODE == "gauss":
            # 旧方式（比較用）：平らな画像で重心を取る
            thresh = u.max() * CENTROID_THRESH_FRAC
            mask = u > thresh
            wsum = u[mask].sum()
            if wsum < 1e-9:
                return 0.0, 0.0, 0.0
            ys, xs = np.nonzero(mask)
            cy_map = float((ys * u[mask]).sum() / wsum)
            cx_map = float((xs * u[mask]).sum() / wsum)
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            h_dir = (cx_map - cx) / cx
            v_dir = -(cy_map - cy) / cy      # 画像の y は下向きなので反転
        else:
            smap = self._collicular_map(u.shape)
            h_dir, v_dir = smap.direction(u, thresh_frac=CENTROID_THRESH_FRAC)
        # 反応の強さ＝中心バイアス後の動きマップの最大値。
        # 画像を [0,1] にそろえたので、これも [0,1] に収まる（_detect_motion 参照）。
        #   念のため上限で切る（中心バイアスの重みは1以下なので理論上は超えない）。
        strength = float(min(activity.max(), 1.0))
        return float(h_dir), float(v_dir), strength
