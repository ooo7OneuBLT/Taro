"""実験の設定を1つの入れ物にまとめる。

【なぜ要るか、2026-07-30】`E/scripts/e_growth_train.py` は設定を**53個のモジュール変数**
（`_LR` `_MUSCLE` `_REWARD` …）として持ち、それを930行の関数の中から直接読んでいた。
そのため：
  ・設定を変えるには**環境変数を経由するしかなかった**（実験ファイルから渡せない）
  ・どの設定がどこで効くのか、読まないと分からなかった
  ・既定値が食い違っても気づけなかった（学習ループだけ関節モードだった事故）

→ 設定を1つのオブジェクトにして、**実験ファイルから作る**のを正式な経路にする。
  環境変数から作る経路（`from_env`）は「古い方式と同じ数値が出るか確かめる」ためだけに残す。

【3つの欄との対応】実験ファイル（E/experiments/*.json）の欄がそのまま入る：
    taro   → 太郎の中身（本能のON/OFF・体の月齢・駆動モード）
    run    → 動かし方（steps / seed / K / checkpoint）
    scene  → 環境（ここには入れない。scene.py が担う）

注意：ここに「測り方」は入れない。測るのはプラグインの仕事。
注意：根拠ラベルは各項目のコメントに残す。値の由来（Tier1=一次文献／Tier3=恣意的）が
  分からなくなると、あとで「この数字はどこから来たのか」を追えなくなる。
"""
import os


# 既定値の表。キー＝実験ファイルの `taro` / `run` で書く名前。
#   値＝(既定値, 説明, 環境変数名)。環境変数名は古い方式との比較用。
#   注意：既定値を変えるときは、必ず理由を研究日誌に書く（過去に既定値の食い違いで事故った）。
TARO_DEFAULTS = {
    # ---- 体 ----------------------------------------------------------------
    # 駆動モード。muscle＝拮抗筋2本/関節・引くだけ・行動180次元[0,1]。
    # joint（90関節を独立に駆動）は逸脱リスト「逸脱5」の逸脱
    #   （Hadders-Algra et al. 1992［Tier1］＝新生児は拮抗筋を同時に力ませる）。
    "actuation":     ("muscle", "駆動モード muscle/joint", "E_MUSCLE"),
    "age_months":    (None, "体の月齢。None＝シーンの値を使う", "E_AGE"),
    "age_to":        (None, "終わりの月齢。None＝体を育てない", "E_AGE_TO"),
    "age_start":     (0, "何回目の学習から月齢を変え始めるか", "E_AGE_START"),
    "age_ramp":      (0, "何回かけて月齢を変えるか。0＝一瞬で", "E_AGE_RAMP"),
    "age_every":     (500, "何回ごとに月齢を見直すか", "E_AGE_EVERY"),
    # ---- 感覚 --------------------------------------------------------------
    # 注意：触覚ONで体を育てるには somatosensory=true が要る（項75・項86）
    "touch":         (False, "触覚を足す", "E_TOUCH"),
    "touch_mode":    ("target", "触覚を予測対象にするか input/target", "E_TOUCH_MODE"),
    "somatosensory": (False, "触覚を視床VPL+S1相当の経路にする", "E_SOMATOSENSORY"),
    "vision":        (True, "視覚を入力に入れる", "E_E1_VISION"),
    # 反射。シーンを組むときに渡す（run/plugins/common/scene.py が読む）。
    #   注意：ここに無いと measure では使えて train/view では弾かれる、という
    #     非対称が起きる（2026-07-30 の点検で発覚）。
    "vor":           (True, "前庭動眼反射（頭が動いても視線を保つ）", None),
    "orienting_reflex": (False, "視線誘導反射（動くものへ目を向ける）", None),
    # 予測対象。"0"=固有感覚のみ／"vision"=+視覚／"all"=+前庭+触覚+視覚
    "target":        ("0", "予測対象 0/vision/all", "E_E1_TARGET"),
    "lam_v":         (1.0, "視覚ブロックの重み[Tier3]", "E_LAM_V"),
    # ---- 本能（脳の中の機構）------------------------------------------------
    "lr":            (0.005, "学習率", "E_LR"),
    # progress＝学習進度（Oudeyer の好奇心）。predict は大行動バイアスの既知欠陥あり
    "reward":        ("progress", "内発的動機 progress/predict", "E_REWARD"),
    "ne_relative":   (True, "ノルアドレナリンを相対基準で出す", "E_NE_RELATIVE"),
    "replay":        (True, "睡眠中の経験リプレイ（記憶定着）", "E_REPLAY"),
    # 遠心性コピー＝直前の行動を次tickの入力(prev_a)として渡す仕組み。
    #   Falseだとprev_aが更新されなくなる（常にゼロのまま固定）。
    #   【なぜ、2026-08-06】C側(run_c_metrics_ac_lr.py)のC_EFFCOPYに相当する切替が
    #   E側（runシステム）に無かったため新設。既定Trueで既存実験の挙動は変わらない。
    "efference_copy": (True, "直前の行動を次tickの入力(prev_a)として渡す遠心性コピー。"
                       "Falseだとprev_aが更新されなくなる（常にゼロのまま固定）",
                       "E_EFFCOPY"),
    "cerebellum":    (True, "運動小脳（自動化・結晶化）", "E_CEREBELLUM"),
    "mature":        (False, "学習進行に合わせて探索を結晶化", "E_MATURE"),
    # 【taro-C5】努力コスト＝代謝コストを報酬から引く[Tier3・二乗の形は恣意的]
    "effort_cost":   (0.0, "努力コストの重み λ", "E_EFFORT"),
    # 行動の急変ペナルティ（CAPS, Mysore et al. 2021）[Tier3]
    "caps":          (0.0, "行動の滑らかさペナルティ λ", "E_CAPS"),
    # ---- 探索（運動性喃語）--------------------------------------------------
    # white＝白色ガウス／colored＝1/f^β の色付きノイズ＋粗いシナジー
    "noise":         ("white", "探索の性質 white/colored", "E_NOISE"),
    # β＝0.686(8週)〜0.877(30週)。López et al. 2026［Tier1・実測］
    "beta":          (0.8, "色付き度 β", "E_BETA"),
    "synergy":       (False, "粗いシナジー（まとめてしか動かせない）", "E_SYNERGY"),
    "syn_w":         (0.6, "シナジー追従の強さ[Tier3]", "E_SYN_W"),
    "antagonist":    (False, "拮抗筋モード（要 actuation=muscle）", "E_ANTAGONIST"),
    "coactivation":  (0.3, "共収縮の度合い[Tier3]", "E_COACTIVATION"),
    # ---- 目標指向の探索（Goal Babbling）------------------------------------
    # 注意：2026-07-30 の実測で**有害**と判明（接触−32%・margin +30.7%→+17.2%・
    #   persist 1000%）。原典（Rolf, Steil & Gienger 2010）とは目標空間が違う
    #   （原典＝手先位置の低次元／太郎＝固有感覚621次元まるごと）。作り直し予定。
    "goal_babbling": (False, "目標指向の探索（現状は有害と判明）", "E_GOALBABBLE"),
    "goal_switch":   ("pe", "探索/目標の切替 fixed/ne/pe", "E_GB_SWITCH"),
    "closed_loop_reach": (False, "目標を保持してにじり寄る（goal_space=prop_full用）", "E_CLTRAIN"),
    # ---- 手先位置の目標表現（案C・Goal Babbling段階1、2026-08-02）------------
    # 設計：作業記録（非公開）
    # 座標を作らず、腕の固有感覚(7)＋自己接触(15)＝22次元を目標にする。
    "goal_space":     ("prop_full", "目標表現 prop_full(旧・固有感覚まるごと、既定)/"
                       "reach_self(新・腕7+自己接触15、案C)", "E_GOAL_SPACE"),
    "goal_trajectory": (True, "区分線形の軌道で目標に近づく(reach_selfのみ有効)"
                        "[Tier1: Rolf 2011実測]", "E_GOAL_TRAJ"),
    "goal_traj_len":  (25, "区分線形軌道のステップ数L[Tier1: Rolf 2011実測]", "E_GOAL_TRAJ_L"),
    "goal_home_prob": (0.1, "確率でホーム姿勢へ戻る[Tier1: Rolf 2011実測]", "E_GOAL_HOME_P"),
    "goal_negative_control": (False, "陰性対照：目標をランダムなダミーに置き換える"
                              "（検証用・reach_selfのみ）", "E_GOAL_NEGCTRL"),
    "reach_arm_side": ("right", "reach_selfで目標にする腕 right/left", "E_REACH_ARM"),
    # ---- 頭へのダブルタッチを報酬に直結する（2026-08-03）---------------------
    # 仕様：作業記録（非公開）
    # `run/plugins/common/double_touch.py`（測定専用）で6000ステップ実測した結果、
    #   「頭」でのダブルタッチだけが reach_success.py の頭タッチ回数と完全一致する
    #   信頼できる指標だった（胸は座面confound、反対の手は未検出）。
    #   reach_space（goal_babbling and goal_space=="reach_self"）が有効なときだけ
    #   taro_setup.py が taro.double_touch を構築する（既定は None のまま）。
    "double_touch_threshold": (0.5, "頭への自己接触presenceのしきい値[Tier3・工学的判断、"
                               "既存プラグインと同じ値]", "E_DTOUCH_THRESH"),
    "double_touch_bonus": (0.0, "頭へのダブルタッチが起きたtickに足す報酬ボーナス"
                           "[Tier3・工学的判断。既定0.0＝OFF。実験ファイルで明示的に"
                           "指定したときだけ有効にする（他の実験的な機構と同じ既定OFFの"
                           "流儀に揃えた。2026-08-03、既定0.2だったのは設計ミスと判明し訂正）。"
                           "有効時の目安0.2＝progress報酬の典型値≈0.04の約5倍で埋もれない"
                           "大きさとして暫定的に選んだだけで文献的根拠は無い", "E_DTOUCH_BONUS"),
    # ---- ダブルタッチ対象部位の全身一般化（2026-08-05）-----------------------
    # 設計：作業記録（非公開）
    # 仕様：作業記録（非公開）
    #   [Tier3・工学的判断]「機構として複数部位（list）を扱えるようにしたこと」
    #   自体に文献根拠は無い。Rochat(1998)の定義は「手の皮膚が顔（頭）の皮膚に
    #   触れる」という組み合わせを他の全ての接触と対比させて定義したものであり、
    #   「顔以外の自分の体の部位」への一般化を許すものではない。既定は頭のみの
    #   まま変えていない＝実験ファイルで明示指定しない限り既存実験の挙動は
    #   1ビットも変わらない。胸などを加える場合は座面confound等、部位ごとの
    #   個別の混同源の確認が要る。「目」は防御的な瞬目反射が知られ、人間の
    #   赤ちゃんが自分の目を触れて報われる行動として確立していないため対象に
    #   加えないこと（`前提.md`「人間の赤ちゃんがしないことは入れない」）。
    "double_touch_touched_groups": (["head"], "ダブルタッチの「触れられる側」の"
                                    "部位一覧（触覚グループ名のリスト、OR判定）"
                                    "[Tier3・工学的判断。既定は頭のみ]。"
                                    "list型のため実験ファイルからのみ指定可能"
                                    "（環境変数からは指定できない、envname=None）",
                                    None),
    # ---- 自己接触の興味度ボーナス（reach_self専用、2026-08-03）--------------
    # 設計：作業記録（非公開）
    # レビュー：同フォルダ\2026-08-03_reach_self新奇性報酬_レビュー.md
    #   （レビューの修正1により、設計の"novelty"という名前を"interest"に統一した。
    #    根拠：Baranes & Oudeyer 2013 SAGG-RIACはこの量を「興味度(interest)」と呼び、
    #    著者自身は"novelty"という語を使っていない。理論分類上もこの量（速い/遅い
    #    移動平均の差の絶対値＝予測器の出力）は "novelty"（記憶に無いことの検出）
    #    ではなく"surprise"（予測との食い違い）に分類される（Barto, Mirolli &
    #    Baldassarre 2013）。名前は依拠する文献（SAGG-RIAC）の用語に合わせた）。
    #   double_touch_bonus（固定値の下駄）とは独立に有効化できる別項目。
    #   既定0.0＝OFF。頭へのダブルタッチが起きたtickに、触覚チャンネルだけを
    #   切り出した局所的な学習進度の絶対値（SAGG-RIAC式・符号を捨てて興味度にする）
    #   を報酬に足す。progress本体（t.lp）には一切触れない、reach_self専用の
    #   実験段階の上書き。
    "self_touch_interest_bonus": (0.0, "頭への自己接触tickに足す興味度ボーナス"
                                  "（触覚チャンネルだけの局所学習進度の絶対値に比例）"
                                  "[Tier3・工学的近似。SAGG-RIAC(Baranes & Oudeyer 2013)の"
                                  "興味度の式を借りた。既定0.0＝OFF]", "E_SELFTOUCH_INTEREST"),
    # ---- progress報酬のsurpriseボーナス（機構1、2026-08-04）------------------
    # 設計：作業記録（非公開）
    #   （案C・機構1のみ。設計は"novelty"と呼んでいるが、この量は理論分類上
    #   surprise（予測との食い違い）であり、novelty（記憶に無いことの検出）とは
    #   異なる（Barto, Mirolli & Baldassarre 2013。self_touch_interest_bonusの
    #   命名訂正と同じ理由）。よってconfigキー名・コードとも"surprise"に統一する。
    "progress_surprise_bonus": (0.0, "レアな予測誤差の急上昇を検出し一時的にprogressへ"
                                "加算するボーナスの利得（trace更新式のgainを兼ねる）"
                                "[Tier3・工学的近似。Kakade&Dayan 2002の二相性反応の"
                                "第一相のみを模す。既定0.0＝OFF]", "E_PROGRESS_SURPRISE_BONUS"),
    "progress_surprise_decay": (0.9, "surprise_bonusのtraceの減衰率[Tier3・恣意的]",
                                "E_PROGRESS_SURPRISE_DECAY"),
    "progress_surprise_var_tau": (0.99, "surprise検出に使う分散の移動平均率[Tier3・恣意的]",
                                  "E_PROGRESS_SURPRISE_VAR_TAU"),
    "progress_surprise_threshold": (5.0, "zスコア（devを直近のばらつきの標準偏差で"
                                    "割った値）が何σを超えたら\"レアな驚き\"とみなすかの"
                                    "しきい値。しきい値を超えた分だけ(超過量型・ヒンジ)を"
                                    "traceに足す[Tier3・工学的近似。"
                                    "2026-08-04実データ検証(check_progress_surprise.py、"
                                    "reach_self・4シード・4500step)で当初案の既定4.0は"
                                    "noneカテゴリtrace平均0.061（要件0.05未満に不合格）、"
                                    "4.5は0.045（4シード集計では合格だが個別には2/4シードが"
                                    "0.05を超えていた）だったため、5.0"
                                    "（集計0.034、個別も1/4シードのみ僅かに超過）を既定にした]",
                                    "E_PROGRESS_SURPRISE_THRESHOLD"),
    # ---- 感覚運動の伝達遅延（候補5、2026-08-02）------------------------------
    # 【なぜ既定0のままか（人間との逸脱を明示）】太郎は運動指令を出したその
    #   env.step()で即座に力が出て、感覚もその場で脳に届く。人間には末梢神経の
    #   伝導に時間がかかる（新生児は特に神経が未髄鞘化で成人よりさらに遅い）。
    #   これは逸脱リスト「感覚運動遅延（神経伝導の遅れ）が無い」に既に登録済みの
    #   既知の逸脱で、太郎の自発運動が人間の2倍細かく震える問題の候補5
    #   （arXiv:2606.17456 のDiscussionが"speculate"として挙げた、遅延が
    #   振動を抑える可能性）。
    #   ⇒ 既定は0（今までと同じ）のまま。実験ファイルで明示的に指定したときだけ
    #   run/plugins/common/scene.py が配線する（run/tools/check_sensorimotor_delay.py
    #   で既定0で1ビットも挙動が変わらないことを確認済み）。
    #
    # 【値の根拠。乳児に「正しい」と確立した決定版の数値は無い（2026-08-02 調査）】
    #   MIMo grows!（López et al. 2025 IEEE ICDL, arXiv:2509.09805）IV-Cは
    #     感覚遅延0/50/200msの3条件を比較（成人の反応時間文献からの転用値[Tier2]）。
    #     運動遅延は最小値1タイムステップ(5ms)に固定し、振っていない。
    #     乳児(0〜4ヶ月)向けの発達段階別の推奨値ではない。
    #   新生児のモロー反射の実測（孫引き含む）：
    #     目視判定 約450ms（Bijesh et al. 2013, Indian Pediatrics、原文確認済み[Tier1だが粗い測定]）
    #     光学式モーションキャプチャ 右腕117.0ms/左腕129.2ms
    #       （Rönnqvist 1995, Neuropsychologia、孫引き[Tier2]）
    #   旧記述「人間には神経伝導の0.1〜0.2秒の遅れがある」（doc/参考文献リスト.md）は
    #     一次文献の出典が確認できず、2026-08-02に訂正した（逸脱リスト参照）。
    #   ⇒ 根拠の強さが違う複数の値を[Tier2]候補として実験ファイルで選べるようにする。
    #     候補：50ms（MIMo grows!の中間値）/120ms（モロー反射・機器計測の平均に近い、
    #     乳児実測に最も近い）/200ms（MIMo grows!の最大値）。
    #
    # 【単位の変換】MIMoEnv(mimo_env.py)のsensory_delay/motor_delayは
    #   「env.step()を何回分遅らせるか」という整数。実測 dt=timestep(5ms)*frame_skip(2)
    #   =10ms/env.step()。ミリ秒→ステップ数は round(delay_ms/1000/dt)（scene.pyで変換）。
    #   50ms→5step, 120ms→12step, 200ms→20step（run/tools/check_sensorimotor_delay.pyで検算済み）。
    "sensory_delay_ms": (0, "感覚が脳に届くまでの遅延（ミリ秒）。既定0=現状と同じ", "E_SENSORY_DELAY_MS"),
    "motor_delay_ms":  (0, "指令が体に届くまでの遅延（ミリ秒）。既定0=現状と同じ", "E_MOTOR_DELAY_MS"),
    # ---- 筋活性化の時定数（震え問題への感度確認、2026-08-07）-----------------
    # MuscleModel（MIMo/mimoActuation/muscle.py）内の一次遅れ（ローパスフィルタ）の
    #   時定数。既定はNone（未指定）＝MuscleModelのハードコード値0.01秒（10ms）の
    #   まま変えない。太郎の自発運動が人間の乳児より2倍細かく震える問題の候補として、
    #   この時定数への感度を確かめるために実験ファイルから振れるようにした
    #   [Tier3・muscle.py既定の0.01秒自体が実測較正されたものではないと既に注記済み]。
    #   actuation=joint（SpringDamperModel）には tau 属性が無いため無視される
    #   （run/plugins/common/scene.py で hasattr確認のうえ配線）。
    "muscle_tau":      (None, "筋活性化の一次遅れの時定数（秒）。既定None=MuscleModelの"
                        "既定0.01秒のまま", "E_MUSCLE_TAU"),
    # ---- モデルの読み書き ---------------------------------------------------
    "model":         (None, "続きから学習するモデルのパス", "E_LOADMODEL"),
    "save":          (None, "学習後にモデルを保存するパス", "E_SAVEMODEL"),
}

RUN_DEFAULTS = {
    "steps":      (600, "学習回数（判断の回数）", None),
    "seed":       (0, "乱数の種", None),
    # K＝1判断あたりの物理ステップ数。K=10＝10Hz（皮質μ律動）。
    # 注意：K=100（1Hz）は人間の最遅神経発火7Hzより遅い＝生物学的に成立しない
    "K":          (10, "1判断あたりの物理ステップ数", "E_K"),
    "checkpoint": (600, "何回ごとに測るか", "E_CKPT"),
    "n_eval":     (80, "自己モデルの評価に使う試行数", None),
    "type":       ("train", "動かし方 train/view/measure/edit", None),
    "log":        (None, "画面に出た文字をそのまま残す先（古い経路のみ）", None),
    "csv":        (None, "チェックポイントの数値を残す先（.csv）", None),
    # ---- 目視（run.type=view）--------------------------------------------
    # 既定で探索ON：決定的な行動だと**ゆらぎが一切出ず自発運動が見えない**
    "view_explore": (True, "自発運動（探索のゆらぎ）を出す", "E_VIEW_EXPLORE"),
    # 学習中の std は 0.05 + ne*0.45。学習初期は ne≒0.275 なので std≒0.174
    "view_std":     (0.174, "探索の揺らぎの大きさ", "E_VIEW_STD"),
    # 2つ目の脳。指定すると Viewer 実行中に ; ' / のキーで往復して見比べられる。
    #   注意：別プロセスで2本立ち上げて見比べるのは当てにならない（乱数も姿勢も違う）。
    #   同じ体・同じ姿勢のまま脳だけ入れ替えるのが正しい比べ方。
    #   （2026-07-30、C_seed0 が固まっているかを C_seed1 と見比べるために追加）
    "view_model_b": (None, "見比べる2つ目のモデルのパス", "E_VIEW_MODEL_B"),
    "view_goal_babbling": (False, "目標指向の動きを見る", "E_VIEW_GOALBABBLE"),
    # 注意：再生では予測誤差を計算しないので学習ループと同じ切替ができない＝割合を直接指定
    "view_gb_rate": (0.5, "目標指向にする割合（学習ループとは違う近似）", "E_VIEW_GB_RATE"),
}

# 環境変数から読むときの型変換
_BOOLS = {k for k, (d, _, _) in TARO_DEFAULTS.items() if isinstance(d, bool)}
_FLOATS = {"lr", "effort_cost", "caps", "beta", "syn_w", "coactivation", "lam_v",
           "age_months", "age_to", "goal_home_prob",
           "sensory_delay_ms", "motor_delay_ms", "muscle_tau",
           "double_touch_threshold", "double_touch_bonus", "self_touch_interest_bonus",
           "progress_surprise_bonus", "progress_surprise_decay", "progress_surprise_var_tau",
           "progress_surprise_threshold"}
_INTS = {"age_start", "age_ramp", "age_every", "steps", "seed", "K", "checkpoint",
         "n_eval", "goal_traj_len"}


class Config:
    """実験の設定。属性で読む（`cfg.lr` `cfg.reward`）。

    注意：作ったあとは**変えない**（読むだけ）。学習の途中で設定が変わると、
      ログのどこから条件が違うのかが追えなくなる。
    """

    def __init__(self, taro=None, run=None, *, scene=None, name=None):
        # 実験の名前。設定ではないが、絵の見出しや記録に使うので持ち回る
        #   （2026-07-31：ダッシュボードの見出しがフォルダ名になっていたので追加）
        self.name = name
        self._taro, self._run = dict(taro or {}), dict(run or {})
        # 知らないキーはここで止める（書き間違いを黙って無視しない）
        for name, d, given in (("taro", TARO_DEFAULTS, self._taro),
                               ("run", RUN_DEFAULTS, self._run)):
            unknown = set(given) - set(d)
            if unknown:
                raise ValueError(
                    f"実験ファイルの {name} 欄に知らない設定がある: {sorted(unknown)}\n"
                    f"  使えるもの: {sorted(d)}")
        for key, (dflt, _doc, _envname) in TARO_DEFAULTS.items():
            setattr(self, key, self._taro.get(key, dflt))
        for key, (dflt, _doc, _envname) in RUN_DEFAULTS.items():
            setattr(self, key, self._run.get(key, dflt))
        self.scene = scene
        self._check()

    # ------------------------------------------------------------------ 作る
    @classmethod
    def from_spec(cls, spec, *, steps_override=None):
        """実験ファイル（辞書）から作る。これが正式な経路。"""
        run = dict(spec.get("run", {}))
        if steps_override is not None:
            run["steps"] = int(steps_override)
        return cls(spec.get("taro"), run, scene=spec.get("scene"),
                   name=spec.get("name"))

    @classmethod
    def from_env(cls):
        """環境変数から作る。注意古い方式と同じ数値が出るか確かめるためだけに使う。

        新しい実験でこれを使わない（実験ファイルが唯一の指定手段）。
        """
        taro, run = {}, {}
        for key, (dflt, _doc, envname) in list(TARO_DEFAULTS.items()):
            if not envname or envname not in os.environ:
                continue
            v = os.environ[envname]
            if key == "actuation":
                taro[key] = "muscle" if v == "1" else "joint"
            elif key in _BOOLS:
                taro[key] = (v == "1")
            elif key in _FLOATS:
                taro[key] = float(v) if v != "" else None
            elif key in _INTS:
                taro[key] = int(v)
            else:
                taro[key] = v if v != "" else None
        # 注意：既定値の型で分ける。全部 int() にすると E_VIEW_STD=0.174 で落ちる
        #   （2026-07-30 の点検で発覚）。
        for key, (dflt, _doc, envname) in RUN_DEFAULTS.items():
            if not envname or envname not in os.environ:
                continue
            v = os.environ[envname]
            if isinstance(dflt, bool):
                run[key] = (v == "1")
            elif isinstance(dflt, float):
                run[key] = float(v)
            else:
                run[key] = int(v)
        if os.environ.get("E_SCENE"):
            return cls(taro, run, scene=os.environ["E_SCENE"])
        return cls(taro, run)

    # ------------------------------------------------------------ 確かめる
    def _check(self):
        """組み合わせとして成り立たない設定をここで止める。"""
        # 注意：語彙は run/plugins/common/scene.py と同じにする（片方だけ通ると
        #   measure では動いて train では弾かれる、という非対称になる）
        if str(self.actuation).lower() not in ("muscle", "muscles", "joint", "spring",
                                               "springdamper", "torque"):
            raise ValueError(f"actuation が不明: {self.actuation}（muscle / joint）")
        if self.antagonist and not self.is_muscle:
            raise ValueError("antagonist（拮抗筋モード）は actuation=muscle が要る")
        if str(self.goal_space) not in ("prop_full", "reach_self"):
            raise ValueError(f"goal_space が不明: {self.goal_space}（prop_full / reach_self）")
        if str(self.reach_arm_side) not in ("right", "left"):
            raise ValueError(f"reach_arm_side が不明: {self.reach_arm_side}（right / left）")
        # goal_space=reach_self は自己接触(q_touch)を目標に使うので touch/somatosensory が要る。
        #   （落とし穴チェックリスト「通ってはいけない条件」。静かに壊れた値を返さず、
        #    学習開始前に明確な ValueError で止める）
        if self.goal_babbling and self.goal_space == "reach_self" \
                and not (self.touch and self.somatosensory):
            raise ValueError(
                "goal_space=reach_self には touch=true, somatosensory=true が要る。\n"
                "  自己接触(q_touch, head/chest/opposite_palmの3部位)を目標に使うため。")
        # 体を育てる設定なのに育たない組み合わせを止める。
        #   【なぜ、2026-07-30】`age_every<=0` だと月齢を見直す処理が**一度も走らない**のに、
        #     起動時には「月齢 0.0 → 4.0」と表示される＝典型的な「黙って壊れる」。
        if self.age_to is not None and self.age_every <= 0:
            raise ValueError(
                f"age_to={self.age_to} を指定しているのに age_every={self.age_every} です。\n"
                "  age_every は「何回ごとに月齢を見直すか」なので、0以下だと\n"
                "  **体が一度も育たないまま**学習が終わります（既定は500）。")
        if self.age_to is not None and self.age_start >= self.steps:
            raise ValueError(
                f"age_start={self.age_start} が steps={self.steps} 以上です。\n"
                "  月齢を変え始める前に学習が終わるので、**体が育ちません**。")
        # 触覚ONで体を育てるときの条件（2026-07-31 に緩和）
        #
        # 【もとの禁止】センサ点が月齢で変わるので観測次元がずれ、黙って壊れる。
        #   0ヶ月1,608点 → 4ヶ月3,268点（落とし穴 項75）
        #
        # 【2026-07-31 に分かったこと】これは表現の問題であって構造的限界ではない。
        #   実測すると、センサーを持つ body の数は月齢によらず一定
        #   （0ヶ月も4ヶ月も同じ。点数だけが倍になる）。
        #   somatosensory=true にすると SomatosensoryCortex（視床VPL+S1相当）が
        #   部位ごとにまとめるので、月齢が変わっても脳への入力次元は固定される。
        #   確かめる道具：run/tools/check_touch_growth.py
        #
        # 【2026-07-31 に一度緩和して、同日に戻した】
        #   somatosensory=true なら大丈夫だと考えて禁止を外したが、実測すると
        #   ★体を育てても**エラーが出ずに黙って壊れる**ことが分かった。
        #
        #   SomatosensoryCortex は「どのインデックスがどの部位か」の表を
        #   作った時点の体で固定して持つ（forward の index_select）。
        #   体が育つと触覚の配列が長くなる（0ヶ月4,824 → 4ヶ月9,804）が、
        #   古いインデックスは範囲内に収まるので**例外にならない**。
        #   結果として、別の部位のデータを読み続ける。
        #   実測：0ヶ月で作った脳に4ヶ月の観測を入れたら、落ちずに値が返った。
        #
        #   さらに trainer の _regrow は observation キーの次元しか見ておらず、
        #   触覚は別キー（touch）なので、次元が倍になっても検出されない。
        #
        # 【2026-07-31 の作り直しで解決】(b) を実装した。
        #   SomatosensoryCortex の部位ごとの出力を
        #   「有無・強さ・重心xyz」の5つ（点数によらず固定）に変えた。
        #   センサ点数に依存する重みが1つも無くなったので、体が育っても層の形が変わらない。
        #   成長時は trainer._regrow → taro.on_body_change() で地図だけ差し替える。
        #   ⇒ touch=true かつ somatosensory=true なら体を育てられる。
        #     somatosensory=false（1枚の巨大変換層）のままでは今も育てられない。
        if self.touch and self.age_to is not None and not self.somatosensory:
            raise ValueError(
                "触覚ONで体を育てるには somatosensory=true が要る。\n"
                "  センサ点が月齢で変わる（0ヶ月4,824 → 4ヶ月9,804次元）ため、\n"
                "  1枚の巨大変換層（somatosensory=false）では入力次元が合わなくなる。\n"
                "  somatosensory=true なら部位ごとの要約（有無/強さ/重心）になり、\n"
                "  点数が変わっても層の形が変わらない。\n"
                "  落とし穴チェックリスト 項75・項86")
        # 自己接触の興味度ボーナス（reach_self専用、2026-08-03）のバリデーション。
        #   設計レビューの修正1〜3反映（作業記録（非公開）
        #   2026-08-03_reach_self新奇性報酬_レビュー.md）。
        if self.self_touch_interest_bonus:
            if self.self_touch_interest_bonus < 0:
                raise ValueError(
                    f"self_touch_interest_bonus={self.self_touch_interest_bonus} は負の値です。\n"
                    "  加点専用（SAGG-RIAC式の興味度＝絶対値）として設計されており、\n"
                    "  罰として使うことは想定していません。0以上の値にしてください。")
            if str(self.touch_mode) != "target":
                raise ValueError(
                    f"self_touch_interest_bonus を有効にするには touch_mode=\"target\" が要る"
                    f"（今 touch_mode={self.touch_mode!r}）。\n"
                    "  この機構は t.blocks の中の touch_embed ブロックを直接スライスして\n"
                    "  誤差を取る（touch_mode=\"input\" だと touch_embed が予測対象の"
                    "ブロックに無く、対象が見つからない）。")
            # 【レビュー修正3】体を育てる実験（cfg.grows、age_toが指定されている）との
            #   同時使用をコードで機械的に止める。ドキュメントの言葉だけに頼らない。
            #   理由：新設する self._self_touch_lp は、成長イベント時に t.lp が持つ
            #   resync()相当の手当てを一切受けない（trainer._regrow はこの新しい
            #   トラッカーに触れない）。そのままだと、成長直後の予測しづらい期間が
            #   自己接触と同じ機序（速い平均だけが跳ねる）で構造的に歪む恐れがある
            #   （レビュー報告2節「体が育っても壊れないか」）。この制限を外すときは、
            #   trainer._regrow に self._self_touch_lp.resync() を足してから外すこと。
            if self.grows:
                raise ValueError(
                    f"self_touch_interest_bonus を有効にした状態で、体を育てる設定"
                    f"（age_to={self.age_to}）を同時に指定することはできません。\n"
                    "  新設のトラッカー(self._self_touch_lp)は成長イベントでの手当て"
                    "（t.lpのresync()相当）を持たないため、成長直後の期間が\n"
                    "  自己接触と同じ機序で構造的に歪む恐れがあります。\n"
                    "  体を固定した実験にとどめるか、age_to=None にしてください。")
        # progress報酬のsurpriseボーナス（機構1、2026-08-04）のバリデーション。
        #   self_touch_interest_bonusと同じ「ゲートしてから中身を検証する」パターン。
        if self.progress_surprise_bonus:
            if self.progress_surprise_bonus < 0:
                raise ValueError(
                    f"progress_surprise_bonus={self.progress_surprise_bonus} は負の値です。\n"
                    "  レアな驚きを罰にしないための加点専用の機構です。0以上にしてください。")
            if not (0.0 <= self.progress_surprise_decay <= 1.0):
                raise ValueError(
                    f"progress_surprise_decay={self.progress_surprise_decay} は0〜1の範囲外です。")
            if not (0.0 <= self.progress_surprise_var_tau <= 1.0):
                raise ValueError(
                    f"progress_surprise_var_tau={self.progress_surprise_var_tau} は0〜1の範囲外です。")
            if self.progress_surprise_threshold < 0:
                raise ValueError(
                    f"progress_surprise_threshold={self.progress_surprise_threshold} は"
                    "負の値です。しきい値は0以上にしてください"
                    "（負にすると平常時のブレまで拾ってしまい、修正前の不具合が再現します）。")
        if self.goal_babbling and self.goal_space == "reach_self":
            print("注意[config] goal_babbling(goal_space=reach_self) を有効にした。"
                  "2026-08-02時点で本番の効果は未検証（診断段階）。"
                  "2026-07-30の『有害』判定は旧実装(goal_space=prop_full)に対するもの"
                  "であり、この新実装には当てはまらない（設計：2026-08-02_案Cの実装設計.md）",
                  flush=True)
        elif self.goal_babbling:
            print("注意[config] goal_babbling を有効にした（goal_space=prop_full、旧実装）。"
                  "2026-07-30 の実測で**有害**（おもちゃへの接触−32%、"
                  "margin +30.7%→+17.2%、persist 1000%）。"
                  "原典と目標空間が違う（原典＝手先位置の低次元）", flush=True)
        if self.reward == "predict":
            print("注意[config] reward=predict は【既知の欠陥】（大行動バイアス＝"
                  "大きく動くほど得なので暴れる。実測でうつ伏せ57.5%・jerk2501）", flush=True)
        if not self.is_muscle:
            print("注意[config] 関節モード（90関節を独立に駆動）＝逸脱リスト 逸脱5 の逸脱を"
                  "選んでいます。人間の新生児は拮抗筋を同時に力ませる[Tier1]", flush=True)
        # ダブルタッチ対象部位の全身一般化（2026-08-05）のバリデーション。
        #   触覚グループ名として実際に存在するかどうかは、taro_core側の
        #   DoubleTouchDetector.__init__（構築時、学習開始前）のAssertionErrorに
        #   委ねる（config.py単体は taro_core の部位一覧を知らないため、実行時の
        #   検証は与えない。実装の裁量、仕様4節）。ここでは型（空でない
        #   文字列のリスト）だけを確認する。
        if not self.double_touch_touched_groups or \
                not all(isinstance(nm, str) for nm in self.double_touch_touched_groups):
            raise ValueError(
                f"double_touch_touched_groups は空でない文字列のリストである必要がある: "
                f"{self.double_touch_touched_groups!r}")
        if self.K >= 100:
            print(f"[!] K={self.K}（{100/self.K:.0f}Hz）＝人間の最遅神経発火7Hzより遅い。"
                  f"比較・再現目的でなければ K=10 を使うこと", flush=True)

    # ---------------------------------------------------------- 便利な読み
    @property
    def is_muscle(self):
        return str(self.actuation).lower() in ("muscle", "muscles", "")

    @property
    def target_kind(self):
        """予測対象の種類。'0' / 'vision' / 'all'。"""
        return str(self.target).lower()

    @property
    def target_has_vision(self):
        return self.target_kind in ("1", "vision", "all")

    @property
    def target_has_all(self):
        return self.target_kind == "all"

    @property
    def grows(self):
        """体を育てる実験か。"""
        return self.age_to is not None

    def age_at(self, step):
        """学習が step 回進んだ時点の月齢。age_to が無ければ常に同じ。

        注意：[Tier3] 月齢を学習回数の**線形**で進めることに直接の根拠は無い。
          taro_core 側の体の補正（taper_weight / target_ratio_for_age）が線形なのに
          合わせてある。主張できるのは「崖が無い」ことだけ。
        """
        if self.age_to is None:
            return self.age_months
        a0 = float(self.age_months if self.age_months is not None else 0.0)
        a1 = float(self.age_to)
        if self.age_ramp <= 0:                  # 一瞬で切り替える（＝「急に変える」条件）
            return a1 if step >= self.age_start else a0
        f = (step - self.age_start) / float(self.age_ramp)
        return a0 + (a1 - a0) * max(0.0, min(1.0, f))

    def summary(self):
        """既定から変えた項目だけを並べる（ログの1行目に出す）。"""
        out = []
        for key, (dflt, _doc, _e) in TARO_DEFAULTS.items():
            v = getattr(self, key)
            if v != dflt:
                out.append(f"{key}={v}")
        return " ".join(out) or "(すべて既定)"

    def as_dict(self):
        """モデルの保存に入れる用（あとから条件を確認できるように）。"""
        d = {k: getattr(self, k) for k in TARO_DEFAULTS}
        d.update({k: getattr(self, k) for k in ("steps", "seed", "K", "checkpoint")})
        d["scene"] = self.scene
        return d


def touch_setting_of(model_path, verbose=True):
    """保存されたモデルを覗いて、触覚の設定（touch / somatosensory）を言い当てる。

    【なぜ要るか、2026-07-31】学習した脳を後から開く道具（Viewer・速度の計測など）は
    触覚の設定を持っていないことが多い。設定が食い違うと
    **触覚の層だけ白紙**の脳になる。しかも例外は出ない
    （`Taro._load` は形の合う層だけ読む strict=False）。
    ⇒ 実験ファイルの書き方に頼らず、**モデル自身に聞く**。

    見分け方：
        fusion_touch が無い          → 触覚なし
        fusion_touch に part_weight  → SomatosensoryCortex（部位ごとの要約）
        それ以外                     → TouchEncoder（1枚の巨大変換層）

    Returns:
        dict: Config の taro 欄に混ぜて使う。読めなければ空 dict。
    """
    import torch
    try:
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:      # noqa: BLE001
        if verbose:
            print(f"注意[脳] 触覚の設定を読み取れません: {e}。触覚なしとして開きます",
                  flush=True)
        return {}
    ft = blob.get("fusion_touch")
    if ft is None:
        if verbose:
            print("  [脳] このモデルは触覚なしで学習されています", flush=True)
        return {"touch": False, "somatosensory": False}
    soma = any(k.startswith("part_weight") for k in ft)
    if verbose:
        print(f"  [脳] このモデルは触覚ありで学習されています"
              f"（{'部位ごとの要約' if soma else '1枚の変換層'}）", flush=True)
    return {"touch": True, "somatosensory": soma}
