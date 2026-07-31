"""太郎の中身の地図 ── 何が実装されていて、どの実験で何が効いているか。

【なぜ要るか、2026-07-30】ユーザーの要望：

> 今のプロジェクトで実装したものはすべて表示して、実験に使ってないものは薄くするとか？
> その図でどの経路を通ってどの判断がされたのか。みたいなのが分かるといい！

太郎の設定は30個以上あり、「この実験は何が違うのか」が読まないと分からない。
実際そこで事故が起きた（学習は関節モード・測定は筋肉モードという**別の体**）。
⇒ **1枚の絵で、実装したもの全部と、この実験で効いているものを見せる。**

【このファイルの役割】絵の**中身の定義だけ**を持つ。描くのは `run/tools/wiring.py`。
  ・NODES … 機構の一覧（日本語・英語・置き場所・根拠・説明）
  ・EDGES … どこからどこへ繋がるか
  ・`is_on(node, cfg)` … その実験で効いているか

注意：根拠のラベル（Tier）は**確実に分かっているものだけ**入れる。
  推測で付けると `doc/人間模倣からの逸脱リスト.md` と食い違い、
  「文献の裏付けがある」と誤解させる。分からないものは None のままにして、
  絵にも「未記載」と出す（＝宿題が見える）。

【根拠の段階（プロジェクトの流儀）】
  Tier1  一次文献の**実測値**がある
  Tier2  文献に定性的な裏付けはあるが数値は自分で決めた
  Tier3  恣意的（工学的な近似）＝必ず複数値で振って感度を確かめる対象
  逸脱   人間模倣から外れていると自覚している（逸脱リストに登録済み）
  未実装 中身が無い／繋がっていない
"""

# 置き場所（絵の列）。left→right で「感覚 → 脳 → 体」の流れになる
COLUMNS = [
    ("sense",  "感覚", "senses"),
    ("fuse",   "まとめる", "fusion"),
    ("brain",  "脳の中", "brain"),
    ("motor",  "動きを作る", "motor"),
    ("body",   "体", "body"),
]
# 下の段（報酬と調節の輪）
LOWER = [
    ("reward", "報酬をつくる", "reward"),
    ("modul",  "神経で調節する", "neuromodulation"),
    ("memory", "覚える・固める", "memory"),
]

# ---------------------------------------------------------------- 機構の一覧
# (id, 日本語, 英語, 列, 根拠, 説明, ONの判定)
#   ONの判定は Config を受け取って True/False を返す関数。None なら常にON。
N = None
NODES = [
    # ---- 感覚 ---------------------------------------------------------
    ("prop", "固有感覚", "proprioception", "sense", "Tier1",
     "関節の角度・速度・筋の張り。筋肉モードで801次元、関節モードで621次元", N),
    ("vest", "前庭感覚", "vestibular", "sense", "Tier1",
     "三半規管（回転）と耳石器（傾き・直線の加速）", N),
    ("touch", "触覚", "touch", "sense", "逸脱",
     "体を育てる実験では使えない（センサ点が月齢で変わり観測次元がずれる）",
     lambda c: bool(c.touch)),
    ("vision", "視覚（両眼）", "vision", "sense", "Tier2",
     "新生児の視力に合わせてぼかす。眼球が常に寄り目＝遠くを見られない（既知）",
     lambda c: bool(c.vision)),
    ("intero", "内受容感覚", "interoception", "sense", "未実装",
     "空腹・眠気・不快・覚醒。E1では中身がほぼ定数（逸脱リストに登録済み）", N),
    # ---- まとめる -----------------------------------------------------
    ("insula", "島皮質", "insula", "fuse", "Tier2",
     "内受容を脳が扱える形に変える", N),
    ("fusion", "感覚をまとめる層", "sensory fusion", "fuse", "Tier3",
     "各感覚を**等重み**で足す＝理論的な正当化のない仮定（逸脱リストに登録済み）", N),
    ("soma", "体性感覚野", "somatosensory cortex", "fuse", "Tier2",
     "触覚を部位別にまとめる（視床VPL＋S1相当）",
     lambda c: bool(c.somatosensory and c.touch)),
    ("tgtfuse", "正解側の凍結した層", "frozen target encoder", "fuse", "Tier3",
     "予測の「正解」を作る側。学習中の層を使うと出力を平坦にする抜け道で崩れる（RND式）", N),
    # ---- 脳の中 -------------------------------------------------------
    ("gru", "前回の行動を再帰へ", "efference copy → GRU", "brain", "Tier1",
     "「自分が出した命令」を次の予測に入れる＝遠心性コピー", N),
    ("pc", "予測符号化の潜在変数", "predictive coding latent", "brain", "Tier1",
     "いまの状況を表す内部表現 z を作る", N),
    ("fwd", "順モデル", "forward model", "brain", "Tier1",
     "「この行動をしたらこう感じるはず」を予測する。自己モデルの本体", N),
    ("target", "予測する対象", "prediction target", "brain", "Tier3",
     "既定は固有感覚のみ。予測対象に入っていない感覚は内部表現から捨てられる（実測）",
     N),
    ("blockpe", "次元数の影響を除く", "per-block error", "brain", "Tier1",
     "ブロックごとに平均してから足す（Ohata & Tani 2020 ほか）",
     lambda c: c.target_has_vision),
    # ---- 動きを作る ---------------------------------------------------
    ("cortex", "運動野", "motor cortex", "motor", "Tier1",
     "内部表現から運動の指令を作る", N),
    ("cereb", "小脳", "cerebellum", "motor", "Tier2",
     "うまくいった運動を真似て自動化する（実測で約26%ブレンド）",
     lambda c: bool(c.cerebellum)),
    ("cpg", "脊髄の運動性喃語", "spinal CPG", "motor", "Tier1",
     "色付きノイズ 1/f^β。β＝0.686(8週)〜0.877(30週)［López et al. 2026・実測］",
     lambda c: str(c.noise) == "colored"),
    ("synergy", "粗いシナジー", "muscle synergy", "motor", "Tier1",
     "新生児は手足をまとめてしか動かせない（文献が一致）",
     lambda c: bool(c.synergy) and str(c.noise) == "colored"),
    ("antag", "拮抗筋の共収縮", "co-activation", "motor", "Tier1",
     "新生児は拮抗筋を同時に力ませる［Hadders-Algra 1992］",
     lambda c: bool(c.antagonist)),
    ("goal", "目標指向の探索", "goal babbling", "motor", "逸脱",
     "2026-07-30 の実測で**有害**（接触−32%・persist 1000%）。原典と目標空間が違う",
     lambda c: bool(c.goal_babbling)),
    ("inv", "逆モデル（行動を逆算）", "inverse model", "motor", "Tier3",
     "順モデルを反転して「望む感覚に届く行動」を推論する",
     lambda c: bool(c.goal_babbling)),
    # ---- 体 -----------------------------------------------------------
    ("muscle", "筋肉モード（拮抗筋2本/関節）", "muscle model", "body", "Tier1",
     "引くだけ・活性化ダイナミクス・長さと速度で力が変わる。行動180次元[0,1]",
     lambda c: c.is_muscle),
    ("joint", "関節モード", "torque model", "body", "逸脱",
     "90関節を独立に駆動＝逸脱リスト 逸脱5。新生児は拮抗筋を同時に力ませる[Tier1]",
     lambda c: not c.is_muscle),
    ("bodyshape", "新生児の体つき", "infant body proportions", "body", "Tier1",
     "頭が大きく四肢が短い。上肢/下肢＝1.07（成人は0.77）", N),
    ("grow", "体が育つ", "body growth", "body", "Tier3",
     "月齢を学習回数の線形で進める（直接の根拠は無い。言えるのは「崖が無い」だけ）",
     lambda c: c.grows),
    # ---- 報酬をつくる -------------------------------------------------
    ("progress", "学習進度（好奇心）", "learning progress", "reward", "Tier1",
     "予測誤差が**減っている**ことを求める（Oudeyer の内発的動機）",
     lambda c: c.reward == "progress"),
    ("predict", "予測しやすさ", "predictability", "reward", "逸脱",
     "大行動バイアス（大きく動くほど得＝暴れる）。実測でうつ伏せ57.5%",
     lambda c: c.reward == "predict"),
    ("homeo_in", "内臓の恒常性", "interoceptive homeostasis", "reward", "Tier2",
     "つらさが下がったら報酬。空腹・眠気・不快から作る", N),
    ("effort", "努力コスト", "effort cost", "reward", "Tier3",
     "活性化の二乗×筋サイズ。二乗の形とλは恣意的（Selinger 2015 が動機）",
     lambda c: bool(c.effort_cost)),
    ("caps", "行動の滑らかさ", "action smoothing (CAPS)", "reward", "Tier3",
     "急に違うことをするのは損［Mysore et al. 2021］。λは恣意的",
     lambda c: bool(c.caps)),
    # ---- 神経で調節する -----------------------------------------------
    ("dop", "ドーパミン", "dopamine", "modul", "Tier1",
     "報酬予測誤差（思ったより良かったか）で方策を学ぶ", N),
    ("ne", "ノルアドレナリン（青斑核）", "locus coeruleus / NE", "modul", "Tier1",
     "探索と活用の切り替え［Aston-Jones & Cohen 2005］。探索の揺らぎ＝0.05+ne×0.45", N),
    ("homeo_sc", "恒常性スケーリング", "homeostatic scaling", "modul", "Tier1",
     "神経の活動量を一定に保つ（暴走も沈黙も防ぐ）", N),
    ("mature", "探索の結晶化", "NE maturation", "modul", "Tier3",
     "学習が進むほど探索を減らす。成熟までの回数にその実験の長さを流用している",
     lambda c: bool(c.mature)),
    ("clock", "発達時計", "developmental clock", "modul", "Tier2",
     "時間を3軸に分ける（現実の秒／シミュレーションの秒／学習回数＝発達年齢）", N),
    # ---- 覚える・固める -----------------------------------------------
    ("hippo", "海馬", "hippocampus", "memory", "Tier1",
     "直近の経験をためる（上限つき）", lambda c: bool(c.replay)),
    ("replay", "睡眠中の経験リプレイ", "sleep replay", "memory", "逸脱",
     "睡眠そのものは実装しない判断。定着だけを周期的なリプレイで行う",
     lambda c: bool(c.replay)),
]

# --------------------------------------------------------------- 経路（線）
# (from, to, ラベル, 太さの元になる数え方のキー ※段階3で使う)
EDGES = [
    ("prop", "fusion", "", None),
    ("vest", "fusion", "", None),
    ("touch", "soma", "", None),
    ("soma", "fusion", "", None),
    ("vision", "fusion", "", None),
    ("intero", "insula", "", None),
    ("insula", "fusion", "", None),
    ("fusion", "gru", "感覚", None),
    ("gru", "pc", "", None),
    ("pc", "fwd", "", None),
    ("pc", "cortex", "", None),
    ("tgtfuse", "target", "正解", None),
    ("target", "blockpe", "", None),
    ("fwd", "blockpe", "予測", None),
    ("cortex", "cereb", "", None),
    ("cereb", "muscle", "自動化", "cereb_blend"),
    ("cortex", "muscle", "指令", None),
    ("cortex", "joint", "指令", None),
    ("cpg", "muscle", "ゆらぎ", None),
    ("synergy", "cpg", "", None),
    ("antag", "cpg", "", None),
    ("goal", "inv", "", "goal_steps"),
    ("inv", "cortex", "逆算した行動", None),
    ("muscle", "prop", "動いた結果", None),
    ("joint", "prop", "動いた結果", None),
    ("grow", "bodyshape", "月齢", None),
    ("blockpe", "progress", "予測誤差", None),
    ("blockpe", "predict", "予測誤差", None),
    ("progress", "dop", "報酬", None),
    ("predict", "dop", "報酬", None),
    ("homeo_in", "dop", "", None),
    ("effort", "dop", "引く", "effort_mean"),
    ("caps", "dop", "引く", "caps_mean"),
    ("dop", "cortex", "方策を直す", None),
    ("progress", "ne", "タスクの出来", None),
    ("ne", "cpg", "探索の強さ", "noise"),
    ("ne", "cortex", "ゆらぎの幅", None),
    ("mature", "ne", "", None),
    ("clock", "mature", "発達年齢", None),
    ("homeo_sc", "fusion", "活動量を保つ", None),
    ("hippo", "replay", "", None),
    ("replay", "fwd", "復習で固める", "replay_count"),
    ("blockpe", "hippo", "経験をためる", None),
]

# 根拠のラベルごとの色（絵の枠に使う）
TIER_STYLE = {
    "Tier1":  ("#276749", "一次文献の実測値がある", "grounded in measured data"),
    "Tier2":  ("#2b6cb0", "定性的な裏付けはあるが数値は自分で決めた", "qualitative support"),
    "Tier3":  ("#b7791f", "恣意的（工学的な近似）＝感度を確かめる対象", "arbitrary"),
    "逸脱":   ("#c53030", "人間模倣から外れていると自覚している", "known deviation"),
    "未実装": ("#718096", "中身が無い／繋がっていない", "not implemented"),
    None:     ("#a0aec0", "根拠が未記載＝確かめる宿題", "unlabeled"),
}


def node_index():
    return {n[0]: n for n in NODES}


def is_on(node, cfg):
    """この実験でその機構が効いているか。cfg は run/config.py の Config。"""
    fn = node[6]
    if fn is None:
        return True
    try:
        return bool(fn(cfg))
    except Exception:
        return False        # 設定が足りない実験ファイルでも絵は出す


# 人が決めた箱の位置（配線図のHTMLで箱をドラッグ →「位置を書き出す」で作る）。
#   空のままなら自動の並び（列ごとに上から順）で描く。
#   【なぜ人が決めるか、2026-07-30】自動での並べ替え（バリセンター法）を試したが
#   **悪化した**（線が箱を横切る 35→42件）。この図は綺麗な層構造ではないため。
#   ⇒ 見やすさは人が決めるのがいちばん速い。決めた値をここに貼れば固定される。
POSITIONS = {}
