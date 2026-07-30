"""実験の設定を1つの入れ物にまとめる。

【なぜ要るか、2026-07-30】`E/scripts/e_growth_train.py` は設定を**53個のモジュール変数**
（`_LR` `_MUSCLE` `_REWARD` …）として持ち、それを930行の関数の中から直接読んでいた。
そのため：
  ・設定を変えるには**環境変数を経由するしかなかった**（実験ファイルから渡せない）
  ・どの設定がどこで効くのか、読まないと分からなかった
  ・★既定値が食い違っても気づけなかった（学習ループだけ関節モードだった事故）

→ 設定を1つのオブジェクトにして、**実験ファイルから作る**のを正式な経路にする。
  環境変数から作る経路（`from_env`）は「古い方式と同じ数値が出るか確かめる」ためだけに残す。

【3つの欄との対応】実験ファイル（E/experiments/*.json）の欄がそのまま入る：
    taro   → 太郎の中身（本能のON/OFF・体の月齢・駆動モード）
    run    → 動かし方（steps / seed / K / checkpoint）
    scene  → 環境（ここには入れない。scene.py が担う）

⚠️★ここに「測り方」は入れない。測るのはプラグインの仕事。
⚠️★根拠ラベルは各項目のコメントに残す。値の由来（Tier1=一次文献／Tier3=恣意的）が
  分からなくなると、あとで「この数字はどこから来たのか」を追えなくなる。
"""
import os


# ★既定値の表。キー＝実験ファイルの `taro` / `run` で書く名前。
#   値＝(既定値, 説明, 環境変数名)。環境変数名は古い方式との比較用。
#   ⚠️既定値を変えるときは、必ず理由を研究日誌に書く（過去に既定値の食い違いで事故った）。
TARO_DEFAULTS = {
    # ---- 体 ----------------------------------------------------------------
    # 駆動モード。muscle＝拮抗筋2本/関節・引くだけ・行動180次元[0,1]。
    # ★joint（90関節を独立に駆動）は逸脱リスト「逸脱5」の逸脱
    #   （Hadders-Algra et al. 1992［Tier1］＝新生児は拮抗筋を同時に力ませる）。
    "actuation":     ("muscle", "駆動モード muscle/joint", "E_MUSCLE"),
    "age_months":    (None, "体の月齢。None＝シーンの値を使う", "E_AGE"),
    "age_to":        (None, "終わりの月齢。None＝体を育てない", "E_AGE_TO"),
    "age_start":     (0, "何回目の学習から月齢を変え始めるか", "E_AGE_START"),
    "age_ramp":      (0, "何回かけて月齢を変えるか。0＝一瞬で", "E_AGE_RAMP"),
    "age_every":     (500, "何回ごとに月齢を見直すか", "E_AGE_EVERY"),
    # ---- 感覚 --------------------------------------------------------------
    # ⚠️触覚ONでは体を育てられない（センサ点が月齢で変わり観測次元がずれる。項75）
    "touch":         (False, "触覚を足す", "E_TOUCH"),
    "touch_mode":    ("target", "触覚を予測対象にするか input/target", "E_TOUCH_MODE"),
    "somatosensory": (False, "触覚を視床VPL+S1相当の経路にする", "E_SOMATOSENSORY"),
    "vision":        (True, "視覚を入力に入れる", "E_E1_VISION"),
    # 予測対象。"0"=固有感覚のみ／"vision"=+視覚／"all"=+前庭+触覚+視覚
    "target":        ("0", "予測対象 0/vision/all", "E_E1_TARGET"),
    "lam_v":         (1.0, "視覚ブロックの重み[Tier3]", "E_LAM_V"),
    # ---- 本能（脳の中の機構）------------------------------------------------
    "lr":            (0.005, "学習率", "E_LR"),
    # progress＝学習進度（Oudeyer の好奇心）。predict は★大行動バイアスの既知欠陥あり
    "reward":        ("progress", "内発的動機 progress/predict", "E_REWARD"),
    "ne_relative":   (True, "ノルアドレナリンを相対基準で出す", "E_NE_RELATIVE"),
    "replay":        (True, "睡眠中の経験リプレイ（記憶定着）", "E_REPLAY"),
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
    # ⚠️★2026-07-30 の実測で**有害**と判明（接触−32%・margin +30.7%→+17.2%・
    #   persist 1000%）。原典（Rolf, Steil & Gienger 2010）とは目標空間が違う
    #   （原典＝手先位置の低次元／太郎＝固有感覚621次元まるごと）。作り直し予定。
    "goal_babbling": (False, "★目標指向の探索（現状は有害と判明）", "E_GOALBABBLE"),
    "goal_switch":   ("pe", "探索/目標の切替 fixed/ne/pe", "E_GB_SWITCH"),
    "closed_loop_reach": (False, "目標を保持してにじり寄る", "E_CLTRAIN"),
    # ---- モデルの読み書き ---------------------------------------------------
    "model":         (None, "続きから学習するモデルのパス", "E_LOADMODEL"),
    "save":          (None, "学習後にモデルを保存するパス", "E_SAVEMODEL"),
}

RUN_DEFAULTS = {
    "steps":      (600, "学習回数（判断の回数）", None),
    "seed":       (0, "乱数の種", None),
    # K＝1判断あたりの物理ステップ数。K=10＝10Hz（皮質μ律動）。
    # ⚠️K=100（1Hz）は人間の最遅神経発火7Hzより遅い＝生物学的に成立しない
    "K":          (10, "1判断あたりの物理ステップ数", "E_K"),
    "checkpoint": (600, "何回ごとに測るか", "E_CKPT"),
    "n_eval":     (80, "自己モデルの評価に使う試行数", None),
    "type":       ("train", "動かし方 train/view/measure", None),
    "log":        (None, "画面に出た文字をそのまま残す先（★古い経路のみ）", None),
    "csv":        (None, "チェックポイントの数値を残す先（.csv）", None),
    # ---- 目視（run.type=view）--------------------------------------------
    # ★既定で探索ON：決定的な行動だと**ゆらぎが一切出ず自発運動が見えない**
    "view_explore": (True, "自発運動（探索のゆらぎ）を出す", "E_VIEW_EXPLORE"),
    # 学習中の std は 0.05 + ne*0.45。学習初期は ne≒0.275 なので std≒0.174
    "view_std":     (0.174, "探索の揺らぎの大きさ", "E_VIEW_STD"),
    "view_goal_babbling": (False, "目標指向の動きを見る", "E_VIEW_GOALBABBLE"),
    # ⚠️再生では予測誤差を計算しないので学習ループと同じ切替ができない＝割合を直接指定
    "view_gb_rate": (0.5, "目標指向にする割合（★学習ループとは違う近似）", "E_VIEW_GB_RATE"),
}

# 環境変数から読むときの型変換
_BOOLS = {k for k, (d, _, _) in TARO_DEFAULTS.items() if isinstance(d, bool)}
_FLOATS = {"lr", "effort_cost", "caps", "beta", "syn_w", "coactivation", "lam_v",
           "age_months", "age_to"}
_INTS = {"age_start", "age_ramp", "age_every", "steps", "seed", "K", "checkpoint",
         "n_eval"}


class Config:
    """実験の設定。属性で読む（`cfg.lr` `cfg.reward`）。

    ⚠️★作ったあとは**変えない**（読むだけ）。学習の途中で設定が変わると、
      ログのどこから条件が違うのかが追えなくなる。
    """

    def __init__(self, taro=None, run=None, *, scene=None):
        self._taro, self._run = dict(taro or {}), dict(run or {})
        # ★知らないキーはここで止める（書き間違いを黙って無視しない）
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
        """実験ファイル（辞書）から作る。★これが正式な経路。"""
        run = dict(spec.get("run", {}))
        if steps_override is not None:
            run["steps"] = int(steps_override)
        return cls(spec.get("taro"), run, scene=spec.get("scene"))

    @classmethod
    def from_env(cls):
        """環境変数から作る。⚠️★古い方式と同じ数値が出るか確かめるためだけに使う。

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
        for key, (dflt, _doc, envname) in RUN_DEFAULTS.items():
            if envname and envname in os.environ:
                run[key] = int(os.environ[envname])
        if os.environ.get("E_SCENE"):
            return cls(taro, run, scene=os.environ["E_SCENE"])
        return cls(taro, run)

    # ------------------------------------------------------------ 確かめる
    def _check(self):
        """組み合わせとして成り立たない設定をここで止める。"""
        if str(self.actuation).lower() not in ("muscle", "joint", "spring",
                                               "springdamper", "torque"):
            raise ValueError(f"actuation が不明: {self.actuation}（muscle / joint）")
        if self.antagonist and not self.is_muscle:
            raise ValueError("antagonist（拮抗筋モード）は actuation=muscle が要る")
        # ⚠️触覚ONで体を育てると観測次元がずれて黙って壊れる（落とし穴 項75）
        if self.touch and self.age_to is not None:
            raise ValueError(
                "★触覚ONでは体を育てられない（センサ点が月齢で変わり観測次元がずれる）。\n"
                "  0ヶ月1734点 → 4ヶ月4274点。落とし穴チェックリスト 項75")
        if self.goal_babbling:
            print("⚠️[config] ★goal_babbling を有効にした。2026-07-30 の実測で"
                  "**有害**（おもちゃへの接触−32%、margin +30.7%→+17.2%、persist 1000%）。"
                  "原典と目標空間が違う（原典＝手先位置の低次元）", flush=True)
        if self.reward == "predict":
            print("⚠️[config] reward=predict は【既知の欠陥】（大行動バイアス＝"
                  "大きく動くほど得なので暴れる。実測でうつ伏せ57.5%・jerk2501）", flush=True)
        if not self.is_muscle:
            print("⚠️[config] ★関節モード（90関節を独立に駆動）＝逸脱リスト 逸脱5 の逸脱を"
                  "選んでいます。人間の新生児は拮抗筋を同時に力ませる[Tier1]", flush=True)
        if self.K >= 100:
            print(f"[!] K={self.K}（{100/self.K:.0f}Hz）＝人間の最遅神経発火7Hzより遅い。"
                  f"比較・再現目的でなければ K=10 を使うこと", flush=True)

    # ---------------------------------------------------------- 便利な読み
    @property
    def is_muscle(self):
        return str(self.actuation).lower() == "muscle"

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

        ⚠️[Tier3] 月齢を学習回数の**線形**で進めることに直接の根拠は無い。
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
        """★既定から変えた項目だけを並べる（ログの1行目に出す）。"""
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
