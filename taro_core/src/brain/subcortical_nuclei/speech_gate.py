# -*- coding: utf-8 -*-
"""発話の門（SpeechGate） — 「言うかどうか」を反応の随伴で学ぶ大脳基底核の門。

【設計】F/docs/設計_発話の動機.md（2026-08-31）。
太郎が声を出したあと、窓の時間内に親の声が返ってきたら報酬1、返らなければ0。
この報酬から「言う確率π」を学ぶ。人間側の対応：大脳基底核の Go/NoGo 門
（行動を出すか止めるかの選択）＋ドーパミンの報酬予測誤差。

【人間模倣の位置づけ（文献調査 2026-08-31 で確認した限界）】
- 随伴的な親の反応が乳児の発声を強化する証拠は強い（Goldstein & Schwade 2008 ほか）
  が、対象は喃語。命名への適用は類推〔Tier2〕
- 報酬予測誤差での定式化は鳥の歌学習（Gadagkar et al. 2016）からの理論借用〔Tier2〕
- πが場面によらないロジット1本なのは初版の単純化〔Tier3・工学的判断〕。
  場面依存（誰がいるか・何を見ているか）は将来、状態を入力に取る形へ拡張する

【学習則・2026-08-31改（Go/NoGo方式）】
初版はロジット1本のREINFORCEだったが、πが端（0.96）に近づくと更新が (1-π) 倍に
潰れて動かなくなり、静止顔実験の第3幕（期待が育った後の無反応）でπが下がらなかった
（F/logs/F2-22_静止顔/図_第3幕の問題.png）。大脳基底核の実際のモデルに合わせ、
**「言う価値 Q_go」と「黙る価値 Q_nogo」を別々に覚えて差で決める**形に差し替えた：
    取った行動の Q を実入りへ向けて少しずつ動かす（指数移動平均）
    π = sigmoid((Q_go - Q_nogo) / T)     T=温度（差の効き方の鋭さ）
人間側の対応：線条体の直接路（D1・Go）と間接路（D2・NoGo）が行動の価値を並行に
持つ、というBG標準モデル（Frank 2005 系）。飽和が原理的に起きず、価値が落ちれば
πも素直に落ちる。ドーパミン（RPE）は Q の更新誤差そのものに対応する。
"""
import math


class SpeechGate:
    """言う価値 q_go と黙る価値 q_nogo を別々に持ち、その差で言う確率を決める門。

    初版は「1本のロジット」方式だったが、言う側だけが更新されて黙る側が
    育たなかったため、2つの価値を別々に覚える形に差し替えた
    （経緯はモジュール冒頭の説明文）。
    """

    def __init__(self, init_prob=0.9, lr=0.1, dopamine=None, speak_cost=0.1,
                 temperature=0.1):
        # 初期πは高め＝導入直後は従来の「必ず言う」に近い挙動から出発する。
        # Q_go の初期値を「πがinit_probになる差」から逆算する（Q_nogo=0起点）。
        p = min(max(float(init_prob), 1e-4), 1.0 - 1e-4)
        self.temperature = float(temperature)
        self.q_go = math.log(p / (1.0 - p)) * self.temperature
        self.q_nogo = 0.0
        # lr＝価値の指数移動平均の速さ。0.1なら約10回の経験で新しい実入りに追いつく
        #   （人間の静止顔実験で乳児の行動が数分＝十数回の発話機会で変わるのに対応）
        self.lr = float(lr)
        # 【本能11・努力コスト】声を出すこと自体の負担。これが無いと
        #   「言っても言わなくても報酬0なら差が無い」となり、無反応の親の前でも
        #   πが下がらない（F2-22第3幕・コスト無し版で実測）。人間の乳児が無視されると
        #   黙るのは報酬ゼロだからではなく、出すだけ損だから。値は暫定〔Tier3・要較正〕。
        self.speak_cost = float(speak_cost)
        self.dop = dopamine            # neuromodulator.dopamine.Dopamine
        self.n_decisions = 0
        self.n_speak = 0
        self.n_rewarded = 0
        self.history = []              # (sim_sec, prob, action, reward) 静止顔実験の記録用

    def prob(self):
        """q_goとq_nogoの差をtemperatureで割ってシグモイド関数に通し、「言う」を選ぶ確率を計算して返す。引数は無い。"""
        return 1.0 / (1.0 + math.exp(-(self.q_go - self.q_nogo) / self.temperature))

    def decide(self, rng):
        """言うかどうかを引く。rng は環境の np_random（再現性のため）。"""
        a = bool(rng.random() < self.prob())
        self.n_decisions += 1
        if a:
            self.n_speak += 1
        return a

    def resolve(self, action, reward, sim_sec=None):
        """結果（窓内に反応が来たか）で1回ぶん学習する。

        action: そのとき言ったか（True/False）。言わなかった決定も報酬0で
        resolve する＝「言わなければ反応も無い」が方策に織り込まれる。
        """
        p = self.prob()
        net = float(reward) - (self.speak_cost if action else 0.0)
        # 取った行動の価値だけを、実際の実入りへ向けて動かす（δ＝予測誤差）。
        if action:
            delta = net - self.q_go
            self.q_go += self.lr * delta
        else:
            delta = net - self.q_nogo
            self.q_nogo += self.lr * delta
        # ドーパミンのbaselineは観測用に更新し続ける（学習はQの誤差が担う）
        if self.dop is not None:
            self.dop.compute_rpe(net)
        a = 1.0 if action else 0.0
        if reward > 0:
            self.n_rewarded += 1
        self.history.append((float(sim_sec) if sim_sec is not None else -1.0,
                             p, a, float(reward)))

    # ------------------------------------------------------------ 保存・復元
    def state(self):
        """q_go・q_nogo・temperature・logit（(q_go-q_nogo)/temperature）・baseline（dopamineがあればその値、無ければ0.0）・n_decisions・n_speak・n_rewarded・historyのコピーを持つ辞書を返す。引数は無い。
        """
        return {"q_go": self.q_go, "q_nogo": self.q_nogo,
                "temperature": self.temperature,
                "logit": (self.q_go - self.q_nogo) / self.temperature,   # 後方互換の読み出し用
                "baseline": self.dop.baseline if self.dop is not None else 0.0,
                "n_decisions": self.n_decisions,
                "n_speak": self.n_speak,
                "n_rewarded": self.n_rewarded,
                # πの軌跡（sim_sec, π, 言ったか, 報酬）。静止顔実験の図の材料。
                # 復元では読まない（続きの学習で軌跡が混ざるのを避ける）
                "history": list(self.history)}

    def load_state(self, st):
        """辞書stからq_go・q_nogoを復元する（stに'q_go'があればそのまま使い、無く'logit'があれば logit*temperature をq_go・q_nogo=0として復元する）。dopamineがあれば'baseline'も復元し、n_decisions・n_speak・n_rewardedも復元する。historyは復元しない。戻り値は無い。
        """
        if "q_go" in st:
            self.q_go = float(st["q_go"])
            self.q_nogo = float(st["q_nogo"])
        elif "logit" in st:
            # 旧形式（ロジット1本）からの移行：差へ換算して引き継ぐ
            self.q_go = float(st["logit"]) * self.temperature
            self.q_nogo = 0.0
        if self.dop is not None and "baseline" in st:
            self.dop.baseline = float(st["baseline"])
        self.n_decisions = int(st.get("n_decisions", 0))
        self.n_speak = int(st.get("n_speak", 0))
        self.n_rewarded = int(st.get("n_rewarded", 0))
