"""筋活性化ダイナミクス（一次遅れ）を既存のトルクモータに足す実行層（taro-C5 実験①）。

【なぜ必要か】太郎の運動は「1秒(K=100)ごとに新しい目標トルクを1回サンプルし、そのまま
1秒保持→次の瞬間に別のトルクへ瞬時ジャンプ」という段階関数（スナップ＆ホールド）だった。
目標Dで眼球視界を初めて動画で確認したところ視野が激しく揺れ、Prechtl's General Movements
Assessment（発達運動学の確立した評価体系）の基準では、健常な乳児の自発運動（なめらかに
始まり終わる・波打つように緩やかに変化）ではなく、むしろ"カクつき(jerkiness)が過大＝軽度
神経学的機能不全の臨床サイン"に近いと判明した（詳細：doc/人間模倣からの逸脱リスト.md
2026-07-17）。

【機構の根拠（人間模倣）】本物の筋肉は、神経指令(興奮)が届いても実際に出す力を瞬時には
変えられない。カルシウムイオンの動態により、力は一次遅れでしか立ち上がり／収まりできない
（筋活性化ダイナミクス、Zajac 1989; Winters & Woo 1990）。標準形は

    da/dt = (u - a) / τ        u=脳が決めた目標トルク、a=実際に印加するトルク

で、力を上げるとき(活性化)は速く、収めるとき(脱活性化)は遅い（時定数が非対称）。
物理ステップごとに更新する。

【なぜ MuscleModel をそのまま使わないか】MIMo純正の MuscleModel はこの活性化ダイナミクスを
内蔵する（同じ tau=0.01 を使用）が、①行動次元が拮抗筋ペアで倍増（90→180）②関節の
硬さ・減衰も変える③較正ファイルが要る、ため学習済みモデル(c_pred_abs_seed0.pt)が読めず
物理も一度に複数変わる。ここでは「1変数ずつ」の原則に従い、行動次元・物理・脳をそのままに、
"力の立ち上がり/収まりが一次遅れになる"部分だけを既存トルクモデルに足す最小版とする。

【⚠️根拠ラベル】機構（一次遅れの活性化ダイナミクス・時定数の非対称）は文献に基づく【人間模倣】。
時定数の具体値 10ms/40ms は OpenSim 等の標準デフォルト（Zajac 1989; Winters 1990、MIMoの
MuscleModel も tau=0.01 を採用）であり、太郎の身体に対して実測較正したものではない
＝⚠️感度確認の対象。また「トルクの符号付き大きさ|·|で活性化/脱活性化を判定」する近似は、
制御レンジが 0 中心で対称（SpringDamperModel の前提）であることに依存する。
"""
import numpy as np

from mimoActuation.actuation import SpringDamperModel


class SmoothTorqueModel(SpringDamperModel):
    """SpringDamperModel（瞬時トルク）に一次遅れの活性化ダイナミクスを足した版。

    行動空間・観測次元・関節物理は SpringDamperModel と完全に同一。唯一の違いは、
    脳が決めた目標トルクを data.ctrl へ即座に書かず、物理ステップごとに一次遅れで
    近づける点（＝力が瞬時ジャンプしなくなる）。
    """

    TAU_ACT = 0.010    # 力を上げる（活性化）の時定数[s]。Zajac 1989; Winters 1990 の標準値。
    TAU_DEACT = 0.040  # 力を収める（脱活性化）の時定数[s]。活性化より遅い（非対称）。

    def __init__(self, env, actuators):
        super().__init__(env, actuators)
        # applied: 実際に印加している（なめらかに追従する）制御値。target: 脳が決めた目標値。
        self.applied = np.zeros(self.action_space.shape, dtype=np.float64)
        self.target = np.zeros(self.action_space.shape, dtype=np.float64)

    def action(self, action):
        """脳の指令を「目標トルク」として受け取る（＝ここでは即座に印加しない）。

        SpringDamperModel は data.ctrl に即セットしていたが、ここでは target に保持するだけ。
        実際の印加は substep_update（物理ステップごと）が一次遅れで行う。observations() や
        cost() が参照する control_input は SpringDamper と同じく「指令(目標)」の意味に保つ
        （＝観測の次元も意味も変えない。実際に印加された力は simulation_torque が data.ctrl
        から読むので、そちらには一次遅れが反映される）。
        """
        self.target = np.clip(action, self.action_space.low,
                              self.action_space.high).astype(np.float64)
        self.control_input = self.target
        self.env.data.ctrl[self.actuators] = self.applied  # 現在の（まだ追従途中の）印加値

    def substep_update(self):
        """物理ステップごとに、印加トルクを目標へ一次遅れで近づける（活性化ダイナミクス）。"""
        dt = self.env.model.opt.timestep
        # 力の"大きさ"が増える向き＝活性化(速い)、減る向き＝脱活性化(遅い)。
        increasing = np.abs(self.target) > np.abs(self.applied)
        tau = np.where(increasing, self.TAU_ACT, self.TAU_DEACT)
        self.applied += dt * (self.target - self.applied) / tau
        self.env.data.ctrl[self.actuators] = self.applied

    def reset(self):
        """印加値・目標値を 0（無張力）に戻す。"""
        self.applied = np.zeros(self.action_space.shape, dtype=np.float64)
        self.target = np.zeros(self.action_space.shape, dtype=np.float64)
        self.env.data.ctrl[self.actuators] = self.applied
