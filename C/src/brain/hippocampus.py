"""
海馬（Hippocampus） — 運動性自己モデル用の短期経験バッファ＆睡眠リプレイ

【人間模倣】海馬は覚醒中の経験を一時保存し、睡眠中のシャープ波リプルで大脳皮質へ
転送して長期記憶に定着させる（McClelland et al., 1995 CLS理論）。目標Bの言語用海馬
（B/src/taro/brain/hippocampus.py）と同じ原理を、**感覚運動（自己内部モデル）経験**に
適用したもの。

⚠️ 言語用海馬を「コピー」してはいない：あちらが保存するのは発話トークン
（full_tokens, body_state, satiety_target）で、こちらが保存するのは感覚運動テンソル
（融合ベクトル・行動・固有感覚など）。共有しているのは「FIFOバッファ→覚醒中record→
睡眠でリプレイ→clear」という型（＝CLSの原理）だけで、中身とリプレイの学習ロジックは別。

なぜ必要か（目標C2で実証）：太郎の順モデルはオンライン1回きり学習で、同じ経験を
二度と復習しないため予測の質が頭打ちになる（corr 0.44 / 持続予測比 82%）。睡眠中に
貯めた経験を何度も再生して定着させると頭打ちを突破する（corr 0.56 / 持続予測比 69%、
オフライン多周回で到達可能な上限にほぼ一致）。人間の睡眠中の記憶固定そのもの。

保存内容（1エピソード）: 覚醒中の1ステップの順モデル予測に必要な材料一式。
  sv       : 融合感覚ベクトル（学習中エンコーダの出力）
  prev_a   : 直前の行動（GRU入力の遠心性コピー）
  action   : このステップで実際に取った行動（予測ヘッドへの条件づけ）
  cf       : 固定ターゲットエンコーダの出力（再構成の"正解"、崩壊回避用）
  clp      : 現在の固有感覚（残差予測の基準）
  nlp      : 次の固有感覚（予測の教師）
  hidden   : このステップのGRU隠れ状態（覚醒時の再帰文脈を再現するため）
"""


class MotorHippocampus:
    """運動性自己モデル用の海馬。覚醒中の感覚運動経験を蓄積し、睡眠移行時にリプレイする。

    定着（重み更新）のロジックは呼び出し側の consolidate に置く（言語用海馬が
    core_b.consolidate に定着を任せているのと同じ分担）。海馬自体はバッファに徹する。
    """

    def __init__(self, max_capacity=3600):
        # 容量＝睡眠までに貯めうる経験数。超過時は最古を捨てる（FIFO）。
        self.max_capacity = max_capacity
        self.episodes = []

    def record(self, sv, prev_a, action, cf, clp, nlp, hidden):
        """覚醒中の1ステップを記録する。全テンソルは detach 済みで渡すこと。"""
        if len(self.episodes) >= self.max_capacity:
            self.episodes.pop(0)
        self.episodes.append((sv, prev_a, action, cf, clp, nlp, hidden))

    def replay(self):
        """蓄積した全経験をリストで返す。"""
        return list(self.episodes)

    def clear(self):
        """睡眠後にクリア（皮質への転送完了）。"""
        self.episodes.clear()

    def __len__(self):
        return len(self.episodes)
