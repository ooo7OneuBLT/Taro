"""
小脳（Cerebellum） — 運動スキルの記憶（順モデル＋逆モデル）

【人間模倣】DIVAモデル（Guenther, Boston University）に基づく。
赤ちゃんは喃語期に「口を動かす→音が出る→聞く」を繰り返し、
「この口の動き→この音」の対応（順モデル）を体験的に蓄積する。

順モデル（forward model）：口の動き → 出る音の予測
逆モデル（inverse model）：出したい音 → 口の動きの検索（順モデルの逆引き）

声道シミュレータ（vocal_tract.py）は「実際の口の物理構造」。
小脳の順モデルは「太郎の脳が体験から学んだ知識」。
声道のテーブルは最初から完全だが、順モデルはゼロから体験で蓄積する。
"""


class Cerebellum:
    """
    小脳。口の動きと音の対応を体験から学ぶ。

    順モデル：(place, manner, voicing, vowel) → 文字
    逆モデル：文字 → (place, manner, voicing, vowel)
    """

    def __init__(self):
        self.forward_map = {}
        self.inverse_map = {}
        self.experience_count = {}
        # B5-3：運動の自動化（VMS＝vocal motor scheme）。よく練習した口の動きほど
        # 自動化され、実行時のばらつき（運動ノイズ）が減って形が安定する＝結晶化。
        # 報酬ではなく「反復回数そのもの」で固める（Vihman；DIVA/小脳の運動学習）。
        # ⚠️定数：自動化の練習スケール（この回数でおよそ半分自動化）。未検証・後で
        # 感度確認と除去テスト（数を入れる原則）。
        self.automatization_scale = 2000

    def automatization(self, place, manner, voicing, vowel):
        """
        この口の動きの自動化度（0=不慣れ 〜 1=ほぼ完全自動）。練習回数で飽和する。

        【人間模倣】運動スキルは反復練習で自動化し、実行のばらつきが減る
        （DIVAモデル／小脳・大脳基底核の手続き的学習）。自動化度が高いほど
        generate時のNEノイズ（運動ノイズ）を抑え「形の安定＝結晶化」を表す。
        """
        n = self.experience_count.get((place, manner, voicing, vowel), 0)
        return n / (n + self.automatization_scale)

    def learn_from_experience(self, place, manner, voicing, vowel, produced_char):
        """
        体験から学ぶ。「この口の動きでこの音が出た」を記録する。

        喃語のたびに呼ばれる。同じ組み合わせを何度も経験すると
        確信度（experience_count）が上がる。
        """
        motor_key = (place, manner, voicing, vowel)
        self.forward_map[motor_key] = produced_char
        self.experience_count[motor_key] = self.experience_count.get(motor_key, 0) + 1

        if produced_char not in self.inverse_map:
            self.inverse_map[produced_char] = motor_key
        else:
            existing_key = self.inverse_map[produced_char]
            existing_count = self.experience_count.get(existing_key, 0)
            new_count = self.experience_count[motor_key]
            if new_count > existing_count:
                self.inverse_map[produced_char] = motor_key

    def predict_sound(self, place, manner, voicing, vowel):
        """
        順モデル：この口の動きをしたら何の音が出るか予測する。

        体験したことがない組み合わせはNoneを返す（予測できない）。
        """
        return self.forward_map.get((place, manner, voicing, vowel), None)

    def lookup_motor(self, target_char):
        """
        逆モデル：この音を出すにはどう口を動かせばいいか検索する。

        体験したことがない音はNoneを返す（出し方を知らない）。
        """
        return self.inverse_map.get(target_char, None)

    def lookup_motor_sequence(self, target_text):
        """
        逆モデル：文字列全体に対して口の動きの列を検索する。

        知っている文字の動きだけ返す。知らない文字はスキップ。
        """
        motor_sequence = []
        for char in target_text:
            motor = self.lookup_motor(char)
            if motor is not None:
                motor_sequence.append(motor)
        return motor_sequence

    def get_known_sounds(self):
        """今まで出したことのある音の一覧を返す。"""
        return list(self.inverse_map.keys())

    def get_experience_summary(self):
        """体験の統計を返す。"""
        return {
            "forward_entries": len(self.forward_map),
            "inverse_entries": len(self.inverse_map),
            "total_experiences": sum(self.experience_count.values()),
        }
