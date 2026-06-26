"""
ブローカ野（Broca's Area） — 左前頭葉にある発話計画の中枢

【人間模倣】ブローカ野は発話の計画・順序づけ・実行を担当する。
1861年にPierre Paul Brocaが発見した、「言葉の座」。

GODIVAモデル（Guenther）に基づき、発話をモーラ単位でバッファリングし
順に実行する。太郎は日本語母語なので計画の単位はモーラ（≒ひらがな1文字）。

流れ：
  ① 親の発話を聞く
  ② 小脳の逆モデルで各モーラの口の動きを検索 → バッファに入れる
  ③ バッファから1モーラずつ取り出して実行（NEノイズつき）
  ④ バッファが空になったら止まる

A2-10で旧speech_planner.pyから改名。人間の脳の部品名に合わせた。

参考文献：
- GODIVA model (Guenther, PMC 2021)
- Broca's area (Brodmann areas 44, 45)
"""


class BrocasArea:
    """
    ブローカ野。音韻バッファと構造バッファを持つ。

    音韻バッファ：「どんな音を出すか」の列（モーラ単位の口の動き）
    構造バッファ：「いくつ出すか」（モーラ数）
    """

    def __init__(self):
        self.motor_buffer = []
        self.structure = 0
        self.current_index = 0

    def plan(self, parent_text, cerebellum, vocal_tract):
        """
        親の発話から発話計画を立てる。

        ① 親の発話を1モーラずつ分解（日本語：1文字＝1モーラ）
        ② 各モーラの口の動きを小脳の逆モデルで検索
        ③ 見つかったものをバッファに入れる
        ④ 見つからないモーラは「未知」として、探索的な動きを入れる
        """
        self.motor_buffer = []
        self.current_index = 0

        for char in parent_text:
            motor = cerebellum.lookup_motor(char)
            if motor is not None:
                self.motor_buffer.append({"motor": motor, "known": True, "target": char})
            else:
                heard_params = vocal_tract.hear(char)
                if heard_params is not None:
                    self.motor_buffer.append({"motor": heard_params, "known": False, "target": char})
                else:
                    self.motor_buffer.append({"motor": None, "known": False, "target": char})

        self.structure = len(self.motor_buffer)

    def has_next(self):
        return self.current_index < self.structure

    def next_motor(self):
        if not self.has_next():
            return None
        item = self.motor_buffer[self.current_index]
        self.current_index += 1
        return item

    def get_plan_length(self):
        return self.structure

    def reset(self):
        self.motor_buffer = []
        self.structure = 0
        self.current_index = 0
