"""
発話計画（Speech Planner） — GODIVAモデルに基づく発話のバッファリングと実行

【人間模倣】人間は1文字ずつ「次は何にしよう」と考えながら話すのではなく、
「何を言うか」を先に計画してから口を動かす（GODIVA: Guenther）。

太郎は日本語母語なので、計画の単位はモーラ（≒ひらがな1文字）。

流れ：
  ① 親の発話を聞く
  ② 小脳の逆モデルで各モーラの口の動きを検索 → バッファに入れる
  ③ バッファから1モーラずつ取り出して実行（NEノイズつき）
  ④ バッファが空になったら止まる

参考文献：
- GODIVA model (Guenther, PMC 2021)
- Chunking of phonological units (PMC, 2019)
"""


class SpeechPlanner:
    """
    発話計画。音韻バッファと構造バッファを持つ。

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
        """バッファにまだ実行すべきモーラが残っているか。"""
        return self.current_index < self.structure

    def next_motor(self):
        """
        バッファから次のモーラの口の動きを取り出す。

        戻り値: {"motor": (place, manner, voicing, vowel) or None,
                 "known": bool, "target": str}
        """
        if not self.has_next():
            return None
        item = self.motor_buffer[self.current_index]
        self.current_index += 1
        return item

    def get_plan_length(self):
        """計画の長さ（モーラ数）。"""
        return self.structure

    def reset(self):
        """計画をクリアする。"""
        self.motor_buffer = []
        self.structure = 0
        self.current_index = 0
