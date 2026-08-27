# --- コピー元: Taro (github.com/ooo7OneuBLT/Taro) ---
# --- 元パス: B/src/taro/brain/left_frontal_lobe/brocas_area.py
#     （F2-1で taro_core へ移植・2026-08-23。A/src/taro/brain/brocas_area.py の
#     SpeechPlanner は A2-10 で改名される前の旧版なので移植元にしていない） ---
# --- 設計: F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部 実装作業②・⑧ ---

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

【F2-1・決定3／実装作業⑧】`vocal_tract.hear()` は文字→調音パラメータの
100%正確な「大人の対応表」の辞書引き。帳面（小脳の逆モデル）が空のまま
ここへフォールバックすると、赤ちゃんがいきなり大人の発音になってしまう
（親設計 `設計_F2_初語.md` が却下した「設計X」と同型）。そのため
`use_hear_fallback` 引数を新設し、**既定Falseで切る**。B原本は常時ON
（＝use_hear_fallback=True相当）だったが、taro_core は目標横断で共有する
ため既定は安全側（フォールバックしない＝未知の文字は探索的な動きに任せる）
に倒した。F2-1では plan() をまだ呼び出す配線をしていない（喃語のみ）。
F2-2で語を言わせる段になって初めて呼ばれる想定。
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

    def plan(self, parent_text, cerebellum, vocal_tract, use_hear_fallback=False):
        """
        親の発話から発話計画を立てる。

        ① 親の発話を1モーラずつ分解（日本語：1文字＝1モーラ）
        ② 各モーラの口の動きを小脳の逆モデルで検索
        ③ 見つかったものをバッファに入れる
        ④ 見つからないモーラは、use_hear_fallback が真なら
           vocal_tract.hear()（大人の正解表）で埋める。偽（既定）なら
           「未知」のまま（motor=None）にして、generate() 側の探索的な
           動き（大脳皮質が自力で選ぶ）に委ねる。
        """
        self.motor_buffer = []
        self.current_index = 0

        for char in parent_text:
            motor = cerebellum.lookup_motor(char)
            if motor is not None:
                self.motor_buffer.append({"motor": motor, "known": True, "target": char})
            elif use_hear_fallback:
                heard_params = vocal_tract.hear(char)
                if heard_params is not None:
                    self.motor_buffer.append({"motor": heard_params, "known": False, "target": char})
                else:
                    self.motor_buffer.append({"motor": None, "known": False, "target": char})
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
