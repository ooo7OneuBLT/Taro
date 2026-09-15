"""測る道具：喃語モード（F2-1）の発話を記録する。

【なぜ要るか、2026-08-23】既存の `word_production` プラグインは
`ctx.last_produce`（語を言うモードの発話イベント）しか読まない。喃語モード
（`produce.mode="babble"`）の発話は `ctx.last_babble` に置かれるため、
`word_production` を付けて喃語走行を回しても発話が1件も記録されなかった
（実測：300秒走行で「発話数」0、発話イベントCSVも作られず）。
既存プラグインを直すと word モードの記録が壊れる恐れがあるため、
喃語モード専用の測定器を新設した。

【設計】F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「測るもの」。

【役割】読むだけ（`run/plugins/base.py` の規約）。太郎も環境も変えない。乱数は
  一切消費しない（辞書の読み取りと集計だけ）。
  読むのは：
    ctx.last_babble  … run/trainer.py の _apply_babble が判断ごとに置く喃語
                        イベント（不成立ならNone）。
                        {"generated_word", "jaw_cycles", "length"}
    ctx.taro.produce_cerebellum … 帳面の持ち主（taro_core/src/brain/cerebellum.py
                        の Cerebellum。forward_map/inverse_map/experience_count）
    ctx.taro.produce_vocal_tract … 文字→調音パラメータの変換（VocalTract.hear()）。
                        調音点(place)が0（「なし」＝母音のみ）かどうかを見るのに使う

【出す指標（設計第4部の4項目）】
  ① 発話ごとの記録   … events_out（1行1発話）
  ② 帳面の成長       … growth_out（一定間隔ごとの forward/inverse/experience）
  ③ 音ごとの練習回数 … practice_out（走行の最後に1回、1行1文字）
  ④ 子音率の時間推移 … ②と同じ間隔・同じ行に同居させる（時点が同じため。
       consonant_rate_cumulative=開始からの累積、consonant_rate_window=前回の
       記録時点からの区間だけ、の2列）

  実験ファイルでの書き方:
    "plugins": {"babble_probe": {
        "events_out": "F/logs/.../発話イベント.csv",
        "growth_out": "F/logs/.../帳面の成長.csv",
        "practice_out": "F/logs/.../音ごとの練習回数.csv",
        "growth_every": 100
    }}
  各 *_out は省略可（省略した出力だけ作らない）。growth_every の既定は100ステップ。
"""
import csv
import os

from run.plugins.base import Plugin


def _abs_path(p):
    if os.path.isabs(p):
        return p
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    return os.path.join(root, p)


class BabbleProbe(Plugin):
    name = "babble_probe"

    def setup(self, ctx):
        """ctx を受け取り、出力先(events_out/growth_out/practice_out)とgrowth_every・記録用リスト・子音率カウンタを初期化する。戻り値は無い。
        """
        self.events_out = self.config.get("events_out")
        self.growth_out = self.config.get("growth_out")
        self.practice_out = self.config.get("practice_out")
        self.growth_every = int(self.config.get("growth_every", 100))

        self.event_rows = []
        self.growth_rows = []

        # ④ 子音率：累積用・直近区間用を分けて数える（design：両方出す）。
        self._cum_cons = 0
        self._cum_total = 0
        self._win_cons = 0
        self._win_total = 0

    def on_step(self, ctx):
        """ctx を受け取り、喃語イベントがあれば1発話分を記録して生成語の各文字の調音点から子音率を更新し、growth_everyステップごとに小脳の帳面の件数と子音率も記録する。戻り値は無い。
        """
        ev = getattr(ctx, "last_babble", None)
        if ev:
            word = ev.get("generated_word", "")
            self.event_rows.append({
                "step": ctx.step, "sim_sec": round(ctx.sim_sec, 3),
                "generated_word": word, "char_count": len(word),
                "jaw_cycles": ev.get("jaw_cycles"),
            })
            vt = getattr(ctx.taro, "produce_vocal_tract", None)
            if vt is not None:
                for ch in word:
                    params = vt.hear(ch)
                    if params is None:
                        # 声道の対応表に無い文字（実際には speak() の出力なので
                        # 起きない想定）。分母に含めない＝妥当な範囲だけを数える。
                        continue
                    place = params[0]
                    is_consonant = place != 0   # 0="なし"＝母音のみ（design 第4項）
                    self._cum_total += 1
                    self._win_total += 1
                    if is_consonant:
                        self._cum_cons += 1
                        self._win_cons += 1

        if self.growth_every > 0 and ctx.step % self.growth_every == 0:
            cereb = getattr(ctx.taro, "produce_cerebellum", None)
            if cereb is not None:
                cum_rate = (self._cum_cons / self._cum_total) if self._cum_total else 0.0
                win_rate = (self._win_cons / self._win_total) if self._win_total else 0.0
                self.growth_rows.append({
                    "step": ctx.step, "sim_sec": round(ctx.sim_sec, 3),
                    "forward_entries": len(cereb.forward_map),
                    "inverse_entries": len(cereb.inverse_map),
                    "total_experiences": sum(cereb.experience_count.values()),
                    "consonant_rate_cumulative": round(cum_rate, 4),
                    "consonant_rate_window": round(win_rate, 4),
                })
                # 区間だけの分は次の区間のためにリセット（累積側は消さない）。
                self._win_cons = 0
                self._win_total = 0

    def metrics(self, ctx):
        """ctx を受け取り、記録が無ければNoneを返し、それ以外は発話数と（あれば）帳面の最新の件数を辞書で返す。"""
        if not self.event_rows and not self.growth_rows:
            return None
        out = {"babble_probe.events": len(self.event_rows)}
        if self.growth_rows:
            last = self.growth_rows[-1]
            out["babble_probe.forward_entries"] = last["forward_entries"]
            out["babble_probe.inverse_entries"] = last["inverse_entries"]
        return out

    def report(self, ctx):
        """ctx を受け取り、指定された出力先へ発話イベント・帳面の成長・音ごとの練習回数をそれぞれCSVで書き出し、喃語の発話数・練習した音の種類数・練習総数を辞書で返す。
        """
        if self.event_rows and self.events_out:
            path = _abs_path(self.events_out)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_sec", "generated_word", "char_count", "jaw_cycles"])
                for r in self.event_rows:
                    w.writerow([r["step"], r["sim_sec"], r["generated_word"],
                               r["char_count"], r["jaw_cycles"]])

        if self.growth_rows and self.growth_out:
            path = _abs_path(self.growth_out)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_sec", "forward_entries", "inverse_entries",
                           "total_experiences", "consonant_rate_cumulative",
                           "consonant_rate_window"])
                for r in self.growth_rows:
                    w.writerow([r["step"], r["sim_sec"], r["forward_entries"],
                               r["inverse_entries"], r["total_experiences"],
                               r["consonant_rate_cumulative"], r["consonant_rate_window"]])

        # ③ 音ごとの練習回数：cerebellum.experience_count は (place,manner,voicing,
        #   vowel) をキーにした練習回数。forward_map で motor_key→文字が引けるので、
        #   同じ文字に対応する motor_key の練習回数を合算する（speak() は決定的な
        #   ので、同じ motor_key は常に同じ文字になり二重計上は起きない）。
        #   これで合計は experience_count の総和と一致する（別集計をしていない）。
        practice_counts = {}
        cereb = getattr(ctx.taro, "produce_cerebellum", None)
        if cereb is not None:
            for motor_key, count in cereb.experience_count.items():
                ch = cereb.forward_map.get(motor_key)
                if ch is None:
                    continue
                practice_counts[ch] = practice_counts.get(ch, 0) + count

        if practice_counts and self.practice_out:
            path = _abs_path(self.practice_out)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["char", "practice_count"])
                for ch, n in sorted(practice_counts.items(), key=lambda kv: -kv[1]):
                    w.writerow([ch, n])

        return {
            "喃語_発話数": len(self.event_rows),
            "喃語_練習した音の種類数": len(practice_counts),
            "喃語_練習総数": sum(practice_counts.values()) if practice_counts else 0,
        }
