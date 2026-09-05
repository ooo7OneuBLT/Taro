"""測る道具：太郎自身の産出（F2「見た物の名前を言う」）を記録する。

【設計】F/docs/設計_F2_初語（見た物の名前を言う）.md 第2部／
  F/docs/設計_F2-2_見た物の名前を言う.md 第3部「測るもの」。

【役割】読むだけ（`run/plugins/base.py` の規約）。太郎も環境も変えない。
  読むのは：
    ctx.last_produce  … run/trainer.py の _apply_word_production が判断ごとに
                        置く産出イベント（トリガー不成立ならNone）。
                        {"target_word", "sim", "generated_word", "reward",
                         "plan_length", "known_moras"}
                        reward は produce.learn（既定False）がFalseのときは
                        None（F2-2・実装作業④で学習を通さない既定にしたため。
                        設計第2部決定1）。
    ctx.last_parent_utterance … 「語→見ていた物(toy1/toy2)」の対応表を作るのに
                        使う（word_learningプラグインと同じ手法。太郎には触れない）

【F2-2・実装作業⑥・2026-08-23】既存のこのプラグインで実際に記録されるかを
  短い走行（run/tools/check_f2_2_short_run.py）で確かめたところ、wordモードの
  トリガー（視覚・注視・クールダウン）が成立すれば ctx.last_produce が置かれ、
  このプラグインで記録できることを確認した（babble_probeが必要だった
  ctx.last_babble とは別経路＝babble_probeを新設した2026-08-23の事情はwordモード
  には当てはまらない）。よってwordモード用に新設はせず、このプラグインを
  F2-2の指標（plan_length・known_moras・帳面参照率）に合わせて拡張した。

【出す指標】
  発話回数（トリガーが成立した回数）
  一致度(reward)の推移（学習ON時のみ意味を持つ） → CSVに書き出して図で見る
  ⑤ 帳面の参照率：known_moras/plan_length の平均（計画のうち何モーラが
    帳面から引けたか。設計第3部「測るもの」⑤）

【発話イベントCSV】いつ(step/sim_sec)・何を見ていたときに(toy)・言いたかった語
  (target_word)・確信度(sim)・実際に出た音(generated_word)・一致度(reward)・
  計画長(plan_length)・帳面から引けたモーラ数(known_moras)・目標語と完全一致したか
  (exact_match、設計第3部④「言えた率」)。
  実験ファイルでの書き方:
    "plugins": {"word_production": {"events_out": "E/logs/.../発話イベント.csv"}}
  events_out を省略した場合はCSVを書かない（メモリ上の集計だけ返す）。
"""
import csv
import os

from run.plugins.base import Plugin


class WordProduction(Plugin):
    name = "word_production"

    def setup(self, ctx):
        self.events_out = self.config.get("events_out")
        self.word_to_target = {}     # 語(text) -> 見ていた物(toy1/toy2)
        self.rows = []

    def on_step(self, ctx):
        # word_learningプラグインと同じ手法：親の発話イベントから
        # 「語→対象」の対応表を育てる（太郎の状態には一切触れない）。
        for ev in (getattr(ctx, "last_parent_utterance", None) or []):
            text, target = ev.get("text"), ev.get("target")
            if text and target:
                self.word_to_target.setdefault(text, target)
        ev = getattr(ctx, "last_produce", None)
        if not ev:
            return
        toy = self.word_to_target.get(ev["target_word"])
        reward = ev.get("reward")
        plan_length = ev.get("plan_length")
        known_moras = ev.get("known_moras")
        self.rows.append({
            "step": ctx.step, "sim_sec": round(ctx.sim_sec, 3),
            "toy": toy if toy is not None else "",
            "target_word": ev["target_word"], "sim": round(ev["sim"], 4),
            "generated_word": ev["generated_word"],
            # 【段階2・2026-09-02】語の選択の内訳（gru_hippo方式のときだけ入る）
            "choice_gru": (ev.get("choice") or {}).get("gru"),
            "choice_hippo": (ev.get("choice") or {}).get("hippo"),
            "chosen": (ev.get("choice") or {}).get("chosen"),
            "reward": round(reward, 4) if reward is not None else "",
            "plan_length": plan_length if plan_length is not None else "",
            "known_moras": known_moras if known_moras is not None else "",
            "exact_match": int(ev["generated_word"] == ev["target_word"]),
        })

    def metrics(self, ctx):
        out = {"word_production.count": len(self.rows)}
        if self.rows:
            rewards = [r["reward"] for r in self.rows if r["reward"] != ""]
            if rewards:
                out["word_production.last_reward"] = rewards[-1]
            out["word_production.exact_match_rate"] = (
                sum(r["exact_match"] for r in self.rows) / len(self.rows))
        return out

    def report(self, ctx):
        if self.rows and self.events_out:
            path = (self.events_out if os.path.isabs(self.events_out) else
                    os.path.join(os.path.abspath(os.path.join(
                        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)),
                        self.events_out))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_sec", "toy", "target_word", "sim",
                           "generated_word", "reward", "plan_length",
                           "known_moras", "exact_match",
                           "choice_gru", "choice_hippo", "chosen"])
                for r in self.rows:
                    w.writerow([r["step"], r["sim_sec"], r["toy"], r["target_word"],
                               r["sim"], r["generated_word"], r["reward"],
                               r["plan_length"], r["known_moras"], r["exact_match"],
                               r.get("choice_gru"), r.get("choice_hippo"), r.get("chosen")])
        exact = [r for r in self.rows if r["exact_match"]]
        return {
            "発話回数": len(self.rows),
            "完全一致数": len(exact),
        }
