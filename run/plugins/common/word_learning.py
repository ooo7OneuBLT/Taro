"""測る道具：親のfollow-in labeling（F1-3）で語彙がどれだけ育っているかを見る。

【設計】F/docs/仕様_F1-3_親のfollow-in labelingと耳の配線.md 技術付録「部品3：測定」。

【役割】読むだけ（`run/plugins/base.py` の規約）。太郎も環境も変えない。
  読むのは：
    ctx.last_parent_utterance  … run/trainer.py が判断ごとに置く発話イベントのリスト
                                （{"text","target","state"}。"state"は太郎が実際に
                                 lexicon.observe に渡したのと同じ視覚特徴＝64次元）
    ctx.taro.lexicon           … 連合器（assoc()で読み出すだけ、書き込まない）
    ctx.taro.hearing           … 耳（hear()でトークン化するだけ）

【出す指標】
  label_count_toy1 / label_count_toy2 : 親がその物に対して発話した累計回数
  assoc_sep : 連合の分離度（2語平均）。
    cos(assoc(語), 正解物の特徴EMA) − cos(assoc(語), 不正解物の特徴EMA)
    正なら「その語を聞いたときに連合が思い浮かべる見え方」が正しい物へ寄っている。

  特徴EMA（「正解物の素の見え方」の目安）はこのプラグインが独自に保持する
  （太郎には触れない＝仕様書の指示どおり）。ParentLabelingは注視が確認できてから
  しか発話しない設計なので、発話イベントに乗ってくる視覚特徴（state）は
  「そのとき見ていたもの＝その物の見え方」の妥当なサンプルになる。
  lexicon.assoc()自体も同じイベント列の平均だが、EMA（指数移動平均・直近重視）と
  lexiconの累積平均は式が異なるため、独立した目安として使える
  [Tier3・厳密な held-out 検証ではない。EMA更新率は _EMA_ALPHA 参照]。

【発話イベントCSV】いつ(step/sim_sec)・何を(text)・どちらを見ていた時に(target)。
  実験ファイルでの書き方:
    "plugins": {"word_learning": {"events_out": "E/logs/.../発話イベント.csv"}}
  events_out を省略した場合はCSVを書かない（メモリ上の集計・metricsだけ返す）。
"""
import csv
import math
import os

from run.plugins.base import Plugin

# [Tier3・恣意的] 特徴EMAの更新率。値そのものに強い根拠はなく、
#   「lexicon.assoc()の累積平均とは独立に動く目安を作る」ことが本質。
_EMA_ALPHA = 0.2


def _cos(a, b):
    """コサイン類似度。どちらかがゼロベクトルならNone。"""
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da < 1e-12 or db < 1e-12:
        return None
    return num / (da * db)


class WordLearning(Plugin):
    name = "word_learning"

    def setup(self, ctx):
        """ctx を受け取り、対象物ごとの発話回数・視覚特徴のEMA・記録行のリストを初期化する。戻り値は無い。"""
        self.events_out = self.config.get("events_out")
        self.label_count = {"toy1": 0, "toy2": 0}
        self.feat_ema = {"toy1": None, "toy2": None}
        self.rows = []          # 発話イベントCSV用（1発話=1行）

    def on_step(self, ctx):
        """ctx を受け取り、親の発話イベントから対象物ごとの発話回数と記録行を積み、対象物ごとの視覚特徴のEMAを更新する。戻り値は無い。"""
        events = getattr(ctx, "last_parent_utterance", None) or []
        for ev in events:
            target = ev.get("target")
            if target not in self.label_count:
                # 【2026-09-01修理・F2-33①】従来ここで toy1/toy2 以外を捨てていた
                #   ため、10択の世界では8枠ぶんの名づけがイベントCSVにも件数にも
                #   残らなかった（提示バランスの検証が不可能だった）。全枠を数える。
                self.label_count[target] = 0
                self.feat_ema[target] = None
            self.label_count[target] += 1
            # 【親の言い直し・2026-09-03】cause/correctがあれば記録。無い行は空欄。
            self.rows.append({"step": ctx.step, "sim_sec": round(ctx.sim_sec, 3),
                              "text": ev.get("text"), "target": target,
                              "cause": ev.get("cause") or "", "correct": ev.get("correct")
                              if ev.get("correct") is not None else ""})
            state = ev.get("state")
            if not state:
                continue
            prev = self.feat_ema[target]
            self.feat_ema[target] = (list(state) if prev is None else
                                     [_EMA_ALPHA * s + (1.0 - _EMA_ALPHA) * p
                                      for s, p in zip(state, prev)])

    def metrics(self, ctx):
        """ctx を受け取り、対象物toy1/toy2それぞれの発話回数と assoc_sep（現状は常にNone）を辞書で返す。"""
        out = {"word_learning.label_count_toy1": self.label_count["toy1"],
               "word_learning.label_count_toy2": self.label_count["toy2"],
               "word_learning.assoc_sep": self._assoc_sep(ctx)}
        return out

    def _assoc_sep(self, ctx):
        taro = getattr(ctx, "taro", None)
        lexicon = None   # 【2026-09-15】意味の表は削除（assoc_sep は常に None）
        hearing = getattr(taro, "hearing", None) if taro is not None else None
        if lexicon is None or hearing is None or not self.rows:
            return None
        # 語(text)→物(target)の対応。ParentLabelingは1つの物に常に同じ語しか
        #   割り当てないので、記録済みイベントから引けば十分（設定を別途持たない）。
        word_for = {}
        for r in self.rows:
            word_for.setdefault(r["target"], r["text"])
        targets = [t for t in ("toy1", "toy2") if t in word_for]
        seps = []
        for t in targets:
            others = [o for o in targets if o != t]
            if not others or self.feat_ema[t] is None or self.feat_ema[others[0]] is None:
                continue
            tokens = hearing.hear(word_for[t])
            # 【2026-09-15】意味の表（lexicon.assoc）を削除したため、
            #   assoc_sep（正解物と不正解物の見えの分離度）は計算できない。
            #   常に None を返す。
            return None
        return (sum(seps) / len(seps)) if seps else None

    def report(self, ctx):
        """ctx を受け取り、events_out指定時は発話イベントをCSVに書き出し、発話イベント数・対象物ごとの発話回数・assoc_sepをまとめた辞書を返す。"""
        if self.rows and self.events_out:
            path = (self.events_out if os.path.isabs(self.events_out) else
                    os.path.join(os.path.abspath(os.path.join(
                        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)),
                        self.events_out))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                # 【親の言い直し・2026-09-03】cause/correct列を追加（従来行は空欄）。
                w.writerow(["step", "sim_sec", "text", "target", "cause", "correct"])
                for r in self.rows:
                    w.writerow([r["step"], r["sim_sec"], r["text"], r["target"],
                               r.get("cause", ""), r.get("correct", "")])
        return {"発話イベント数": len(self.rows),
                "label_count_toy1": self.label_count["toy1"],
                "label_count_toy2": self.label_count["toy2"],
                "assoc_sep": self._assoc_sep(ctx)}
