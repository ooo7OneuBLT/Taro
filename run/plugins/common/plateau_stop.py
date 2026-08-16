"""小脳の予測誤差(cereb_err)が頭打ちに達したら、学習を打ち切る道具。

【なぜ、2026-08-15】太郎の内発的動機は「予測が上手くなった対象は学ぶ余地が
減って飽きる」設計。次のリーチング実験に進む前に「自分の体を学びきったか」を
確かめたい。既存ログ（13本、いずれも18000ステップ）では、小脳の予測誤差
（`t.cereb.err_ema` の移動平均、`run/trainer.py:587` が記録している値）が
最後まで下がり続けており、頭打ちに達していない。

「頭打ちに達した瞬間」を捉えるには、頭打ちになるまで回し続ける必要がある。
このプラグインは一定間隔（`run.checkpoint`）ごとに cereb_err を見て、

  1. 直近K件の平均が、その前のK件の平均から十分に改善していなければ
     「頭打ち（plateau）」と判定する
  2. 上限ステップ数(max_steps)に達したら、頭打ちに未到達でも「上限到達
     （max_steps）」として打ち切る

のどちらかが起きたら `ctx.stop_requested = True` を立てる。実際に学習を
止めるのは `run/trainer.py` 側（このプラグインは ctx を読み書きする
唯一の例外＝停止シグナルの土台。それ以外の値は測るだけで太郎を変えない）。

置き場所の判定（落とし穴チェックリスト項28）：これは「外から測って、
学習の続行可否を判断する道具」であり太郎の中身ではない。よって
taro_coreではなくrun/plugins/commonに置く。

実験ファイルでの書き方：

    "plugins": {
      "plateau_stop": {
        "window": 5,
        "min_improve_frac": 0.02,
        "min_checkpoints": 10,
        "max_steps": 60000,
        "out_path": "E/logs/xxx/plateau_stop.json"
      }
    }

パラメータの数値そのもの（何%を「改善が止まった」とするか等）はまだ誰も
実測しておらず未確定 [Tier3・未較正、実験ファイルから必ず指定する想定]。
`max_steps` だけはデフォルトを置かず必須（未指定なら setup() で例外）。
"""

import json
import os

from run.plugins.base import Plugin


class PlateauStop(Plugin):
    """cereb_err が頭打ちになったら、または上限ステップに達したら学習を止める。"""

    name = "plateau_stop"

    def setup(self, ctx):
        cfg = self.config
        # window: 直近K回・その前のK回を比べる、その「K」
        #   [Tier3・未較正]。デフォルト5は「何かを置かないとゼロ除算等で
        #   落ちてしまう」ための工学的な仮置きで、根拠となる実測値は無い。
        self.window = int(cfg.get("window", 5))
        if self.window < 1:
            raise ValueError(f"plateau_stop.window は1以上である必要がある: {self.window}")
        # min_improve_frac: 「改善が止まった」とみなす改善率の閾値
        #   [Tier3・未較正]。0.02=2%未満の改善なら頭打ちとみなす、という仮置き。
        self.min_improve_frac = float(cfg.get("min_improve_frac", 0.02))
        # min_checkpoints: 判定を始めるまでの最低チェックポイント数
        #   （序盤の不安定な時期の誤判定防止）。2*window以上を推奨。
        self.min_checkpoints = int(cfg.get("min_checkpoints", 2 * self.window))
        if self.min_checkpoints < 2 * self.window:
            print(f"警告[plateau_stop] min_checkpoints={self.min_checkpoints} は "
                  f"2*window={2 * self.window} 未満。直近K件と前のK件が両方揃う前に "
                  "判定が始まる可能性がある（仕様の推奨に反する）。", flush=True)
        # max_steps: ★必須。実験ファイルで未指定なら例外で止める
        #   （固定のデフォルト値は埋め込まない。仕様2節の明記どおり）。
        if "max_steps" not in cfg:
            raise ValueError(
                "plugins.plateau_stop.max_steps が未指定。上限ステップ数は必ず"
                "実験ファイルから明示的に指定すること（固定のデフォルト値は置かない）。")
        self.max_steps = int(cfg["max_steps"])
        if self.max_steps <= 0:
            raise ValueError(f"plateau_stop.max_steps は正の値である必要がある: {self.max_steps}")
        self.out_path = cfg.get("out_path")

        # cereb_err の履歴。(step, 値) のタプルのリスト。
        self.history = []
        # cerebellum=False のときは判定できない旨を1回だけ警告する（仕様1節）。
        self._cerebellum_warned = False
        # 打ち切りが発火したかどうか・その詳細（report() で使う）。
        self._stopped = False
        self._stop_info = None

    def on_checkpoint(self, ctx):
        # すでに打ち切りを決めた後（防御的：trainer側は break するはずだが、
        #   万一この後もう一度呼ばれても二重発火・二重printしない）。
        if self._stopped:
            return

        taro = ctx.taro
        if not taro.cfg.cerebellum:
            # cerebellum=False なら小脳が学習されておらず cereb_err が動かない
            #   ので頭打ち判定はできない。以後は上限ステップ数の判定のみ行う。
            if not self._cerebellum_warned:
                print("警告[plateau_stop] taro.cerebellum=False のため cereb_err が"
                      "存在せず頭打ち判定はできない。以後は上限ステップ数(max_steps)"
                      "の判定のみ行う。", flush=True)
                self._cerebellum_warned = True
            if ctx.step >= self.max_steps:
                self._trigger(ctx, "max_steps", {"cereb_err": None})
            return

        val = float(taro.cereb.err_ema.item())
        self.history.append((ctx.step, val))

        if len(self.history) >= self.min_checkpoints and len(self.history) >= 2 * self.window:
            K = self.window
            recent_vals = [v for (_s, v) in self.history[-K:]]
            prev_vals = [v for (_s, v) in self.history[-2 * K:-K]]
            recent_avg = sum(recent_vals) / K
            prev_avg = sum(prev_vals) / K
            # 改善率 = (前の平均 - 直近の平均) / 前の平均。
            #   分母が0（=誤差が完全に0になった、実際にはほぼ起きない）なら
            #   ゼロ除算を避けて「改善なし(0.0)」扱いにする。
            improve_frac = ((prev_avg - recent_avg) / prev_avg) if prev_avg != 0 else 0.0
            if improve_frac < self.min_improve_frac:
                self._trigger(ctx, "plateau", {
                    "cereb_err": val,
                    "recent_avg": recent_avg,
                    "prev_avg": prev_avg,
                    "improve_frac": improve_frac,
                })
                return

        if ctx.step >= self.max_steps:
            self._trigger(ctx, "max_steps", {"cereb_err": val})

    def _trigger(self, ctx, reason, detail):
        """打ち切りを決める。理由は必ず plateau／max_steps を区別できる値にする。"""
        self._stopped = True
        info = {"reason": reason, "step": int(ctx.step), "detail": detail or {}}
        self._stop_info = info
        ctx.stop_requested = True
        ctx.stop_reason = info
        label = ("頭打ち（plateau）" if reason == "plateau"
                 else "上限到達（max_steps。頭打ちには未到達）")
        print(f"[plateau_stop] 学習を打ち切ります: {label}  "
              f"step={info['step']}  詳細={info['detail']}", flush=True)

    def metrics(self, ctx):
        # CSV(run.csv) にも「監視中か／打ち切りが発火したか」が分かる列を残す。
        row = {"plateau_stop_active": 0 if self._stopped else 1,
               "plateau_stop_reason": (self._stop_info["reason"] if self._stopped else "")}
        if self.history:
            row["plateau_stop_cereb_err"] = self.history[-1][1]
        return row

    def line(self, ctx):
        if self._stopped:
            return f"plateau_stop=STOPPED({self._stop_info['reason']})"
        if self.history:
            return f"plateau_stop=watching(cereb_err={self.history[-1][1]:.5f})"
        return "plateau_stop=watching"

    def report(self, ctx):
        out = {"打ち切りが発火したか": self._stopped}
        if self._stopped:
            out["理由"] = self._stop_info["reason"]
            out["理由表示"] = ("頭打ち（plateau）" if self._stop_info["reason"] == "plateau"
                             else "上限到達（max_steps。頭打ちには未到達）")
            out["何ステップ目か"] = self._stop_info["step"]
            out["詳細"] = self._stop_info["detail"]
        else:
            # run.steps の方が先に尽きて、プラグインが一度も打ち切りを判定
            #   しなかった場合（仕様「report(ctx)で最後に必ず分かるようにする」節）。
            out["備考"] = "plugin未発火のまま run.steps 到達"
        out["監視した履歴の件数"] = len(self.history)
        if not self.out_path:
            return out
        # 【なぜこの解決の仕方か】block_progress_probe.pyのout_path解決と同じ
        #   ロジック（相対パスならリポジトリルート基準に直す）。
        path = self.out_path if os.path.isabs(self.out_path) else os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir, os.pardir)), self.out_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(out)
        payload["history"] = [{"step": s, "cereb_err": v} for (s, v) in self.history]
        payload["params"] = {"window": self.window, "min_improve_frac": self.min_improve_frac,
                             "min_checkpoints": self.min_checkpoints, "max_steps": self.max_steps}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        out["出力パス"] = path
        return out
