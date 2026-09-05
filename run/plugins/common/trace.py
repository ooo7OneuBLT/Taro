"""毎ステップの内部の値の「指紋」を残す。同じシードで結果がばらつく原因を追うため。

【なぜ要るか、2026-07-30】同じ設定・同じ乱数の種で学習を2回回すと結果が違う
（落とし穴チェックリスト 項79）。「結果が違う」しか分かっていないので、
**どのステップで、何が最初に食い違うか**を記録して原因の場所を絞る。

指紋が食い違った量から、原因の性質が決まる：
    obs_in の interoception  → 内臓（泣く・寝る・うとうと）
    obs_in の eye_left/right → 視覚のレンダリング
    obs_in の observation    → 物理（関節の状態）
    sv                       → 感覚をまとめる層（fusion）
    z                        → 脳の内部（予測符号化の乱数など）
    mean / std               → 行動を作るところ（運動野＋小脳）
    a                        → 探索のゆらぎ
    W                        → 学習の計算（浮動小数の順序）だけが違う

注意：このプラグインは ctx を**読むだけ**。太郎も環境も変えない。
  読むのは `ctx.last`（`run/trainer.py` が毎ステップ置いている参照）。

実験ファイルでの書き方:
    "plugins": {"trace": {"out": "E/logs/trace/trace_1.csv"}}
使い方（2回実行して比べる）:
    run/tools/check_divergence.py が呼ぶ
"""
import hashlib
import os

import numpy as np

from run.plugins.base import Plugin

# 指紋を取る対象。obs は中身のキーごとに分けて取る（どの感覚が違うか知りたいので）
OBS_KEYS = ("observation", "interoception", "vestibular", "touch",
            "eye_left", "eye_right")
VEC_KEYS = ("sv", "cf", "clp", "z", "mean", "std", "a", "pred", "nlp")


def _fp(v):
    """配列・テンソルを短い文字列にする。中身から作る（実行ごとに変わらない）。

    注意：Python の組み込み `hash()` は使わない。文字列のハッシュは起動ごとに変わるので、
      中身が同じでも指紋が違い「非決定的だ」と誤診する（2026-07-30 に実際に踏んだ）。
    """
    if v is None:
        return ""
    a = v.detach().numpy() if hasattr(v, "detach") else np.asarray(v)
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()
                        ).hexdigest()[:10]


class Trace(Plugin):
    name = "trace"

    def setup(self, ctx):
        self.out = self.config.get("out")
        self.rows = []
        self.keys = None
        # 指紋だけでは「どこが違うか」まで分からないときの生保存（2026-09-01追加）。
        #   dump_dir を指定すると、dump_keys の生配列を dump_until ステップまで npy で残す
        self.dump_dir = self.config.get("dump_dir")
        self.dump_keys = self.config.get("dump_keys", [])
        self.dump_until = int(self.config.get("dump_until", 0))
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

    def on_step(self, ctx):
        last = getattr(ctx, "last", None)
        if last is None:
            return          # 脳を通さない実行（measure）では何も置かれない
        row = {"step": ctx.step}
        for nm in ("obs_in", "obs_out"):
            o = last.get(nm)
            if not isinstance(o, dict):
                continue
            for k in OBS_KEYS:
                if k in o:
                    row[f"{nm}.{k}"] = _fp(o[k])
        for k in VEC_KEYS:
            row[k] = _fp(last.get(k))
        if self.dump_dir and ctx.step <= self.dump_until:
            # 全身＋全物体＋目の生の物理状態。指紋のobsは体の一部しか覆っていない
            env = getattr(ctx, "env", None)
            if env is not None:
                dd = env.unwrapped.data
                np.save(os.path.join(self.dump_dir, "step%04d_qpos.npy" % ctx.step),
                        np.array(dd.qpos, dtype=np.float64))
                np.save(os.path.join(self.dump_dir, "step%04d_ctrl.npy" % ctx.step),
                        np.array(dd.ctrl, dtype=np.float64))
                mm = env.unwrapped.model
                for nm, arr in (("xpos", dd.xpos), ("cam_xpos", dd.cam_xpos),
                                ("cam_xmat", dd.cam_xmat), ("mocap_pos", dd.mocap_pos),
                                ("geom_rgba", mm.geom_rgba), ("qvel", dd.qvel),
                                ("act", dd.act)):
                    np.save(os.path.join(self.dump_dir,
                                         "step%04d_%s.npy" % (ctx.step, nm)),
                            np.array(arr, dtype=np.float64))
            for full in self.dump_keys:
                nm, _, k = full.partition(".")
                o = last.get(nm)
                v = o.get(k) if isinstance(o, dict) else None
                if v is not None:
                    a = v.detach().numpy() if hasattr(v, "detach") else np.asarray(v)
                    np.save(os.path.join(self.dump_dir,
                                         "step%04d_%s.npy" % (ctx.step, full)), a)
        # 脳の重みは全部の合計で見る（毎ステップ全パラメータのハッシュを取るのは重い）。
        #   合計が一致していても中身が違う可能性は残るが、**違えば確実に違う**ので
        #   「どこで分岐したか」を絞る用途には足りる。
        taro = getattr(ctx, "taro", None)
        if taro is not None:
            row["W"] = f"{sum(float(p.sum()) for p in taro.brain.parameters()):.12e}"
            row["Wf"] = f"{sum(float(p.sum()) for p in taro.fusion.parameters()):.12e}"
        self.rows.append(row)

    def report(self, ctx):
        if not self.rows or not self.out:
            return None
        path = self.out if os.path.isabs(self.out) else os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir, os.pardir)), self.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cols = list(self.rows[0])
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(cols)
            for r in self.rows:
                w.writerow([r.get(c, "") for c in cols])
        return {"記録したステップ数": len(self.rows), "出力": self.out}
