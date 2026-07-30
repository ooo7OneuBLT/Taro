"""物体（おもちゃ）に手が触れたかを測る。

判定の実体は `E/scripts/e_toy_touch.py` にある（2026-07-30 に一本化したもの）。
ここはそれをプラグインの形に**包むだけ**。
⚠️判定を2箇所に書かない。中身を直すときは e_toy_touch.py を直す。

実験ファイルでの書き方:
    "plugins": {"toy_touch": true}
"""
import os
import sys

from run.plugins.base import Plugin

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
_E_SCRIPTS = os.path.join(_ROOT, "E", "scripts")
if _E_SCRIPTS not in sys.path:
    sys.path.insert(0, _E_SCRIPTS)


class ToyTouch(Plugin):
    name = "toy_touch"

    def setup(self, ctx):
        from e_toy_touch import ToyTouchProbe
        self.probe = ToyTouchProbe(ctx.model, ctx.data)
        if not self.probe.ok:
            print("[toy_touch] おもちゃが無い環境＝接触は測らない", flush=True)
            self.probe = None

    def on_step(self, ctx):
        if self.probe is not None:
            self.probe.update(ctx.model, ctx.data)

    def on_body_change(self, ctx):
        # ★体を作り直したら geom の id を引き直す（溜めた回数は保つ）
        if self.probe is not None:
            self.probe.rebind(ctx.model)

    def line(self, ctx):
        return self.probe.line(ctx.dt) if self.probe is not None else None

    def report(self, ctx):
        if self.probe is None:
            return {"おもちゃ": "無し"}
        s = self.probe.summary(ctx.dt)
        return {"触れた回数": s["touches"],
                "1分あたり": round(s["touch_per_min"], 3),
                "最接近cm": {k: round(v, 2) for k, v in s["min_cm"].items()},
                "sim秒": round(s["sim_sec"], 1)}
