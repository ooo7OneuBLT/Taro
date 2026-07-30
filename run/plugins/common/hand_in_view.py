"""手が視界に入っている割合を測る。

判定の実体は `E/scripts/e_hand_in_view.py`（既に一本化されている）。
ここは包むだけ。

実験ファイルでの書き方:
    "plugins": {"hand_in_view": true}
"""
import os
import sys

from run.plugins.base import Plugin

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
for _p in (os.path.join(_ROOT, "E", "scripts"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class HandInView(Plugin):
    name = "hand_in_view"

    def setup(self, ctx):
        from e_hand_in_view import hand_in_view
        self._fn = hand_in_view
        self.hit = 0.0
        self.tot = 0

    def on_step(self, ctx):
        self.hit += float(self._fn(ctx.model, ctx.data))
        self.tot += 1

    def line(self, ctx):
        if not self.tot:
            return None
        return f"hand_in_view={100.0 * self.hit / self.tot:.1f}%"

    def report(self, ctx):
        if not self.tot:
            return None
        return {"手が視界に入っていた割合%": round(100.0 * self.hit / self.tot, 2),
                "測ったステップ数": self.tot}
