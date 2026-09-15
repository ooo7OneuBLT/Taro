"""おもちゃ（test_object1）が視界に入っている割合を測る。

判定の幾何は `E/scripts/e_hand_in_view.py` の eye_angles を流用（一本化を保つ）。
片目でも視野内（角度 <= fovy/2）なら「見えている」とする（hand_in_view と同じ基準）。
脳を使わないので run.type=measure でも動く。

実験ファイルでの書き方:
    "plugins": {"toy_in_view": true}
"""
import os
import sys

import numpy as np

from run.plugins.base import Plugin

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
for _p in (os.path.join(_ROOT, "E", "scripts"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TOY_BODY = "test_object1"


class ToyInView(Plugin):
    name = "toy_in_view"

    def setup(self, ctx):
        """e_hand_in_view の eye_angles・VISION_FOVY_HALF を取り込み、通し・区間のヒット数とステップ数、視線とのずれ角度を貯めるリストを初期化する。戻り値は無い。
        """
        from e_hand_in_view import eye_angles, VISION_FOVY_HALF
        self._eye_angles = eye_angles
        self._half = VISION_FOVY_HALF
        self.hit = 0.0
        self.tot = 0
        self.angs = []          # 視線からのずれ（両目の小さい方）[度]
        self._seg_hit, self._seg_tot = 0.0, 0

    def on_step(self, ctx):
        """毎ステップ呼ばれる。おもちゃ(test_object1)の位置が取れなければ何もしない。取れれば両目からの視線角度を計算し、片目でも視野角の半分以内なら「視界に入っている」として通し・区間のカウンタとずれ角度のリストに加える。戻り値は無い。
        """
        try:
            p = np.array(ctx.data.body(_TOY_BODY).xpos, dtype=float)
        except Exception:
            return                      # おもちゃの無いシーンでは何もしない
        ang = self._eye_angles(ctx.model, ctx.data, p)
        v = 1.0 if any(a <= self._half for a in ang.values()) else 0.0
        self.hit += v
        self.tot += 1
        self.angs.append(min(ang.values()))
        self._seg_hit += v
        self._seg_tot += 1

    def metrics(self, ctx):
        """区間内にステップが無ければ None を返す。区間内でおもちゃが視界に入っていた割合(%)をまとめた辞書を返し、区間の集計をリセットする。"""
        if not getattr(self, "_seg_tot", 0):
            return None
        v = 100.0 * self._seg_hit / self._seg_tot
        self._seg_hit, self._seg_tot = 0.0, 0
        return {"toy_in_view": round(v, 3)}

    def line(self, ctx):
        """まだ測定していなければ None を返す。通しでの視界に入っていた割合(%)と視線とのずれの最小値を短い文字列にして返す。"""
        if not self.tot:
            return None
        return (f"toy_in_view={100.0 * self.hit / self.tot:.1f}%"
                f" ずれmin={min(self.angs):.1f}°")

    def report(self, ctx):
        """まだ測定していなければ None を返す。通しでの視界に入っていた割合(%)、視線とのずれの平均・最小、測ったステップ数をまとめた辞書を返す。"""
        if not self.tot:
            return None
        a = np.asarray(self.angs)
        return {"おもちゃが視界に入っていた割合%": round(100.0 * self.hit / self.tot, 2),
                "視線とのずれ度_平均": round(float(a.mean()), 1),
                "視線とのずれ度_最小": round(float(a.min()), 1),
                "測ったステップ数": self.tot}
