"""測る道具：輻輳反射がいま何度を目標にし、実際に両目が何度ずれているかを記録する。

【なぜ要るか、2026-08-28・F2-15】輻輳反射をONにしても、両目の中心窩に写るものが
揃わなかった（図＝F/logs/F2-15_輻輳ON/図_輻輳のONとOFF.png）。原因の候補が3つあり
（①視差の測定が失敗 ②符号が逆で開散している ③効きが弱い）、外からは区別できない。
輻輳角の目標値・実測値・測った視差を並べて記録し、どれなのかを切り分ける。

【役割】読むだけ（run/plugins/base.py の規約）。太郎も環境も変えない。
  乱数を消費しない。読むのは env.unwrapped._vergence（既に生成済みのもの）と
  data.qpos（左右の眼球関節の角度）。
"""
import csv
import os

import numpy as np

from run.plugins.base import Plugin


def _abs_path(p):
    if os.path.isabs(p):
        return p
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    return os.path.join(root, p)


class VergenceProbe(Plugin):
    name = "vergence_probe"

    def setup(self, ctx):
        self.out = self.config.get("out")
        self.rows = []

    def on_step(self, ctx):
        u = ctx.env.unwrapped
        v = getattr(u, "_vergence", None)
        d = getattr(u, "data", None)
        if d is None:
            return
        row = {"sim_sec": round(ctx.sim_sec, 2)}
        # 実際の左右の眼球角度（関節角）。輻輳反射が無くても測れる
        for side in ("left", "right"):
            try:
                j = u.model.joint("robot:%s_eye_horizontal" % side)
                row["%s_deg" % side] = round(float(np.degrees(d.qpos[j.qposadr[0]])), 3)
            except Exception:
                row["%s_deg" % side] = ""
        if row.get("left_deg") != "" and row.get("right_deg") != "":
            # 【2026-08-28】左右の眼球関節は回転軸の符号が逆（MIMo_modelv2.xml:334/344）。
            #   そのため世界座標では
            #     共同運動（両目が同じ方向）＝ 関節角の【差】
            #     輻輳・開散（両目が逆方向）＝ 関節角の【和】
            #   になる。両方を記録して取り違えを防ぐ。
            row["diff_deg"] = round(row["left_deg"] - row["right_deg"], 3)
            row["sum_deg"] = round(row["left_deg"] + row["right_deg"], 3)
        if v is not None:
            row["target_deg"] = round(float(getattr(v, "vergence_deg", 0.0)), 3)
            for attr in ("last_disparity_deg", "_last_disparity_deg",
                         "last_err_deg", "_last_err_deg"):
                if hasattr(v, attr):
                    row["disparity_deg"] = round(float(getattr(v, attr)), 3)
                    break
            # 【2026-08-28】測定の信頼度も記録する（測れなかった割合を見るため）
            if hasattr(v, "last_score"):
                row["corr"] = round(float(v.last_score), 3)
            if hasattr(v, "last_overlap_px"):
                row["overlap_px"] = int(v.last_overlap_px)
            if hasattr(v, "last_measurable"):
                row["measurable"] = int(bool(v.last_measurable))
        self.rows.append(row)

    def report(self, ctx):
        if not self.rows:
            return {"輻輳の記録": 0}
        keys = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        if self.out:
            p = _abs_path(self.out)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", newline="", encoding="utf-8-sig") as fp:
                w = csv.DictWriter(fp, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        out = {"輻輳の記録": len(self.rows)}
        for k in ("diff_deg", "sum_deg", "target_deg", "disparity_deg",
                  "corr", "measurable"):
            vals = [r[k] for r in self.rows if isinstance(r.get(k), (int, float))]
            if vals:
                out["%s_平均" % k] = round(sum(vals) / len(vals), 2)
                out["%s_最小" % k] = round(min(vals), 2)
                out["%s_最大" % k] = round(max(vals), 2)
        return out
