# -*- coding: utf-8 -*-
"""測る道具：眼球への命令が、どの段階で・どれだけ入ったかを1歩ごとに記録する。

【なぜ要るか・2026-09-16】`eye_motion` で眼球が**100%の歩で動いている**
（中央値117度/秒・可動域いっぱい）と分かったが、視線誘導反射のサッケードは
**0発**だった。＝眼球を振り回しているのは反射ではない。では何か。
`run/world/toy_env.py` の `step()` で、方策が出した `action` は

    方策 → VOR（前庭動眼反射）→ 視線誘導反射 → 輻輳反射 → 物理

の順に上書きされていく。**各段階の値を並べれば、どこで大きな命令が
入ったかが分かる**。反射は「上書き」ではなく「加算」する作りなので、
段と段の差がそのまま各反射の寄与になる。

【この道具がやらないこと】「犯人はこれ」と判定しない（`base_取説.md`
「道具は列を出す。数えるのはあと」）。段ごとの値と差を列にするだけ。

【読むだけ】`env._eye_cmd`（toy_env が控えた辞書）を読むだけ。

【実験ファイルでの書き方】
    "plugins": {"eye_command": {"out": "F/logs/<実験>/眼球への命令.csv"}}

【出る列】各段（方策 / VORのあと / 定位反射のあと / 輻輳のあと）×（左目_水平 / 左目_垂直）と、
    隣り合う段の差（VORの寄与 / 定位反射の寄与 / 輻輳の寄与）、
    最後に実際にかかったトルク（実トルク_）。
    筋モデルでは1関節に筋が2本あるので、`_縮` `_伸` の生の値も出す。
    見出しに何も付かない列は**正味＝伸−縮**（-1〜1）。
"""
import csv
import os

from run.plugins.base import Plugin

段 = ["方策", "VORのあと", "定位反射のあと", "輻輳のあと"]
軸 = ["左目_水平", "左目_垂直"]
寄与 = [("VORの寄与", "方策", "VORのあと"),
        ("定位反射の寄与", "VORのあと", "定位反射のあと"),
        ("輻輳の寄与", "定位反射のあと", "輻輳のあと")]

# 【筋モデル・2026-09-16】`actuation="muscle"` では1関節に筋が2本（縮む側・伸びる側）
#   あり、命令は両方に出る。`正味` は 伸−縮。`data.ctrl` は常に1のダミーなので
#   見ても意味がなく（MIMo/mimoActuation/muscle.py の _apply_torque 参照）、
#   代わりに実際にかかったトルク `model.actuator_gear` を `実トルク_` として出す。
列 = (["step", "sim_sec"]
      + ["%s_%s" % (s, a) for s in 段 for a in 軸]
      + ["%s_%s_縮" % (s, a) for s in 段 for a in 軸]
      + ["%s_%s_伸" % (s, a) for s in 段 for a in 軸]
      + ["%s_%s" % (n, a) for n, _x, _y in 寄与 for a in 軸]
      + ["実トルク_%s" % a for a in 軸]
      # VOR が「頭がこれだけ回っている」と思っている値（三半規管を通したあと）
      + ["頭の角速度_rad毎秒", "頭の角速度x", "頭の角速度y", "頭の角速度z",
         "VORが見た頭の回転_左目水平", "VORの目標速度_左目水平"])


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)), p)


class EyeCommand(Plugin):
    """眼球への命令を段階ごとに記録する（読むだけ）。"""

    name = "eye_command"
    intervenes = None

    def setup(self, ctx):
        """出力先を読み、行を貯める入れ物を用意する。戻り値は無い。"""
        self.out = self.config.get("out")
        self.rows = []

    def on_step(self, ctx):
        """毎ステップ。toy_env が控えた段階ごとの命令を1行にして積む。戻り値は無い。"""
        u = getattr(ctx.env, "unwrapped", ctx.env)
        c = getattr(u, "_eye_cmd", None) or {}
        行 = {"step": ctx.step, "sim_sec": round(ctx.sim_sec, 2)}
        for k in 列[2:]:
            v = c.get(k)
            if v is not None:
                行[k] = round(v, 5)
        for s in 段:
            for a in 軸:
                v = c.get("%s_%s" % (s, a))
                行["%s_%s" % (s, a)] = "" if v is None else round(v, 5)
        # VOR が見ている頭の回転（読むだけ・2026-09-16）
        v = getattr(u, "_vor", None)
        w = getattr(v, "last_w_world", None) if v is not None else None
        if w is not None:
            行["頭の角速度x"], 行["頭の角速度y"], 行["頭の角速度z"] = (round(float(x), 5) for x in w)
            行["頭の角速度_rad毎秒"] = round(float(sum(float(x) ** 2 for x in w) ** 0.5), 5)
        lu = getattr(v, "last_unit", None) if v is not None else None
        if lu:
            for 名2, t in lu.items():
                if 名2.endswith("left_eye_horizontal"):
                    行["VORが見た頭の回転_左目水平"] = round(float(t[0]), 5)
                    行["VORの目標速度_左目水平"] = round(float(t[1]), 5)
        for 名, 前, 後 in 寄与:
            for a in 軸:
                x = c.get("%s_%s" % (前, a))
                y = c.get("%s_%s" % (後, a))
                行["%s_%s" % (名, a)] = ("" if x is None or y is None
                                        else round(y - x, 5))
        self.rows.append(行)

    def report(self, ctx):
        """out があれば列をCSVへ書き、生の事実だけを返す（判定はしない）。"""
        if self.rows and self.out:
            p = _abs_path(self.out)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(列)
                for r in self.rows:
                    w.writerow([r.get(c, "") for c in 列])
        out = {"記録した歩数": len(self.rows)}
        # 各段・各寄与について「0でなかった歩」の数だけ返す（平均や順位は出さない）
        for k in 列[2:]:
            n = sum(1 for r in self.rows
                    if r.get(k) != "" and abs(float(r.get(k) or 0.0)) > 1e-6)
            if n:
                out["0でない歩_" + k] = n
        if self.out:
            out["出力"] = self.out
        return out
