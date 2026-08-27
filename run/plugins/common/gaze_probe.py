"""測る道具：太郎の視線が的（おもちゃ）にどれだけ・どう留まるかを記録する。

【なぜ要るか、2026-08-26・F2-9】親の名付けの「注視が◯秒続いたら言う」判定が、
学習中の太郎では hold=1.0秒ですら一度も成立しなかった（実測：300秒で発話0回。
hold=0.3秒なら9回）。静止した太郎では hold=3.0秒でも成立していた（発話5回/30秒）
＝学習中の体・首の動きで視線が暴れているのが原因と推測されるが、実際に
「的への視線角度が時間とともにどう動くか」「連続で判定内に留まる区間は
何秒くらいか」を測った記録が無い。判定の設計（連続か累積か・角度の閾値）を
数字で決めるための測定器。

【役割】読むだけ（run/plugins/base.py の規約）。太郎も環境も変えない。
  乱数は一切消費しない。読むのは：
    ctx.env.unwrapped._gaze_angle_to("test_object1"/"test_object2")
        … 視線と各おもちゃの間の角度[度]（e_toy_env.py。親の判定と同じ関数）
    ctx.env.unwrapped._parent_labeling … 親の状態（state/target。無ければ空欄）

【出す指標】
  ① 毎判断（0.1秒ごと）の視線角度 … angles_out（1行=1判断）
  ② 連続注視区間の分布 … report に要約（10度以内に連続で留まった区間長の
     中央値・最大・本数）。区間の切れ目は「10度を超えた判断」。

  実験ファイルでの書き方:
    "plugins": {"gaze_probe": {"angles_out": "F/logs/.../視線角度.csv"}}
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


class GazeProbe(Plugin):
    name = "gaze_probe"

    GAZE_DEG = 10.0     # 親の判定と同じ閾値（parent_labeling.py の gaze_deg 既定）

    def setup(self, ctx):
        self.angles_out = self.config.get("angles_out")
        self.rows = []
        # 的（親が振っている方）への連続注視の区間長[判断数]を集める
        self.runs = []          # 終わった区間の長さ
        self._cur_run = 0

    def on_step(self, ctx):
        u = ctx.env.unwrapped
        a1 = u._gaze_angle_to("test_object1")
        a2 = u._gaze_angle_to("test_object2")
        pl = getattr(u, "_parent_labeling", None)
        state = getattr(pl, "_state", "") if pl is not None else ""
        target = getattr(pl, "_target", "") if pl is not None else ""
        # 的への角度（親が誰も選んでいなければ近い方）
        at = None
        if target == "toy1":
            at = a1
        elif target == "toy2":
            at = a2
        elif a1 is not None and a2 is not None:
            at = min(a1, a2)
        self.rows.append((round(ctx.sim_sec, 2),
                          None if a1 is None else round(a1, 2),
                          None if a2 is None else round(a2, 2),
                          state, target))
        # 連続区間の集計（親が振っている間だけ数える＝判定と同じ土俵）
        if state == "shake" and at is not None:
            if at <= self.GAZE_DEG:
                self._cur_run += 1
            else:
                if self._cur_run > 0:
                    self.runs.append(self._cur_run)
                self._cur_run = 0
        else:
            if self._cur_run > 0:
                self.runs.append(self._cur_run)
            self._cur_run = 0

    def report(self, ctx):
        if self._cur_run > 0:
            self.runs.append(self._cur_run)
        if self.angles_out:
            p = _abs_path(self.angles_out)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", newline="", encoding="utf-8-sig") as fp:
                w = csv.writer(fp)
                w.writerow(["sim_sec", "angle_toy1_deg", "angle_toy2_deg",
                            "parent_state", "parent_target"])
                w.writerows(self.rows)
        out = {"連続注視の区間数": len(self.runs)}
        if self.runs:
            rs = sorted(self.runs)
            sec = 0.1   # 1判断=K tick=0.1秒
            out["区間長中央値_sec"] = round(rs[len(rs) // 2] * sec, 2)
            out["区間長最大_sec"] = round(rs[-1] * sec, 2)
            out["1秒以上の区間数"] = sum(1 for r in rs if r * sec >= 1.0)
            out["3秒以上の区間数"] = sum(1 for r in rs if r * sec >= 3.0)
        return out
