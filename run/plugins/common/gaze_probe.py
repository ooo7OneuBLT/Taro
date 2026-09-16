"""測る道具：太郎の視線が的（おもちゃ）にどれだけ・どう留まるかを記録する。

【なぜ要るか、2026-08-26・F2-9】親の名付けの「注視が◯秒続いたら言う」判定が、
学習中の太郎では hold=1.0秒ですら一度も成立しなかった（実測：300秒で発話0回。
hold=0.3秒なら9回）。静止した太郎では hold=3.0秒でも成立していた（発話5回/30秒）
＝学習中の体・首の動きで視線が暴れているのが原因と推測されるが、実際に
「的への視線角度が時間とともにどう動くか」「連続で判定内に留まる区間は
何秒くらいか」を測った記録が無い。判定の設計（連続か累積か・角度の閾値）を
数字で決めるための測定器。

【注意・2026-09-15】以前ここには「似た道具として `gaze_command.py`（2026-09-11・K0）がある」と
書いてあったが、**その名前のファイルはリポジトリのどこにも無い**（作られなかったか、
別の名前になったか）。このファイルは読むだけの測定器である、という趣旨は変わらない。

【役割】読むだけ（run/plugins/base.py の規約）。太郎も環境も変えない。
  乱数は一切消費しない。読むのは **ctx だけ**（2026-09-16に変えた）：
    ctx.視線角   … スロットごとの視線のずれ[度]。**場面のスロット数に自動で合う**
    ctx.視線の先 … 遮られていないものの中で、いちばん視線に近い物
    ctx.親の状態 / ctx.親の的
  以前は `ctx.env.unwrapped._gaze_angle_to("test_object1"/"test_object2")` を
  直接呼び、**2択を決め打ち**していた。目標Fの8択の場面では存在しない2つを
  見ていたことになる。事実の作り手は run/context.py の1箇所だけにした。

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
        """config から angles_out（角度ログの出力先）を読み、行を貯める self.rows、連続注視区間の長さを貯める self.runs、現在の区間長 self._cur_run を初期化する。戻り値は無い。
        """
        self.angles_out = self.config.get("angles_out")
        self.rows = []
        # 的（親が振っている方）への連続注視の区間長[判断数]を集める
        self.runs = []          # 終わった区間の長さ
        self._cur_run = 0

    def on_step(self, ctx):
        """毎ステップ呼ばれる。環境から test_object1・test_object2 への視線角度と親のラベリング状態・対象を読み、対象への角度（無ければ小さい方）を求めて1行を self.rows に積む。親の状態が「shake」かつ角度がGAZE_DEG以下なら連続注視区間を延ばし、そうでなければ区間を締めて self.runs に記録する。戻り値は無い。
        """
        # 【2026-09-16】環境を直接触るのをやめ、ctx の「導いた事実」を読む。
        #   【何が壊れていたか】ここは test_object1/2 を決め打ちしていた。
        #   目標Fの場面は8択（test_object5〜11）なので、**存在しない2つを見て
        #   いた**＝角度も「的への角度」も意味の無い値になっていた。
        #   同じ決め打ちが word_production にもあり、そちらは 2026-09-15 に
        #   直したが、ここは残っていた。事実を1箇所（run/context.py）にまとめて
        #   両方がそこを読む形にすることで、片方だけ直る状態をなくす。
        視線角 = ctx.視線角                 # {"toy8": 12.3, ...} スロットの数は場面で決まる
        state = ctx.親の状態
        target = ctx.親の的
        # 的への角度（親が誰も選んでいなければ、いちばん近い物への角度）
        at = 視線角.get(target)
        if at is None and 視線角:
            at = min(視線角.values())
        self.rows.append({"sim_sec": round(ctx.sim_sec, 2),
                          "親の状態": state, "親の的": target,
                          "視線角_親の的": "" if at is None else round(at, 2),
                          "視線の先": ctx.視線の先,     # 遮蔽を見た値（光線）
                          "視線角": dict(視線角)})
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
        """終わっていない連続区間があれば締めて self.runs に加える。angles_out が設定されていれば self.rows をCSVに書き出す。連続注視区間の本数と、区間長の中央値・最大・1秒以上/3秒以上の区間数をまとめた辞書を返す。
        """
        if self._cur_run > 0:
            self.runs.append(self._cur_run)
        if self.angles_out:
            p = _abs_path(self.angles_out)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", newline="", encoding="utf-8-sig") as fp:
                w = csv.writer(fp)
                # 【2026-09-16】列は場面で決まる（2択なら2列、8択なら8列）。
                #   決め打ちの angle_toy1_deg / angle_toy2_deg をやめた。
                スロット = sorted({k for r in self.rows for k in r["視線角"]})
                w.writerow(["sim_sec", "親の状態", "親の的", "視線角_親の的", "視線の先"]
                           + ["視線角_%s" % s for s in スロット])
                for r in self.rows:
                    w.writerow([r["sim_sec"], r["親の状態"], r["親の的"],
                                r["視線角_親の的"], r["視線の先"]]
                               + [r["視線角"].get(s, "") for s in スロット])
        out = {"連続注視の区間数": len(self.runs)}
        if self.runs:
            rs = sorted(self.runs)
            sec = 0.1   # 1判断=K tick=0.1秒
            out["区間長中央値_sec"] = round(rs[len(rs) // 2] * sec, 2)
            out["区間長最大_sec"] = round(rs[-1] * sec, 2)
            out["1秒以上の区間数"] = sum(1 for r in rs if r * sec >= 1.0)
            out["3秒以上の区間数"] = sum(1 for r in rs if r * sec >= 3.0)
        return out
