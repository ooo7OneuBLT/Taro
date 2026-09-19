# -*- coding: utf-8 -*-
"""測る道具：眼球と頭が実際に何度動いたかを1歩ごとに記録する。

【なぜ要るか・2026-09-16】`produce_gates` で「サッケードを1発も撃っていない」と
出たので「目が動いていない」と報告したところ、**ユーザーが動画を見て「目は
動いている」と指摘した**。撃った回数（`orienting.n_saccades`）は
**反射が命令を出した回数**であって、**眼球が動いたかどうかではない**。
眼球は反射のほかに、輻輳（`vergence.py`）・眼球を中央へ戻すバネ
（`apply_eye_centering_spring`）・頭や体の動きでも動く。
「動いた／動いていない」は**関節の角度そのもの**を見ないと言えない。

【この道具がやらないこと】速さの平均・「暴れている」の判定はしない
（`run/plugins/base_取説.md`「道具は列を出す。数えるのはあと」）。

【読むだけ】ctx.model / ctx.data を読むだけ。太郎にも環境にも触らない。
乱数を消費しない。

【実験ファイルでの書き方】
    "plugins": {"eye_motion": {"out": "F/logs/<実験>/眼球の動き.csv"}}

【出る列】
    step / sim_sec
    左目_水平度 / 左目_垂直度 / 右目_水平度 / 右目_垂直度
    左目_動いた度 / 左目_速さ度毎秒      ← 前の歩からの差
    頭_左右度 / 頭_上下度 / 頭_傾き度
    頭_動いた度
    サッケード残り秒 / 撃った回数        ← 反射の命令と突き合わせるため
"""
import csv
import math
import os

from run.plugins.base import Plugin

_眼 = [("左目_水平度", "robot:left_eye_horizontal"),
       ("左目_垂直度", "robot:left_eye_vertical"),
       ("右目_水平度", "robot:right_eye_horizontal"),
       ("右目_垂直度", "robot:right_eye_vertical")]
_頭 = [("頭_左右度", "robot:head_swivel"),
       ("頭_上下度", "robot:head_tilt"),
       ("頭_傾き度", "robot:head_tilt_side")]

列 = (["step", "sim_sec"] + [n for n, _ in _眼]
      + ["左目_動いた度", "左目_速さ度毎秒"]
      + [n for n, _ in _頭] + ["頭_動いた度",
                               # 【2026-09-16】関節の角度だけでは足りない。VOR は
                               #   頭の**世界での**角速度（data.cvel）を見て眼を動かす。
                               #   「実際に世界で何度回ったか」と「cvelが何度/秒と
                               #   言っているか」を並べる。食い違えば幻の回転。
                               "頭の世界回転_度毎歩", "頭のcvel_度毎秒", "体の根のqvel_度毎秒",
                               "最も速い関節", "その速さ_度毎秒", "眼以外で最も速い関節", "眼以外の速さ_度毎秒",
                               "頭の向きを決める自由度の速さ_度毎秒", "そのうち最速の関節",
                               "サッケード残り秒", "撃った回数"])


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)), p)


class EyeMotion(Plugin):
    """眼球・頭の関節角を度で記録する（読むだけ）。"""

    name = "eye_motion"
    intervenes = None

    def setup(self, ctx):
        """出力先を読み、関節の番号を引いておく。戻り値は無い。"""
        self.out = self.config.get("out")
        self.rows = []
        self._prev = None
        self._idx = {}
        m = getattr(ctx, "model", None)
        for 名, j in _眼 + _頭:
            try:
                self._idx[名] = m.joint(j).qposadr[0]
            except Exception:       # noqa: BLE001  関節が無い体でも走行は止めない
                self._idx[名] = None

    def _角度(self, ctx, 名):
        """その関節の角度[度]。引けなければ None。"""
        a = self._idx.get(名)
        if a is None:
            return None
        return math.degrees(float(ctx.data.qpos[a]))

    def on_step(self, ctx):
        """毎ステップ。眼球・頭の角度と、前の歩からの変化を1行にして積む。戻り値は無い。"""
        行 = {"step": ctx.step, "sim_sec": round(ctx.sim_sec, 2)}
        いま = {}
        for 名, _j in _眼 + _頭:
            v = self._角度(ctx, 名)
            いま[名] = v
            行[名] = "" if v is None else round(v, 4)
        # 左目の動き（水平・垂直をまとめた大きさ）と、頭の動き
        def 差(名たち):
            if self._prev is None:
                return None
            s = 0.0
            for 名 in 名たち:
                a, b = いま.get(名), self._prev.get(名)
                if a is None or b is None:
                    return None
                s += (a - b) ** 2
            return math.sqrt(s)
        d目 = 差(["左目_水平度", "左目_垂直度"])
        d頭 = 差(["頭_左右度", "頭_上下度", "頭_傾き度"])
        行["左目_動いた度"] = "" if d目 is None else round(d目, 4)
        行["左目_速さ度毎秒"] = ("" if d目 is None or not ctx.dt
                               else round(d目 / float(ctx.dt), 3))
        行["頭_動いた度"] = "" if d頭 is None else round(d頭, 4)
        self._prev = いま
        # 頭が世界で実際に何度回ったか（xmat の前向き軸の角度差）と、
        # VOR が読む cvel（角速度）を並べて出す。
        try:
            u2 = getattr(ctx.env, "unwrapped", ctx.env)
            bid = int(ctx.model.body("head").id)
            R = [float(x) for x in ctx.data.xmat[bid]]
            f = (R[0], R[3], R[6])          # 前向き軸（1列目）
            pf = getattr(self, "_prev_fwd", None)
            if pf is not None:
                dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(f, pf))))
                行["頭の世界回転_度毎歩"] = round(math.degrees(math.acos(dot)), 4)
            self._prev_fwd = f
            cv = ctx.data.cvel[bid][:3]
            行["頭のcvel_度毎秒"] = round(math.degrees(
                math.sqrt(sum(float(x) ** 2 for x in cv))), 3)
            # 【2026-09-16】体の根（freejoint）の角速度そのもの。_pin_root が毎歩
            #   これを0にしている。ここが0なのに cvel が大きければ、cvel は
            #   **留める前の値のまま更新されていない**＝VORは幻の回転を読んでいる。
            # 【2026-09-16】頭の cvel が幻なのか本物なのかの決め手。
            #   cvel は qvel から導かれるので、**どこかの関節が速く動いていれば本物**。
            #   眼球そのものは高速で動くので、眼を除いた最速も別に出す。
            m2 = ctx.model
            速 = []
            for jj in range(m2.njnt):
                nm = m2.joint(jj).name or ""
                adr = int(m2.jnt_dofadr[jj])
                nd = {0: 6, 1: 3, 2: 1, 3: 1}.get(int(m2.jnt_type[jj]), 1)
                w = math.sqrt(sum(float(ctx.data.qvel[adr + z]) ** 2
                                  for z in range(min(nd, 3))))
                速.append((w, nm))
            速.sort(reverse=True)
            if 速:
                行["その速さ_度毎秒"] = round(math.degrees(速[0][0]), 3)
                行["最も速い関節"] = 速[0][1]
                他 = [x for x in 速 if "eye" not in x[1]]
                if 他:
                    行["眼以外の速さ_度毎秒"] = round(math.degrees(他[0][0]), 3)
                    行["眼以外で最も速い関節"] = 他[0][1]
            # 【2026-09-16・決め手】頭の向きは「世界→頭」の連なりの関節だけで決まる
            #   （mimo_orientation・hip_lean/rot/bend 1と2・head_swivel/tilt/tilt_side）。
            #   ここが全部0なら cvel の36度/秒は**留める前の値のまま＝幻**。
            #   0でなければ頭は本当に（1歩の中で）速く振動している。
            ch = getattr(self, "_head_chain", None)
            if ch is None:
                ch = []
                b2 = bid
                while b2 > 0:
                    for jj in range(int(m2.body_jntnum[b2])):
                        jid = int(m2.body_jntadr[b2]) + jj
                        nd2 = {0: 6, 1: 3, 2: 1, 3: 1}.get(int(m2.jnt_type[jid]), 1)
                        adr2 = int(m2.jnt_dofadr[jid])
                        nm2 = m2.joint(jid).name or "(無名)"
                        # 回転の自由度だけ（freejoint は後ろ3つが回転）
                        rng = range(adr2 + 3, adr2 + 6) if nd2 == 6 else range(adr2, adr2 + nd2)
                        ch.append((nm2, list(rng)))
                    b2 = int(m2.body_parentid[b2])
                self._head_chain = ch
            合 = 0.0
            最 = ("", 0.0)
            for nm2, rng in ch:
                w2 = math.sqrt(sum(float(ctx.data.qvel[z]) ** 2 for z in rng))
                合 += w2 ** 2
                if w2 > 最[1]:
                    最 = (nm2, w2)
            行["頭の向きを決める自由度の速さ_度毎秒"] = round(math.degrees(math.sqrt(合)), 5)
            行["そのうち最速の関節"] = 最[0]
            da = getattr(u2, "_root_dadr", None)
            if da is not None:
                rv = ctx.data.qvel[int(da) + 3:int(da) + 6]
                行["体の根のqvel_度毎秒"] = round(math.degrees(
                    math.sqrt(sum(float(x) ** 2 for x in rv))), 5)
        except Exception:       # noqa: BLE001
            pass
        # 反射の命令のほうも一緒に控える（「命令」と「実際の動き」を並べて見るため）
        o = getattr(getattr(ctx.env, "unwrapped", ctx.env), "_orienting", None)
        行["サッケード残り秒"] = ("" if o is None
                                else round(float(getattr(o, "_sacc_remaining", 0.0) or 0.0), 4))
        行["撃った回数"] = "" if o is None else int(getattr(o, "n_saccades", 0) or 0)
        self.rows.append(行)

    def report(self, ctx):
        """out があれば列をCSVへ書き、生の事実だけを返す（平均や判定はしない）。"""
        if self.rows and self.out:
            p = _abs_path(self.out)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(列)
                for r in self.rows:
                    w.writerow([r.get(c, "") for c in 列])
        動 = [r["左目_動いた度"] for r in self.rows if r.get("左目_動いた度") != ""]
        out = {"記録した歩数": len(self.rows),
               "左目が動いた歩": sum(1 for v in 動 if v > 0.01),
               "左目が動いた歩の分母": len(動)}
        if 動:
            out["左目の1歩あたり最大度"] = round(max(動), 3)
        if self.rows:
            out["終わりの撃った回数"] = self.rows[-1].get("撃った回数")
        if self.out:
            out["出力"] = self.out
        return out
