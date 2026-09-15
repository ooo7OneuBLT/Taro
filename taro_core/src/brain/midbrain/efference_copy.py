# -*- coding: utf-8 -*-
"""段0 座標の安定化（遠心性コピー）── 目を動かす命令の写しを、感覚側へ先回りで送る。

【仕様】F/docs/二語文/仕様_見る側の段構成_実装_2026-09-09.md 1節・
F/docs/二語文/仕様_見る側の設計_段構成_2026-09-09.md 後半A（段0）。

【符号の導出（3行、仕様0節の手がかりから）】
1. 水平：orienting.py 294-296行「eye_h角度を上げる→視線は左を向く→正面の物体は
   画像の右へ動く」。よってΔeye_h>0のとき画像の中身は列（+x=右）へ動く＝
   sign_h=+1.0（dx=sign_h・f・tan(Δh)、+x=右／img配列の列が増える向き）。
2. 垂直：同299行「関節のプラス方向は…垂直は「上」」なので、水平と同じ論法で
   Δeye_v>0（視線が上）のとき中身は画像の行（+y=下）へ動く（1013-1016行の
   目標角の式・1637行「v_dirは上が正」から見ても、目標へ目が動くほど元の
   位置の中身は中央（逆向き）へ寄る関係は水平・垂直で対称）。
3. 行・列ともimg配列の素直な向き（+x=列が増える＝右、+y=行が増える＝下）で
   そろえるとsign_v=+1.0（dy=sign_v・f・tan(Δv)）。結論：既定sign_h=+1.0,
   sign_v=+1.0（F2-107 JSONの既定と一致）。走行中の自己検査（object_files.py
   の`_run_efference_self_check`）で反対と分かれば設定で反転する。

【役割】orienting（`OrientingReflexV2`）の目標角・実際角から、次の4つを返す：
  1. 命令の写し（`shift_pred`）を先に感覚側へ送る＝次コマまでに起きる
     画像のずれの先回り
  2. 物の位置を先回りで更新する（段3が`shift_pred`を使う）
  3. 動いている最中は感度を落とす（`moving`）
  4. 実際の変化から予告分を引いて残りを外の変化とみなす
     （段1が`shift_actual`で前コマをずらしてから差分を取る）
Duhamel 1992・Wurtz 2008（サル・成人で神経基盤まで確定）[Tier1・機序]。
実装はMIMoの目の関節角の差分から画素へ回転投影するだけの近似
（平行移動は無視＝仮置きラベル、doc/人間模倣からの逸脱リストC節）。
"""
import math


class EfferenceCopy:
    """毎tick呼ぶ（検出コマでなくても）。orienting が None なら全部 0/False。"""

    def __init__(self, f_px, sign_h=1.0, sign_v=1.0, moving_extra_ticks=1):
        self.f_px = float(f_px)
        self.sign_h = float(sign_h)
        self.sign_v = float(sign_v)
        self.moving_extra_ticks = int(moving_extra_ticks)
        self._prev_eye_h = None
        self._prev_eye_v = None
        self._moving_extra_left = 0
        # 【2026-09-10】前に使われてから今までの、画像のずれの合計。
        #   使う側（検出コマ）が consume_shift() で受け取り、そこで 0 に戻す。
        self._accum = [0.0, 0.0]

    def consume_shift(self):
        """前に呼ばれてから今までに溜まった「画像の中身のずれ」を返し、0 に戻す。

        【なぜ必要か・2026-09-10】この器官は毎tick呼ばれて **1tickぶん** の
        ずれを出すが、使う側（物体ファイルの照合・地図のずらし）は
        **2〜3tickに1回**しか読まない。1tickぶんだけ渡すと、その間に起きた
        残りのずれが抜け落ちる。実測（F2-112pre）：予告÷実測の中央値が 0.52
        ＝約半分しか予告できていなかった。検出は600歩で262回なので
        1÷2.3＝0.43 とほぼ一致する。

        【人間はどうか】補正はサッケードの開始と同時に、**1回のまとまった
        ジャンプ**として起きる（中間地点を経由しない）。運ばれるのは「動いた」
        という合図ではなく**振幅そのもの**（Sommer & Wurtz 2006 Nature ほか、
        `doc/文献調査/二語文/2026-09-10_遠心性コピーの大きさと較正_人間側.md`）。
        ＝「使うときに、その間ぶんをまとめて1回渡す」がこの形にあたる。
        """
        out = (float(self._accum[0]), float(self._accum[1]))
        self._accum[0] = 0.0
        self._accum[1] = 0.0
        return out

    def update(self, orienting, t):
        if orienting is None:
            self._prev_eye_h = None
            self._prev_eye_v = None
            self._moving_extra_left = 0
            self._accum = [0.0, 0.0]
            return {"shift_pred": (0.0, 0.0), "shift_actual": (0.0, 0.0),
                    "shift_accum": (0.0, 0.0),
                    "moving": False, "eye_h": 0.0, "eye_v": 0.0,
                    "d_eye_h": 0.0, "d_eye_v": 0.0}

        eye_h = float(orienting._version_h_deg())
        eye_v = float(orienting._angle_deg(orienting.eye_qadr["v"]))

        if self._prev_eye_h is None:
            d_eye_h, d_eye_v = 0.0, 0.0
        else:
            d_eye_h = eye_h - self._prev_eye_h
            d_eye_v = eye_v - self._prev_eye_v
        self._prev_eye_h, self._prev_eye_v = eye_h, eye_v

        dx_actual = self.sign_h * self.f_px * math.tan(math.radians(d_eye_h))
        dy_actual = self.sign_v * self.f_px * math.tan(math.radians(d_eye_v))
        # 使われるまで足し合わせておく（consume_shift() で受け取られて 0 に戻る）
        self._accum[0] += dx_actual
        self._accum[1] += dy_actual

        tgt = getattr(orienting, "_tgt", None) or {}
        tgt_h = tgt.get("eye_h", eye_h)
        tgt_v = tgt.get("eye_v", eye_v)
        d_pred_h = tgt_h - eye_h
        d_pred_v = tgt_v - eye_v
        dx_pred = self.sign_h * self.f_px * math.tan(math.radians(d_pred_h))
        dy_pred = self.sign_v * self.f_px * math.tan(math.radians(d_pred_v))

        sacc_active = getattr(orienting, "_sacc_remaining", 0.0) > 0.0
        if sacc_active:
            self._moving_extra_left = self.moving_extra_ticks
            moving = True
        elif self._moving_extra_left > 0:
            self._moving_extra_left -= 1
            moving = True
        else:
            moving = False

        return {"shift_pred": (dx_pred, dy_pred), "shift_actual": (dx_actual, dy_actual),
                "shift_accum": (float(self._accum[0]), float(self._accum[1])),
                "moving": moving, "eye_h": eye_h, "eye_v": eye_v,
                "d_eye_h": d_eye_h, "d_eye_v": d_eye_v}
