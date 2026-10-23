"""目だけの場面の親。

親の決まり（いつ、どの物を、どう見せ、いつ名前を言い、いつ「○○ないね」と言うか、太郎の声にどう応えるか）は
run/world/parent.py の ParentLabeling をそのまま使う（状態機械・時間・確率・言い方は1つも変えない）。
変えるのは、物を動かす・隠す所だけ：
* MIMo の環境は毎歩すべての物を定位置へ置き直し、親がその上から書き直していた。隠すときは地下へ移して透明にしていた
* ここでは親が env.parent_hold（持って振る・差し出す）・env.hide（1つだけ見せる／消える場面）を呼ぶ。
  物はそれ以外では動かない。隠した物は描かれない（影も無い）
語の再生装置（WordSchedule）は、環境の時計と _gaze_angle_to しか使わないので、そのまま使う（env.py）。
"""

from __future__ import annotations

import numpy as np

from run.world.parent import ParentLabeling


class EyeParentLabeling(ParentLabeling):

    def _slots(self, env):
        d = {"toy1": {"body": env.toy_body["toy1"], "shape": getattr(env, "_toy_shape", "")}}
        if getattr(env, "_toy2", False):
            d["toy2"] = {"body": env.toy_body["toy2"], "shape": getattr(env, "_toy2_shape", "")}
        for k, sl in env.present_slots.items():
            if k in self.utterances:          # 名前の決まった物だけ（場面に出していない物を隠し直さない）
                d[k] = {"body": sl["body"], "shape": sl["shape"]}
        return d

    def _apply_shake(self, env):
        """ParentLabeling._apply_shake と同じ式（揺れ＝目の右向き×振幅×sin、振る2往復→止まる。基準＝定位置か
        視線の先）。置くのは env.parent_hold（親の手）。"""
        cid = int(env.model.camera("eye_left").id)
        right = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 0]
        phase = self._shake_t % (self.shake_burst_sec + self.shake_pause_sec)
        if phase < self.shake_burst_sec:
            wob = right * self.shake_amp_m * np.sin(2 * np.pi * self.shake_hz * phase)
        else:
            wob = right * 0.0
        if self.follow_gaze:
            base = self._follow_step(env)
        elif self._target == "toy1":
            base = getattr(env, "_rest_pos", None)
        else:
            base = getattr(env, "_rest_pos2", None)
        if base is None:
            return
        sl = self._slots(env).get(self._target)
        if sl is None:
            return
        quat = self._present_quat(env, sl["shape"]) if self.follow_gaze else None
        env.parent_hold(sl["body"], np.array(base, dtype=float) + wob, quat, cause=f"parent:{self._target}")

    def _hide_other(self, env):
        for k, sl in self._slots(env).items():
            if k != self._target:
                env.hide(sl["body"], cause="parent_solo")

    def _hide_all(self, env):
        for k, sl in self._slots(env).items():
            env.hide(sl["body"], cause="parent_vanish")
