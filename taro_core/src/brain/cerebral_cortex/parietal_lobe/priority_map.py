# -*- coding: utf-8 -*-
"""段6 注意の優先度地図 ── 「下からの目立ち」と「上からの目的」を1枚の地図で競わせる。

【仕様】F/docs/二語文/仕様_見る側の段構成_実装_2026-09-09.md 5節・
F/docs/二語文/仕様_見る側の設計_段構成_2026-09-09.md 後半A（段6）。
Bisley & Goldberg（LIP＝下からの目立ち＋上からの目的）、Kidd 2012（驚きは
逆U字）、Itti-Koch（復帰抑制IOR）[Tier2]。

【既定不変】全部の重みが0・switch_delay_s=0・hysteresis=0のときは
「候補（misses==0）の中で area が最大のもの」を選ぶだけになる
（`priority_i = w_size・area_i` のみが残るため）。中央56px内への絞りは
呼び出し側（object_files.py の`_process_attention`、既存843-845行）が
候補を渡す前に行う＝本クラスの外。
"""
import math


class PriorityMap:
    def __init__(self, w_size=1.0, w_center=0.0, w_motion=0.0, w_z=0.0, w_nov=0.0,
                 w_ior=0.0, ior_tau_s=2.0, w_hab=0.0, hysteresis=0.0, switch_delay_s=0.0,
                 explore_interval_s=None, explore_ior_s=3.0, cell=32):
        self.w_size = float(w_size)
        self.w_center = float(w_center)
        self.w_motion = float(w_motion)
        self.w_z = float(w_z)
        self.w_nov = float(w_nov)
        self.w_ior = float(w_ior)
        self.ior_tau_s = float(ior_tau_s)
        self.w_hab = float(w_hab)
        self.hysteresis = float(hysteresis)
        self.switch_delay_s = float(switch_delay_s)
        self.explore_interval_s = (None if explore_interval_s is None
                                    else float(explore_interval_s))
        self.explore_ior_s = float(explore_ior_s)
        self.cell = int(cell)

        self._current_id = None
        self._last_attended_t = {}     # file_id -> t（注意が最後に離れた時刻）
        self._hab_start_t = {}         # file_id -> t（現在の連続注視の開始時刻）
        self._pending_best = None
        self._pending_since_t = None
        self._last_explore_t = float("-inf")
        self._explore_cell_t = {}      # (i, j) -> 最後に探索した時刻

    @staticmethod
    def _inv_u(z, z0=3.0):
        """逆U字（Kidd 2012）：予測できなさすぎる物からは離れる。"""
        z = float(z)
        return z * math.exp(-z / z0)

    def update(self, files, t, motion_by_id, static_sal, surprise_trace, z_vec_by_id,
               goal_bonus, moving, dt):
        candidates = [f for f in files if getattr(f, "misses", 0) == 0]
        CENTER_X, CENTER_Y = 112.0, 112.0
        surprise_trace = surprise_trace or {}
        z_vec_by_id = z_vec_by_id or {}
        goal_bonus = goal_bonus or {}

        priority = {}
        for f in candidates:
            fx, fy = f.pos
            dist = math.hypot(fx - CENTER_X, fy - CENTER_Y)
            center_term = 1.0 - dist / 158.0
            motion_term = motion_by_id.get(f.id, 0.0)
            z_state = surprise_trace.get(f.id, 0.0)
            z_vec = z_vec_by_id.get(f.id, 0.0)
            gbonus = goal_bonus.get(f.id, 0.0)
            last_t = self._last_attended_t.get(f.id)
            ior_term = 0.0 if last_t is None else math.exp(-(t - last_t) / self.ior_tau_s)
            hab_start = self._hab_start_t.get(f.id, t)
            hab_term = max(0.0, t - hab_start) if f.id == self._current_id else 0.0

            priority[f.id] = (self.w_size * f.area + self.w_center * center_term
                               + self.w_motion * motion_term + self.w_z * self._inv_u(z_state)
                               + self.w_nov * z_vec + gbonus
                               - self.w_ior * ior_term - self.w_hab * hab_term)

        switch_signal = False
        best_id = max(priority, key=lambda i: priority[i]) if priority else None

        if moving:
            # 動作中は候補の更新だけ行い、切り替えはしない（仕様5節）。
            pass
        elif self._current_id is not None and self._current_id not in priority:
            # 候補から消えた（misses>0になった等）＝従来どおり保持（消えた物を追う）。
            pass
        elif self._current_id is None:
            if best_id is not None:
                self._current_id = best_id
                self._hab_start_t[best_id] = t
                switch_signal = True
                self._pending_best = None
                self._pending_since_t = None
        elif best_id is not None and best_id != self._current_id:
            cur_p = priority.get(self._current_id, float("-inf"))
            if priority[best_id] - cur_p > self.hysteresis:
                if self._pending_best != best_id:
                    self._pending_best = best_id
                    self._pending_since_t = t
                if (t - self._pending_since_t) >= self.switch_delay_s:
                    self._last_attended_t[self._current_id] = t
                    self._current_id = best_id
                    self._hab_start_t[best_id] = t
                    switch_signal = True
                    self._pending_best = None
                    self._pending_since_t = None
            else:
                self._pending_best = None
                self._pending_since_t = None
        else:
            self._pending_best = None
            self._pending_since_t = None

        explore_point = self._compute_explore(files, static_sal, t)

        ior = {fid: (0.0 if lt is None else math.exp(-(t - lt) / self.ior_tau_s))
               for fid, lt in self._last_attended_t.items()}

        return {"attended_id": self._current_id, "switch_signal": switch_signal,
                "priority": priority, "explore_point": explore_point, "ior": ior}

    def _compute_explore(self, files, static_sal, t):
        """仕様5節「探索」：explore_interval_sが数値のとき、static_salの中で
        「どのカードのbboxにも入らず、最近explore_ior_s秒以内に探索していない
        セル」の最大をexplore_pointとして返す（前回の探索からexplore_interval_s
        以上経っているとき）。"""
        if self.explore_interval_s is None or static_sal is None:
            return None
        if (t - self._last_explore_t) < self.explore_interval_s:
            return None
        cell = static_sal.shape[0]
        img_size = 224.0
        cs = img_size / cell
        best_val, best_ij = None, None
        for i in range(cell):
            cy = (i + 0.5) * cs
            for j in range(cell):
                cx = (j + 0.5) * cs
                in_file = False
                for f in files:
                    fx, fy = f.pos
                    r = math.sqrt(max(f.area, 0.0)) * 224.0 * 0.75 + 8.0
                    if math.hypot(cx - fx, cy - fy) <= r:
                        in_file = True
                        break
                if in_file:
                    continue
                last_t = self._explore_cell_t.get((i, j))
                if last_t is not None and (t - last_t) < self.explore_ior_s:
                    continue
                val = float(static_sal[i, j])
                if best_val is None or val > best_val:
                    best_val = val
                    best_ij = (i, j)
        if best_ij is None:
            return None
        i, j = best_ij
        self._last_explore_t = t
        self._explore_cell_t[(i, j)] = t
        return ((j + 0.5) * cs, (i + 0.5) * cs)
