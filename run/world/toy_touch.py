"""おもちゃに手が触れたか・手とおもちゃの距離を測る。判定の実体はここだけ。

【なぜ分けたか、2026-07-30】リーチングが起きたかを測るのに、
  ・`e_reach_babble_check.py` は色付きノイズで自発運動を作るだけ＝**脳を使わない**
  ・学習ループ（`e_growth_train.py`）はおもちゃとの距離を**記録していない**
という状態だった。学習済みの太郎で測るには学習ループ側に記録が要るが、
判定を2箇所に書くと食い違う（`e_hand_in_view.py` に一本化したのと同じ理由）。

注意：「触れた」の定義：**おもちゃのgeom と 手のgeom の接触**。
  おもちゃのgeomが接触リストに入っているかだけを見ると、体や床に当たったのも
  数えてしまう（このシーンではおもちゃは胸の上に浮いているので床には触れないが、
  胸には触れうる）。＝手であることを必ず確かめる。

使い方（他のスクリプトから）:
    from e_toy_touch import ToyTouchProbe
    probe = ToyTouchProbe(model, data)     # 環境を作った直後に1回
    ...
    probe.update(model, data)              # 毎ステップ
    probe.summary()                        # {"touches":n, "min_cm":{...}, ...}
"""
import numpy as np

TOY_BODY = "test_object1"          # おもちゃ。名指しで取る（落とし穴 項67）
HAND_BODIES = ("right_hand", "left_hand")
# 指まで含めるか。手のひらだけだと「指先が触れた」を落とす
FINGER_HINTS = ("_ff_", "_mf_", "_rf_", "_lf_", "_th_", "_thumb", "hand")


def _geoms_of_body(model, bid, include_children=False):
    """body に属する geom の id 集合。include_children なら子孫も含める。"""
    ids = set()
    targets = {bid}
    if include_children:
        for b in range(model.nbody):
            p = b
            while p != 0:
                if p == bid:
                    targets.add(b)
                    break
                p = model.body_parentid[p]
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in targets:
            ids.add(g)
    return ids


class ToyTouchProbe:
    """おもちゃへの接触と最接近距離を溜める。"""

    def __init__(self, model, data):
        self.ok = True
        try:
            tb = int(model.body(TOY_BODY).id)
        except Exception:
            self.ok = False            # おもちゃの無い環境
            return
        self.toy_bid = tb
        self.toy_geoms = _geoms_of_body(model, tb, include_children=True)
        # 手（指を含む）のgeom。左右別に持つ
        self.hand_geoms = {}
        self.hand_bid = {}
        for hb in HAND_BODIES:
            try:
                bid = int(model.body(hb).id)
            except Exception:
                continue
            side = hb.split("_")[0]
            self.hand_bid[side] = bid
            self.hand_geoms[side] = _geoms_of_body(model, bid, include_children=True)
        self.touches = 0               # 触れた回数（立ち上がりだけ数える）
        self.steps_touching = 0        # 触れていたステップ数
        self.steps = 0
        self.min_dist = {s: 1e9 for s in self.hand_geoms}
        self._prev = False

    def rebind(self, model):
        """体を作り直したあとに id を引き直す。溜めた回数・最接近は**消さない**。

        【なぜ要るか、2026-07-30】体を育てる実験（月齢を進める）では学習の途中で
        env を作り直す。そのとき geom / body の id は原理的に変わりうるので、
        引き直さないと「接触を1回も検出しない」という静かな失敗になる。
        注意：元の学習ループ（e_growth_train.py）はここを引き直しておらず、
          さらに `env.unwrapped` を作り直し前のまま参照していた。
        """
        if not self.ok:
            return
        try:
            self.toy_bid = int(model.body(TOY_BODY).id)
        except Exception:
            self.ok = False
            return
        self.toy_geoms = _geoms_of_body(model, self.toy_bid, include_children=True)
        for hb in HAND_BODIES:
            try:
                bid = int(model.body(hb).id)
            except Exception:
                continue
            side = hb.split("_")[0]
            self.hand_bid[side] = bid
            self.hand_geoms[side] = _geoms_of_body(model, bid, include_children=True)

    def update(self, model, data):
        """model と data を受け取り、毎ステップ呼び出す。おもちゃと両手それぞれの最接近距離を更新し、おもちゃのgeomと手のgeomが接触リストに同時に含まれているかで「触れているか」を判定する。触れ始めた瞬間（直前のステップは触れていなかった場合）だけ touches を1増やし、触れているステップ数も数える。戻り値はない。
        """
        if not self.ok:
            return
        self.steps += 1
        toy = np.array(data.xpos[self.toy_bid], dtype=float)
        for side, bid in self.hand_bid.items():
            p = np.array(data.xpos[bid], dtype=float)
            self.min_dist[side] = min(self.min_dist[side],
                                      float(np.linalg.norm(toy - p)))
        hit = False
        for c in range(int(data.ncon)):
            g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
            # おもちゃ側と手側の**両方**が揃っていることを確かめる
            if ((g1 in self.toy_geoms and any(g2 in gs for gs in self.hand_geoms.values()))
                    or (g2 in self.toy_geoms and any(g1 in gs for gs in self.hand_geoms.values()))):
                hit = True
                break
        if hit:
            self.steps_touching += 1
            if not self._prev:
                self.touches += 1
        self._prev = hit

    def summary(self, dt_per_step=0.1):
        """dt_per_step: 1ステップの秒数（学習ループなら K*DT=0.1秒）"""
        if not self.ok:
            return {"ok": False}
        sim_sec = self.steps * dt_per_step
        return {"ok": True,
                "touches": self.touches,
                "steps_touching": self.steps_touching,
                "steps": self.steps,
                "sim_sec": sim_sec,
                "touch_per_min": (self.touches / (sim_sec / 60.0)) if sim_sec else float("nan"),
                "min_cm": {s: v * 100.0 for s, v in self.min_dist.items()}}

    def line(self, dt_per_step=0.1):
        """dt_per_step（1ステップの秒数、既定0.1秒）を受け取り、summary() の結果をもとに1行の文字列を返す。おもちゃが無い場合は「toy=なし」を返し、それ以外は接触回数・1分あたりの接触回数・各手の最接近距離（未測定なら「-」）・経過シミュレーション秒数をまとめた文字列を返す。
        """
        s = self.summary(dt_per_step)
        if not s["ok"]:
            return "toy=なし"
        # 注意：まだ1回も測っていないと min_dist は 1e9 のまま＝「1000億cm」と表示されて
        #   意味不明になる（2026-07-30 の試運転で実際に出た）。未測定は "-" と出す。
        mn = " ".join(f"{k[0]}{'-' if v > 1e8 else f'{v:.1f}cm'}"
                      for k, v in sorted(s["min_cm"].items()))
        return (f"toy_touch={s['touches']}回({s['touch_per_min']:.2f}/分) "
                f"min[{mn}] sim={s['sim_sec']:.0f}s")
