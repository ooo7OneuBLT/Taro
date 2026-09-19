"""実行中の「いま」をまとめた入れ物。プラグインはこれだけを受け取る。

【なぜ入れ物にするか、2026-07-30】プラグインが env や太郎の中身へ勝手に手を伸ばすと、
「測る道具が太郎を変えてしまう」事故が起きる（＝観測装置が対象を変える）。
渡すものを1つに絞り、**読むだけ**という約束を明示する。

注意：プラグインは ctx の中身を**書き換えない**。測る・見るだけ。
  太郎を変える設定は実験ファイルの `taro` 欄が担う。
"""


class Ctx:
    """実行中の状態。プラグインへ渡す唯一の入れ物。

    属性:
        env      : 環境（HybridEnv 等）。model/data は env.unwrapped から
        model    : MuJoCo の model（よく使うので取り出してある）
        data     : MuJoCo の data
        brain    : 太郎の脳（無い実験＝measure のみ なら None）
        spec     : 実験ファイル（辞書）そのもの
        scene    : シーンの辞書
        step     : いまのステップ数
        n_steps  : 全体のステップ数
        dt       : 1ステップの秒数（K × DT）
        log      : 追記用の関数 log(行の辞書)
    """

    def __init__(self, *, env, spec, scene, n_steps, dt, brain=None, log=None):
        self.env = env
        u = env.unwrapped
        self.model = u.model
        self.data = u.data
        self.brain = brain
        self.spec = spec
        self.scene = scene
        self.step = 0
        self.n_steps = n_steps
        self.dt = dt
        self._log = log or (lambda row: None)
        # 【なぜ、2026-08-15】プラグインが「学習を打ち切りたい」と伝える手段が
        #   無かった（学習ループには早期終了の仕組みが無かった）。
        #   停止シグナルの土台として2つの属性を追加する（run/trainer.py が
        #   on_checkpointの直後にstop_requestedを見てbreakする）。
        #   注意：既存のどのプラグインもこの属性を読み書きしていないので、
        #   既存挙動は変わらない（仕様「作るもの」1節）。
        self.stop_requested = False
        self.stop_reason = None


    # 【段B-2a・2026-09-13】太郎が自分の作業台（taro.tick）へ書いた値を、
    #   道具からは今までどおり `ctx.<名前>` で読めるようにする。
    #   **写さない＝参照する。** 写すと、写した瞬間と道具が読む瞬間がずれて
    #   1周期古い値を読むことになる（道具は tick の途中で読むため）。
    #   自分（Ctx）に直接置かれた値があればそちらが勝つ＝
    #   `taro.visual_attention` を使わない古い実験（道具が自分で作って
    #   ctx へ置く経路）もそのまま動く。
    _BRAIN_KEYS = frozenset(['attended_object', 'gone_object', 'attention_point', 'attention_switch_t', 'efference', 'goal_point', 'last_babble', 'last_produce', 'last_vision_vec', 'last_world_pred', 'object_files', 'priority_map_result', 'priority_map_result_t', 'produce_gate', 'salience_map', 'surprise_trace', 'vanish_misses', 'world_pred_by_file', 'world_pred_inputs', 'world_predictor_grad_report'])

    def __getattr__(self, name):
        # 通常の属性探索が失敗したときだけ呼ばれる
        if name in Ctx._BRAIN_KEYS:
            taro = self.__dict__.get("taro")          # 再帰を避けるため辞書から直接
            tick = getattr(taro, "tick", None) if taro is not None else None
            if tick is not None:
                try:
                    return getattr(tick, name)
                except AttributeError:
                    pass
        raise AttributeError(name)

    def log(self, row):
        """行の辞書 row を受け取り、生成時に渡されたログ用の関数へそのまま渡す。戻り値は無い。"""
        self._log(row)

    @property
    def sim_sec(self):
        """シミュレーション上の経過秒数。"""
        return self.step * self.dt

    # ======================================================================
    # 導いた事実（2026-09-16）
    # ======================================================================
    # 【なぜここに置くか・ユーザー指摘「別々に出してるのがおかしい」】
    #   Ctx には生の状態（step・model・data）しか無く、そこから導いた事実
    #   （太郎が喋ったか／親が何を差し出しているか／太郎が何を見ているか）は
    #   **道具ごとに各自で計算していた**。実測：
    #     gaze_probe.py:61-62   u._gaze_angle_to("test_object1"/"test_object2")
    #     word_production.py    同じ計算をもう一度（2026-09-15に8択へ直した）
    #   同じ事実が2箇所にあり、**片方だけ直した**状態になっていた
    #   （gaze_probe は2択決め打ちのままで、8択の場面では間違った答えを出す）。
    #
    # 【視線は角度ではなく光線で出す】run/world/visibility.py は自分の説明文に
    #   「①角度だけ gaze_angle() … 遮蔽を見ないので単独では使わないこと」と
    #   書いている（2026-07-26、柵の向こうの物を「視界内100%」と報告した事故）。
    #   にもかかわらず測定器31本のうち visibility.py を使っているものは0本だった。
    #   ここでは②visible_by_ray（遮蔽を見る）を使う。
    #   ③visible_in_image は**使えない**：対象を指定する引数が無く、赤い画素の
    #   総数を数えるだけ＝目標Eの「赤い球1個」専用で、8択では物を区別できない。
    #
    # 値は1ステップに1度だけ計算して持ち回す（同じ歩で何本の道具が読んでも同じ値）。

    def _事実(self):
        """このステップの導いた事実をまとめて作る（1歩に1度だけ計算する）。"""
        c = self.__dict__.get("_事実キャッシュ")
        if c is not None and c[0] == self.step:
            return c[1]
        f = {"親の的": "", "親の状態": "", "スロット": {}, "視線角": {}, "視線の先": ""}
        u = getattr(self.env, "unwrapped", self.env)
        pl = getattr(u, "_parent_labeling", None)
        f["親の的"] = getattr(pl, "_target", None) or ""
        f["親の状態"] = str(getattr(pl, "_state", "") or "")
        # スロットの数は場面で決まる（2択のことも8択のこともある）。決め打ちしない。
        try:
            for slot, info in (pl._slots(u) or {}).items():
                body = info.get("body")
                if body:
                    f["スロット"][slot] = body
        except Exception:       # noqa: BLE001  親のいない場面では空のまま
            pass
        if f["スロット"]:
            import sys as _sys
            import os as _os
            _w = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "world")
            if _w not in _sys.path:
                _sys.path.insert(0, _w)
            try:
                import visibility as _vis
            except Exception:   # noqa: BLE001
                _vis = None
            見える = []
            for slot, body in f["スロット"].items():
                try:
                    bid = int(self.model.body(body).id)
                    ang = round(float(_vis.gaze_angle(self.model, self.data, bid)), 2)
                except Exception:   # noqa: BLE001
                    continue
                f["視線角"][slot] = ang
                try:
                    ok, _hit = _vis.visible_by_ray(self.model, self.data, bid)
                except Exception:   # noqa: BLE001
                    ok = False
                if ok:
                    見える.append((ang, slot))
            # 見えている（遮られていない）ものの中で、いちばん視線に近いもの。
            #   1つも見えていなければ空欄＝「何も見ていない」。角度で代用しない。
            f["視線の先"] = min(見える)[1] if 見える else ""
        self.__dict__["_事実キャッシュ"] = (self.step, f)
        return f

    @property
    def 親の的(self):
        """親がいま差し出している物のスロット名（例 "toy8"）。無ければ空文字。"""
        return self._事実()["親の的"]

    @property
    def 親の状態(self):
        """親のいまの状態（_PICK/_SHAKE など）。無ければ空文字。"""
        return self._事実()["親の状態"]

    @property
    def スロット(self):
        """この場面にあるスロットと物の名前 {"toy8": "test_object8", ...}。"""
        return self._事実()["スロット"]

    @property
    def 視線角(self):
        """スロットごとの視線のずれ[度] {"toy8": 12.3, ...}。**遮蔽は見ていない**。"""
        return self._事実()["視線角"]

    @property
    def 視線の先(self):
        """太郎がいま見ている物のスロット名。遮られているものは選ばない。

        何も見えていなければ空文字（角度が近いだけの物で代用しない）。
        """
        return self._事実()["視線の先"]

    @property
    def 発話した(self):
        """この歩で太郎が実際に声を出したか。

        【なぜ要るか・2026-09-15に踏んだ間違い】`last_produce` は**黙った歩にも**
        入っている（`generated_word` が空文字）。それを1発話として数えたため、
        F2-130b の報告は 95.8%、実際は 86.1% だった。さらに
        `exact_match = (generated_word == target_word)` が空文字どうしの比較で
        1（正解）になり、分母と分子の両方が水増しされていた。
        """
        ev = getattr(self, "last_produce", None)
        return bool(ev) and bool((ev or {}).get("generated_word"))
