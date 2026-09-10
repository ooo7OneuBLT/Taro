# -*- coding: utf-8 -*-
"""【退避したコード・2026-09-11】視線誘導反射の「自前の動き検出」一式。

`taro_core/src/brain/midbrain/orienting.py` から外した。走行では使っていない。

【なぜ外したか】実測（F2-114pre、600歩・種91）で、この経路が撃った回数は **0 回**。
同じ走行で、地図を読む経路が 218 回すべてを撃っていた。加えて、サッケード後に
眼球をその位置に留める力（保持）の発動条件がこの経路にぶら下がっており、
保持が一度も働かない原因になっていた。

【中身】画像から動きを見つける（複数の時間スケールの差分）／中心視野の重み／
上丘の格子へ写して競合させる／勝った山で重心を取る。

【人間側】上丘は自分でも目立ちを計算しており（White et al. 2017、上丘 65ms <
頭頂 70-75ms < 一次視覚野 139ms）、この経路が人間模倣として不要になったわけでは
ない。ただし上丘の地図は1枚で特徴ごとの地図を持たない（Krauzlis ら、
Gandhi & Katnani 2011 の総説で優勢）ため、太郎が上丘の中に2枚持っている形が
人間と違う。地図を1枚にまとめる設計が決まったら、地図の側へ統合して戻す。

【元の場所】`taro_core/src/brain/midbrain/orienting.py`（コミット 1d73d12 時点）
【関連】`F/docs/二語文/現状_見る側_2026-09-10.md`
"""

# 以下、退避したメソッド本体（OrientingReflexV2 のメソッドとして書かれている）。
# 定数は元ファイル（orienting.py）の冒頭にそのまま残してある。

    def _detect_motion(self, image):
        """現フレームと 1・5・20 前のフレームとの差を混ぜて動きマップを返す。

        フレームが揃わないうちは、揃っているスケールだけ使う（穴埋めしない）。
        RGB画像はチャネル平均で単一マップにする。
        注意：ON（明転）とOFF（暗転）を区別せず絶対値を取る。文献調査（2026-07-26）で
        「ON優位」を支持する定量値は無く、V1はむしろOFF優位（Jansen 2019, 65:35）、
        行動レベルの非対称もパラダイム依存でバラバラと判明したため、
        **対称（1:1）が根拠のない前提を最小にする**という判断。
        """
        # 2026-07-26修正：画像を [0,1] にそろえてから差を取る。
        #   それまで uint8（0〜255）のまま差分していたため、動きマップの値が
        #   輝度スケールに乗ってしまい、`strength` が 50〜140 になっていた。
        #   発火の閾値 SACCADE_MIN_STRENGTH=0.02 は 0〜1 を想定した値なので、
        #   **常に2500倍の値が来て必ず発火**していた（6.7秒で32発＝撃ちっぱなし。
        #   おもちゃが存在しない条件でも同じ数だけ撃っていた）。
        #   → 落とし穴チェックリスト 項36（指標を作るとき向きと尺度も書く）
        arr = np.asarray(image)
        cur = (arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8
               else arr.astype(np.float32))
        cur_gray = cur.mean(axis=-1) if cur.ndim == 3 else cur

        self._frame_buffer.append(cur_gray)
        while len(self._frame_buffer) > self._max_scale + 1:
            self._frame_buffer.pop(0)

        motion = np.zeros_like(cur_gray, dtype=np.float32)
        used = 0
        for scale in self.time_scales:
            if len(self._frame_buffer) > scale:
                past = self._frame_buffer[-1 - scale]
                motion += np.abs(cur_gray - past)
                used += 1
        if used == 0:
            return np.zeros_like(cur_gray)
        return motion / used

    # ------------------------------------------------------------
    # ステップ1.6：静的顕著性（中心-周辺差分＝DoG）
    # ------------------------------------------------------------
    def _static_contrast(self, image):
        """輝度画像の局所コントラスト地図を返す（中心-周辺差分の絶対値＝DoG）。

        Itti-Koch系 luminance contrast の最小実装。網膜神経節細胞の
        中心-周辺拮抗型受容野に対応する（機構の存在は[Tier2]、σの具体値は
        文献に無く手順0の絵で調整した暫定値[Tier3・ARBITRARY]）。
        """
        arr = np.asarray(image)
        cur = (arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8
               else arr.astype(np.float32))
        gray = cur.mean(axis=-1) if cur.ndim == 3 else cur
        h, w = gray.shape
        # 視野角 → 画素（retina.object_motion と同じ変換：画像1辺=fovy_deg）。
        scale = max(h, w) / VISION_FOVY_DEG
        sigma_c = STATIC_SIGMA_CENTER_DEG * scale
        sigma_s = STATIC_SIGMA_SURROUND_DEG * scale
        center = gaussian_filter(gray, sigma_c)
        surround = gaussian_filter(gray, sigma_s)
        return np.abs(center - surround)

    # ------------------------------------------------------------
    # ステップ2：中心視野バイアス
    # ------------------------------------------------------------
    def _apply_center_bias(self, motion):
        """中心視野の優位を反映する。

        既定（collicular）：上丘のマグニフィケーション（面積）を重みにする。
          M_area(R) ∝ (Bu·Bv)/(R+A)² ＝「視野の1度が上丘で何mm²を占めるか」。
          ガウス窓と違い**べき乗則で裾が重い**ので、周辺の刺激も捨てない。
        旧（gauss）：ガウス窓。形も強さも根拠がなかった。比較用に残す。
        """
        if CENTER_BIAS_MODE == "grid":
            # 格子方式では**何も掛けない**。中心視野の優位は、次の段で
            #   上丘の格子へ写すときに座標変換そのものから出る。
            #   ここで重みを掛けると二重になる。
            return motion

        if CENTER_BIAS_MODE == "gauss":
            if self._center_weight is None or self._center_weight.shape != motion.shape:
                h, w = motion.shape
                cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
                sigma = CENTER_BIAS_SIGMA_FRAC * max(h, w)
                ys = np.arange(h)[:, None] - cy
                xs = np.arange(w)[None, :] - cx
                self._center_weight = np.exp(
                    -(ys ** 2 + xs ** 2) / (2 * sigma ** 2)).astype(np.float32)
            return motion * self._center_weight

        smap = self._collicular_map(motion.shape)
        if self._center_weight is None or self._center_weight.shape != motion.shape:
            w = smap.mag_area.astype(np.float32)
            self._center_weight = w / max(float(w.max()), 1e-12)   # 中心を1に正規化
        return motion * self._center_weight

    def _collicular_map(self, shape):
        """画像の形に対応する上丘の地図を作る（一度だけ計算してキャッシュ）。"""
        if self._smap is None or self._smap.h != shape[0] or self._smap.w != shape[1]:
            import sys as _s, os as _o
            _b = _o.path.abspath(_o.path.join(
                _o.path.dirname(_o.path.abspath(__file__)), _o.pardir, _o.pardir,
                "taro_core", "src", "brain"))
            if _b not in _s.path:
                _s.path.insert(0, _b)
            from superior_colliculus import CollicularMap
            self._smap = CollicularMap(width=shape[1], height=shape[0],
                                       fovy_deg=VISION_FOVY_DEG)
        return self._smap

    # ------------------------------------------------------------
    # ステップ2＋3（格子方式）：上丘の地図へ写し、受容野でまとめ、そこで競合する
    # ------------------------------------------------------------
    def _select_on_collicular_grid(self, activity):
        """画像を上丘の格子へ写し、受容野でぼかしてから競合させ、重心を返す。

        実際の上丘で起きている順序をそのままなぞる：
            網膜の像 → 上丘の地図へ（中心視野が引き伸ばされる）
                     → 受容野でまとめる（1ニューロンが視野の広い範囲を担当）
                     → 上丘の組織の上で側方抑制の競合
                     → 集団の重心（population vector）→ 視野の向きへ逆変換
        """
        smap = self._collicular_map(activity.shape)
        g = smap.to_grid(activity)                       # [2, nv, nu]

        # 受容野：上丘の上では一様な広がり（RF_SIGMA_MM）。視野角で見ると
        # 中心では狭く周辺では広くなる＝偏心度依存が座標変換から自動的に出る。
        sv, su = smap.rf_sigma_cells
        g = gaussian_filter(g, (0.0, sv, su))
        # 【2026-08-21修正】抑制（IOR・疲労）を引く**前**の地形の最大値を覚えておく。
        #   以後の正規化（競合入力・疲労の駆動信号）はこの値を基準にする。
        #   従来は抑制後のg自身のmaxで割っていたため、疲労で地形全体がカスカスに
        #   なるとカスを1.0へ再増幅して全力で選挙し続けた（焦土化の正体の片割れ。
        #   F日誌2026-08-21追記2）。抑制OFF時は g0max==g.max() で従来と完全一致。
        g0max = float(g.max())
        # ステップ5（IOR）：受容野でまとめた直後の上丘の活動（＝競合入力）から
        #   抑制地図を引く。負は0でクリップする。OFF時は呼ばない
        #   （self.ior が False なら _ior_map は一度も割り当てられないので
        #   このブロックには入らない）。
        if self.ior and self._ior_map is not None and self._ior_map.shape == g.shape:
            g = np.clip(g - IOR_STRENGTH * self._ior_map, 0, None)
        # 改訂3（2026-08-20）項1・2：中央固定の抑制（旧・馴化スカラーによる
        #   grid_u=0,grid_v=0への一律ガウス抑制）は廃止した。実際のtoy1の山は
        #   中央から数mmずれた場所にでき、中央固定の抑制は届かず空撃ちしていた
        #   （実測：`F/logs/F1-4d_視線探索_2026-08-20/B_停止時_競合入力ヒートマップ.png`）。
        #   代わりに、**いま活動が高い場所ほど疲労する**地図 H で g を減衰させる
        #   （文献の式の乗算適用。本ファイル冒頭「改訂3」節・_update_fatigue参照）。
        #   ここでは①直近のHでg_effを作り②次の蓄積に使う駆動信号（正規化済みg。
        #   Hの更新そのものは apply()->_update_fatigue が毎stepオイラー法で回す。
        #   vision は物理stepより粗い周期でしか来ないため、ここでは駆動信号を
        #   キャッシュするだけ）を用意する。
        #   OFF時（self.habituation=False）は呼ばない＝既存挙動と完全一致。
        if self.habituation:
            if self._fatigue_map is None or self._fatigue_map.shape != g.shape:
                self._fatigue_map = np.zeros_like(g, dtype=np.float32)
            # 【2026-08-21修正】駆動信号の基準を抑制前の地形の最大値 g0max に変更。
            #   本当に目立っている場所だけが疲れる（弱い残りカスは疲れも遅い）。
            self._fatigue_drive = ((g / g0max) if g0max > 1e-9
                                    else np.zeros_like(g, dtype=np.float32))
            h_clip = np.clip(self._fatigue_map, 0.0, 1.0)
            g = g * (1.0 - h_clip)
        self.sc_input = g
        if g.max() < 1e-9:
            self.competed_map = g
            return 0.0, 0.0, 0.0

        # 【2026-08-21修正】競合入力の基準も抑制前の g0max。疲労が地形を削ったら
        #   競合はその分だけ静かになる（削りカスを1.0へ再増幅しない）。
        inp = g / g0max
        # 【F1-4e・2026-08-20】場の記憶（綱引きの持ち越し）。既定OFF。
        #   従来は毎フレーム u = inp.copy() で競合を仕切り直していたため、
        #   対称な2標的ではノイズが一瞬対称を破っても次のフレームで消え、
        #   永遠に決着しなかった（実測：左右両視野に山が両立するフレーム72%・
        #   toy2方向を勝者が指した回数ゼロ。日誌2026-08-20追記7）。
        #   動的神経場の標準形は場を時間の中で連続進化させ、ノイズが破った対称が
        #   自己強化されて1山に収束する（Wilimzig et al. 2006・原文精読：対称2標的の
        #   選択には確率的な対称性の破れが必須）。ONのときは前フレームの場を
        #   初期値に使う（固視中の持ち越し。サッケード発火時は網膜座標がずれるため
        #   apply() 側で場をリセットする＝新しい視点で仕切り直し）。
        if USE_FIELD_MEMORY and self._field is not None and self._field.shape == inp.shape:
            u = self._field
        else:
            u = inp.copy()
        se = (0.0, LI_SIGMA_EXC_MM / smap.dv, LI_SIGMA_EXC_MM / smap.du)
        si = (0.0, LI_SIGMA_INH_MM / smap.dv, LI_SIGMA_INH_MM / smap.du)
        # 【2026-08-21修正】ノイズを入力の強さに比例させる（正規化の床と同じ思想）。
        #   従来は固定振幅で、疲労が地形を0.1程度まで削った瞬間にノイズが選挙を
        #   支配していた。神経活動の揺らぎは発火率に応じて増える（Poisson的）＝
        #   相対ノイズ一定が生理的[Tier2]。入力が満額（inp.max()==1、抑制なしの
        #   既定経路は常にこれ）なら従来と完全一致。
        noise_scale = self.noise * np.sqrt(LI_RATE) * float(inp.max())
        for _ in range(LI_N_ITER):
            f = np.clip(u, 0, None)
            exc = gaussian_filter(f, se) * LI_W_EXC
            inh = gaussian_filter(f, si) * LI_W_INH
            # 全体抑制＝左右の上丘のあいだの抑制も含む（Munoz & Istvan 1998）
            inh_global = LI_W_GLOBAL * float(f.mean())
            u = u + LI_RATE * (-u + inp + exc - inh - inh_global)
            if noise_scale > 0:
                u = u + noise_scale * self.rng.standard_normal(u.shape)
        u = np.clip(u, 0, None)
        if USE_FIELD_MEMORY:
            self._field = u.copy()      # 次フレームへ持ち越す（F1-4e）
        self.competed_map = u
        if USE_WINNER_ONLY and u.max() > 1e-12:
            # 最大値を含む連結成分（＝勝者の山）だけを残して重心を取る（F1-4e）。
            #   場の持ち越し(self._field)には触れない＝競合の力学は変えず読み出しのみ。
            from scipy.ndimage import label as _ndlabel
            mask = u >= CENTROID_THRESH_FRAC * u.max()
            side, iv, iu = np.unravel_index(int(np.argmax(u)), u.shape)
            lab, _n = _ndlabel(mask[side])
            keep = lab == lab[iv, iu]
            u_win = np.zeros_like(u)
            u_win[side][keep] = u[side][keep]
            u = u_win
        h_dir, v_dir = smap.grid_direction(u, thresh_frac=CENTROID_THRESH_FRAC)
        strength = float(min(activity.max(), 1.0))
        return float(h_dir), float(v_dir), strength

    # ------------------------------------------------------------
    # ステップ3：側方抑制で1箇所を選び、その周辺で重心を取る
    # ------------------------------------------------------------
    def _select_and_centroid(self, activity):
        """メキシカンハット型の側方相互作用を数回まわして競合させ、
        残った活動の重心を (h_dir, v_dir, strength) として返す。

        h_dir: 右が正、左が負（[-1, 1] 相当）
        v_dir: 上が正、下が負（画像座標の y は下向きなので符号を反転）
        strength: 反応の強さ（動きが無ければ 0）
        """
        if activity.max() < 1e-6:
            self.competed_map = np.zeros_like(activity)
            self.sc_input = None
            return 0.0, 0.0, 0.0

        if CENTER_BIAS_MODE == "grid":
            return self._select_on_collicular_grid(activity)

        h, w = activity.shape
        sigma_exc = LI_SIGMA_EXC_FRAC * max(h, w)
        sigma_inh = LI_SIGMA_INH_FRAC * max(h, w)

        inp = activity / activity.max()      # 入力を [0,1] に正規化
        u = inp.copy()
        # ノイズは確率微分方程式の慣例に従い sqrt(dt) でスケールする
        noise_scale = self.noise * np.sqrt(LI_RATE)
        for _ in range(LI_N_ITER):
            f = np.clip(u, 0, None)          # 発火率（負は出力しない）
            exc = gaussian_filter(f, sigma_exc) * LI_W_EXC
            inh = gaussian_filter(f, sigma_inh) * LI_W_INH
            inh_global = LI_W_GLOBAL * float(f.mean())   # 視野全体に届く抑制
            # du/dt = -u + input + 近傍興奮 - 近距離抑制 - 全体抑制 + ノイズ
            u = u + LI_RATE * (-u + inp + exc - inh - inh_global)
            if noise_scale > 0:
                u = u + noise_scale * self.rng.standard_normal(u.shape)
        u = np.clip(u, 0, None)
        self.competed_map = u

        if u.max() < 1e-6:
            return 0.0, 0.0, 0.0

        # 勝った山の裾を切ってから、上丘の座標で重心（population vector）を取る。
        #   Lee, Rohrer & Sparks (1988) Nature 332:357-360 ＝ 上丘のサッケードは
        #   活動している集団の重心で決まる（一部を薬理的に止める直接実験）。
        #
        #   注意：「平らな画像で重心を取る」のと「歪んだ地図で重心を取ってから逆変換」は
        #     数学的に別物（Jensen の不等式）。実際の上丘は後者
        #     （Goossens & Van Opstal 2012 J Neurophysiol）。
        #     対象が視野の端にあるほど出力が中心寄りに潰れるのは、
        #       バグではなく**この写像の性質そのもの**。
        if u.max() <= 1e-12:
            return 0.0, 0.0, 0.0
        if CENTER_BIAS_MODE == "gauss":
            # 旧方式（比較用）：平らな画像で重心を取る
            thresh = u.max() * CENTROID_THRESH_FRAC
            mask = u > thresh
            wsum = u[mask].sum()
            if wsum < 1e-9:
                return 0.0, 0.0, 0.0
            ys, xs = np.nonzero(mask)
            cy_map = float((ys * u[mask]).sum() / wsum)
            cx_map = float((xs * u[mask]).sum() / wsum)
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            h_dir = (cx_map - cx) / cx
            v_dir = -(cy_map - cy) / cy      # 画像の y は下向きなので反転
        else:
            smap = self._collicular_map(u.shape)
            h_dir, v_dir = smap.direction(u, thresh_frac=CENTROID_THRESH_FRAC)
        # 反応の強さ＝中心バイアス後の動きマップの最大値。
        # 画像を [0,1] にそろえたので、これも [0,1] に収まる（_detect_motion 参照）。
        #   念のため上限で切る（中心バイアスの重みは1以下なので理論上は超えない）。
        strength = float(min(activity.max(), 1.0))
        return float(h_dir), float(v_dir), strength
