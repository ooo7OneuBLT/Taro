# -*- coding: utf-8 -*-
"""物体ファイル ── 「これ」を物として保つ最小の仕組み。

【置き場所について、2026-09-04】この機能（動く物の位置を予測し、新しい見えと
突き合わせて同じ物と認識し続ける）を人間で担うのは、頭頂葉の中でも頭頂間溝
（intraparietal sulcus, IPS）だとされる。複数の物を同時に目で追う課題
（multiple object tracking）で一貫して活動が見られる部位で、動きを扱う視覚野
（MT/MST）とも接続が強い[Tier2・複数のfMRI研究で報告されている関連づけだが、
「IPSがカルマンフィルタと同じ計算をしている」という意味ではない]。

【誤解しないための注意】このファイルの名前が実在の脳部位と同じでも、それは
「IPSが担うとされる働き（予測してから新しい見えと突き合わせる）を、計算の形で
真似ている」という意味であり、「IPSの神経回路を忠実に再現した」という意味では
ない。2026-09-04の調査（`F/docs/太郎の作り方/`）で、この2つを混同すること
（構造の名前を実装したことと、機能を再現したことを同一視する誤り）が、既存の
脳型AI研究で繰り返し批判されてきたと確認した。太郎は学術的な主張をしないため、
この一線を越えない限り、その批判は太郎には当てはまらない。

【なぜ、2026-09-03・設計の経緯】
人間の予測は視野全体にかかるのではなく、**物体に束ねられて**かかる（「ボールは
落ちてバウンドする」）。この束の最小単位が物体ファイル（Kahneman & Treisman 1992・
Object files: A model for integrating information about objects [Tier1]）。
名前より先に、位置に紐づく一時的な入れ物として存在し、乳児も言葉の前から持つ
（Xu & Carey 1996 ほか [Tier1・WebSearchで確認]）。

**発達の順序**：10ヶ月までは時空間情報（位置・動きの連続性）だけで物を個体化し、
見た目の違いは使わない。12ヶ月から見た目・種類でも個体化できる
（Xu & Carey 1996 [Tier1]）。本実装は `appearance_weight` を0にすれば10ヶ月相当、
正の値にすれば12ヶ月相当になる（**場合分けで書き分けない**。連続パラメータ1本）。

【方式：標準的な多物体追跡（tracking-by-detection）をそのまま使う】
2026-09-03の議論で「時間で保持する（数コマ）」「隠れたか/視野外か/消えたかを
場合分けする」という初期案はどちらも却下された（恣意的な値・場合分けの直書き）。
代わりに、多物体追跡の標準手法（SORT/DeepSORT系 [Tier2・工学的に確立した手法。
人間の生理機構の直接の模写ではない]）をそのまま採用する：

  1. カルマンフィルタ（等速モデル）で各物体ファイルの次の位置を予測する
  2. 予測位置と見た目のベクトルから、今のコマの検出との「コスト」を作る
     （位置のずれ＋見た目の遠さ。重みは appearance_weight で連続的に調整）
  3. ハンガリアン法（scipy.optimize.linear_sum_assignment）で
     コスト最小の組み合わせを解く。場合分けは無い —— 全部同じ1つの最適化問題
  4. 一定コマ対応がつかない物体ファイルは削除する（これは知覚的な判断ではなく
     単なるメモリ管理。max_missed の値そのものに意味を持たせない）

**遮蔽（隠れる）は特別扱いしない。** 対応がつかなくても物体ファイルは
`max_missed` コマの間は残るので、検出が一時的に途切れても
（何かの後ろに隠れた／認識に失敗した、を区別せず）予測位置と見た目が近ければ
再び対応がつく。DeepSORT系がこの性質を持つのは、位置だけのカルマン予測は遮蔽が
続くと逸走するため、見た目の一致で長い遮蔽に耐える設計になっているから
（2026-09-03 WebSearch で確認・出典は設計文書側に記録）。

**予測違反**：`reappeared_with_gap` に该当した組（一定コマ以上途切れた後に
再び対応がついた）の中で、予測位置からの実際のずれが大きいものを返す。
「見た目は同じだがあり得ない場所に現れた」を検出するための最小の形。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# 等速モデルのカルマンフィルタ（状態=[x, y, vx, vy]）。標準形。値そのものに
# 恣意性は無い（プロセスノイズ・観測ノイズの大きさだけが自由パラメータ）。
_F = np.array([[1, 0, 1, 0],
               [0, 1, 0, 1],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], dtype=np.float64)
_H = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0]], dtype=np.float64)


class ObjectFile:
    """1つの物体ファイル。位置・速度（カルマン状態）と見た目を持つ。名前は持たない。"""

    def __init__(self, id_, pos, appearance, area, process_noise, meas_noise):
        self.id = id_
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 30.0
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * meas_noise
        self.appearance = np.asarray(appearance, dtype=np.float64)
        self.area = float(area)
        self.age = 0             # 何コマ存在するか
        self.hits = 1            # 対応がついた回数
        self.misses = 0          # 連続して対応がつかなかったコマ数
        self.since_seen = 0      # 直近で対応がついてから何コマ経ったか

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    def predict(self):
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + self.Q

    def innovation_cov(self):
        """観測前の予測から決まるイノベーション共分散 S = H P H^T + R。
        predict() 済みの P があれば、対応づけを決める前でも計算できる。"""
        return _H @ self.P @ _H.T + self.R

    def mahalanobis(self, pos):
        """予測位置と候補位置の距離を、予測の不確実性(P)で正規化した値。
        【2026-09-04・調査で確認】画素の生の距離だと、遮蔽が続いて予測が
        当てにならなくなっても許容幅が変わらない。マハラノビス距離を使うと、
        Pが遮蔽中に自動的に増える（predict()が呼ばれるたび加算される）ため、
        同じ画素のずれでも遮蔽が長いほど許容されるようになる。これは
        乳児の物理推論のベイズモデル（Téglás et al.）と数学的に同型の性質。"""
        y = np.asarray(pos, dtype=np.float64) - _H @ self.x
        S = self.innovation_cov()
        return float(np.sqrt(y @ np.linalg.inv(S) @ y))

    def update(self, pos, appearance, area, appearance_lr=0.3):
        z = np.asarray(pos, dtype=np.float64)
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + self.R
        mahal = float(np.sqrt(y @ np.linalg.inv(S) @ y))
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ _H) @ self.P
        # 見た目は指数移動平均で更新（1枚のノイズに引きずられないため）
        self.appearance = (1 - appearance_lr) * self.appearance + appearance_lr * np.asarray(appearance, dtype=np.float64)
        self.area = float(area)
        self.hits += 1
        self.misses = 0
        self.since_seen = 0
        return mahal      # 予測とのずれ（不確実性で正規化・単位なし）＝マハラノビス距離


MAHAL_GATE95 = 2.4477468306808161      # sqrt(chi2 95%点, 自由度2) ≈ sqrt(5.991)。
                                        # 2次元の位置観測を、95%の確率質量が収まる
                                        # 範囲までは同じ物とみなす、という標準的な統計量[Tier1]。


class ObjectFileSystem:
    """物体ファイルの集合を1コマぶん進める。

    Args:
        appearance_weight: 対応づけのコストに見た目をどれだけ使うか（0〜1程度）。
            0＝位置だけ（10ヶ月相当）。正＝見た目も使う（12ヶ月相当）。[Tier2・
            発達的に妥当な範囲付けだが連続値そのものは工学的な調整対象]
        mahal_gate: 位置の対応づけを許すマハラノビス距離（予測の不確実性で正規化
            した距離）の上限。これを超えたら別の物体とみなす（＝対応づけない）。
            既定はカイ二乗分布から出る統計的な境界（Tier1）で、画素の固定値ではない。
            【2026-09-04・旧 max_dist からの変更】旧版は画素の固定上限だったため、
            遮蔽が続いて予測が当てにならなくなっても許容幅が変わらなかった。
            マハラノビス距離なら、predict() のたびに増えるPに応じて自動的に
            広がる（ObjectFile.mahalanobis 参照）。
        max_missed: 何コマ連続で対応がつかなかったら物体ファイルを削除するか。
            知覚的な判断ではなくメモリ管理。値は寛容にとる。
        reappear_gap: これ以上コマが空いた後の再対応を「戻ってきた」として
            予測違反の候補にする閾値。
        revive_window_s: 【2026-09-09・連続性】消えたカードを覚えておく秒数。
            None＝従来どおり墓場を使わない（既定不変）。Kahneman & Treisman の
            物体ファイル・Spelke の時空間連続性[Tier1]の近似。
        revive_cos: 復活を許す見た目のコサイン類似度の下限。
        revive_dist_px: 復活を許す、消えた位置からの距離の上限（画素）。
        max_files: 【2026-09-09・枚数の上限】同時に持てるファイル数。None＝
            従来どおり無制限（既定不変）。Feigenson & Carey（乳児3個）・
            Pylyshyn（成人4個）[Tier1]の近似。
    """

    def __init__(self, appearance_weight=0.4, mahal_gate=MAHAL_GATE95, max_missed=20,
                 reappear_gap=4, process_noise=4.0, meas_noise=6.0, gate=1.0,
                 revive_window_s=None, revive_cos=0.8, revive_dist_px=40.0, max_files=None):
        self.appearance_weight = float(appearance_weight)
        self.mahal_gate = float(mahal_gate)
        self.max_missed = int(max_missed)
        self.reappear_gap = int(reappear_gap)
        self.process_noise = float(process_noise)
        self.meas_noise = float(meas_noise)
        self.gate = float(gate)          # 対応づけを許すコストの上限（大きいほど甘い）
        self.revive_window_s = None if revive_window_s is None else float(revive_window_s)
        self.revive_cos = float(revive_cos)
        self.revive_dist_px = float(revive_dist_px)
        self.max_files = None if max_files is None else int(max_files)
        self.files = []
        self._next_id = 0
        # 【2026-09-09・連続性】消えたカードの控え。revive_window_sがNoneのままなら
        # 一度も使われない（要素を足す箇所が全てrevive_window_s is not Noneで
        # 守られている）＝既定不変。
        self._graveyard = []
        self.last_revived = []   # 直近のstep()で復活したfile_idのリスト（属性で公開）

    def _make_room(self, protect_id, t):
        """【2026-09-09・枚数の上限】max_filesがNoneなら何もしない（既定不変）。
        上限に達していたら、protect_id以外でmissesが最大（同点ならsince_seenが
        大きい）ファイルを1枚削除し、revive_window_sがNoneでなければ墓場に送る。
        削除したfile_idを返す（削除しなければNone）。"""
        if self.max_files is None or len(self.files) < self.max_files:
            return None
        candidates = [f for f in self.files if f.id != protect_id]
        if not candidates:
            # 【仕様に無かった判断】全ファイルがprotect_id（1枚しか無くそれが
            # 注意中）なら押し出せない。この稀なケースでは上限を一時的に超える。
            return None
        victim = max(candidates, key=lambda f: (f.misses, f.since_seen))
        self.files = [f for f in self.files if f.id != victim.id]
        if self.revive_window_s is not None:
            self._graveyard.append({
                "id": victim.id, "pos": victim.pos,
                "appearance": np.array(victim.appearance, dtype=np.float64, copy=True),
                "area": victim.area, "hits": victim.hits, "age": victim.age,
                "t_lost": t if t is not None else 0.0,
            })
        return victim.id

    def predict_only(self):
        """【2026-09-09・仕様_予測して確かめる検出】確認（confirm）専用：対応づけは
        せず、各ファイルの予測位置だけを1コマぶん進める。`step()` 冒頭の2行
        （`f.predict()`・`f.age += 1`）と同じ処理を切り出しただけで、`step()`
        自体は無改修（既存の呼び出し元の挙動は不変）。呼び出し元（object_files.py）
        が確認用の点プロンプトを作る前に、いまの予測位置 `f.pos` を得るために使う。"""
        for f in self.files:
            f.predict()
            f.age += 1

    def step(self, detections, t=None, protect_id=None):
        """detections = [{"pos": (x,y), "area": float, "appearance": vec}, ...]（1コマぶん）。
        `t`（sim秒。連続性の墓場の期限管理に使う）と `protect_id`（注意中の
        file_id。枚数の上限で押し出さない）は任意引数（省略すれば従来どおり）。

        Returns:
            dict: matched=[(file_id, det_idx, residual)], created=[file_id,...]
                  （復活は含まない）, revived=[file_id,...]（連続性で復活した
                  もの。既存キーではなく新設のキー＝既存の読み手を壊さない）,
                  lost=[file_id,...]（このコマで削除された。max_missed超過に
                  加え、枚数の上限で押し出されたものも含む＝仕様に無かった判断。
                  CSV等の「lost」イベントとして自然に見えるようにするため）,
                  prediction_violations=[(file_id, residual, gap)]
                  （空白が reappear_gap 以上あった後に対応がつき、かつ
                   予測とのずれが mahal_gate を超えたもの＝見た目で救われた組）
        """
        for f in self.files:
            f.predict()
            f.age += 1

        n_f, n_d = len(self.files), len(detections)
        matched, unmatched_f, unmatched_d = [], list(range(n_f)), list(range(n_d))
        violations = []

        if n_f and n_d:
            cost = np.zeros((n_f, n_d))
            for i, f in enumerate(self.files):
                for j, d in enumerate(detections):
                    pos_cost = f.mahalanobis(d["pos"]) / self.mahal_gate  # 上限で切り詰めない
                    av = d["appearance"] / (np.linalg.norm(d["appearance"]) + 1e-9)
                    fv = f.appearance / (np.linalg.norm(f.appearance) + 1e-9)
                    app_cost = 1.0 - float(av @ fv)
                    cost[i, j] = (1 - self.appearance_weight) * pos_cost + self.appearance_weight * app_cost
            ri, ci = linear_sum_assignment(cost)
            used_f, used_d = set(), set()
            for i, j in zip(ri, ci):
                if cost[i, j] <= self.gate:
                    f, d = self.files[i], detections[j]
                    gap = f.since_seen
                    residual = f.update(d["pos"], d["appearance"], d["area"])
                    matched.append((f.id, j, residual))
                    if gap >= self.reappear_gap and residual > self.mahal_gate:
                        violations.append((f.id, residual, gap))
                    used_f.add(i); used_d.add(j)
            unmatched_f = [i for i in range(n_f) if i not in used_f]
            unmatched_d = [j for j in range(n_d) if j not in used_d]

        for i in unmatched_f:
            self.files[i].misses += 1
            self.files[i].since_seen += 1

        # ---- 連続性：墓場の期限切れを捨てる（revive_window_sがNoneなら墓場は空のまま）----
        if self.revive_window_s is not None and t is not None and self._graveyard:
            self._graveyard = [g for g in self._graveyard
                                if (t - g["t_lost"]) <= self.revive_window_s]

        # ---- 連続性：消えたカードの復活 -----------------------------------------
        revived = []
        remaining_d = unmatched_d
        if self.revive_window_s is not None and self._graveyard:
            used_grave = set()
            remaining_d = []
            for j in unmatched_d:
                d = detections[j]
                dv = np.asarray(d["appearance"], dtype=np.float64)
                dv_n = dv / (np.linalg.norm(dv) + 1e-9)
                best_idx, best_cos = None, None
                for gi, g in enumerate(self._graveyard):
                    if gi in used_grave:
                        continue
                    gv = np.asarray(g["appearance"], dtype=np.float64)
                    gv_n = gv / (np.linalg.norm(gv) + 1e-9)
                    cos = float(dv_n @ gv_n)
                    if cos < self.revive_cos:
                        continue
                    dist = float(np.hypot(d["pos"][0] - g["pos"][0], d["pos"][1] - g["pos"][1]))
                    if dist > self.revive_dist_px:
                        continue
                    if best_cos is None or cos > best_cos:
                        best_idx, best_cos = gi, cos
                if best_idx is not None:
                    g = self._graveyard[best_idx]
                    used_grave.add(best_idx)
                    self._make_room(protect_id, t)
                    nf = ObjectFile(g["id"], d["pos"], d["appearance"], d["area"],
                                    self.process_noise, self.meas_noise)
                    nf.hits = g["hits"]
                    nf.age = g["age"]
                    self.files.append(nf)
                    revived.append(nf.id)
                else:
                    remaining_d.append(j)
            if used_grave:
                self._graveyard = [g for gi, g in enumerate(self._graveyard) if gi not in used_grave]

        # ---- 新規作成（枚数の上限：max_filesがNoneなら従来どおり無制限）----------
        created = []
        evicted = []
        for j in remaining_d:
            d = detections[j]
            ev = self._make_room(protect_id, t)
            if ev is not None:
                evicted.append(ev)
            nf = ObjectFile(self._next_id, d["pos"], d["appearance"], d["area"],
                            self.process_noise, self.meas_noise)
            self._next_id += 1
            self.files.append(nf)
            created.append(nf.id)

        lost = [f.id for f in self.files if f.misses > self.max_missed]
        if lost:
            lost_set = set(lost)
            if self.revive_window_s is not None and t is not None:
                for f in self.files:
                    if f.id in lost_set:
                        self._graveyard.append({
                            "id": f.id, "pos": f.pos,
                            "appearance": np.array(f.appearance, dtype=np.float64, copy=True),
                            "area": f.area, "hits": f.hits, "age": f.age,
                            "t_lost": t,
                        })
            self.files = [f for f in self.files if f.id not in lost_set]

        lost = lost + evicted
        self.last_revived = revived

        return {"matched": matched, "created": created, "revived": revived,
                "lost": lost, "prediction_violations": violations}
