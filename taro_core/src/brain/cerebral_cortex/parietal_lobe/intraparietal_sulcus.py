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

    def __init__(self, id_, pos, appearance, area, process_noise, meas_noise, t=None, emb=None):
        self.id = id_
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 30.0
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * meas_noise
        self.appearance = np.asarray(appearance, dtype=np.float64)
        self.area = float(area)
        # 【なぜ、2026-09-09・予測して確かめる検出「追記3」1節】appearance
        #   （DINOv2、384次元・見回りの粗い見た目、EMA更新）とは別に、MobileSAMの
        #   画像埋め込み由来の256次元ベクトルを持つ。確認(confirm)の「本当に
        #   その物か」の照合に使う（見た目の同一性、Xu & Carey 1996[Tier1]）。
        #   emb=None（既定）＝従来どおり照合しない・生成しない検出からは
        #   常にNoneのまま（既定不変）。
        self.emb = None if emb is None else np.asarray(emb, dtype=np.float64)
        self.age = 0             # 何コマ存在するか
        self.hits = 1            # 対応がついた回数
        self.misses = 0          # 連続して対応がつかなかったコマ数
        self.since_seen = 0      # 直近で対応がついてから何コマ経ったか
        # 【2026-09-09・仕様_物体ファイルの重複をなくす「後半」1節】最後に一致した
        #   （＝実際に見えた）sim時刻。coast_max_s の「最後に一致した時刻から
        #   何秒まで等速で延長するか」の起点。coast_max_s=None（既定）では
        #   一度も参照されない＝既定不変。
        self.last_seen_t = t

    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    def predict(self, t=None, coast_max_s=None, coast_min_speed_px_s=None, frame_dt_s=1.0):
        """coast_max_s（【2026-09-09・仕様_物体ファイルの重複をなくす「後半」1節】）：
        Noneなら従来どおり永遠に等速（既定不変）。秒数を渡すと、`misses>0`（隠れて
        いる）かつ「最後に一致した時刻(last_seen_t)からt秒」がcoast_max_sを超えた
        カードは、この呼び出し以降 vx=vy=0 にして位置を止める（Pの増加だけ続く）。
        文献（von Hofsten & Rosander、Bremner 2005）：可視区間の速度をそのまま
        延長して再出現を予期し、減速は支持されない。延長の上限は規格（1秒）。

        coast_min_speed_px_s（【2026-09-09・追記「直し」2節】）：Noneなら従来どおり
        （既定不変）。値を渡すと、`misses>0`のカードは、カルマン状態の速度
        `hypot(vx,vy)`を`frame_dt_s`（1コマ何秒か）でpx/秒に換算し、この閾値未満
        （＝静止物のマスク重心の揺れが生んだ見せかけの速度）なら vx=vy=0 にして
        位置を保持する（coast_max_sの秒数内でも、動いていなければ延長しない）。
        文献：静止した物はその場に留まる（Baillargeon 1985）、動く物だけ延長
        （von Hofsten & Rosander）。F2-105本番の実測（追記「実測」）：確認は0.2秒
        ごとで、マスク重心の数画素の揺れから15〜25px/秒相当の見せかけの速度が
        推定され、静止物なのに等速延長で流れていた。"""
        if self.misses > 0:
            if coast_min_speed_px_s is not None:
                speed_px_s = float(np.hypot(self.x[2], self.x[3])) / frame_dt_s
                if speed_px_s < coast_min_speed_px_s:
                    self.x[2] = 0.0
                    self.x[3] = 0.0
            if coast_max_s is not None:
                if t is None:
                    # 【仕様「後半」1節】「tが無ければ検出コマ数×interval_s相当で
                    #   近似しない」＝coast_max_sを使うなら呼び出し元は必ずtを渡す。
                    raise ValueError("coast_max_s を使うときは t が必須")
                if (self.last_seen_t is not None
                        and (t - self.last_seen_t) > coast_max_s):
                    self.x[2] = 0.0
                    self.x[3] = 0.0
        elif coast_max_s is not None and t is None:
            # 【既定不変】misses==0でも従来どおりtの必須チェックだけは行う
            #   （coast_max_sを渡すならtも渡す、という契約自体は変えない）。
            raise ValueError("coast_max_s を使うときは t が必須")
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

    def update(self, pos, appearance, area, appearance_lr=0.3, t=None, emb=None):
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
        if emb is not None:
            # 【なぜ、2026-09-09・予測して確かめる検出「追記3」1節】emb=None
            #   （既定・確認(confirm)の更新）では一切触らない＝既定不変。
            #   見回り(scan)がembを持つ検出で一致したときだけ単純に置き換える
            #   （appearanceと違いEMAにしない。1枚の埋め込みが1回のforwardで
            #   決定的に決まる値であり、時間平滑化する理由が無いため）。
            self.emb = np.asarray(emb, dtype=np.float64)
        self.hits += 1
        self.misses = 0
        self.since_seen = 0
        if t is not None:
            # 【2026-09-09・重複をなくす「後半」1節】実際に一致した＝見えた時刻。
            self.last_seen_t = t
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
        coast_max_s: 【2026-09-09・仕様_物体ファイルの重複をなくす「後半」1節】
            隠れた物が等速で進む延長の上限（秒）。None＝従来どおり永遠に等速
            （既定不変）。超えたら位置を止めてPの増加だけ続ける。
        uncertainty_penalty: 【同「後半」2節】照合コストに不確かさの大きさ
            （0.5*log(det(S)/det(R))）を足す係数。0.0＝従来どおり（既定不変）。
            不確かな（長く一致していない）カードほど選ばれやすくなる計算間違いの修正。
        exclusive_dist_px: 【同「後半」3節】固体性：新規作成（復活を含む）の前に、
            生きているカードのうちこの距離(px)以内かつ見た目が近いものがあれば
            作らず、その検出をそのカードに吸収させる。None＝従来どおり（既定不変）。
        exclusive_cos: exclusive_dist_px と併用する見た目のコサイン類似度の下限。
            【2026-09-09・追記「直し」1節】Noneを渡すと見た目の条件を外し、位置
            だけで吸収を決める（人間側：同一性は位置優先＝Xu & Carey 1996）。
            既定0.7＝従来どおり見た目も条件にする（既定不変）。
        exclusive_scale_by_size: 【同「追記」1節】Trueなら、吸収を許す距離を
            `max(exclusive_dist_px, 0.5*sqrt(f.area)*224)`（f＝比較している既存
            カードの面積。物の半径の近似）に広げる。既定False＝従来どおり
            exclusive_dist_pxそのまま（既定不変）。
        coast_min_speed_px_s: ObjectFile.predict()docstring参照。None＝従来どおり
            （既定不変）。
        frame_dt_s: 1コマ（predict()の1呼び出し）が何秒に相当するか。
            coast_min_speed_px_sをpx/秒に換算するために使う。既定1.0。
        confirm_gate: 【2026-09-09・仕様_予測して確かめる検出「追記4」1節】
            `confirm_update()` が結果を採用するかどうかの位置ゲートの種類。
            "mahal"（既定・既定不変）＝従来どおりマハラノビス距離
            （`f.mahalanobis(pos) <= mahal_gate`）。一致が続いてPが小さくなると
            マスク重心の数画素の揺れでゲートを超えてしまう問題がF2-104pre実測
            （追記4「実測」）で見つかった。"size"＝物の大きさで決める
            （`dist(重心, 予測位置) <= sqrt(area)*224*0.75 + 8px`。確認のマスクは
            頼んだカードの予測位置に打った点から出たものなので、位置の同一性は
            構造上ほぼ担保される＝人間側は見ている物が数画素ずれても同じ物と
            扱う、位置の許容は物の大きさに比例という近似）。"mahal"以外・"size"
            以外の値は仕様に無いためValueErrorにする。
    """

    def __init__(self, appearance_weight=0.4, mahal_gate=MAHAL_GATE95, max_missed=20,
                 reappear_gap=4, process_noise=4.0, meas_noise=6.0, gate=1.0,
                 revive_window_s=None, revive_cos=0.8, revive_dist_px=40.0, max_files=None,
                 coast_max_s=None, uncertainty_penalty=0.0, exclusive_dist_px=None,
                 exclusive_cos=0.7, confirm_gate="mahal", exclusive_scale_by_size=False,
                 coast_min_speed_px_s=None, frame_dt_s=1.0):
        if confirm_gate not in ("mahal", "size"):
            raise ValueError("confirm_gate は 'mahal' か 'size' のどちらか（%r）" % (confirm_gate,))
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
        self.coast_max_s = None if coast_max_s is None else float(coast_max_s)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.exclusive_dist_px = None if exclusive_dist_px is None else float(exclusive_dist_px)
        # 【2026-09-09・追記「直し」1節】Noneなら見た目の条件を外す（既定0.7＝
        #   従来どおり見た目も使う＝既定不変）。
        self.exclusive_cos = None if exclusive_cos is None else float(exclusive_cos)
        self.exclusive_scale_by_size = bool(exclusive_scale_by_size)
        self.coast_min_speed_px_s = (None if coast_min_speed_px_s is None
                                      else float(coast_min_speed_px_s))
        self.frame_dt_s = float(frame_dt_s)
        self.confirm_gate = confirm_gate
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

    def predict_only(self, t=None):
        """【2026-09-09・仕様_予測して確かめる検出】確認（confirm）専用：対応づけは
        せず、各ファイルの予測位置だけを1コマぶん進める。`step()` 冒頭の2行
        （`f.predict()`・`f.age += 1`）と同じ処理を切り出しただけで、`step()`
        自体は無改修（既存の呼び出し元の挙動は不変）。呼び出し元（object_files.py）
        が確認用の点プロンプトを作る前に、いまの予測位置 `f.pos` を得るために使う。
        `t` は coast_max_s（重複をなくす「後半」1節）を使うときに必要（Noneなら
        従来どおり coast_max_s も渡らないので使われない＝既定不変）。"""
        for f in self.files:
            f.predict(t=t, coast_max_s=self.coast_max_s,
                      coast_min_speed_px_s=self.coast_min_speed_px_s,
                      frame_dt_s=self.frame_dt_s)
            f.age += 1

    def confirm_update(self, results, t=None):
        """【2026-09-09・仕様_予測して確かめる検出_2026-09-09「追記2」1節】
        確認（confirm）専用の更新。`predict_only()` で既に1コマぶん進めた
        予測位置を、点プロンプトの確認結果と突き合わせる。`step()` と違い
        ハンガリアン法は使わない —— 各結果は呼び出し元（object_files.py）が
        どの `file_id` への問い合わせだったかを既に知っているので、1対1で
        そのカードにだけ結ぶ。**確認から新しいカードは作らない**（作成は
        見回り・横取りの全体切り出し＝`step()` だけ）。

        【なぜ、2026-09-09】F2-101pre2/F2-102pre の実測：確認コマで4枚の
        カードが全部同じボールを検出し、`step()`（ハンガリアン法）に渡すと
        1つの検出を1枚のカードにしか対応づけられないため、残り3枚が
        「対応がつかなかった」として新規カード作成に回っていた（確認モードで
        作成25／消滅25、寿命の中央値4.65秒）。頼んだカードにだけ結べば、
        この事故は起きない。

        Args:
            results: [{"file_id": id, "matched": bool,
                       "pos": (x,y)（matched時のみ）, "area": float（matched時
                       のみ）}, ...]（呼び出し元が「同じ物を複数のカードが確認
                       したら1枚だけ」の重複整理を済ませたあとの、カードごとの
                       最終判定。file_idはself.filesのidと対応する）。
            t: sim秒（last_seen_tの更新用。coast_max_sを使うときの起点）。
        Returns:
            `step()` と同じ形の辞書。created/revived/absorbedは常に空リスト
            （確認からは新規作成しない）。matchedのdet_idxはNone固定
            （呼び出し元はfile_idしか読まないため）。加えて
            `gate_rejected`（【2026-09-09・追記4】matched=Trueで来たが位置の
            ゲートで落ちたfile_idのリスト。呼び出し元がCSVの
            `conf_reject`="gate"を書くために使う。既定"mahal"のときも従来どおり
            返る＝新しいキーが増えるだけで既定不変）。
        """
        by_id = {f.id: f for f in self.files}
        matched = []
        gate_rejected = []
        for r in results:
            f = by_id.get(r.get("file_id"))
            if f is None:
                continue
            if r.get("matched"):
                # 【追記2「直し」1節→追記4「直し」1節】自分のカードのゲート内か
                #   だけを見る。ハンガリアン法のコスト最小化とは別の、
                #   「この点はこのカードのものか」という単純な足切り。
                #   confirm_gate="mahal"（既定）は従来どおりマハラノビス距離。
                #   "size"は物の大きさ基準（追記4「直し」1節）：確認のマスクは
                #   頼んだカードの予測位置に打った点から出たものなので、位置の
                #   同一性は構造上ほぼ担保される＝人間側は数画素のずれを同じ物
                #   として扱う（位置の許容は物の大きさに比例）。
                if self.confirm_gate == "size":
                    dx = r["pos"][0] - f.pos[0]
                    dy = r["pos"][1] - f.pos[1]
                    dist = float(np.hypot(dx, dy))
                    # 【仕様に無かった判断】「物の大きさ」に使うareaは、確認で
                    #   実際に測れたr["area"]ではなくカード自身のf.area
                    #   （＝confirm_pointsに期待面積として渡した値）にした。
                    #   r["area"]は測定ノイズを含み、ゲートの基準がコマごとに
                    #   揺れるのを避けるため。差があるなら作業記録に書く。
                    allowed = float(np.sqrt(max(f.area, 0.0))) * 224.0 * 0.75 + 8.0
                    gate_ok = dist <= allowed
                else:
                    gate_ok = f.mahalanobis(r["pos"]) <= self.mahal_gate
                if gate_ok:
                    residual = f.update(r["pos"], f.appearance, r["area"], t=t)
                    matched.append((f.id, None, residual))
                    continue
                gate_rejected.append(f.id)
            f.misses += 1
            f.since_seen += 1

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

        self.last_revived = []
        return {"matched": matched, "created": [], "revived": [], "lost": lost,
                "prediction_violations": [], "absorbed": [], "gate_rejected": gate_rejected}

    def step(self, detections, t=None, protect_id=None, skip_predict=False):
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
                   予測とのずれが mahal_gate を超えたもの＝見た目で救われた組）,
                  absorbed=[(det_idx, file_id), ...]（【2026-09-09・重複をなくす
                  「後半」3節】固体性：既存カードに吸収された検出。exclusive_dist_px
                  がNone（既定）なら常に空リスト＝既定不変）
        `skip_predict`（【2026-09-09・仕様_予測して確かめる検出「追記」3節】）：
            Trueなら冒頭の predict() を飛ばす（呼び出し元が `predict_only()` で
            既に1コマぶん進めている確認(confirm)コマ用。1コマにpredict()が
            2回進むのを防ぐ）。既定False＝従来どおり（既定不変）。
        """
        if not skip_predict:
            for f in self.files:
                f.predict(t=t, coast_max_s=self.coast_max_s,
                          coast_min_speed_px_s=self.coast_min_speed_px_s,
                          frame_dt_s=self.frame_dt_s)
                f.age += 1

        n_f, n_d = len(self.files), len(detections)
        matched, unmatched_f, unmatched_d = [], list(range(n_f)), list(range(n_d))
        violations = []

        if n_f and n_d:
            cost = np.zeros((n_f, n_d))
            for i, f in enumerate(self.files):
                for j, d in enumerate(detections):
                    mahal = f.mahalanobis(d["pos"])
                    if self.uncertainty_penalty:
                        # 【2026-09-09・重複をなくす「後半」2節】採用済みのベイズ
                        #   模型の計算間違いの修正：不確かさの大きさ(log|S|)の項を
                        #   足す。確率として正しい形（S=innovation_cov, Rは観測雑音）。
                        #   uncertainty_penalty=0.0（既定）なら従来と1ビットも
                        #   変わらない（既定不変）。
                        S = f.innovation_cov()
                        penalty = self.uncertainty_penalty * 0.5 * float(
                            np.log(np.linalg.det(S) / np.linalg.det(f.R)))
                    else:
                        penalty = 0.0
                    pos_cost = (mahal + penalty) / self.mahal_gate  # 上限で切り詰めない
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
                    residual = f.update(d["pos"], d["appearance"], d["area"], t=t, emb=d.get("emb"))
                    matched.append((f.id, j, residual))
                    if gap >= self.reappear_gap and residual > self.mahal_gate:
                        violations.append((f.id, residual, gap))
                    used_f.add(i); used_d.add(j)
            unmatched_f = [i for i in range(n_f) if i not in used_f]
            unmatched_d = [j for j in range(n_d) if j not in used_d]

        for i in unmatched_f:
            self.files[i].misses += 1
            self.files[i].since_seen += 1

        # ---- 固体性：既存カードに近い検出は新規作成せず吸収する（重複をなくす
        #      「後半」3節。exclusive_dist_pxがNoneなら従来どおり何もしない＝
        #      既定不変）。新規作成（revive含む）の前に行う。 ---------------------
        absorbed = []
        if self.exclusive_dist_px is not None and unmatched_d:
            still_remaining = []
            for j in unmatched_d:
                d = detections[j]
                dv = np.asarray(d["appearance"], dtype=np.float64)
                dv_n = dv / (np.linalg.norm(dv) + 1e-9)
                best_f, best_dist = None, None
                for f in self.files:
                    dist = float(np.hypot(d["pos"][0] - f.pos[0], d["pos"][1] - f.pos[1]))
                    thresh = self.exclusive_dist_px
                    if self.exclusive_scale_by_size:
                        # 【2026-09-09・追記「直し」1節】既存カード(f)の面積から
                        #   物の半径を近似し、吸収を許す距離をそれに合わせて広げる
                        #   （固体性の範囲は物の大きさに比例させる、という近似）。
                        radius_px = 0.5 * float(np.sqrt(max(f.area, 0.0))) * 224.0
                        thresh = max(thresh, radius_px)
                    if dist > thresh:
                        continue
                    if self.exclusive_cos is not None:
                        # 【2026-09-09・追記「直し」1節】exclusive_cos=None（追記の
                        #   短い走行）では見た目の条件を外し、位置だけで決める
                        #   （同一性は位置優先＝Xu & Carey 1996）。
                        fv = f.appearance / (np.linalg.norm(f.appearance) + 1e-9)
                        cos = float(dv_n @ fv)
                        if cos < self.exclusive_cos:
                            continue
                    if best_dist is None or dist < best_dist:
                        best_f, best_dist = f, dist
                if best_f is not None:
                    best_f.update(d["pos"], d["appearance"], d["area"], t=t, emb=d.get("emb"))
                    absorbed.append((j, best_f.id))
                else:
                    still_remaining.append(j)
            unmatched_d = still_remaining

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
                                    self.process_noise, self.meas_noise, t=t, emb=d.get("emb"))
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
                            self.process_noise, self.meas_noise, t=t, emb=d.get("emb"))
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
                "lost": lost, "prediction_violations": violations, "absorbed": absorbed}
