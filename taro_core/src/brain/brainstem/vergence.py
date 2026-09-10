"""輻輳反射：左右の周辺カメラ画像のズレ（両眼視差）から、目をどれだけ寄せるかを決める。

設計：`F/docs/設計_F2-15_焦点を合わせる（輻輳反射）.md`
斥候レポート：`F/docs/輻輳/斥候_眼球制御の現状_2026-08-28.md`

【何を足すか（設計より）】
    いまある反射   定位反射    左目の画像を見て「どっちを見るか」を決める   ← 触らない
         ↓
    新しく足す     輻輳反射    左右の画像のズレを見て「どれだけ寄せるか」を決める
         ↓
    左目の指令 ＝ 共同運動の成分 ＋ 輻輳の成分 ÷ 2
    右目の指令 ＝ 共同運動の成分 － 輻輳の成分 ÷ 2

`e_vor.py` の `override()` が「方策の眼球出力を捨てて置き換える」のに対し、こちらは
**加算**（additive）。定位反射（左右共通の指令）を上書きせず、左右差だけを足す。

【視差の測り方（設計2節）】
周辺カメラ（eye_left/eye_right、視野60度・128px）だけを使う。中心窩カメラ（視野15度）は
使わない——ずれが31.8度あるとき中心窩どうしには共通して写っているものが無く、相関が
取れないため。
    ① 左右の画像をグレースケール化
    ② 右画像を水平に -D〜+D 画素ずらしながら、左画像との一致度（正規化相互相関）を計算
    ③ 一致度が最大になるずらし量 d*[画素] を求める
    ④ 画素→角度に変換： 視差角[度] = d* × (視野角[度] / 画像幅[px])
判断（0.1秒＝10Hz）ごとに1回だけ計算する（潜時150〜200msと整合する）。
【この方式の位置づけ】人間のV1にある視差検出細胞は局所相関で視差を求めるが、本実装は
画像全体で1つの視差を求める簡略版【Tier2・原理レベル】。物体ごとの視差（奥行き地図）は
作らない＝立体視は実装しない（設計「今回は決めないこと」3）。

【制御則（設計3節・パラメータは文献値）】
    誤差 e = 測った視差角[度]（0が「左右が同じものを見ている」状態）
    目標輻輳角の更新: vergence_deg += clip(K * e, -v_max, +v_max) * dt
  K=3.0[1/秒]・v_max=20[度/秒]・潜時0.18[秒]・不感帯0.5[度]・範囲±35[度]。
  根拠は設計の表を参照（K・v_maxは文献調査②【原文確認】、不感帯は恣意的【Tier3】）。

【実装判断（設計に無い部分・止まらず自分で決めた点。作業記録にも明記）】
1. 潜時0.18秒の実現：設計は「判断2回分の遅延バッファ」とだけ指定。判断間隔0.1秒に
   対し2回分＝0.2秒（0.18秒に最も近い整数倍）としてバッファ長を決めた。
2. 目標輻輳角の積分（vergence_deg += ...*dt）をどこでやるか：update()のシグネチャに
   dt引数が無い（設計1節のコード例のとおり）ため、**update()の中でDT=0.1固定定数を
   使って積分する**（update自体が判断=0.1秒ごとにしか呼ばれない前提。e_toy_env.py側の
   配線で「両目の画像がある時だけ」呼ぶようにするのはこの前提を満たすため）。
   apply()は毎物理stepごとに呼ばれてよい位置フィードバックのみを行う
   （e_orienting_v2.pyのサッケード保持と同じ構造）。
3. apply()の位置フィードバックのゲイン：設計の表に指定が無い。同じ眼球アクチュエータ群
   に対して既に較正済みの `e_orienting_v2.EYE_FB_GAIN`（0.25、位置誤差[度]→筋の活性化、
   4度で飽和・注意[ARBITRARY]）を**そのまま流用**した。新しい定数は増やしていない。
   apply()は「左右の関節角の差(actual_left - actual_right)」を「目標のvergence_deg」に
   近づけるよう、左右へ符号を逆にして加算する（この式は共同運動の成分に影響しない。
   下記 _apply の実装参照）。
4. 左右のアクチュエータの符号とワールド座標での収束方向の対応は、斥候レポートが指摘する
   軸符号の逆転（`e_orienting_v2.py`が未修正）の影響を受ける。**この整合はONにしたときの
   受け入れ条件2〜4（次回・e_orienting_v2.pyの左右分離とセット）で検証する**。今回は
   既定OFFの配線までが範囲。
"""
import numpy as np

# ---- 視差測定パラメータ（設計2節）----------------------------------------
PERIPHERAL_FOVY_DEG = 60.0      # 周辺カメラの視野角[度]
MAX_SHIFT_PX = 70               # 探索範囲 ±D[画素]（設計：31.8度÷(60/128)≒68→安全側70）

# ---- 制御則パラメータ（設計3節の表。すべて文献値／恣意的の注記つき）--------
GAIN_K = 3.0             # [1/秒] Tier2（文献調査②、誤差31.8度で初速95度/秒→頭打ち20度/秒）
MAX_SPEED_DEG_S = 20.0   # [度/秒] Tier2（文献調査②【原文確認】ピーク速度4〜20度/秒）
LATENCY_SEC = 0.18       # [秒] Tier2（文献調査②【原文確認】150〜200ms）
DEADZONE_DEG = 0.5       # [度] Tier3・恣意的（ノイズで震えないための最小限）
VERGENCE_RANGE_DEG = 35.0  # [度] 関節ROM±45度から共同運動の余地を残して設定
DECISION_DT = 0.1        # [秒] 判断間隔（10Hz）。update()はこの間隔でしか呼ばれない前提

# apply()の位置フィードバックゲイン。e_orienting_v2.EYE_FB_GAINを流用（上記コメント3参照）。
_FB_GAIN = 0.25


def _grayscale(img):
    """(H, W, 3) uint8/float 画像を輝度(H, W) floatへ。"""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        return arr.mean(axis=-1)
    return arr


def _corr_at(left_gray, right_gray, d):
    """ずらし量 d のときの正規化相互相関と、重なった画素数を返す。"""
    h, w = left_gray.shape
    if d >= 0:
        l = left_gray[:, d:]
        r = right_gray[:, :w - d] if d > 0 else right_gray
    else:
        l = left_gray[:, :w + d]
        r = right_gray[:, -d:]
    if l.size == 0:
        return -np.inf, 0
    lf = l.ravel() - l.mean()
    rf = r.ravel() - r.mean()
    denom = float(np.linalg.norm(lf) * np.linalg.norm(rf))
    score = float(np.dot(lf, rf) / denom) if denom > 1e-8 else 0.0
    return score, int(l.shape[1])


def _measure_shift_coarse_to_fine(left_gray, right_gray, max_shift, coarse_step=4):
    """粗く探してから細かく合わせる（2026-08-28・A）。

    【なぜ要るか】板が顔から0.086mと近く、必要な視差は約68画素（画像幅128pxの53%）。
    1画素ずつ全域を探すと、重なりが小さいずらし量でも相関が偶然高く出て、
    測定値が探索範囲の端（±32.8度）に張り付いた（実測：F/logs/F2-15_診断/）。
    人間も視差が大きいと融像できず、まず**粗く寄せてから細かく合わせる**
    （近接性輻輳→融像性輻輳。文献調査②）。同じ二段構えにする。

    戻り値: (ずらし量d*, 相関スコア, 重なり画素数)
    """
    h, w = left_gray.shape
    max_shift = int(min(max_shift, w - 1))
    # ① 粗探索：coarse_step 画素おき
    best = (0, -np.inf, 0)
    for d in range(-max_shift, max_shift + 1, coarse_step):
        sc, ov = _corr_at(left_gray, right_gray, d)
        if sc > best[1]:
            best = (d, sc, ov)
    # ② 細探索：粗い答えの周り ±coarse_step を1画素刻みで
    lo = max(-max_shift, best[0] - coarse_step)
    hi = min(max_shift, best[0] + coarse_step)
    for d in range(lo, hi + 1):
        sc, ov = _corr_at(left_gray, right_gray, d)
        if sc > best[1]:
            best = (d, sc, ov)
    return best


def _measure_shift_px(left_gray, right_gray, max_shift):
    """右画像を -max_shift〜+max_shift 画素ずらし、正規化相互相関が最大になる
    ずらし量 d* を返す（設計2節②③）。

    【高速化・2026-09-02・F2-39】旧実装（下の _measure_shift_px_slow）は
    ずらし量ごとに全画素の平均・ノルム・内積を計算し直しており、走行時間の
    約45%を占めていた（cProfile実測：1歩64ms・F/logs/F2-39_速度/）。
    本実装は**同じ定義式**を、FFTの相互相関（全ずらし量の内積を一括）＋
    列和の累積（平均・ノルムをO(1)化）で計算する。**近似はしない**：
    数式は同一で、浮動小数の加算順序だけが変わる（best_dの一致は
    実画像・人工画像でテスト済み。2026-08-28の粗密2段階の失敗とは違い
    探索は全数のまま）。

    定義：d に対し比較するのは left[:, d:] と right[:, :w-d]（d>=0の場合）。
    つまり d* は「left[x] に一致する right の内容が right[x - d*] にある」量。
    d*>0 は「right の内容が left より右へ d* 画素ぶんずれている」ことを意味する
    （下の test_e_vergence.py で構成と対応づけて検証する）。
    """
    h, w = left_gray.shape
    max_shift = int(min(max_shift, w - 1))
    L = np.asarray(left_gray, dtype=np.float64)
    R = np.asarray(right_gray, dtype=np.float64)
    n_fft = 1
    while n_fft < 2 * w:
        n_fft *= 2
    fl = np.fft.rfft(L, n=n_fft, axis=1)
    fr = np.fft.rfft(R, n=n_fft, axis=1)
    # cc[d mod n_fft] = Σ_行 Σ_y L[:, y+d]・R[:, y]（ゼロ埋めなので巻き込みなし）
    cc = np.fft.irfft(fl * np.conj(fr), n=n_fft, axis=1).sum(axis=0)
    colL, colL2 = L.sum(axis=0), (L * L).sum(axis=0)
    colR, colR2 = R.sum(axis=0), (R * R).sum(axis=0)
    cL = np.concatenate(([0.0], np.cumsum(colL)))
    cL2 = np.concatenate(([0.0], np.cumsum(colL2)))
    cR = np.concatenate(([0.0], np.cumsum(colR)))
    cR2 = np.concatenate(([0.0], np.cumsum(colR2)))
    best_d = 0
    best_score = -np.inf
    for d in range(-max_shift, max_shift + 1):
        if d >= 0:
            n_cols = w - d
            sL, sL2 = cL[w] - cL[d], cL2[w] - cL2[d]
            sR, sR2 = cR[n_cols], cR2[n_cols]
        else:
            n_cols = w + d
            sL, sL2 = cL[n_cols], cL2[n_cols]
            sR, sR2 = cR[w] - cR[-d], cR2[w] - cR2[-d]
        n = h * n_cols
        if n <= 0:
            continue
        num = cc[d % n_fft] - sL * sR / n
        var_l = max(sL2 - sL * sL / n, 0.0)
        var_r = max(sR2 - sR * sR / n, 0.0)
        denom = float(np.sqrt(var_l * var_r))
        score = float(num / denom) if denom > 1e-8 else 0.0
        if score > best_score:
            best_score = score
            best_d = d
    return best_d


def _measure_shift_px_slow(left_gray, right_gray, max_shift):
    """旧実装（全数・逐次計算）。高速版との一致テスト専用に残す（2026-09-02）。"""
    h, w = left_gray.shape
    max_shift = int(min(max_shift, w - 1))
    best_d = 0
    best_score = -np.inf
    for d in range(-max_shift, max_shift + 1):
        if d >= 0:
            l = left_gray[:, d:]
            r = right_gray[:, :w - d] if d > 0 else right_gray
        else:
            l = left_gray[:, :w + d]
            r = right_gray[:, -d:]
        if l.size == 0:
            continue
        lf = l.ravel() - l.mean()
        rf = r.ravel() - r.mean()
        denom = float(np.linalg.norm(lf) * np.linalg.norm(rf))
        score = float(np.dot(lf, rf) / denom) if denom > 1e-8 else 0.0
        if score > best_score:
            best_score = score
            best_d = d
    return best_d


# 和(left+right)の+方向が世界座標で「寄り目」か「開散」かの符号。
# 【2026-08-28・実測で確定】走行もフィードバックも使わず、モデルを静止させたまま
#   左右の関節へ既知の角度を入れ、各目のカメラから板がどちらに見えるかを直接計算した
#   （板は実験と同じ 0.086m）。結果：
#       関節角(左,右)=( 0,  0)→和  0度：左目から板は+15.90度、右目から-15.90度（両目とも外れ）
#       関節角(左,右)=(-15,-15)→和-30度：両目ともほぼ0度（＝両目が板を捉える）★正解
#       関節角(左,右)=(-20,-20)→和-40度：行き過ぎ（左-6.60度／右+6.60度）
#   ＝「寄り目」は和のマイナス方向。目標 vergence_deg も同じ向きで持つので反転は不要。
#   （初版の -1.0 は目標と逆方向へ寄せていた。実測：目標-30度に対し和が+4.9度へ動いた）
VERG_SIGN = +1.0

# 測った視差角から目標輻輳角をどちら向きに積むかの符号。
# 【2026-08-28・実測で確定】板を0.30mに置き、駆動側（VERG_SIGN・左右分離）を
#   直したうえで走らせたところ、目標が上限+35度に張り付き、実際の和も+27.5度
#   まで追従した（到達率94%＝駆動は正しく効いている）。だが「寄り目」は和の
#   **マイナス**方向（VERG_SIGN のコメントの実測表を参照）なので、これは
#   **開散しきっていた**。＝視差から目標を積む向きが逆。
#   測定できた割合96%・相関0.79と測定自体は健全だったので、符号だけを反転する。
DISPARITY_SIGN = -1.0
# 視差測定の信頼度のしきい値（2026-08-28）。
#   MIN_CORR：正規化相互相関がこれ未満なら「測れなかった」とみなす
#   MIN_OVERLAP_FRAC：左右画像の重なりが画像幅のこの割合を切ったら同上
#   【Tier3・恣意的】人間の融像限界（パヌムの融合域）に対応する概念だが、
#   太郎の画像サイズに対する妥当値は実測で決めた。ONにしたときの
#   「測れた割合」を見て調整すること。
MIN_CORR = 0.30
MIN_OVERLAP_FRAC = 0.45

class VergenceReflex:
    """左右周辺カメラの視差から目標輻輳角を求め、左右の眼球アクチュエータへ
    加算で反映する（定位反射の共同運動成分は上書きしない）。
    """

    def __init__(self, model, data, gain=GAIN_K, max_speed_deg=MAX_SPEED_DEG_S,
                 latency_sec=LATENCY_SEC, deadzone_deg=DEADZONE_DEG,
                 vergence_range_deg=VERGENCE_RANGE_DEG, dt=DECISION_DT,
                 max_shift_px=MAX_SHIFT_PX, fb_gain=_FB_GAIN):
        self.model = model
        self.data = data
        self.gain = float(gain)
        self.max_speed_deg = float(max_speed_deg)
        self.latency_sec = float(latency_sec)
        self.deadzone_deg = float(deadzone_deg)
        self.range_deg = float(vergence_range_deg)
        self.dt = float(dt)
        self.max_shift_px = int(max_shift_px)
        self.fb_gain = float(fb_gain)

        # 潜時0.18秒 ≒ 判断2回分（0.1秒間隔）の遅延バッファで実現（設計コメント参照）。
        self._n_delay = max(1, int(round(self.latency_sec / self.dt)))
        self._delay_buf = []   # 直近の測定視差角[度]をFIFOで保持

        self.vergence_deg = 0.0     # 目標輻輳角（積分器の状態）
        self.last_disparity_deg = 0.0   # 直近に測った視差角（観察用）
        self.last_score = 0.0           # 直近の相関スコア（観察用）
        self.last_overlap_px = 0        # 直近の重なり画素数（観察用）
        self.last_measurable = False    # 測れたかどうか（観察用）
        self.n_actuator = int(model.nu)

        # 左右の水平アクチュエータ（act:left_eye_horizontal / act:right_eye_horizontal）
        # を名前で直接探す。e_orienting_v2.py の eye_idx["h"] は左右がまだ分離されて
        # いない（次回タスク）ため、ここでは独立に探す。
        self._left_aid = None
        self._right_aid = None
        self._left_qadr = None
        self._right_qadr = None
        for i in range(model.nu):
            name = model.actuator(i).name
            if "eye" not in name or "horizontal" not in name:
                continue
            jid = int(model.actuator_trnid[i, 0])
            if jid < 0:
                continue
            qadr = int(model.jnt_qposadr[jid])
            if "left" in name:
                self._left_aid = i
                self._left_qadr = qadr
            elif "right" in name:
                self._right_aid = i
                self._right_qadr = qadr

        import sys as _s, os as _o
        _core = _o.path.abspath(_o.path.join(
            _o.path.dirname(_o.path.abspath(__file__)), _o.pardir, _o.pardir,
            "taro_core", "src", "brain"))
        if _core not in _s.path:
            _s.path.insert(0, _core)
        from spinal_cord.cpg import write_joint_command
        self._write = write_joint_command

    def __deepcopy__(self, memo):
        """複製しても `self.data`（シミュレーションへの繋がり）は複製しない。

        理由は `taro_core/src/brain/midbrain/orienting.py` の同名メソッドと同じ。
        この反射は今のところ `run/trainer.py` の控えの対象に入っていないが、
        同じ形で data を握っているので、あらかじめ同じ守りを入れておく。
        """
        import copy as _copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "data":
                new.data = v
            else:
                setattr(new, k, _copy.deepcopy(v, memo))
        return new

    def reset(self):
        """前エピソードの状態を持ち越さない（e_orienting_v2.reset()と同じ流儀）。"""
        self._delay_buf = []
        self.vergence_deg = 0.0
        self.last_disparity_deg = 0.0

    def measure_disparity_deg(self, img_left, img_right):
        """左右画像から視差角[度]を測るだけ（積分はしない）。単体テスト用に公開。"""
        gl = _grayscale(img_left)
        gr = _grayscale(img_right)
        w = gl.shape[1]
        # 【2026-08-28】粗密2段階(_measure_shift_coarse_to_fine)を試したが、
        #   人工画像で真のずれを検出できず相関も0.03しか出なかった（実装の誤り）。
        #   元の全探索は単体テストで正しく検出できているため、そちらを使い、
        #   信頼度の判定だけを新たに足す。粗密化は原因を特定してから再挑戦する。
        d_star = _measure_shift_px(gl, gr, self.max_shift_px)
        score, overlap = _corr_at(gl, gr, int(d_star))
        deg_per_px = PERIPHERAL_FOVY_DEG / float(w)
        # 【2026-08-28】測定の信頼度。重なりが狭いと相関が偶然高く出るため、
        #   ①相関がMIN_CORR未満 ②重なりが画像幅のMIN_OVERLAP_FRAC未満
        #   のどちらかなら「測れなかった」として0を返す（動かさない）。
        #   人間も視差が大きすぎると融像できず、無理に寄せない。
        self.last_score = float(score)
        self.last_overlap_px = int(overlap)
        if score < MIN_CORR or overlap < MIN_OVERLAP_FRAC * w:
            self.last_measurable = False
            return 0.0
        self.last_measurable = True
        return float(d_star) * deg_per_px

    def update(self, img_left, img_right):
        """左右の周辺カメラ画像からずれを測り、目標輻輳角(vergence_deg)を更新する。

        判断（0.1秒＝10Hz）ごとに1回だけ呼ばれる前提（e_toy_env.py側で両目の画像が
        新しく描画された時だけ呼ぶ）。dt引数を取らない設計のシグネチャに合わせ、
        積分にはこのクラスのDECISION_DT（既定0.1秒）を使う（上記クラスdocstring
        「実装判断」2参照）。
        """
        e = self.measure_disparity_deg(img_left, img_right)
        self.last_disparity_deg = e

        # 潜時：直近 n_delay 回ぶんの測定をFIFOに積み、一番古い値を使う。
        self._delay_buf.append(e)
        if len(self._delay_buf) > self._n_delay:
            self._delay_buf.pop(0)
        if len(self._delay_buf) < self._n_delay:
            # まだ潜時ぶんのデータが溜まっていない（起動直後）。何もしない。
            return
        e_delayed = self._delay_buf[0]

        if abs(e_delayed) < self.deadzone_deg:
            rate = 0.0
        else:
            rate = float(np.clip(DISPARITY_SIGN * self.gain * e_delayed,
                                 -self.max_speed_deg, self.max_speed_deg))
        self.vergence_deg = float(np.clip(
            self.vergence_deg + rate * self.dt, -self.range_deg, self.range_deg))

    def apply(self, action):
        """actionへ左右差を加算して返す（上書きではなく加算）。

        左目の指令 ＝ 共同運動の成分 ＋ 輻輳の成分÷2
        右目の指令 ＝ 共同運動の成分 － 輻輳の成分÷2
        （設計「1. 全体の構成」）。ここでは「左右の関節角の差
        (actual_left - actual_right) を vergence_deg に近づける」P制御として実装する。
        この誤差は左右の和(共同運動の成分)には影響しない（クラスdocstring「実装判断」3
        の式変形を参照）ので、定位反射の出力を壊さない。
        """
        if self._left_aid is None or self._right_aid is None or self.data is None:
            return action
        out = np.array(action, dtype=float).copy()
        actual_left = float(self.data.qpos[self._left_qadr])
        actual_right = float(self.data.qpos[self._right_qadr])
        # 【重大な訂正・2026-08-28】左右の眼球関節は回転軸の符号が逆
        #   （MIMo/mimoEnv/assets/mimo/MIMo_modelv2.xml:334 axis="0 0 1" と
        #     :344 axis="0 0 -1"）。そのため世界座標では
        #       共同運動（両目が同じ方向を向く） ＝ 関節角の【差】(left - right)
        #       輻輳・開散（両目が逆方向へ動く） ＝ 関節角の【和】(left + right)
        #   になる。初版は差を輻輳角として制御していたため、輻輳ではなく
        #   **共同運動を35度ずらしていた**（実測：ON時に差が29度で固定され、
        #   和は動かず、両目の中心窩が揃わないまま。
        #   F/logs/F2-15_診断/ の輻輳.csv、図＝F/logs/F2-15_輻輳ON/図_輻輳のONとOFF.png）。
        #   よって和を制御し、左右へ**同符号**の指令を書く（軸が逆なので
        #   同じ値を書けば世界では逆向きに動く＝これが輻輳）。
        actual_verg_deg = float(np.degrees(actual_left + actual_right))
        err = self.vergence_deg - actual_verg_deg
        cmd = float(np.clip(self.fb_gain * err, -1.0, 1.0))
        cmd = VERG_SIGN * cmd
        # 【2026-08-28・左右対称に戻した】初版は「右目だけ」を動かしていた。
        #   当時は定位反射が左右へ同じ値を書いており、それが世界座標では
        #   共同運動ではなく**輻輳**を動かしていたため（軸符号が逆）、輻輳を
        #   左右対称に足すと定位反射と綱引きになり、目標30度に対して実測で
        #   10度しか動かなかった。定位反射を左右分離して差（共同運動）だけを
        #   制御するよう直した（e_orienting_v2._write_eye_h）ので、こちらは
        #   左右へ**同符号**で書けば和（輻輳）だけが動き、差には影響しない。
        #   ＝人間と同じ左右対称の配り方に戻せた（Hering の法則）。
        for aid in (self._left_aid, self._right_aid):
            self._write(out, aid, cmd, self.n_actuator,
                        co_activation=0.0, additive=True)
        return out
