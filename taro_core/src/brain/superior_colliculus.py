"""上丘（superior colliculus）— 視野の地図と、そこからサッケードを決める仕組み。

【何をする器官か】中脳にある。「動くもの・目立つものの方へ目と首を向ける」定位反射を作る。
皮質を経由しない皮質下の経路で、★新生児の視覚定位はこの系が主役
（生後0〜2ヶ月は皮質下反射系が優位、2ヶ月以降に皮質系が優勢化。Johnson 1990
 J Cogn Neurosci 2:81-95 [PMID 23972019]、Bronson 1974 の再検討）。

★【2026-07-27 新設】それまで「画像の中心ほど大きい重みを掛ける」という近似で
中心視野の優位を表していたが、重みの形（ガウス窓）も強さ（σ=0.32）も**恣意的**だった。
ユーザーの指摘「その数値を変えるのっていうのは恣意的じゃない？」を受けて調べた結果、
上丘の地図の歪み方には**実測に基づく数式と定数がある**と分かったので置き換える。

═══════════════════════════════════════════════════════════════════
【上丘の地図は歪んでいる】中心視野が広い面積を占める
═══════════════════════════════════════════════════════════════════

視野の座標 (R, Φ)  R＝中心からの角度[度]、Φ＝方位角[rad]
        ↓
上丘の座標 (u, v)  u＝吻側-尾側軸（サッケードの大きさ）、v＝内側-外側軸（向き）

    u = Bu · ln( √(R² + 2AR·cosΦ + A²) / A )     [mm]
    v = Bv · arctan( R·sinΦ / (R·cosΦ + A) )      [mm]

逆変換：
    x = A·(exp(u/Bu)·cos(v/Bv) − 1)      [度]
    y = A·exp(u/Bu)·sin(v/Bv)            [度]

定数（サル。[Tier2]）：
    A  = 3.0 度        中心視野のオフセット
    Bu = 1.4 mm        振幅の軸のスケール
    Bv = 1.8 mm/rad    方向の軸のスケール

出典：Ottes FP, Van Gisbergen JAM, Eggermont JJ (1986)
      "Visuomotor fields of the superior colliculus: a quantitative model"
      Vision Research 26:857-873 [PMID 3750869]
⚠️原論文は有料で本文を確認できていない。定数は複数の独立したモデリング論文
  （PMC5506246 等）が引用する値が一致することで確認した [Tier2]。

ここから「中心ほど広い」度合い（マグニフィケーションファクター）が出る：
    M(R) = du/dR = Bu / (R + A)      [mm/度]
    面積で見ると ∝ (Bu·Bv) / (R + A)²

★これは V1 の古典的な複素対数マグニフィケーション M(E)=M₀/(E+E₂) と**同じ関数形**。

═══════════════════════════════════════════════════════════════════
★【なぜ「重みを掛ける」ではダメなのか】
═══════════════════════════════════════════════════════════════════

①**関数の形が違う**
    ガウス窓        exp(-R²/2σ²)   … 裾が軽い。数σ先でほぼゼロ＝周辺を捨てる
    マグニフィケーション 1/(R+A)ⁿ   … べき乗則。裾が重く、遠くの刺激にも応答が残る
  実測でも、σを大きくするほど対象の位置を正しく出せた（＝裾が重い形に近づいていた）。
  σをどう調整しても、形が違う以上たどり着けない。

②★**平均を取る順序が結果を変える**（Jensen の不等式）
    平らな画像で重心を取る          ≠  歪んだ地図で重心を取ってから逆変換する
  非線形な変換 f では mean(f(x)) ≠ f(mean(x))。
  Goossens HHLM, Van Opstal AJ (2012) "Order of operations for decoding superior
  colliculus activity for saccade generation" J Neurophysiol [DOI 10.1152/jn.00265.2011]
  がこの2つを直接比べ、**サッケードの大きさと向きが変わる**ことを示している。
  実際の上丘は後者（歪んだ地図の上で重心＝population vector）。

  ★つまり「対象が視野の端にあるほど、出力が中心寄りに潰れる」のは
    **バグではなく上丘の本来の性質**。消すのではなく、実測値 A=3° に基づいて
    潰れの強さを決めるのが正しい。

【重心＝population vector で決まる根拠】
  Lee C, Rohrer WH, Sparks DL (1988) "Population coding of saccadic eye movements
  by neurons in the superior colliculus" Nature 332:357-360 [PMID 3352733]
  ＝一部を薬理的に止めても、残った集団の重心の向きにサッケードが出る（直接実験）。

═══════════════════════════════════════════════════════════════════
⚠️【簡略化・逸脱】
═══════════════════════════════════════════════════════════════════
・定数 A=3° は**サル**の値。★ヒト、まして新生児のデータは存在しない
  （2026-07-27 の調査で確認）。上丘の地図が生後いつ成体並みになるかも不明。
・実際の上丘は**左右の半球に分かれている**（左視野→右上丘）。ここでは
  1枚の地図として扱う[簡略化]。定位の向きを決めるだけなら影響は小さいはず。
・深さ方向（表層＝視覚／中間層＝運動）の区別をしていない[簡略化]。
"""
import numpy as np

# ---- Ottes et al. (1986) の定数 [Tier2] -----------------------------------
OTTES_A = 3.0        # 中心視野のオフセット [度]
OTTES_BU = 1.4       # 振幅の軸のスケール [mm]
OTTES_BV = 1.8       # 方向の軸のスケール [mm/rad]

# ---- 受容野（1つのニューロンが担当する視野の広さ）[Tier2] -----------------
# 上丘のニューロンは1画素ではなく**視野の広い範囲**を担当する。
#   浅層で 2〜20度。中心視野の近くで小さく、周辺ほど大きい
#   （Goldberg & Wurtz 1972、Cynader & Berman 1972。中心窩近傍で1度未満、
#     周辺では視野の1象限に達する）
#
# ★これを「上丘の組織の上の長さ」に直すと、ほぼ一定になる：
#     中心 R= 0度   2度 × M(0)=0.467 mm/度   = 0.93 mm
#     周辺 R=40度  20度 × M(40)=0.0326 mm/度 = 0.65 mm
#   ＝**上丘の上では一様な広がり**。だから上丘の地図に写してから
#     一様にぼかすだけで、視野角で見たときの「中心は細かく周辺はざっくり」が出る。
#
# 【なぜ要るか・2026-07-27】この段が無いと、大きな対象の左右の縁が
#   鋭く分離したまま競合に入り、**片方の縁だけが勝って定位の向きが逆になる**
#   （実測：見かけ26度の対象で符号が4/6しか合わない。7度なら6/6）。
#   人間は単一の対象なら縁の情報を統合して正確に中心へ向く
#   （Kilpeläinen & Georgeson 2018 Sci Rep、幅4度の四角形で偏差0.57度以内）。
#   大きい対象で平均化が弱まること自体は人間にもあるが
#   （van der Stigchel et al. 2012 Vision Res 62:108-115）、
#   **対象が存在しない逆方向へ飛ぶことは人間では報告がない**＝実装の欠陥。
RF_DIAMETER_MM = 0.8              # 受容野の直径 [mm]（上の 0.65〜0.93 の中間）
RF_SIGMA_MM = RF_DIAMETER_MM / 2.355   # 半値全幅 → ガウスの標準偏差


def visual_to_collicular(ecc_deg, azim_rad, a=OTTES_A, bu=OTTES_BU, bv=OTTES_BV):
    """視野の位置 → 上丘の位置。

    Args:
        ecc_deg: 中心からの角度 R [度]（0以上）
        azim_rad: 方位角 Φ [rad]（0が右、π/2が上）
        a, bu, bv: Ottes の定数

    Returns:
        (u, v): 上丘の座標 [mm]
    """
    R = np.asarray(ecc_deg, dtype=float)
    P = np.asarray(azim_rad, dtype=float)
    inner = np.sqrt(R ** 2 + 2.0 * a * R * np.cos(P) + a ** 2)
    u = bu * np.log(np.maximum(inner, 1e-12) / a)
    v = bv * np.arctan2(R * np.sin(P), R * np.cos(P) + a)
    return u, v


def collicular_to_visual(u, v, a=OTTES_A, bu=OTTES_BU, bv=OTTES_BV):
    """上丘の位置 → 視野の位置（逆変換）。

    Returns:
        (x, y): 視野の座標 [度]（xが右、yが上）
    """
    e = np.exp(np.asarray(u, dtype=float) / bu)
    ang = np.asarray(v, dtype=float) / bv
    x = a * (e * np.cos(ang) - 1.0)
    y = a * e * np.sin(ang)
    return x, y


def magnification(ecc_deg, a=OTTES_A, bu=OTTES_BU):
    """マグニフィケーションファクター M(R) = Bu/(R+A) [mm/度]。

    「視野の1度が上丘で何mmを占めるか」。中心ほど大きい。
    ★参考：中心（R=0）で Bu/A ≈ 0.47 mm/度。無限大には発散しない。
    """
    return bu / (np.asarray(ecc_deg, dtype=float) + a)


class CollicularMap:
    """画像の各画素と上丘の座標を対応づける地図。

    画像は「太郎の眼球カメラが描いた正方形の画像」を想定する。
    透視投影なので、画素の位置から視野角を出すには arctan を使う
    （画像の端は中心より1画素あたりの角度が小さい）。

    使い方:
        smap = CollicularMap(width=128, height=128, fovy_deg=60.0)
        x, y = smap.population_vector(activity)   # 視野の座標[度]
        h, v = smap.direction(activity)           # -1〜1 に正規化した向き
    """

    def __init__(self, width, height, fovy_deg,
                 a=OTTES_A, bu=OTTES_BU, bv=OTTES_BV, nu=None, nv=None):
        self.w = int(width)
        self.h = int(height)
        self.fovy = float(fovy_deg)
        self.half_fov = self.fovy / 2.0
        self.a, self.bu, self.bv = float(a), float(bu), float(bv)

        # 画素の中心を -1〜1 に正規化（右が正、上が正）
        cx = (self.w - 1) / 2.0
        cy = (self.h - 1) / 2.0
        gx = (np.arange(self.w) - cx) / cx
        gy = -(np.arange(self.h) - cy) / cy      # 画像のyは下向きなので反転
        GX, GY = np.meshgrid(gx, gy)

        # 透視投影：正規化座標 → 視野角[度]
        t = np.tan(np.radians(self.half_fov))
        self.ang_x = np.degrees(np.arctan(GX * t))
        self.ang_y = np.degrees(np.arctan(GY * t))

        # 視野角 → 極座標 → 上丘座標
        self.ecc = np.sqrt(self.ang_x ** 2 + self.ang_y ** 2)
        self.azim = np.arctan2(self.ang_y, self.ang_x)
        # ★左右の上丘は**それぞれ反対側の視野**を担当する。Ottes の式は片側
        #   （方位角 |Φ| ≤ π/2）を前提にしており、そのまま左視野（Φ≈π）に
        #   使うと √(R²−2AR+A²)=|R−A| となり、**R=A=3° で ln(0) が発散する**。
        #   実装ではいったん右側へ折り返して計算し、左視野は符号で戻す。
        self.side = np.where(self.ang_x >= 0, 1.0, -1.0)
        azim_folded = np.arctan2(self.ang_y, np.abs(self.ang_x))
        self.u, self.v = visual_to_collicular(self.ecc, azim_folded,
                                              self.a, self.bu, self.bv)
        # マグニフィケーション（面積）。画素あたりの上丘の面積に比例する重み。
        # ⚠️「重みを掛ける」方式（旧）専用。格子方式では**使わない**（二重になる）。
        self.mag_area = (self.bu * self.bv) / (self.ecc + self.a) ** 2

        # ★格子の細かさ。既定は計算コストとの兼ね合いで決めた（下記）。
        import os as _o
        # ★2026-07-27：96x128 → 64x96。実時間で動かすには 96x128・反復12 が重すぎた
        #   （1回24.7ms、制御周期10msの2.5倍）。48x64 まで落とすと速いが、
        #   環境では定位の精度が落ちた（後半ずれ 0.154 → 0.490）。
        #   静止画1枚の単体テストでは差が 0.002 しか出なかったのに環境で悪化した＝
        #   時系列の挙動（競合の収束と次の入力の相互作用）が変わるため。
        #   64x96・反復12 が速度と精度の折衷。
        nu = int(_o.environ.get("E_SC_NU", "48")) if nu is None else int(nu)
        nv = int(_o.environ.get("E_SC_NV", "64")) if nv is None else int(nv)
        self._build_grid(nu, nv)

    # ------------------------------------------------------------
    # ★上丘の「格子」＝実際の神経組織に相当する升目
    # ------------------------------------------------------------
    def _build_grid(self, nu, nv):
        """上丘の座標 (u,v) を等間隔に区切った升目を作り、各升が画像のどの画素を
        見ているかを先に求めておく（逆写像）。

        ★なぜ画像から写すのでなく、升から画像を引くのか：
          画像を升へ「配る」と、中心視野の1画素が上丘では広い面積に対応するため
          **升に穴が空く**。升の側から引けば穴が空かず、中心視野の1画素が
          自然に複数の升へ広がる＝中心視野が上丘で大きな面積を占めることを、
          重みを掛けずに座標変換そのもので表せる。
        """
        self.nu, self.nv = int(nu), int(nv)
        u_max = float(visual_to_collicular(float(self.ecc.max()), 0.0,
                                           self.a, self.bu, self.bv)[0])
        v_max = self.bv * (np.pi / 2.0)
        self.du = u_max / self.nu
        self.dv = 2.0 * v_max / self.nv
        us = (np.arange(self.nu) + 0.5) * self.du
        vs = -v_max + (np.arange(self.nv) + 0.5) * self.dv
        self.grid_u, self.grid_v = np.meshgrid(us, vs)        # [nv, nu]

        # 升の中心 → 視野角[度]（右側へ折り返した座標）
        gx_deg, gy_deg = collicular_to_visual(self.grid_u, self.grid_v,
                                              self.a, self.bu, self.bv)
        t = np.tan(np.radians(self.half_fov))
        cx = (self.w - 1) / 2.0
        cy = (self.h - 1) / 2.0
        with np.errstate(invalid="ignore"):
            gy = np.tan(np.radians(np.clip(gy_deg, -89.0, 89.0))) / t
            gx_r = np.tan(np.radians(np.clip(gx_deg, -89.0, 89.0))) / t
        # ★折り返した座標なので x<0 は像の外＝無効
        base_ok = (gx_deg >= 0.0) & np.isfinite(gx_r) & np.isfinite(gy)

        self._grid_px = np.zeros((2, self.nv, self.nu), dtype=np.intp)
        self._grid_py = np.zeros((2, self.nv, self.nu), dtype=np.intp)
        self._grid_ok = np.zeros((2, self.nv, self.nu), dtype=bool)
        for si, s in enumerate((1.0, -1.0)):       # 0: 右視野, 1: 左視野
            gx = gx_r * s
            px = np.rint(gx * cx + cx)
            py = np.rint(-gy * cy + cy)            # 画像の y は下向き
            ok = base_ok & (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
            self._grid_px[si] = np.clip(np.nan_to_num(px), 0, self.w - 1).astype(np.intp)
            self._grid_py[si] = np.clip(np.nan_to_num(py), 0, self.h - 1).astype(np.intp)
            self._grid_ok[si] = ok
        # 受容野のぼかし幅を「升いくつぶん」に直す
        self.rf_sigma_cells = (RF_SIGMA_MM / self.dv, RF_SIGMA_MM / self.du)

    def to_grid(self, image):
        """画像 → 上丘の格子 [2, nv, nu]（0:右視野 / 1:左視野）。"""
        a = np.asarray(image, dtype=np.float32)
        g = a[self._grid_py, self._grid_px]
        return np.where(self._grid_ok, g, 0.0)

    def grid_centroid(self, grid, thresh_frac=0.0):
        """格子の活動 → 視野の座標[度]（population vector）。

        左右の上丘それぞれで重心を取り、活動量で重み付けして合成する。
        """
        g = np.clip(np.asarray(grid, dtype=float), 0, None)
        g = np.where(self._grid_ok, g, 0.0)
        mx = g.max()
        if mx <= 1e-12:
            return float("nan"), float("nan")
        if thresh_frac > 0:
            g = np.where(g >= mx * thresh_frac, g, 0.0)
        xs, ys, ws = [], [], []
        for si, s in enumerate((1.0, -1.0)):
            gs = g[si]
            w = float(gs.sum())
            if w <= 1e-12:
                continue
            ub = float((gs * self.grid_u).sum() / w)
            vb = float((gs * self.grid_v).sum() / w)
            x, y = collicular_to_visual(ub, vb, self.a, self.bu, self.bv)
            xs.append(float(x) * s)
            ys.append(float(y))
            ws.append(w)
        if not ws:
            return float("nan"), float("nan")
        tw = sum(ws)
        return (float(sum(x * w for x, w in zip(xs, ws)) / tw),
                float(sum(y * w for y, w in zip(ys, ws)) / tw))

    def grid_direction(self, grid, thresh_frac=0.0):
        """grid_centroid を -1〜1 に正規化して返す（右が正、上が正）。"""
        x, y = self.grid_centroid(grid, thresh_frac)
        if np.isnan(x):
            return 0.0, 0.0
        return (float(np.clip(x / self.half_fov, -1.0, 1.0)),
                float(np.clip(y / self.half_fov, -1.0, 1.0)))

    def population_vector(self, activity, thresh_frac=0.0):
        """★上丘の座標で重心を取り、視野の座標[度]に戻す。

        これが実際の上丘の処理（Lee, Rohrer & Sparks 1988）。
        平らな画像で重心を取るのとは**別の答えになる**（Goossens & Van Opstal 2012）。

        Args:
            activity: 画像と同じ形の活動マップ（負は0として扱う）
            thresh_frac: 最大値のこの割合未満を捨てる（0なら全部使う）

        Returns:
            (x, y): 視野の座標 [度]。見つからなければ (nan, nan)
        """
        a = np.clip(np.asarray(activity, dtype=float), 0, None)
        if a.shape != self.u.shape or a.max() <= 1e-12:
            return float("nan"), float("nan")
        if thresh_frac > 0:
            a = np.where(a >= a.max() * thresh_frac, a, 0.0)
        tot = a.sum()
        if tot <= 1e-12:
            return float("nan"), float("nan")
        # ★左右の上丘は別々の地図なので、それぞれで重心を取ってから足し合わせる。
        #   （実際の上丘も左右の半球に分かれており、両者の集団活動の和で
        #     サッケードが決まる。片側だけが活動していれば、その側へ向く）
        xs, ys, ws = [], [], []
        for s in (1.0, -1.0):
            m = (self.side == s)
            w = float((a * m).sum())
            if w <= 1e-12:
                continue
            u_bar = float((a * m * self.u).sum() / w)
            v_bar = float((a * m * self.v).sum() / w)
            x, y = collicular_to_visual(u_bar, v_bar, self.a, self.bu, self.bv)
            xs.append(float(x) * s)      # 折り返した左視野を戻す
            ys.append(float(y))
            ws.append(w)
        if not ws:
            return float("nan"), float("nan")
        tw = sum(ws)
        return (float(sum(x * w for x, w in zip(xs, ws)) / tw),
                float(sum(y * w for y, w in zip(ys, ws)) / tw))

    def direction(self, activity, thresh_frac=0.0):
        """population_vector を -1〜1 に正規化して返す（右が正、上が正）。

        1.0 が視野の端（半角ぶん）に相当する。
        """
        x, y = self.population_vector(activity, thresh_frac)
        if np.isnan(x):
            return 0.0, 0.0
        return (float(np.clip(x / self.half_fov, -1.0, 1.0)),
                float(np.clip(y / self.half_fov, -1.0, 1.0)))
