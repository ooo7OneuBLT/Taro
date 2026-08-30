# -*- coding: utf-8 -*-
"""基本図形だけで、4カテゴリ × 5個体の物体を組み立てる（F2-17 般化テスト用）。

【なぜ基本図形か、2026-08-30】般化テストの素材を探して既製画像・生成AI・3Dモデルを
調べたが、どれも権利か統制で詰まった（研究日誌 (29)・現在地.md「素材の選択肢」）。
MuJoCo の基本図形（球・楕円体・カプセル・円柱・箱）だけで組めば、
  ・権利は無縁（自作）
  ・統制が完全（「胴の長さだけ変えた」と数値で言える）
  ・メモリ最小（メッシュもテクスチャも無い）
実測で、形が根本的に違うもの（犬・車・りんご・くつ）はカテゴリ内0.708〜0.859／
カテゴリ外0.354〜0.417 で明確に分離することを確認済み（研究日誌 (30)）。

【この方式の限界】同じ骨格で細部が違うもの（犬と猫）は区別できない＝10語程度が上限。
語彙を増やす段階では実物スキャン（Google Scanned Objects）へ移る。
【Tier3・太郎固有の逸脱】人間の乳児は基本図形の犬を見ない。刺激を統制するための単純化。

【使い方】
    .venv/Scripts/python.exe F/scripts/f_make_objects.py            # 図を作って確認
    .venv/Scripts/python.exe F/scripts/f_make_objects.py --xml 出力先  # XMLの断片を書き出す
"""
import os
import sys
import math
import argparse

# 【2026-08-30】contype=0 conaffinity=0 ＝ 衝突しない「見た目だけ」の図形。
#   27個の図形が重なった物体を物理で扱うと接触計算が発散した
#   （実測："Nan, Inf or huge value in QACC. The simulation is unstable."）。
#   太郎は物を掴まない（座位で手が固定）ので衝突は要らない。
#   衝突を切ると物体は落ちず、親が差し出した位置に浮く＝板と同じ挙動になる。
_NC = ' contype="0" conaffinity="0"'
G = '<geom type="{t}" size="{s}" pos="{p}" rgba="{c}"' + _NC + '{extra}/>'
GC = '<geom type="capsule" fromto="{f}" size="{r}" rgba="{c}"' + _NC + '/>'
GCY = '<geom type="cylinder" fromto="{f}" size="{r}" rgba="{c}"' + _NC + '/>'

# 【2026-08-30・実測して決めた】物体の全体倍率。
#   初版は犬の全長が0.304mあり、0.30m先から見ると**53.7度**＝視野60度をほぼ埋めた。
#   中心窩（15度）には胴体の一部しか写らず、板を8.6cmに置いていた頃と同じ
#   「一部しか見えない」状態に戻っていた（F/logs/F2-17_素材/視界確認/両目/）。
#   板（8cm角）は0.30m先で15.2度なので、それに合わせて全長8cm前後になる倍率にする。
#   質量も初版は1.53kgあり、親が差し出す物として重すぎた。
# 【2026-08-30・2回目の調整】0.26 では中心窩（15度）で画面の3分の1しか埋まらなかった。
#   「全長を板と同じ8cmにする」という揃え方が誤り。板は8cm角の平面が正面を向くので
#   面積で画面を埋めるが、立体は横向きの細い形なので同じ全長でも面積は4分の1以下になる。
#   中心窩で板と同程度の面積を占める 0.42 にする（図＝F/logs/F2-17_素材/図_大きさの比較.png）。
SCALE = 0.42
# カテゴリごとの追加倍率。実測した全長（0.30m先での見かけの角度）で揃える：
#   犬 0.079m/15.0度・車 0.069m/13.2度・りんご 0.027m/5.2度・くつ 0.047m/8.9度
# りんごとくつが小さすぎて中心窩（15度）で細部が潰れたため、板（8cm・15.2度）に合わせる。
# 実世界の大きさ比（りんごは犬より小さい）は捨てる。板の時代も全部8cm角だった。
CAT_SCALE = {"dog": 1.00, "car": 1.10, "apple": 2.60, "shoe": 1.55}


# 長さの単位を持つキー（これだけに SCALE を掛ける。色・角度・個数・比率は掛けない）
_LEN_KEYS = {"body_l", "body_r", "head_r", "leg_up", "leg_lo", "leg_r",
             "tail_l", "tail_r", "body_w", "body_h", "roof_l", "roof_h", "roof_x",
             "wheel_r", "r", "stem_l", "len", "width", "upper_h", "sole_h"}


def scaled(p, cat="dog"):
    """寸法だけを SCALE × カテゴリ倍率 倍した辞書を返す。"""
    f = SCALE * CAT_SCALE[cat]
    out = dict(p)
    for k in _LEN_KEYS:
        if k in out:
            out[k] = out[k] * f
    return out


# ---------------------------------------------------------------- 犬
def dog(p):
    """犬。変えるもの：胴の長さ・太さ、脚の長さ、耳の形、鼻の長さ、色。"""
    p = scaled(p, "dog")
    c, c2, ce = p["col"], p["col2"], p["ear_col"]
    bl, br = p["body_l"], p["body_r"]
    z = p["leg_up"] + p["leg_lo"]
    hr = p["head_r"]
    hx, hz = bl * 1.02, br * p["head_up"]
    out = []
    A = out.append
    # 胴：胸・腹・腰の3つでなだらかに
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .44, br * 1.04, br),
               p="%.5f 0 %.5f" % (bl * .40, br * .05), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .42, br, br * .97),
               p="%.5f 0 0" % (-bl * .04), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .38, br * 1.02, br),
               p="%.5f 0 %.5f" % (-bl * .46, br * .03), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .58, br * .60, br * .30),
               p="%.5f 0 %.5f" % (0, -br * .80), c=c2, extra=""))
    # 首と頭
    A(GC.format(f="%.5f 0 %.5f %.5f 0 %.5f" % (bl * .72, br * .14, hx * .99, hz * .90),
                r="%.5f" % (br * .60), c=c))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (hr * 1.05, hr * .90, hr),
               p="%.5f 0 %.5f" % (hx, hz), c=c, extra=""))
    # マズル（鼻先）：犬は前へ長く出る
    ml = p["muzzle"]
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (hr * ml, hr * .32, hr * .28),
               p="%.5f 0 %.5f" % (hx + hr * (.55 + ml * .55), hz - hr * .22), c=c2, extra=""))
    A(G.format(t="sphere", s="%.5f" % (hr * .12),
               p="%.5f 0 %.5f" % (hx + hr * (.55 + ml * 1.05), hz - hr * .18),
               c=".16 .12 .12 1", extra=""))
    for s in (1, -1):
        A(G.format(t="sphere", s="%.5f" % (hr * .11),
                   p="%.5f %.5f %.5f" % (hx + hr * .52, s * hr * .55, hz + hr * .26),
                   c=".10 .09 .08 1", extra=""))
        # 耳：ear_drop が大きいほど下に垂れる
        A(G.format(t="ellipsoid",
                   s="%.5f %.5f %.5f" % (hr * p["ear_w"], hr * .12, hr * p["ear_h"]),
                   p="%.5f %.5f %.5f" % (hx - hr * .16, s * hr * .68,
                                         hz + hr * (.66 + p["ear_h"] * .5) - p["ear_drop"] * hr),
                   c=ce, extra=' euler="%d %d 0"' % (s * 14, p["ear_pitch"])))
    # 脚（ひざあり）と足
    lu, ll, lr = p["leg_up"], p["leg_lo"], p["leg_r"]
    for xs in (bl * .54, -bl * .48):
        for s in (1, -1):
            yb = s * br * .80
            A(GC.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                        % (xs, yb, -br * .40, xs + lr * .5, yb, -br * .40 - lu),
                        r="%.5f" % lr, c=c))
            A(GC.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                        % (xs + lr * .5, yb, -br * .40 - lu, xs, yb, -br * .40 - lu - ll),
                        r="%.5f" % (lr * .85), c=c))
            A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (lr * 1.6, lr * 1.1, lr * .55),
                       p="%.5f %.5f %.5f" % (xs + lr * .6, yb, -br * .40 - lu - ll),
                       c=c2, extra=""))
    # 尾
    tl, tr_, tc = p["tail_l"], p["tail_r"], p["tail_curl"]
    x0, z0 = -bl * .82, br * .28
    for i in range(3):
        x1 = x0 - tl * .33 * math.cos(math.radians(tc * (i + 1) * .5))
        z1 = z0 + tl * .33 * math.sin(math.radians(tc * (i + 1) * .5))
        A(GC.format(f="%.5f 0 %.5f %.5f 0 %.5f" % (x0, z0, x1, z1),
                    r="%.5f" % (tr_ * (1 - i * .15)), c=c))
        x0, z0 = x1, z1
    return "".join(out), z + br * .40


# ---------------------------------------------------------------- 車
def car(p):
    """車。変えるもの：車体の長さ・高さ、屋根の位置と高さ、車輪の大きさ、色。"""
    p = scaled(p, "car")
    c, cg = p["col"], ".55 .70 .82 1"
    bl, bw, bh = p["body_l"], p["body_w"], p["body_h"]
    rl, rh, rx = p["roof_l"], p["roof_h"], p["roof_x"]
    wr = p["wheel_r"]
    out = []
    A = out.append
    A(G.format(t="box", s="%.5f %.5f %.5f" % (bl, bw, bh), p="0 0 0", c=c, extra=""))
    A(G.format(t="box", s="%.5f %.5f %.5f" % (rl, bw * .93, rh),
               p="%.5f 0 %.5f" % (rx, bh + rh), c=c, extra=""))
    # 窓
    A(G.format(t="box", s="%.5f %.5f %.5f" % (rl * .62, bw * .96, rh * .58),
               p="%.5f 0 %.5f" % (rx, bh + rh * 1.15), c=cg, extra=""))
    # バンパー
    for s in (1, -1):
        A(G.format(t="box", s="%.5f %.5f %.5f" % (bl * .05, bw * .95, bh * .35),
                   p="%.5f 0 %.5f" % (s * bl * .98, -bh * .45), c=".82 .82 .84 1", extra=""))
    # ライト
    for s in (1, -1):
        A(G.format(t="sphere", s="%.5f" % (bh * .28),
                   p="%.5f %.5f %.5f" % (bl * .96, s * bw * .62, bh * .25),
                   c=".98 .95 .80 1", extra=""))
    # 車輪（タイヤ＋ホイール）
    for xs in (bl * p["wheel_x"], -bl * p["wheel_x"]):
        A(GCY.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                     % (xs, bw * 1.02, -bh, xs, -bw * 1.02, -bh),
                     r="%.5f" % wr, c=".13 .13 .13 1"))
        for s in (1, -1):
            A(GCY.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                         % (xs, s * bw * 1.03, -bh, xs, s * bw * 1.12, -bh),
                         r="%.5f" % (wr * .52), c=".85 .85 .87 1"))
    return "".join(out), bh + wr


# ---------------------------------------------------------------- りんご
def apple(p):
    """りんご。変えるもの：大きさ、縦横比、へたの向き、葉の有無、色。"""
    p = scaled(p, "apple")
    c = p["col"]
    r, ar = p["r"], p["aspect"]
    out = []
    A = out.append
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r, r, r * ar), p="0 0 0", c=c, extra=""))
    # 上下のくぼみ（背景色の球を埋めて凹ませる代わりに、少し潰した球を重ねる）
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r * .55, r * .55, r * ar * .30),
               p="0 0 %.5f" % (r * ar * .82), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r * .55, r * .55, r * ar * .28),
               p="0 0 %.5f" % (-r * ar * .84), c=c, extra=""))
    # へた
    st = p["stem_tilt"]
    A(GCY.format(f="0 0 %.5f %.5f 0 %.5f"
                 % (r * ar * .90, r * math.sin(math.radians(st)) * .60,
                    r * ar * .90 + p["stem_l"]),
                 r="%.5f" % (r * .06), c=".34 .24 .12 1"))
    if p["leaf"]:
        A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r * .42, r * .16, r * .05),
                   p="%.5f 0 %.5f" % (r * .40, r * ar * .90 + p["stem_l"] * .75),
                   c=".26 .56 .20 1", extra=' euler="0 -18 0"'))
    return "".join(out), r * ar


# ---------------------------------------------------------------- くつ
def shoe(p):
    """くつ。全個体で同じ骨格（細長い舟形＋盛り上がった甲＋立ったかかと）を共有し、
    長さ・幅・甲の高さ・つま先の細さ・かかとの高さ・色の比率だけを変える。

    【2026-08-30・作り直し】初版は「箱の上に楕円を乗せた」だけで、人が見ても靴に
    見えなかった（カテゴリ内の似ている度 0.795・カテゴリ外との差 +0.366 で4語中最低）。
    しかも くつ4 だけヒール付きの別骨格にしたため、形の一貫性が崩れていた。
    靴の形の要点は「横から見た輪郭」＝つま先が低く前へ伸び、甲が盛り上がり、
    かかとが垂直に立ち上がる。楕円体を前後に5つ並べて、その稜線をつくる。
    """
    p = scaled(p, "shoe")
    c, cs = p["col"], p["sole_col"]
    L, W = p["len"], p["width"]
    up = p["upper_h"]
    toe = p["toe"]        # つま先の細さ（小さいほど尖る）
    hb = p["heel_back"]   # かかとの立ち上がりの高さ
    out = []
    A = out.append

    # --- 靴底：前後で幅と高さを変えた箱を3つ並べ、なだらかな舟底にする ---
    sh = p["sole_h"]
    A(G.format(t="box", s="%.5f %.5f %.5f" % (L * .30, W * toe, sh * .85),
               p="%.5f 0 %.5f" % (L * .62, -up * .92), c=cs, extra=""))
    A(G.format(t="box", s="%.5f %.5f %.5f" % (L * .38, W, sh),
               p="%.5f 0 %.5f" % (L * .02, -up * .95), c=cs, extra=""))
    A(G.format(t="box", s="%.5f %.5f %.5f" % (L * .26, W * .92, sh * 1.10),
               p="%.5f 0 %.5f" % (-L * .62, -up * .90), c=cs, extra=""))
    # つま先の先端を丸める
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (L * .10, W * toe, sh),
               p="%.5f 0 %.5f" % (L * .90, -up * .92), c=cs, extra=""))

    # --- 甲：楕円体を5つ、前から後ろへ高さを上げながら並べる（靴の稜線） ---
    #   x     位置（前が正）
    #   rz    その位置での甲の高さ
    prof = [(.86, .30, toe * .92), (.58, .62, toe * .98),
            (.24, .92, 1.00), (-.14, 1.00, 1.00), (-.48, .88, .96)]
    for fx, fz, fw in prof:
        A(G.format(t="ellipsoid",
                   s="%.5f %.5f %.5f" % (L * .26, W * fw, up * fz),
                   p="%.5f 0 %.5f" % (L * fx, -up * (1 - fz) * .55), c=c, extra=""))
    # --- かかと：後ろで垂直に立ち上がる部分 ---
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (L * .16, W * .90, up * hb),
               p="%.5f 0 %.5f" % (-L * .74, up * (hb - 1) * .40), c=c, extra=""))
    A(G.format(t="box", s="%.5f %.5f %.5f" % (L * .12, W * .86, up * hb * .55),
               p="%.5f 0 %.5f" % (-L * .80, up * (hb * .30)), c=c, extra=""))

    # --- 履き口：上面を暗い色でくぼませる ---
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (L * .24, W * .66, up * .16),
               p="%.5f 0 %.5f" % (-L * .40, up * .90), c=".22 .21 .22 1", extra=""))

    # --- 甲の上のひも（前後に並ぶ細い帯） ---
    for i in range(p["laces"]):
        fx = .38 - i * .17
        A(G.format(t="box", s="%.5f %.5f %.5f" % (L * .030, W * .62, up * .05),
                   p="%.5f 0 %.5f" % (L * fx, up * .86), c=p["lace_col"], extra=""))
    return "".join(out), up + p["sole_h"] * 2


# ---------------------------------------------------------------- 個体の定義
DOG_BASE = dict(body_l=.105, body_r=.040, head_r=.038, head_up=1.55, muzzle=.62,
                ear_w=.17, ear_h=.60, ear_pitch=26, ear_drop=.0,
                leg_up=.030, leg_lo=.028, leg_r=.0115,
                tail_l=.062, tail_r=.0105, tail_curl=34,
                col=".68 .50 .32 1", col2=".88 .82 .74 1", ear_col=".52 .37 .23 1")
DOGS = [
    ("犬1", dict(DOG_BASE)),
    ("犬2", dict(DOG_BASE, body_l=.128, body_r=.033, leg_up=.014, leg_lo=.013,
                 muzzle=.78, ear_h=.86, ear_pitch=52, ear_drop=.30, tail_l=.050,
                 col=".42 .27 .16 1", ear_col=".32 .20 .12 1")),           # 胴長短足
    ("犬3", dict(DOG_BASE, body_l=.088, body_r=.046, leg_up=.038, leg_lo=.034,
                 head_r=.042, muzzle=.40, ear_w=.20, ear_h=.44, ear_pitch=10,
                 tail_curl=62, col=".90 .86 .80 1", col2=".96 .94 .90 1",
                 ear_col=".78 .72 .64 1")),                                # 丸くて白い
    ("犬4", dict(DOG_BASE, body_l=.118, body_r=.043, leg_up=.040, leg_lo=.036,
                 head_r=.040, muzzle=.72, ear_h=.72, ear_pitch=40, ear_drop=.18,
                 tail_l=.075, tail_curl=12, col=".28 .24 .22 1",
                 col2=".62 .58 .54 1", ear_col=".20 .17 .16 1")),          # 大きくて黒い
    ("犬5", dict(DOG_BASE, body_l=.094, body_r=.036, leg_up=.024, leg_lo=.022,
                 head_r=.034, muzzle=.55, ear_w=.15, ear_h=.70, ear_pitch=18,
                 tail_curl=50, col=".84 .62 .28 1", col2=".93 .88 .78 1",
                 ear_col=".70 .50 .20 1")),                                # 小さくて明るい
]

CAR_BASE = dict(body_l=.095, body_w=.042, body_h=.020, roof_l=.048, roof_h=.020,
                roof_x=-.012, wheel_r=.026, wheel_x=.58, col=".80 .15 .12 1")
CARS = [
    ("車1", dict(CAR_BASE)),
    ("車2", dict(CAR_BASE, body_l=.082, roof_l=.052, roof_h=.034, roof_x=-.004,
                 wheel_r=.022, col=".18 .32 .72 1")),                      # 背が高い
    ("車3", dict(CAR_BASE, body_l=.112, body_h=.016, roof_l=.040, roof_h=.014,
                 roof_x=-.020, wheel_r=.024, wheel_x=.66, col=".92 .88 .30 1")),  # 低くて長い
    ("車4", dict(CAR_BASE, body_l=.078, body_w=.038, roof_l=.044, roof_h=.026,
                 wheel_r=.028, wheel_x=.52, col=".22 .62 .34 1")),         # 小さい
    ("車5", dict(CAR_BASE, body_l=.104, body_w=.046, body_h=.024, roof_l=.056,
                 roof_h=.024, roof_x=-.016, wheel_r=.030, col=".94 .94 .92 1")),  # 大きくて白い
]

APPLE_BASE = dict(r=.052, aspect=.95, stem_l=.028, stem_tilt=0, leaf=True,
                  col=".85 .10 .08 1")
APPLES = [
    ("りんご1", dict(APPLE_BASE)),
    ("りんご2", dict(APPLE_BASE, r=.044, aspect=1.10, stem_l=.034, stem_tilt=22,
                     leaf=False, col=".72 .14 .10 1")),                    # 細くて濃い
    ("りんご3", dict(APPLE_BASE, r=.060, aspect=.84, stem_l=.020, stem_tilt=-14,
                     col=".90 .30 .12 1")),                                # 平たくて明るい
    ("りんご4", dict(APPLE_BASE, r=.048, aspect=1.00, stem_l=.030, stem_tilt=10,
                     col=".62 .74 .18 1")),                                # 青りんご
    ("りんご5", dict(APPLE_BASE, r=.056, aspect=.90, stem_l=.024, stem_tilt=-8,
                     leaf=False, col=".94 .74 .16 1")),                    # 黄色
]

SHOE_BASE = dict(len=.080, width=.030, upper_h=.022, sole_h=.006,
                 toe=.62, heel_back=1.30, laces=3,
                 col=".92 .92 .90 1", sole_col=".16 .16 .16 1",
                 lace_col=".30 .30 .32 1")
SHOES = [
    ("くつ1", dict(SHOE_BASE)),                                              # 白いスニーカー
    ("くつ2", dict(SHOE_BASE, len=.070, width=.032, upper_h=.030, toe=.72,
                   heel_back=1.55, laces=4, col=".52 .34 .18 1",
                   sole_col=".28 .20 .12 1", lace_col=".78 .70 .55 1")),      # 背の高い革靴
    ("くつ3", dict(SHOE_BASE, len=.094, width=.026, upper_h=.017, toe=.50,
                   heel_back=1.15, laces=2, col=".22 .30 .62 1",
                   sole_col=".85 .85 .84 1")),                                # 細長い
    ("くつ4", dict(SHOE_BASE, len=.072, width=.028, upper_h=.019, toe=.55,
                   heel_back=1.40, laces=2, col=".82 .18 .30 1",
                   sole_col=".62 .14 .22 1", lace_col=".95 .90 .90 1")),      # 小さい赤
    ("くつ5", dict(SHOE_BASE, len=.088, width=.034, upper_h=.026, toe=.68,
                   heel_back=1.25, laces=3, col=".26 .26 .28 1",
                   sole_col=".84 .84 .82 1", lace_col=".70 .70 .72 1")),      # 大きくて黒い
]

CATS = [("わんわん", DOGS, dog), ("ぶーぶー", CARS, car),
        ("りんご", APPLES, apple), ("くつ", SHOES, shoe)]

TPL = """<mujoco>
  <visual>
    <global offwidth="640" offheight="640"/>
    <headlight ambient=".45 .45 .45" diffuse=".35 .35 .35"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1=".96 .96 .96" width="8" height="8"/>
  </asset>
  <worldbody>
    <light pos="0.32 -0.26 0.52" dir="-0.5 0.45 -0.72" diffuse=".75 .75 .75" specular=".18 .18 .18"/>
    <light pos="-0.22 0.28 0.38" dir="0.4 -0.5 -0.5" diffuse=".22 .22 .22"/>
    <camera name="c0" fovy="15" mode="targetbody" target="obj" pos="0.212 -0.163 {cz1}"/>
    <camera name="c1" fovy="15" mode="targetbody" target="obj" pos="0.268 -0.018 {cz2}"/>
    <camera name="c2" fovy="15" mode="targetbody" target="obj" pos="0.018 -0.267 {cz3}"/>
    <body name="obj" pos="0 0 {z}">{g}</body>
  </worldbody>
</mujoco>"""


def build(fn, prm):
    g, lift = fn(prm)
    # カメラは物体の中心から 0.30m（本番の板の距離）に置く。
    # 物体の大きさが変わっても「見かけの角度」が本番と同じ条件で測れる。
    zc = 0.10 + lift
    return TPL.format(z="%.5f" % zc, g=g,
                      cz1="%.4f" % (zc + 0.126), cz2="%.4f" % (zc + 0.135),
                      cz3="%.4f" % (zc + 0.135))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="F/logs/F2-17_素材/図_個体一覧.png")
    a = ap.parse_args()
    os.chdir(r"C:\claude\AI\Taro")
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    for c in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if any(c in f.name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = c
            break

    imgs, small = {}, {}
    for cat, items, fn in CATS:
        for name, prm in items:
            m = mujoco.MjModel.from_xml_string(build(fn, prm))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            rb = mujoco.Renderer(m, height=320, width=320)
            rs = mujoco.Renderer(m, height=128, width=128)
            rb.update_scene(d, camera="c0")
            imgs[name] = rb.render().copy()
            rs.update_scene(d, camera="c0")
            small[name] = rs.render().copy()
            rb.close()
            rs.close()
        print("%s: %d個体" % (cat, len(items)))

    fig, ax = plt.subplots(4, 5, figsize=(15.0, 12.4))
    for r, (cat, items, _) in enumerate(CATS):
        for i, (name, _p) in enumerate(items):
            ax[r, i].imshow(imgs[name])
            ax[r, i].set_xticks([])
            ax[r, i].set_yticks([])
            tag = "学習用" if i < 3 else "テスト用"
            ax[r, i].set_title("%s（%s）" % (name, tag), fontsize=10,
                               color="#333" if i < 3 else "#b34")
        ax[r, 0].set_ylabel(cat, fontsize=14)
    fig.suptitle("基本図形で作った4語 × 5個体（左3つが学習用・右2つがテスト用）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=112)
    print("\n保存:", a.out)

    # 太郎の目に「カテゴリの内と外」が届いているかを測る
    sys.path.insert(0, os.path.abspath("taro_core/src"))
    from senses.vision_backends import get_backend
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})

    def cos(x, y):
        n = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / n) if n > 1e-9 else 0.0

    feats = {n: np.asarray(be.encode(small[n], small[n]), dtype=float) for n in small}
    print("\nカテゴリの中と外（太郎の目で見た近さ）")
    inner_all, outer_all = [], []
    for cat, items, _ in CATS:
        ns = [n for n, _ in items]
        inner = [cos(feats[a_], feats[b_]) for i, a_ in enumerate(ns) for b_ in ns[i + 1:]]
        others = [n for c2, it2, _ in CATS if c2 != cat for n, _ in it2]
        outer = [cos(feats[a_], feats[b_]) for a_ in ns for b_ in others]
        inner_all += inner
        outer_all += outer
        print("  %-8s 中 %.3f（%.3f〜%.3f）  外 %.3f  差 %+.3f"
              % (cat, np.mean(inner), min(inner), max(inner), np.mean(outer),
                 np.mean(inner) - np.mean(outer)))
    print("  %-8s 中 %.3f  外 %.3f  差 %+.3f"
          % ("全体", np.mean(inner_all), np.mean(outer_all),
             np.mean(inner_all) - np.mean(outer_all)))


if __name__ == "__main__":
    main()
