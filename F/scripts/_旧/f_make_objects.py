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
# 【2026-08-31・ユーザー指摘で倍率変更】0.42（全長8cm＝板時代の名残）→ 0.84。
#   現実の乳児のおもちゃは15〜20cmで、30cm先では視野の30〜40度を占める。
#   中心窩(15度)からはみ出すのが人間として正常（ユーザーが以前から指摘していた点。
#   「中心窩に収める」較正は板との互換のための誤った制約だった）。
SCALE = 0.84
# カテゴリごとの追加倍率。実測した全長（0.30m先での見かけの角度）で揃える：
#   犬 0.079m/15.0度・車 0.069m/13.2度・りんご 0.027m/5.2度・くつ 0.047m/8.9度
# りんごとくつが小さすぎて中心窩（15度）で細部が潰れたため、板（8cm・15.2度）に合わせる。
# 実世界の大きさ比（りんごは犬より小さい）は捨てる。板の時代も全部8cm角だった。
CAT_SCALE = {"dog": 1.00, "car": 1.10, "apple": 1.30, "shoe": 1.55,
             # 【2026-08-30・10語化】犬猫・車電車・りんごボールを「そっくりペア」
             #   として足した。ペアの倍率は元カテゴリに揃える（大きさで区別
             #   できてしまうと、形の似ている度を測れなくなるため）。
             #   球系（りんご・ボール）の2.60は誤り：直径11cmになり fovy15度
             #   （0.30m先で7.9cm）から四辺ともはみ出し「ただの色の面」になっていた
             #   （2026-08-30 実測、ボールのカテゴリ内類似 0.657 まで低下）。
             #   直径が視野の9割（約7cm）に収まる値へ下げる。
             "cat": 1.00, "train": 0.95, "ball": 1.50,
             "banana": 2.20, "cup": 1.80, "hat": 1.50}


# 長さの単位を持つキー（これだけに SCALE を掛ける。色・角度・個数・比率は掛けない）
_LEN_KEYS = {"body_l", "body_r", "head_r", "leg_up", "leg_lo", "leg_r",
             "tail_l", "tail_r", "body_w", "body_h", "roof_l", "roof_h", "roof_x",
             "wheel_r", "r", "stem_l", "len", "width", "upper_h", "sole_h",
             "h", "brim_r", "brim_t", "crown_r", "crown_h", "handle_r"}


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
    # 【2026-08-30】へたと葉が上へ突き出す分、見た目の中心が実の中心より上になる。
    #   カメラは body の原点（＝実の中心）を狙うので、上だけ見切れていた（実測）。
    #   実全体をへたの高さの半分だけ下げて、見た目の中心を原点に合わせる。
    dz = -p["stem_l"] * 0.5
    out = []
    A = out.append
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r, r, r * ar),
               p="0 0 %.5f" % dz, c=c, extra=""))
    # 上下のくぼみ（背景色の球を埋めて凹ませる代わりに、少し潰した球を重ねる）
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r * .55, r * .55, r * ar * .30),
               p="0 0 %.5f" % (r * ar * .82 + dz), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r * .55, r * .55, r * ar * .28),
               p="0 0 %.5f" % (-r * ar * .84 + dz), c=c, extra=""))
    # へた
    st = p["stem_tilt"]
    A(GCY.format(f="0 0 %.5f %.5f 0 %.5f"
                 % (r * ar * .90 + dz, r * math.sin(math.radians(st)) * .60,
                    r * ar * .90 + p["stem_l"] + dz),
                 r="%.5f" % (r * .06), c=".34 .24 .12 1"))
    if p["leaf"]:
        A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (r * .42, r * .16, r * .05),
                   p="%.5f 0 %.5f" % (r * .40, r * ar * .90 + p["stem_l"] * .75 + dz),
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

# ---------------------------------------------------------------- 猫
def cat(p):
    """猫。**犬とわざと同じ骨格**（胴・首・頭・4本脚・尾）で作る。

    違うのは4点だけ：耳が三角に立つ・鼻先が短い・目が大きい・尾が細く長い。

    【なぜわざと似せるか・2026-08-30】太郎の語彙は「語ごとの見えの平均」との
    近さで語を選ぶ表引きなので、形が根本から違う4語（犬・車・りんご・くつ）では
    まず間違えない。だが人間の12ヶ月児は猫を見て「わんわん」と言う。
    同じ骨格の2カテゴリを入れて、**表引きがどこで壊れるか**を測るための対照。
    """
    p = scaled(p, "cat")
    c, c2, ce = p["col"], p["col2"], p["ear_col"]
    bl, br = p["body_l"], p["body_r"]
    z = p["leg_up"] + p["leg_lo"]
    hr = p["head_r"]
    hx, hz = bl * 1.00, br * p["head_up"]
    out = []
    A = out.append
    # 胴：犬より細くしなやか
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .44, br * 1.00, br * .98),
               p="%.5f 0 %.5f" % (bl * .38, br * .06), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .44, br * .92, br * .90),
               p="%.5f 0 0" % (-bl * .06), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .36, br * 1.02, br),
               p="%.5f 0 %.5f" % (-bl * .48, br * .08), c=c, extra=""))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (bl * .56, br * .56, br * .28),
               p="0 0 %.5f" % (-br * .78), c=c2, extra=""))
    # 首と頭（犬より丸い）
    A(GC.format(f="%.5f 0 %.5f %.5f 0 %.5f" % (bl * .66, br * .20, hx * .98, hz * .92),
                r="%.5f" % (br * .52), c=c))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (hr * .98, hr * .94, hr),
               p="%.5f 0 %.5f" % (hx, hz), c=c, extra=""))
    # 鼻先：猫は短い（muzzle が犬の半分以下）
    ml = p["muzzle"]
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (hr * ml, hr * .42, hr * .34),
               p="%.5f 0 %.5f" % (hx + hr * (.42 + ml * .42), hz - hr * .24), c=c2, extra=""))
    A(G.format(t="sphere", s="%.5f" % (hr * .10),
               p="%.5f 0 %.5f" % (hx + hr * (.42 + ml * .85), hz - hr * .16),
               c=".84 .50 .52 1", extra=""))
    for s in (1, -1):
        # 目：猫は顔に対して大きい
        A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (hr * .13, hr * .10, hr * .17),
                   p="%.5f %.5f %.5f" % (hx + hr * .50, s * hr * .44, hz + hr * .16),
                   c=p["eye_col"], extra=""))
        # 耳：三角に立つ（縦長の楕円体を外へ倒す）
        A(G.format(t="ellipsoid",
                   s="%.5f %.5f %.5f" % (hr * .24, hr * p["ear_w"], hr * p["ear_h"]),
                   p="%.5f %.5f %.5f" % (hx - hr * .08, s * hr * .50,
                                         hz + hr * (.70 + p["ear_h"] * .55)),
                   c=ce, extra=' euler="%d 0 0"' % (s * p["ear_tilt"])))
    # 脚（犬と同じ作りで細い）
    lu, ll, lr = p["leg_up"], p["leg_lo"], p["leg_r"]
    for xs in (bl * .50, -bl * .46):
        for s in (1, -1):
            yb = s * br * .74
            A(GC.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                        % (xs, yb, -br * .40, xs + lr * .4, yb, -br * .40 - lu),
                        r="%.5f" % lr, c=c))
            A(GC.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                        % (xs + lr * .4, yb, -br * .40 - lu, xs, yb, -br * .40 - lu - ll),
                        r="%.5f" % (lr * .85), c=c))
            A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (lr * 1.4, lr * 1.0, lr * .50),
                       p="%.5f %.5f %.5f" % (xs + lr * .5, yb, -br * .40 - lu - ll),
                       c=c2, extra=""))
    # 尾：細く長く、上へ反る
    tl, tr_, tc = p["tail_l"], p["tail_r"], p["tail_curl"]
    x0, z0 = -bl * .80, br * .30
    for i in range(4):
        x1 = x0 - tl * .26 * math.cos(math.radians(tc * (i + 1) * .42))
        z1 = z0 + tl * .26 * math.sin(math.radians(tc * (i + 1) * .42))
        A(GC.format(f="%.5f 0 %.5f %.5f 0 %.5f" % (x0, z0, x1, z1),
                    r="%.5f" % (tr_ * (1 - i * .10)), c=c))
        x0, z0 = x1, z1
    return "".join(out), z + br * .40


# ---------------------------------------------------------------- 電車
def train(p):
    """電車。**車とわざと同じ骨格**（箱の車体＋屋根＋車輪）で作る。
    違うのは車体が長い・屋根が全長にわたる・窓が横一列に並ぶ・車輪が小さく多いこと。
    犬と猫のペアと同じ趣旨（`cat` のコメント参照）。
    """
    p = scaled(p, "train")
    c, cg = p["col"], ".58 .72 .84 1"
    bl, bw, bh = p["body_l"], p["body_w"], p["body_h"]
    rh = p["roof_h"]
    wr = p["wheel_r"]
    out = []
    A = out.append
    A(G.format(t="box", s="%.5f %.5f %.5f" % (bl, bw, bh), p="0 0 0", c=c, extra=""))
    # 屋根：全長にわたる（車は途中までしかない）
    A(G.format(t="box", s="%.5f %.5f %.5f" % (bl * .98, bw * .96, rh),
               p="0 0 %.5f" % (bh + rh), c=p["roof_col"], extra=""))
    # 窓：横一列に並ぶ
    n = p["windows"]
    for i in range(n):
        fx = (i - (n - 1) * .5) / max(n * .55, 1e-6)
        A(G.format(t="box", s="%.5f %.5f %.5f" % (bl * .07, bw * 1.02, bh * .42),
                   p="%.5f 0 %.5f" % (bl * fx, bh * .30), c=cg, extra=""))
    # 前面の窓と帯
    A(G.format(t="box", s="%.5f %.5f %.5f" % (bl * .02, bw * .96, bh * .40),
               p="%.5f 0 %.5f" % (bl * .99, bh * .34), c=cg, extra=""))
    A(G.format(t="box", s="%.5f %.5f %.5f" % (bl * .99, bw * 1.02, bh * .12),
               p="0 0 %.5f" % (-bh * .42), c=p["band_col"], extra=""))
    # 車輪：小さいものが多数
    nw = p["wheels"]
    for i in range(nw):
        xs = bl * (.80 - 1.60 * i / float(max(nw - 1, 1)))
        A(GCY.format(f="%.5f %.5f %.5f %.5f %.5f %.5f"
                     % (xs, bw * .98, -bh, xs, -bw * .98, -bh),
                     r="%.5f" % wr, c=".14 .14 .15 1"))
    return "".join(out), bh + wr


# ---------------------------------------------------------------- ボール
def ball(p):
    """ボール。球に帯を巻いただけ。**りんごから「へた」と「くぼみ」を取った形**で、
    りんごとのペアになる（`cat` のコメント参照）。
    """
    p = scaled(p, "ball")
    r = p["r"]
    out = []
    A = out.append
    A(G.format(t="sphere", s="%.5f" % r, p="0 0 0", c=p["col"], extra=""))
    # 帯：球の表面すれすれに、少しだけ平たい楕円体を重ねる
    for i in range(p["bands"]):
        ang = 180.0 * i / float(max(p["bands"], 1))
        A(G.format(t="ellipsoid",
                   s="%.5f %.5f %.5f" % (r * 1.005, r * p["band_w"], r * 1.005),
                   p="0 0 0", c=p["band_col"], extra=' euler="0 0 %.1f"' % ang))
    return "".join(out), r


# ---------------------------------------------------------------- ばなな
def banana(p):
    """ばなな。カプセルを弧に沿って並べ、中央を太く両端を細くする。"""
    p = scaled(p, "banana")
    L, r = p["len"], p["r"]
    cv = max(float(p["curve"]), 1.0)
    n = 8
    R = L / math.radians(cv)
    half = math.radians(cv) / 2.0
    pts = []
    for i in range(n + 1):
        a = -half + 2.0 * half * i / float(n)
        pts.append((R * math.sin(a), R * (math.cos(a) - math.cos(half))))
    out = []
    A = out.append
    for i in range(n):
        x0, z0 = pts[i]
        x1, z1 = pts[i + 1]
        u = abs((i + 0.5) / float(n) * 2.0 - 1.0)
        rr = r * (1.0 - p["taper"] * u * u)
        col = p["tip_col"] if (i == 0 or i == n - 1) else p["col"]
        A(GC.format(f="%.5f 0 %.5f %.5f 0 %.5f" % (x0, z0, x1, z1),
                    r="%.5f" % max(rr, r * .18), c=col))
    return "".join(out), r


# ---------------------------------------------------------------- コップ
def cup(p):
    """コップ。円柱＋上面の暗い円（中が空に見える）＋取っ手。"""
    p = scaled(p, "cup")
    r, h, c = p["r"], p["h"], p["col"]
    out = []
    A = out.append
    A(GCY.format(f="0 0 %.5f 0 0 %.5f" % (-h * .5, h * .5), r="%.5f" % r, c=c))
    # 飲み口（中の暗がり）
    A(GCY.format(f="0 0 %.5f 0 0 %.5f" % (h * .44, h * .5),
                 r="%.5f" % (r * .86), c=p["inner_col"]))
    # 底の縁
    A(GCY.format(f="0 0 %.5f 0 0 %.5f" % (-h * .5, -h * .43),
                 r="%.5f" % (r * 1.04), c=p["rim_col"]))
    # 取っ手：弧を3本のカプセルでつくる
    if p["handle"]:
        hr_ = p["handle_r"]
        pts = []
        for i in range(4):
            a = math.radians(-70 + 140 * i / 3.0)
            pts.append((r * .92 + hr_ * math.cos(a), hr_ * math.sin(a)))
        for i in range(3):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            A(GC.format(f="%.5f 0 %.5f %.5f 0 %.5f" % (x0, z0, x1, z1),
                        r="%.5f" % (r * .10), c=c))
    return "".join(out), h * .5


# ---------------------------------------------------------------- ぼうし
def hat(p):
    """ぼうし。薄い円柱のつば＋半球のクラウン＋色の違うバンド。"""
    p = scaled(p, "hat")
    br_, bt = p["brim_r"], p["brim_t"]
    cr, ch = p["crown_r"], p["crown_h"]
    c = p["col"]
    out = []
    A = out.append
    z0 = -ch * .45
    A(GCY.format(f="0 0 %.5f 0 0 %.5f" % (z0 - bt, z0 + bt), r="%.5f" % br_, c=c))
    A(G.format(t="ellipsoid", s="%.5f %.5f %.5f" % (cr, cr, ch),
               p="0 0 %.5f" % (z0 + bt), c=c, extra=""))
    # バンド
    A(GCY.format(f="0 0 %.5f 0 0 %.5f" % (z0 + bt, z0 + bt + ch * p["band_at"]),
                 r="%.5f" % (cr * 1.03), c=p["band_col"]))
    return "".join(out), ch * .45 + bt


# ---------------------------------------------------------------- 個体の定義（追加6語）
CAT_BASE = dict(body_l=.098, body_r=.034, head_r=.036, head_up=1.60, muzzle=.26,
                ear_w=.16, ear_h=.52, ear_tilt=18,
                leg_up=.026, leg_lo=.024, leg_r=.0095,
                tail_l=.080, tail_r=.0085, tail_curl=46,
                col=".58 .56 .54 1", col2=".92 .90 .88 1", ear_col=".46 .44 .42 1",
                eye_col=".30 .62 .32 1")
NEKOS = [
    ("猫1", dict(CAT_BASE)),
    ("猫2", dict(CAT_BASE, body_l=.086, body_r=.030, leg_up=.022, leg_lo=.020,
                 head_r=.033, ear_h=.62, tail_l=.090, tail_curl=64,
                 col=".16 .15 .16 1", col2=".34 .33 .34 1", ear_col=".12 .11 .12 1",
                 eye_col=".86 .74 .22 1")),                                # 黒くて細い
    ("猫3", dict(CAT_BASE, body_l=.106, body_r=.040, leg_up=.030, leg_lo=.026,
                 head_r=.040, muzzle=.32, ear_w=.19, ear_h=.44, tail_l=.070,
                 col=".92 .88 .82 1", col2=".97 .95 .92 1", ear_col=".82 .74 .68 1")),
    ("猫4", dict(CAT_BASE, body_l=.092, body_r=.036, leg_up=.024, leg_lo=.023,
                 ear_h=.58, ear_tilt=24, tail_l=.086, tail_curl=56,
                 col=".80 .58 .28 1", col2=".94 .88 .76 1", ear_col=".66 .46 .20 1")),
    ("猫5", dict(CAT_BASE, body_l=.102, body_r=.032, leg_up=.028, leg_lo=.025,
                 head_r=.034, muzzle=.22, ear_h=.56, tail_l=.094, tail_r=.0075,
                 col=".44 .43 .46 1", col2=".88 .87 .89 1", ear_col=".34 .33 .36 1",
                 eye_col=".34 .58 .82 1")),
]

TRAIN_BASE = dict(body_l=.115, body_w=.036, body_h=.026, roof_h=.008,
                  wheel_r=.014, windows=4, wheels=4,
                  col=".18 .42 .74 1", roof_col=".86 .86 .88 1",
                  band_col=".92 .90 .30 1")
TRAINS = [
    ("電車1", dict(TRAIN_BASE)),
    ("電車2", dict(TRAIN_BASE, body_l=.128, body_h=.022, windows=5, wheels=6,
                   col=".76 .16 .16 1", band_col=".95 .95 .95 1")),
    ("電車3", dict(TRAIN_BASE, body_l=.104, body_w=.040, body_h=.030, windows=3,
                   wheel_r=.016, col=".22 .58 .34 1", roof_col=".62 .62 .66 1",
                   band_col=".95 .92 .40 1")),
    ("電車4", dict(TRAIN_BASE, body_l=.122, body_w=.033, body_h=.024, windows=5,
                   wheels=4, wheel_r=.013, col=".92 .92 .90 1",
                   roof_col=".55 .58 .62 1", band_col=".24 .40 .78 1")),
    ("電車5", dict(TRAIN_BASE, body_l=.110, body_w=.038, body_h=.028, windows=4,
                   wheels=6, wheel_r=.015, col=".94 .60 .16 1",
                   roof_col=".80 .80 .82 1", band_col=".30 .30 .32 1")),
]

BALL_BASE = dict(r=.050, bands=2, band_w=.14, col=".88 .30 .22 1",
                 band_col=".96 .96 .94 1")
BALLS = [
    ("ボール1", dict(BALL_BASE)),
    ("ボール2", dict(BALL_BASE, r=.044, bands=3, band_w=.10,
                     col=".20 .40 .78 1", band_col=".95 .90 .25 1")),
    ("ボール3", dict(BALL_BASE, r=.058, bands=1, band_w=.20,
                     col=".95 .94 .92 1", band_col=".22 .22 .24 1")),
    ("ボール4", dict(BALL_BASE, r=.046, bands=2, band_w=.16,
                     col=".26 .66 .34 1", band_col=".92 .92 .90 1")),
    ("ボール5", dict(BALL_BASE, r=.054, bands=3, band_w=.12,
                     col=".95 .68 .18 1", band_col=".72 .26 .60 1")),
]

BANANA_BASE = dict(len=.088, r=.0135, curve=105, taper=.62,
                   col=".95 .86 .26 1", tip_col=".44 .34 .16 1")
BANANAS = [
    ("ばなな1", dict(BANANA_BASE)),
    ("ばなな2", dict(BANANA_BASE, len=.078, r=.0155, curve=130, taper=.70,
                     col=".92 .80 .20 1")),
    ("ばなな3", dict(BANANA_BASE, len=.098, r=.0120, curve=85, taper=.55,
                     col=".97 .91 .40 1", tip_col=".36 .46 .18 1")),
    ("ばなな4", dict(BANANA_BASE, len=.082, r=.0145, curve=118, taper=.66,
                     col=".90 .74 .18 1", tip_col=".30 .24 .12 1")),
    ("ばなな5", dict(BANANA_BASE, len=.092, r=.0128, curve=95, taper=.58,
                     col=".96 .89 .32 1")),
]

CUP_BASE = dict(r=.030, h=.062, handle=True, handle_r=.016,
                col=".92 .92 .94 1", inner_col=".38 .38 .42 1",
                rim_col=".72 .72 .76 1")
CUPS = [
    ("コップ1", dict(CUP_BASE)),
    ("コップ2", dict(CUP_BASE, r=.026, h=.074, handle=False,
                     col=".26 .50 .80 1", inner_col=".14 .26 .44 1",
                     rim_col=".20 .38 .62 1")),
    ("コップ3", dict(CUP_BASE, r=.036, h=.050, handle_r=.020,
                     col=".90 .34 .26 1", inner_col=".42 .16 .12 1",
                     rim_col=".70 .26 .20 1")),
    ("コップ4", dict(CUP_BASE, r=.028, h=.068, handle=False,
                     col=".30 .66 .40 1", inner_col=".16 .34 .22 1",
                     rim_col=".24 .50 .32 1")),
    ("コップ5", dict(CUP_BASE, r=.033, h=.058, handle_r=.018,
                     col=".95 .82 .30 1", inner_col=".46 .38 .14 1",
                     rim_col=".76 .64 .22 1")),
]

HAT_BASE = dict(brim_r=.052, brim_t=.0035, crown_r=.032, crown_h=.030,
                band_at=.34, col=".36 .34 .32 1", band_col=".88 .86 .82 1")
HATS = [
    ("ぼうし1", dict(HAT_BASE)),
    ("ぼうし2", dict(HAT_BASE, brim_r=.046, crown_r=.034, crown_h=.038,
                     band_at=.28, col=".78 .22 .20 1", band_col=".95 .94 .90 1")),
    ("ぼうし3", dict(HAT_BASE, brim_r=.058, crown_r=.030, crown_h=.024,
                     band_at=.42, col=".92 .88 .74 1", band_col=".42 .34 .24 1")),
    ("ぼうし4", dict(HAT_BASE, brim_r=.048, crown_r=.036, crown_h=.034,
                     band_at=.30, col=".22 .38 .70 1", band_col=".90 .90 .92 1")),
    ("ぼうし5", dict(HAT_BASE, brim_r=.055, crown_r=.031, crown_h=.028,
                     band_at=.38, col=".30 .58 .38 1", band_col=".95 .82 .28 1")),
]

def _mix_params(pa, pb, w):
    """個体2つのパラメータを w:1-w で数値配合する（2026-09-02・訓練個体の増産）。

    - 数値はそのまま線形配合（wが[0,1]の外なら外挿＝親より極端な個体になる）
    - 色（"r g b a" 形式の文字列）は成分ごとに配合して0〜1へクランプ
    - それ以外の文字列は親Aの値を使う
    【統制の維持】乱数を使わない決定的な配合＝「どの親をどの重みで混ぜたか」を
    数値で言える。再現も差し替えも可能。
    """
    out = {}
    for k in set(pa) | set(pb):
        a, b = pa.get(k), pb.get(k)
        if a is None or b is None:
            out[k] = a if a is not None else b
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            v = a * w + b * (1.0 - w)
            # 両親が整数（個数・段数など）なら整数のまま保つ（range()等で使われる）
            out[k] = int(round(v)) if (isinstance(a, int) and isinstance(b, int)
                                       and not isinstance(a, bool)) else v
        elif isinstance(a, str) and isinstance(b, str):
            try:
                va = [float(x) for x in a.split()]
                vb = [float(x) for x in b.split()]
                out[k] = " ".join("%.3f" % min(1.0, max(0.0, x * w + y * (1.0 - w)))
                                  for x, y in zip(va, vb))
            except ValueError:
                out[k] = a
        else:
            out[k] = a
    return out


def _expand_training_individuals(items, cat_ja):
    """訓練用個体6〜10を、**訓練用の個体1〜3だけ**を親に決定配合で作る
    （2026-09-02・F2-40 経験の多様性増強）。

    【なぜ親を1〜3に限るか】個体4・5は般化テスト用の「初見」。それらを親にすると
    テスト個体の特徴が訓練に漏れ、般化の測定が壊れる。
    配合表（5体）：中間形3つ＋外挿形2つ（親より極端な個体）。
    """
    p1, p2, p3 = items[0][1], items[1][1], items[2][1]
    recipes = [
        ("6", _mix_params(p1, p2, 0.5)),      # 1と2の中間
        ("7", _mix_params(p2, p3, 0.5)),      # 2と3の中間
        ("8", _mix_params(p1, p3, 0.5)),      # 1と3の中間
        ("9", _mix_params(p1, p2, 1.4)),      # 1を2から遠ざけた外挿
        ("10", _mix_params(p3, p2, 1.4)),     # 3を2から遠ざけた外挿
    ]
    base = items[0][0].rstrip("0123456789")   # 「犬1」→「犬」
    return [(base + n, prm) for n, prm in recipes]


CATS = [("わんわん", DOGS, dog), ("にゃんにゃん", NEKOS, cat),
        ("ぶーぶー", CARS, car), ("でんしゃ", TRAINS, train),
        ("りんご", APPLES, apple), ("ボール", BALLS, ball),
        ("くつ", SHOES, shoe), ("ばなな", BANANAS, banana),
        ("コップ", CUPS, cup), ("ぼうし", HATS, hat)]
# 【2026-09-02】各カテゴリに訓練用個体6〜10を追加（index 5〜9）。
#   個体4・5（index 3・4）＝般化テスト用の位置は変えない。
for _ja, _items, _fn in CATS:
    _items.extend(_expand_training_individuals(_items, _ja))


# ============================================================ 毛の模様（2026-09-02）
# 【なぜ】単色の犬猫だけでは人間のおもちゃより多様性が乏しい（ユーザー指示で追加）。
# 模様はnumpyで決定的に生成した自作テクスチャ（権利フリー・再現可能・統制は数値のまま）。
# 貼るのは胴・首・頭（rgba=col の最初の5 geom）だけ。脚・耳・尾は単色のまま。
TEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "assets", "textures")
# 個体名 → (種別, 乱数の種, 地の色, 模様色1, 模様色2)。種別: buchi=ぶち / shima=しま
PATTERNS = {
    "犬6":  ("buchi", 61, (0.60, 0.40, 0.22), (0.94, 0.91, 0.85), (0.28, 0.19, 0.12)),
    "犬8":  ("buchi", 83, (0.94, 0.92, 0.88), (0.15, 0.13, 0.12), (0.15, 0.13, 0.12)),
    "犬10": ("buchi", 105, (0.95, 0.93, 0.90), (0.55, 0.36, 0.20), (0.45, 0.30, 0.18)),
    "猫7":  ("shima", 72, (0.62, 0.55, 0.45), (0.32, 0.28, 0.22), None),
    "猫10": ("shima", 104, (0.82, 0.80, 0.78), (0.45, 0.43, 0.42), None),
}
_TEX_KEY = {"犬6": "dog6", "犬8": "dog8", "犬10": "dog10",
            "猫7": "cat7", "猫10": "cat10"}


def _save_png(path, img_u8):
    import struct, zlib
    h, w = img_u8.shape[:2]
    raw = b"".join(b"\x00" + img_u8[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = t + d
        import struct as _s
        return _s.pack(">I", len(d)) + c + _s.pack(">I", zlib.crc32(c))
    with open(path, "wb") as fp:
        fp.write(b"\x89PNG\r\n\x1a\n"
                 + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def ensure_pattern_assets():
    """模様テクスチャPNGを生成する（決定的・既にあれば上書きで同一内容）。"""
    import numpy as _np
    os.makedirs(TEX_DIR, exist_ok=True)
    W = 128
    yy, xx = _np.mgrid[0:W, 0:W]
    for name, (kind, seed, base, m1, m2) in PATTERNS.items():
        rng = _np.random.default_rng(seed)
        img = _np.zeros((W, W, 3)); img[:] = base
        if kind == "buchi":
            for _ in range(6):     # 大きめの斑6個（試作2で確認した見え方）
                cx, cy = rng.integers(0, W, 2)
                r = int(rng.integers(22, 40))
                col = m1 if rng.random() < 0.6 else (m2 or m1)
                dx = _np.minimum(_np.abs(xx - cx), W - _np.abs(xx - cx))
                dy = _np.minimum(_np.abs(yy - cy), W - _np.abs(yy - cy))
                img[dx * dx + dy * dy < r * r] = col
        else:                       # shima：太い縞3本（試作2の細かすぎを修正）
            stripe = (_np.sin(xx / W * 2 * _np.pi * 3
                              + 1.2 * _np.sin(yy / W * 2 * _np.pi)) > 0.2)
            img[stripe] = m1
        _save_png(os.path.join(TEX_DIR, "%s.png" % _TEX_KEY[name]),
                  (img * 255).astype(_np.uint8))


def pattern_assets_xml():
    """シーンXMLの<asset>に入れる texture/material 断片（模様つき個体の分だけ）。"""
    ensure_pattern_assets()
    root = TEX_DIR.replace(os.sep, "/")
    out = []
    for name in PATTERNS:
        k = _TEX_KEY[name]
        out.append('<texture name="tex_%s" type="cube" file="%s/%s.png"/>' % (k, root, k))
        out.append('<material name="mat_%s" texture="tex_%s" texuniform="true"/>' % (k, k))
    return "".join(out)


def apply_pattern(geoms_xml, name, prm):
    """個体名がPATTERNSにあれば、胴・首・頭（rgba=colの最初の5 geom）へ模様を貼る。"""
    if name not in PATTERNS:
        return geoms_xml
    col = prm.get("col")
    if not col:
        return geoms_xml
    return geoms_xml.replace('rgba="%s"' % col,
                             'material="mat_%s"' % _TEX_KEY[name], 5)
# 【2026-08-30・10語化】そっくりペア（犬-猫／車-電車／りんご-ボール）を隣どうしに
#   置いてある。表引きが壊れるとしたらこの3組で、単独の4語（くつ・ばなな・コップ・
#   ぼうし）は壊れないはず、という予想を測る。

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

    nr = len(CATS)
    fig, ax = plt.subplots(nr, 5, figsize=(15.0, 3.1 * nr))
    for r, (cat, items, _) in enumerate(CATS):
        for i, (name, _p) in enumerate(items):
            ax[r, i].imshow(imgs[name])
            ax[r, i].set_xticks([])
            ax[r, i].set_yticks([])
            tag = "学習用" if i < 3 else "テスト用"
            ax[r, i].set_title("%s（%s）" % (name, tag), fontsize=10,
                               color="#333" if i < 3 else "#b34")
        ax[r, 0].set_ylabel(cat, fontsize=14)
    fig.suptitle("基本図形で作った%d語 × 5個体（左3つが学習用・右2つがテスト用）" % nr, fontsize=14)
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
