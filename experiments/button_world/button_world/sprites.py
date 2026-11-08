"""絵の部品（スプライト）を、1マスの画素数 P に合わせて numpy で作る。

起動時に1回だけ作り、表（N, P, P, 3）にして env に渡す。描画は番号からの表引きだけになる。
形は 0〜1 の連続座標で定義し、各画素の中心で判定する。P を変えても同じ形が粗く・細かく描かれる。
"""
import numpy as np

# ---- 色 ----
# ボタンと飾りに使う6色（背景と紛れない鮮やかな色）。番号が「色」の手がかり。
COLORS = np.array(
    [
        [225, 45, 45],  # 0 赤
        [245, 145, 25],  # 1 橙
        [240, 220, 35],  # 2 黄
        [45, 195, 70],  # 3 緑
        [45, 95, 235],  # 4 青
        [175, 65, 215],  # 5 紫
    ],
    dtype=np.float32,
)
N_COLORS = len(COLORS)

# 視点: 0 = 上から、1 = 横から。絵柄はそれぞれ3つ（0・1 が練習用、2 が試し用）。
N_VIEWS = 2
N_THEMES = 3
N_VT = N_VIEWS * N_THEMES  # vt = view * N_THEMES + theme

# 絵柄ごとの背景（床／空、壁／地面、地面の上の段）
THEMES = {
    # 上から 絵柄0: 草地と石垣
    (0, 0): dict(floor=(118, 150, 96), floor2=(108, 140, 88), wall=(110, 110, 118), wall_line=(80, 80, 88)),
    # 上から 絵柄1: 夜の床タイルとレンガ
    (0, 1): dict(floor=(52, 58, 86), floor2=(62, 68, 98), wall=(150, 78, 60), wall_line=(105, 50, 40)),
    # 横から 絵柄0: 昼の空と土
    (1, 0): dict(sky=(150, 200, 240), sky2=(170, 212, 245), ground=(140, 95, 55), ground_top=(95, 160, 70)),
    # 横から 絵柄1: 夕方の空と石
    (1, 1): dict(sky=(70, 45, 95), sky2=(85, 55, 110), ground=(95, 95, 105), ground_top=(140, 140, 150)),
    # 上から 絵柄2（試し用）: 砂の床と木の柵
    (0, 2): dict(floor=(214, 196, 150), floor2=(200, 182, 138), wall=(120, 84, 50), wall_line=(160, 120, 75)),
    # 横から 絵柄2（試し用）: 雪の昼と氷
    (1, 2): dict(sky=(222, 232, 238), sky2=(205, 220, 230), ground=(120, 160, 190), ground_top=(250, 252, 255)),
}
LADDER = {0: (150, 105, 50), 1: (190, 190, 200), 2: (70, 95, 130)}  # 横からの絵柄ごとのはしごの色

# ---- ボタンの形の4特徴 ----
OUTERS = ["円", "四角", "ひし形", "板"]
RINGS = ["なし", "あり"]
HEIGHTS = ["平ら", "盛り上がり"]
MARKS = ["なし", "点", "線"]
N_SHAPES = len(OUTERS) * len(RINGS) * len(HEIGHTS) * len(MARKS)  # 48


def shape_index(outer, ring, height, mark):
    return ((outer * len(RINGS) + ring) * len(HEIGHTS) + height) * len(MARKS) + mark


def shape_features(idx):
    mark = idx % len(MARKS)
    idx //= len(MARKS)
    height = idx % len(HEIGHTS)
    idx //= len(HEIGHTS)
    ring = idx % len(RINGS)
    outer = idx // len(RINGS)
    return outer, ring, height, mark


# ---- 座標 ----
def _grid(P):
    """画素の中心の座標。x は右へ、y は下へ、どちらも -0.5〜0.5。"""
    c = (np.arange(P) + 0.5) / P - 0.5
    y, x = np.meshgrid(c, c, indexing="ij")
    return x, y


def _outer_mask(x, y, outer, scale_y=1.0):
    y = y / scale_y
    if outer == 0:
        return x**2 + y**2 < 0.36**2
    if outer == 1:
        return (np.abs(x) < 0.31) & (np.abs(y) < 0.31)
    if outer == 2:
        return np.abs(x) + np.abs(y) < 0.42
    return (np.abs(x) < 0.43) & (np.abs(y) < 0.15)  # 板


def _paint(img, mask, color):
    img[mask] = color


def button_sprite(bg, P, shape_idx, color, pressed, side):
    """背景 bg（P,P,3）の上にボタンを描く。side=True なら床に置いた形（下寄せ）。"""
    outer, ring, height, mark = shape_features(shape_idx)
    img = bg.copy()
    x, y = _grid(P)
    color = np.asarray(color, dtype=np.float32)
    # 置き方: 横から見る時は床の上に乗るので下へずらす
    dy = 0.08 if side else 0.0
    # 押された形: 縦につぶれて少し下がり、暗くなる（効く・効かないとも同じ）
    sy = 0.85 if pressed else 1.0
    dy += 0.07 if pressed else 0.0
    shade = 0.6 if pressed else 1.0
    yy = y - dy
    body = _outer_mask(x, yy, outer, scale_y=sy)
    if height == 1:  # 盛り上がり: 下に暗い側面を描く
        side_mask = _outer_mask(x, yy - 0.13 * sy, outer, scale_y=sy) & ~body
        _paint(img, side_mask, color * 0.45 * shade)
    _paint(img, body, color * shade)
    if ring == 1:  # 輪: 外形の縁を暗く
        inner = _outer_mask(x * 1.6, yy * 1.6, outer, scale_y=sy)
        _paint(img, body & ~inner, color * 0.35 * shade)
    if mark == 1:  # 点
        _paint(img, (x**2 + ((yy) / sy) ** 2 < 0.11**2) & body, np.array([250, 250, 250]) * shade)
    elif mark == 2:  # 線
        _paint(img, (np.abs(x) < 0.26) & (np.abs(yy / sy) < 0.07) & body, np.array([250, 250, 250]) * shade)
    return img


def deco_sprite(bg, P, color, side):
    """ボタンでない色付きの物。上から＝床の宝石、横から＝花。通れるが何も起きない。"""
    img = bg.copy()
    x, y = _grid(P)
    color = np.asarray(color, dtype=np.float32)
    if side:
        stem = (np.abs(x) < 0.04) & (y > 0.0) & (y < 0.5)
        leaf = ((x - 0.1) ** 2 / 0.12**2 + (y - 0.28) ** 2 / 0.05**2) < 1
        _paint(img, stem | leaf, np.array([40, 130, 50]))
        petals = np.zeros_like(stem)
        for a in range(5):
            cx, cy = 0.13 * np.cos(2 * np.pi * a / 5), -0.1 + 0.13 * np.sin(2 * np.pi * a / 5)
            petals |= (x - cx) ** 2 + (y - cy) ** 2 < 0.09**2
        _paint(img, petals, color)
        _paint(img, x**2 + (y + 0.1) ** 2 < 0.06**2, np.array([250, 230, 120]))
    else:
        gem = (np.abs(x) / 0.22 + np.abs(y) / 0.3) < 1
        _paint(img, gem, color)
        facet = (np.abs(x + 0.06) / 0.07 + np.abs(y + 0.1) / 0.09) < 1
        _paint(img, facet & gem, np.minimum(color * 0.4 + 160, 255))
    return img


def door_sprite(bg, P, opened, side, frame_color):
    img = bg.copy()
    x, y = _grid(P)
    frame = (np.abs(x) < 0.4) & (np.abs(y) < 0.46)
    _paint(img, frame, np.asarray(frame_color))
    inner = (np.abs(x) < 0.3) & (y > -0.36) & (y < 0.46)
    if opened:
        _paint(img, inner, np.array([15, 15, 20]))
    else:
        _paint(img, inner, np.array([150, 100, 50]))
        _paint(img, (np.abs(x) < 0.3) & (np.abs(y - 0.05) < 0.025), np.array([110, 70, 35]))
        _paint(img, (x - 0.18) ** 2 + (y - 0.08) ** 2 < 0.045**2, np.array([240, 210, 80]))
    return img


def agent_sprite(P, side):
    """自分。白と黒だけ（6色と紛れない）。(P,P,3) と、描く所の印 (P,P)。"""
    img = np.zeros((P, P, 3), np.float32)
    x, y = _grid(P)
    if side:
        head = x**2 + (y + 0.27) ** 2 < 0.15**2
        body = (np.abs(x) < 0.17) & (y > -0.12) & (y < 0.3)
        legs = ((np.abs(x - 0.09) < 0.05) | (np.abs(x + 0.09) < 0.05)) & (y >= 0.3) & (y < 0.48)
        mask = head | body | legs
        img[mask] = [245, 245, 245]
        img[legs] = [30, 30, 30]
        eye = (x - 0.06) ** 2 + (y + 0.29) ** 2 < 0.035**2
        img[eye] = [20, 20, 20]
    else:
        # 上から: 上向きの矢印（円も輪も使わない。ボタンの「円・輪あり」と紛れないように）
        head = (y < 0.05) & (y > -0.45) & (np.abs(x) < (y + 0.45) * 0.9)
        stem = (np.abs(x) < 0.13) & (y >= 0.05) & (y < 0.45)
        mask = head | stem
        img[mask] = [245, 245, 245]
        notch = (np.abs(x) < 0.05) & (y > -0.05) & (y < 0.35)  # 真ん中の黒い筋で向きを強調
        img[notch & mask] = [20, 20, 20]
    return img, mask


def background_tiles(P, view, theme):
    """背景4種: 0 床／空、1 壁／地面、2 段の床（横から）、3 はしご（横から）。上からの 2・3 は使わないが形をそろえる。"""
    t = THEMES[(view, theme)]
    x, y = _grid(P)
    tiles = np.zeros((4, P, P, 3), np.float32)
    if view == 0:
        tiles[0] = t["floor"]
        tiles[0][(np.abs(x) > 0.42) | (np.abs(y) > 0.42)] = t["floor2"]  # 目地
        tiles[1] = t["wall"]
        tiles[1][(np.abs(y) < 0.04) | ((np.abs(x) < 0.04) & (y < 0)) | ((np.abs(x - 0.5) < 0.04) & (y > 0)) | ((np.abs(x + 0.5) < 0.04) & (y > 0))] = t["wall_line"]
        tiles[2] = tiles[1]
        tiles[3] = tiles[0]
    else:
        tiles[0] = t["sky"]
        tiles[0][(y > 0.3)] = t["sky2"]
        tiles[1] = t["ground"]
        tiles[1][((x * 7).astype(int) + (y * 7).astype(int)) % 3 == 0] = np.asarray(t["ground"]) * 0.85
        tiles[2] = t["ground"]
        tiles[2][y < -0.2] = t["ground_top"]
        tiles[3] = tiles[0]
        rails = (np.abs(np.abs(x) - 0.3) < 0.06)
        rungs = (np.abs(x) < 0.3) & (np.abs(((y + 0.5) * 4) % 1 - 0.5) < 0.14)
        tiles[3][rails | rungs] = LADDER[theme]
    return tiles


DOOR_FRAME = {(0, 0): (70, 70, 78), (0, 1): (90, 40, 30), (0, 2): (90, 60, 35), (1, 0): (90, 60, 30), (1, 1): (55, 55, 62), (1, 2): (60, 90, 120)}
N_BG = 4


# ---- 表の番号の並び ----
def layout():
    """表の中の各部品の番号の始まり。env と同じ式を使う。"""
    bg0 = 0  # vt*N_BG + kind
    door0 = bg0 + N_VT * N_BG  # vt*2 + opened
    deco0 = door0 + N_VT * 2  # vt*N_COLORS + color
    btn0 = deco0 + N_VT * N_COLORS  # ((vt*N_SHAPES + shape)*N_COLORS + color)*2 + pressed
    total = btn0 + N_VT * N_SHAPES * N_COLORS * 2
    return dict(bg0=bg0, door0=door0, deco0=deco0, btn0=btn0, total=total)


SUPER = 32  # 形はこの細かさ（1マスあたりの画素）以上で描いてから、面積の平均で P に縮める


def _factor(P):
    return max(1, -(-SUPER // P))


def _down(img, P, K):
    return img.reshape(P, K, P, K, *img.shape[2:]).mean(axis=(1, 3))


def build_tables(P):
    """全部の部品の表 (N,P,P,3) uint8 と、自分の絵 (2,P,P,3) と、描く濃さ (2,P,P) 0〜1。

    細かく描いてから縮めるので、P が小さくても特徴が色の濃淡として残る。
    """
    K = _factor(P)
    Q = P * K
    L = layout()
    table = np.zeros((L["total"], P, P, 3), np.float32)
    for view in range(N_VIEWS):
        for theme in range(N_THEMES):
            vt = view * N_THEMES + theme
            bgs = background_tiles(Q, view, theme)
            for k in range(N_BG):
                table[L["bg0"] + vt * N_BG + k] = _down(bgs[k], P, K)
            floor = bgs[0]
            for opened in (0, 1):
                table[L["door0"] + vt * 2 + opened] = _down(door_sprite(floor, Q, opened, view == 1, DOOR_FRAME[(view, theme)]), P, K)
            for c in range(N_COLORS):
                table[L["deco0"] + vt * N_COLORS + c] = _down(deco_sprite(floor, Q, COLORS[c], view == 1), P, K)
            for s in range(N_SHAPES):
                for c in range(N_COLORS):
                    for pr in (0, 1):
                        idx = L["btn0"] + ((vt * N_SHAPES + s) * N_COLORS + c) * 2 + pr
                        table[idx] = _down(button_sprite(floor, Q, s, COLORS[c], pr, view == 1), P, K)
    agents = np.zeros((N_VIEWS, P, P, 3), np.float32)
    alphas = np.zeros((N_VIEWS, P, P), np.float32)
    for view in range(N_VIEWS):
        img, mask = agent_sprite(Q, view == 1)
        a = _down(mask.astype(np.float32), P, K)
        col = _down(img * mask[..., None], P, K)
        agents[view] = np.where(a[..., None] > 0, col / np.maximum(a[..., None], 1e-6), 0)
        alphas[view] = a
    return np.clip(np.round(table), 0, 255).astype(np.uint8), np.clip(np.round(agents), 0, 255).astype(np.uint8), alphas


def button_on_neutral(P, shape_idx, color_idx=2):
    """形の似方を測るための、灰色の背景の真ん中に置いたボタン（押す前）。視点や絵柄の違いを含めない。"""
    K = _factor(P)
    Q = P * K
    bg = np.full((Q, Q, 3), 128, np.float32)
    return _down(button_sprite(bg, Q, shape_idx, COLORS[color_idx], 0, False), P, K)


def shape_pixel_diff(a, b, P=5):
    """2つの形の画素の差（0〜255 の平均の絶対差）。"""
    return float(np.abs(button_on_neutral(P, a) - button_on_neutral(P, b)).mean())
