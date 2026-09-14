"""「太郎に見えているか」を判定する共通の道具。

【なぜ作ったか】2026-07-26、視界の判定を
    視線ベクトルと対象方向のなす角 < 視野の半角(30度)
だけで行っていた。おもちゃがベビーサークルの柵の**向こう側**（X=0.353・柵は0.310）に
あっても、なす角は 4.5度なので「視界内 100%」と報告し続けた。
ユーザーの目視「今 視界にないおもちゃが、柵に引っかかって」で初めて分かった。
→ 検証の落とし穴チェックリスト 項51

【3つの判定を用意する。強さの順】

  ①角度だけ         `gaze_angle()`
      視線と対象のなす角。**遮蔽を見ない**ので単独では使わない。補助。

  ②光線を飛ばす     `visible_by_ray()`
      目から対象へ光線を飛ばし、最初に当たる物体が対象かを見る。
      幾何学的に確実。注意ただし「太郎の脳に届く情報」ではない（視力・解像度を経ていない）。

  ③画像に映っているか `visible_in_image()`   これが本命
      太郎の眼球カメラが実際に描いた画像に、対象の色が何画素あるかを数える。
      視力のぼかし（acuity）・解像度・視野を全部通った後なので、
      **太郎が実際に受け取っている情報そのもの**。

②と③は独立に効く。②が真で③が偽なら「幾何学的には見えるが、視力が足りず
埋もれている」＝太郎にとっては見えていない、と読める。
"""
import numpy as np
import mujoco

# 太郎の視野の半角[度]（fovy=60）。1画素 = 60/128 = 0.469度。
HALF_FOV = 30.0

# おもちゃの色（e_toy_env の TOY_RGBA_OFF / _ON）。
TOY_RGB_OFF = np.array([0.9, 0.2, 0.15])     # 通常＝赤
TOY_RGB_ON = np.array([1.0, 1.0, 0.45])      # 接触中＝明るい黄

# 色で拾うときの許容幅。注意視力のぼかし（acuity）で色が薄まるので緩めに取る。
# 「赤が飛び抜けて強い画素」を拾う条件にして、しきい値の恣意性を減らす。
RED_DOMINANCE = 1.35     # R が G・B の何倍以上なら「赤い」とみなすか
RED_MIN = 0.30           # R の下限（暗すぎる画素は拾わない）


def gaze_angle(model, data, toy_bid, camera="eye_left"):
    """視線と対象のなす角[度]。注意遮蔽を見ないので単独で使わないこと。"""
    cid = int(model.camera(camera).id)
    cpos = data.cam_xpos[cid]
    fwd = -np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
    v = np.array(data.xpos[toy_bid], dtype=float) - cpos
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(fwd, v / n), -1, 1))))


def visible_by_ray(model, data, toy_bid, camera="eye_left", exclude_head=True):
    """目から対象へ光線を飛ばし、遮られていないかを見る。

    Returns:
        (bool, str): 見えているか／最初に当たった物体の名前
    """
    cid = int(model.camera(camera).id)
    pnt = np.array(data.cam_xpos[cid], dtype=float)
    target = np.array(data.xpos[toy_bid], dtype=float)
    vec = target - pnt
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return True, "(重なっている)"
    vec = vec / dist

    # 自分の頭（まぶた・鼻など）に当たるのを避ける。カメラは頭の中にあるため。
    bodyexclude = int(model.body("head").id) if exclude_head else -1
    geomid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, pnt, vec, None, 1, bodyexclude, geomid)
    if geomid[0] < 0:
        return False, "(何にも当たらない)"
    gid = int(geomid[0])
    hit_bid = int(model.geom_bodyid[gid])
    name = model.geom(gid).name or f"<{model.body(hit_bid).name}の無名geom>"
    # 対象より手前で何かに当たっていたら遮られている
    if hit_bid == toy_bid:
        return True, name
    if hit >= dist - 1e-4:
        return True, "(手前に何もない)"
    return False, name


def red_mask(image):
    """画像から「赤が飛び抜けて強い画素」を拾う。おもちゃ（赤）の検出用。

    注意：太郎の視覚は視力のぼかしを通っているので、色は薄まる。
      絶対値でなく**チャンネル間の比**で見ることで、ぼけに強くする。
    """
    a = np.asarray(image, dtype=float)
    if a.max() > 1.5:
        a = a / 255.0
    if a.ndim != 3 or a.shape[2] < 3:
        return np.zeros(a.shape[:2], dtype=bool)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > RED_MIN) & (r > RED_DOMINANCE * g) & (r > RED_DOMINANCE * b)


def visible_in_image(image, min_pixels=1):
    """太郎の目に実際に映っているか。赤い画素の数と、その重心を返す。

    Returns:
        dict: seen(bool) / n_pixels(int) / frac(画面に占める割合) /
              cx, cy(重心。画像の中心を0とし、-1〜1で正規化)
    """
    m = red_mask(image)
    n = int(m.sum())
    h, w = m.shape[:2]
    out = dict(seen=n >= int(min_pixels), n_pixels=n, frac=n / float(h * w),
               cx=float("nan"), cy=float("nan"))
    if n > 0:
        ys, xs = np.nonzero(m)
        out["cx"] = float((xs.mean() - (w - 1) / 2.0) / ((w - 1) / 2.0))
        out["cy"] = float((ys.mean() - (h - 1) / 2.0) / ((h - 1) / 2.0))
    return out


# ---- 描き分けによる正確な判定（2026-07-27 追加）------------------------
# 色による判定（red_mask）は**本当の画素の46%しか拾えていなかった**
#   本当のおもちゃ 2352画素 → 色で拾えたのは 1086画素、取りこぼし 1266画素
#   ユーザーの目視「赤い部分があるのに緑に光ってないときがある」で発覚
# 取りこぼしは**縁に集中**する（ぼかしで背景と混ざり「赤が飛び抜けている」と
# 言えなくなる）ため、重心が中心寄りにずれていた。
#
# MuJoCo の segmentation rendering は「どの画素がどの geom か」を正確に返す。
# ただし**太郎が受け取る情報ではない**（視力のぼかしを通っていない）。
#   使い分け：
#     「おもちゃが本当はどこにあるか」  → segment_mask（正確）
#     「太郎に見分けられるか」          → red_mask（ぼかし後の画像）
_SEG_CACHE = {}


def segment_mask(model, data, body_id, camera="eye_left", size=128):
    """MuJoCo に描き分けさせて、対象の画素を正確に取り出す。

    Returns:
        np.ndarray: True/False の2次元マップ（対象の画素が True）
    """
    key = (id(model), camera, int(size))
    ren = _SEG_CACHE.get(key)
    if ren is None:
        ren = mujoco.Renderer(model, height=int(size), width=int(size))
        ren.enable_segmentation_rendering()
        _SEG_CACHE[key] = ren
    ren.update_scene(data, camera=camera)
    seg = ren.render()
    gids = [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) == int(body_id)]
    return np.isin(seg[..., 0], gids)


def visible_by_segment(model, data, body_id, camera="eye_left", size=128,
                       min_pixels=1):
    """描き分けで「対象が視野に何画素映っているか」と重心を返す。

    Returns:
        dict: seen / n_pixels / frac / cx, cy（画像中心を0、端を±1）
    """
    msk = segment_mask(model, data, body_id, camera, size)
    n = int(msk.sum())
    h, w = msk.shape[:2]
    out = dict(seen=n >= int(min_pixels), n_pixels=n, frac=n / float(h * w),
               cx=float("nan"), cy=float("nan"))
    if n > 0:
        ys, xs = np.nonzero(msk)
        out["cx"] = float((xs.mean() - (w - 1) / 2.0) / ((w - 1) / 2.0))
        out["cy"] = float((ys.mean() - (h - 1) / 2.0) / ((h - 1) / 2.0))
    return out


def close_renderers():
    """作った Renderer を片付ける（環境を作り直す前に呼ぶ）。"""
    for r in _SEG_CACHE.values():
        try:
            r.close()
        except Exception:
            pass
    _SEG_CACHE.clear()


def report(model, data, toy_bid, image=None, camera="eye_left"):
    """3つの判定をまとめて返す。食い違いがあれば、それ自体が情報になる。"""
    ang = gaze_angle(model, data, toy_bid, camera)
    ray_ok, hit = visible_by_ray(model, data, toy_bid, camera)
    out = dict(angle=ang, in_fov=bool(ang < HALF_FOV), ray_ok=ray_ok, ray_hit=hit,
               pix_seen=None, n_pixels=0, frac=0.0, cx=float("nan"), cy=float("nan"),
               seg_seen=None, seg_pixels=0, seg_cx=float("nan"), seg_cy=float("nan"))
    # 描き分けによる正確な位置（色の判定は本当の46%しか拾えない）
    try:
        sz = 128
        if image is not None:
            sz = int(np.asarray(image).shape[0])
        sv = visible_by_segment(model, data, toy_bid, camera, size=sz)
        out.update(seg_seen=sv["seen"], seg_pixels=sv["n_pixels"],
                   seg_cx=sv["cx"], seg_cy=sv["cy"])
    except Exception:
        pass
    if image is not None:
        v = visible_in_image(image)
        out.update(pix_seen=v["seen"], n_pixels=v["n_pixels"], frac=v["frac"],
                   cx=v["cx"], cy=v["cy"])
    return out


def describe(r):
    """report() の結果を1行の日本語にする。"""
    s = f"なす角{r['angle']:5.1f}° "
    s += "視野内 " if r["in_fov"] else "視野外 "
    s += "遮蔽なし " if r["ray_ok"] else f"{r['ray_hit']}に遮られている "
    if r.get("seg_seen") is not None:
        s += (f"実際に{r['seg_pixels']:4d}画素(ずれ{np.hypot(r['seg_cx'], r['seg_cy']):.2f}) "
              if r["seg_seen"] else "視野に入っていない ")
    if r["pix_seen"] is not None:
        s += (f"／色で拾えたのは{r['n_pixels']:4d}画素" if r["pix_seen"]
              else "／色では拾えない")
    return s
