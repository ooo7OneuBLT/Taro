"""目の入口（eyein）の重い部分（DINOv2・目立ちの地図）を、PC でもクラウドでも同じ形で作る。

目だけの場面は、目の入口の中身を変えない。eyein.setup() がそのまま DINOv2（torch.hub の facebookresearch/dinov2、
dinov2_vits14）と目立ちの地図（taro_core の SalienceMap）を作る。ここでは、作る前に「この機械で作れるか」を
1回で調べて、足りない物をはっきり言う（途中で黙って別の目に替えることはしない）。

* PC（GPU）：今すぐ動く（重みは torch.hub の置き場にある）
* クラウド（2026-10-18 時点）：動かない。足りない物は ① DINOv2 の重み（dl.fbaipublicfiles.com・github の
  torch.hub が通信の決まりで止められている。ユーザーに許可を頼んである）② scipy（目立ちの地図と物の切り出しが使う。
  このコンテナに入っていない。mujoco 以外は入れない決まり）。GPU は無いので CPU で回る（1回 数百ミリ秒の見込み）
* 代わりの目（TableEye、段階9。**DINOv2 ではない／画像から物を切り出さない**）：クラウドで段階9の場面の配線を確かめる
  ためだけの目。物の切り出しの代わりに、場面に出ている段階9の個体の中心をカメラへ写した場所（環境の正解）を使い、
  見た目は見た目の素の表（PC の DINOv2 で作った 384、`core_v2/data/段階9_見た目の素.pt`）から、その個体・その向きの行を引く。
  そのあとは目の入口と同じ（tagging.appearance_tag で 64 に縮め、同じ雑音を足し、物の中心に山、目立ちの地図、画面のずれ）。
  目立ちの地図は物の山の最大（本物の目の SalienceMap の代わり）。画面のずれは本物と同じ（Eyein.ego_shift：指令の写し＋画像の位相相関）
使い方：
  python -m core_v2.world_eye.eye_backend        … この機械で目の入口を作れるかを表示
  from core_v2.world_eye.eye_backend import setup_eye; setup_eye(brain)   … 調べてから eyein.setup()
  install_eye(brain, "dinov2"|"table", env_ref=..., bank=...)   … 段階9の道具が使う。どちらでも同じ道具が回る
"""

from __future__ import annotations

import importlib.util
import os


def check() -> dict:
    import torch
    hub = torch.hub.get_dir()
    repo = os.path.join(hub, "facebookresearch_dinov2_main")
    ckpt = os.path.join(hub, "checkpoints", "dinov2_vits14_pretrain.pth")
    out = {
        "scipy": importlib.util.find_spec("scipy") is not None,
        "dinov2_code": os.path.isdir(repo),
        "dinov2_weights": os.path.isfile(ckpt),
        "cuda": bool(torch.cuda.is_available()),
        "hub_dir": hub,
    }
    out["ready"] = out["scipy"] and out["dinov2_code"] and out["dinov2_weights"]
    return out


def missing_text(c: dict) -> str:
    miss = []
    if not c["scipy"]:
        miss.append("scipy（目立ちの地図と物の切り出しが使う）")
    if not c["dinov2_code"]:
        miss.append(f"DINOv2 のコード（{c['hub_dir']}/facebookresearch_dinov2_main。torch.hub が github から取る）")
    if not c["dinov2_weights"]:
        miss.append(f"DINOv2 の重み（{c['hub_dir']}/checkpoints/dinov2_vits14_pretrain.pth。dl.fbaipublicfiles.com から取る）")
    return "、".join(miss)


def setup_eye(brain, allow_download: bool = False, **kw) -> dict:
    """足りない物があれば止める（allow_download=True なら、コードと重みは torch.hub に取りに行かせる）。"""
    c = check()
    if not c["scipy"] or (not c["ready"] and not allow_download):
        raise RuntimeError("この機械では目の入口（DINOv2）を作れない。足りない物：" + missing_text(c))
    brain.boxes["eyein"].setup(**kw)
    c["device"] = brain.boxes["eyein"].device
    return c


# ---------------------------------------------------------------- 段階9：代わりの目
DEFAULT_TABLE = {"made": "core_v2/data/段階9_見た目の素.pt", "real": "core_v2/data/段階9_見た目の素_本物.pt"}


def load_table(path: str):
    """見た目の素の表 → (app[20,4,24,384] float32, found[20,4,24] bool, meta)。"""
    import torch
    from ..tools.growth import REPO
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    blob = torch.load(p, map_location="cpu", weights_only=False)
    return blob["app"].float().numpy(), blob["found"].numpy(), dict(blob.get("meta", {}), path=path)


def project(env, name: str):
    """物の体の中心を左目の画像（px×px）へ写した (列, 行)。後ろ・画像の外なら None。"""
    import numpy as np
    cid = env._cam_id
    R = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)
    v = np.array(env.data.body(name).xpos, dtype=float) - np.array(env.data.cam_xpos[cid], dtype=float)
    x, y, z = R.T @ v
    if z >= -1e-6:
        return None
    f = 0.5 * env.px / np.tan(np.radians(env.fovy) / 2.0)
    cx, cy = 0.5 * env.px + f * x / -z, 0.5 * env.px - f * y / -z
    if not (0.0 <= cx < env.px and 0.0 <= cy < env.px):
        return None
    s = 224.0 / env.px
    return float(cx * s), float(cy * s)


def table_dets(env, app, found) -> list[dict]:
    """場面に出ている段階9の個体ごとに {"appearance": 384, "pos": (列, 行), "name"}（画像の中に中心がある物だけ）。"""
    out = []
    for nm, m in env.n9.items():
        if not env.visible(nm) or nm not in env.n9_view:
            continue
        pos = project(env, nm)
        if pos is None:
            continue
        i, k, v = m["i"], m["k"], int(env.n9_view[nm])
        ok = [u for u in range(found.shape[2]) if found[i, k, u]]
        if not ok:
            continue
        if v not in ok:      # 拾えなかった向きは、いちばん近い番号の向き
            v = min(ok, key=lambda u: min(abs(u - v), found.shape[2] - abs(u - v)))
        out.append({"appearance": app[i, k, v].astype("float64"), "pos": pos, "name": nm})
    return out


def _table_see(self, img):
    """Eyein.see の代わり（上の説明）。物の切り出しと DINOv2 と SalienceMap だけを、環境の正解と表に替える。"""
    import numpy as np
    import torch
    from ..boxes.eyein import bump
    from ..formats import DTYPE, GRID, MAX_OBJECTS
    from ..tagging import appearance_tag, noisy
    img224 = np.asarray(img)
    if img224.shape[:2] != (224, 224):
        from PIL import Image
        img224 = np.array(Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8)).resize((224, 224)))
    gray = img224.astype(np.float32).mean(axis=2)
    self.ego_shift(gray)            # 目が動いた分（指令の写し＋画像の直し。本物の目の入口と同じ）
    dets = table_dets(self._table_env(), self._table_app, self._table_found)
    sal = torch.zeros(GRID, GRID, dtype=DTYPE)
    for d in dets:
        sal = torch.maximum(sal, bump(*d["pos"], sigma=2.0))
    step = 224.0 / GRID
    for d in dets:
        d["sal"] = float(sal[int(min(d["pos"][1] // step, GRID - 1)), int(min(d["pos"][0] // step, GRID - 1))])
    dets = sorted(dets, key=lambda d: -d["sal"])[:MAX_OBJECTS]
    self.last_dets = dets
    self.last_mode = "table"
    return [(noisy(appearance_tag(d["appearance"])), bump(*d["pos"])) for d in dets], sal


def install_eye(brain, backend: str, env_ref=None, table: str | None = None, objects: str = "made") -> dict:
    """段階9の道具の目。backend="dinov2"：本物の目の入口（setup_eye）。backend="table"：代わりの目（上の説明）。
    env_ref は今の環境を返す関数（部屋ごとに環境が替わるため）。返り値は写しの条件に書く情報。"""
    import types
    eyein = brain.boxes["eyein"]
    if backend == "dinov2":
        c = setup_eye(brain)
        return {"eye": "dinov2", "device": c.get("device")}
    if backend != "table":
        raise ValueError(f"目は dinov2 か table（{backend}）")
    path = table or DEFAULT_TABLE[objects]
    app, found, meta = load_table(path)
    eyein._table_env = env_ref
    eyein._table_app, eyein._table_found = app, found
    eyein.see = types.MethodType(_table_see, eyein)
    eyein._ready = True
    return {"eye": "table", "table": path, "table_objects": meta.get("objects", "made"),
            "note": "代わりの目（DINOv2 の表を引く。画像から切り出さない）"}


if __name__ == "__main__":
    c = check()
    print(c)
    print("作れる" if c["ready"] else "作れない：" + missing_text(c))
