"""家の場面の目の入口の作り方。

* dinov2（本物の目、PC）：目だけの場面と同じ eye_backend.install_eye(brain, "dinov2")。AI2-THOR の本当の絵から
  DINOv2 で見た目を取り、物を切り出し、目立ちの地図を作る（目の入口の中身は変えない）。
  段階9の代わりの目（table）は使えない（見た目の素の表は目だけの場面の自作・本物の物のもの。家の物とは違う）
* seg（クラウドの確かめ用の代わりの目。**DINOv2 ではない／絵から物を切り出さない**）：DINOv2 の重みが無いクラウドで、
  家の場面の配線（目が動く・親が持ってくる・名前を言う・動画）を脳ごと確かめるためだけの目。物の切り出しの代わりに
  環境の正解（色分けの絵 env.object_centres）の見えている所の真ん中を使い、見た目は物の種類ごとに決まった乱数の 384
  （同じ種類は同じ見た目。DINOv2 の見た目の代わり）。そのあとは目の入口と同じ（tagging.appearance_tag で 64 に縮め、
  同じ雑音を足し、物の中心に山、目立ちの地図は物の山の最大、画面のずれは本物と同じ Eyein.ego_shift）
"""

from __future__ import annotations

import hashlib

import numpy as np

APP_DIM = 384


def type_appearance(obj_type: str) -> np.ndarray:
    """物の種類ごとに決まった 384 の見た目（seg の目だけが使う）。"""
    seed = int.from_bytes(hashlib.sha256(("thor:" + obj_type).encode("utf-8")).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(APP_DIM)
    return v / np.linalg.norm(v)


def seg_dets(env) -> list[dict]:
    """写っている、名前を言える物ごとに {"appearance": 384, "pos": (列, 行)（224 の座標）, "name": id}。"""
    out = []
    for oid, (c, r, _n) in env.object_centres(px=224).items():
        out.append({"appearance": type_appearance(env.objects[oid]["objectType"]).astype("float64"),
                    "pos": (float(c), float(r)), "name": oid})
    return out


def _seg_see(self, img):
    """Eyein.see の代わり（eye_backend._table_see と同じ形。見た目と場所だけを環境の正解から）。"""
    import torch
    from ..boxes.eyein import bump
    from ..formats import DTYPE, GRID, MAX_OBJECTS
    from ..tagging import appearance_tag, noisy
    img224 = np.asarray(img)
    if img224.shape[:2] != (224, 224):
        from PIL import Image
        img224 = np.array(Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8)).resize((224, 224)))
    gray = img224.astype(np.float32).mean(axis=2)
    self.ego_shift(gray)
    dets = seg_dets(self._thor_env())
    sal = torch.zeros(GRID, GRID, dtype=DTYPE)
    for d in dets:
        sal = torch.maximum(sal, bump(*d["pos"], sigma=2.0))
    step = 224.0 / GRID
    for d in dets:
        d["sal"] = float(sal[int(min(d["pos"][1] // step, GRID - 1)), int(min(d["pos"][0] // step, GRID - 1))])
    dets = sorted(dets, key=lambda d: -d["sal"])[:MAX_OBJECTS]
    self.last_dets = dets
    self.last_mode = "seg"
    return [(noisy(appearance_tag(d["appearance"])), bump(*d["pos"])) for d in dets], sal


def install_thor_eye(brain, backend: str, env_ref=None) -> dict:
    """backend="dinov2"：本物の目の入口。backend="seg"：上の代わりの目（env_ref は今の環境を返す関数）。"""
    import types
    if backend == "dinov2":
        from ..world_eye.eye_backend import install_eye
        return install_eye(brain, "dinov2")
    if backend != "seg":
        raise ValueError(f"家の場面の目は dinov2 か seg（{backend}）")
    eyein = brain.boxes["eyein"]
    eyein._thor_env = env_ref
    eyein.see = types.MethodType(_seg_see, eyein)
    eyein._ready = True
    return {"eye": "seg", "note": "代わりの目（環境の色分けの正解と種類ごとの乱数の見た目。DINOv2 ではない）"}
