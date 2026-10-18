"""目の入口（eyein）の重い部分（DINOv2・目立ちの地図）を、PC でもクラウドでも同じ形で作る。

目だけの場面は、目の入口の中身を変えない。eyein.setup() がそのまま DINOv2（torch.hub の facebookresearch/dinov2、
dinov2_vits14）と目立ちの地図（taro_core の SalienceMap）を作る。ここでは、作る前に「この機械で作れるか」を
1回で調べて、足りない物をはっきり言う（途中で黙って別の目に替えることはしない）。

* PC（GPU）：今すぐ動く（重みは torch.hub の置き場にある）
* クラウド（2026-10-18 時点）：動かない。足りない物は ① DINOv2 の重み（dl.fbaipublicfiles.com・github の
  torch.hub が通信の決まりで止められている。ユーザーに許可を頼んである）② scipy（目立ちの地図と物の切り出しが使う。
  このコンテナに入っていない。mujoco 以外は入れない決まり）。GPU は無いので CPU で回る（1回 数百ミリ秒の見込み）
使い方：
  python -m core_v2.world_eye.eye_backend        … この機械で目の入口を作れるかを表示
  from core_v2.world_eye.eye_backend import setup_eye; setup_eye(brain)   … 調べてから eyein.setup()
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


if __name__ == "__main__":
    c = check()
    print(c)
    print("作れる" if c["ready"] else "作れない：" + missing_text(c))
