"""段階9：本物の物の一覧（core_v2/data/段階9_本物の物.json）から、目だけの場面で使うファイルを作る（PC で1回）。

* 元の物を取ってくる（--fetch、無い物だけ）：
    GSO … kevinzakka/mujoco_scanned_objects の models/<名前>/model.obj・texture.png を G/assets/gso/models/<名前>/ へ
    Objaverse … allenai/objaverse（Hugging Face）の glb を G/assets/objaverse/glbs/<uid>.glb へ（場所は object-paths.json.gz）
* 直す（world_eye/real_objects.convert）：G/assets/real/<種類>_<個体>/ に部品ごとの obj＋png と parts.json。
  一覧の color・rgb・orig_size_m・n_parts を書き足す（色は直したときに決まる）
* 一覧の絵（--sheet）：20語×4個体を左目で 35cm・下20度・横30度から描いて1枚に（4個目は TEST の印）

使い方（リポジトリの一番上で）：
  python -m core_v2.tools.make_real_objects [--fetch] [--sheet doc/段階9_本物の物_一覧.png]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import urllib.request

from ..world_eye import instances as ins
from ..world_eye import real_objects as ro

GSO_RAW = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/main/models/"
OBJV = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


def fetch(it: dict) -> None:
    if it["source"] == "gso":
        d = os.path.join(ro.ASSET_DIR, "gso", "models", it["id"])
        for f in ("model.obj", "texture.png"):
            p = os.path.join(d, f)
            if not os.path.exists(p):
                os.makedirs(d, exist_ok=True)
                urllib.request.urlretrieve(GSO_RAW + it["id"] + "/" + f, p)
    else:
        p = os.path.join(ro.ASSET_DIR, "objaverse", "glbs", it["id"] + ".glb")
        if not os.path.exists(p):
            idx = os.path.join(ro.ASSET_DIR, "objaverse", "object-paths.json.gz")
            if not os.path.exists(idx):
                os.makedirs(os.path.dirname(idx), exist_ok=True)
                urllib.request.urlretrieve(OBJV + "object-paths.json.gz", idx)
            paths = json.load(gzip.open(idx))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(OBJV + paths[it["id"]], p)


def build(do_fetch: bool) -> dict:
    data = ro.load_list()
    for it in data["items"]:
        if do_fetch:
            fetch(it)
        kind = ins.KIND_OF[it["word"]]
        info = ro.convert(it["source"], it["id"], float(it["size_m"]), ro.obj_dir(kind, int(it["k"])),
                          float(it.get("tilt_deg", 0.0)))
        it.update({"color": info["color"], "rgb": info["rgb"], "orig_size_m": info["orig_size_m"],
                   "n_parts": len(info["parts"])})
        print(f"{it['word']} {it['k']} {it['source']} {it['id'][:40]} → {info['color']} 部品{len(info['parts'])}", flush=True)
    with open(ro.LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return data


def sheet(path: str) -> str:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from .make_appearance_bank import PX, Stage
    ins.use_objects("real")
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)
    except OSError:
        font = ImageFont.load_default()
    view = {"yaw": 30.0, "pitch": 0.0, "dist": 0.35}
    half = (len(ins.WORDS) + 1) // 2
    img = Image.new("RGB", (8 * PX, half * PX), (255, 255, 255))
    for i, w in enumerate(ins.WORDS):
        for k in range(ins.N_INST):
            inst = ins.instance(w, k)
            st = Stage(inst)
            t = Image.fromarray(np.asarray(st.render(view)))
            st.close()
            dr = ImageDraw.Draw(t)
            dr.rectangle([0, 0, PX, 17], fill=(0, 0, 0))
            lab = f"{w} {k} {inst['color']} {int(round(inst['size_m'] * 100))}cm"
            dr.text((2, 1), lab, fill=(255, 255, 255), font=font)
            if k == ins.TEST_INST:
                dr.rectangle([PX - 46, PX - 22, PX - 2, PX - 2], fill=(200, 0, 0))
                dr.text((PX - 42, PX - 20), "TEST", fill=(255, 255, 255), font=font)
            img.paste(t, ((i // half) * 4 * PX + k * PX, (i % half) * PX))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    print("一覧の絵：", path, flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args(argv)
    if not a.no_build:
        build(a.fetch)
    if a.sheet:
        sheet(a.sheet if os.path.isabs(a.sheet) else os.path.join(ro.REPO, a.sheet))


if __name__ == "__main__":
    main()
