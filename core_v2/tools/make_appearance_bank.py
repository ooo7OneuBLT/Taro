"""段階9：見た目の素の表を作る（PC で1回だけ。ユーザー決定「DINOv2 の見た目の素は PC で1回作って表をクラウドへ持ってくる」）。

仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` 付録「試験場」。20語×4個体（world_eye/instances.py、全部自作の形）を、
目だけの場面の目（左目のカメラ、224×224、画角60度）で、いろいろな向きと距離から1つずつ描き、目の入口と同じ切り出し
（floor_objects.detect：DINOv2 vits14 のマスの特徴＋床らしさ）にかけて、物の見た目（384、目の入口が tagging.appearance_tag へ
渡すのと同じ数）を保存する。クラウドの体なしの試験場（tools/noun_bed.py）はこの表を読む。

* 向き（VIEWS、24通り）：左右 15度ずつ（0〜345度）、上下 −20・0・+20度、距離 25・35・45cm（仕様書 §2「左右0〜360度、上下±20度、
  距離25〜45cm」）。番号が隣の向きは左右15度違い（試験場は「同じ物を続けて見た2つの見え」を隣の向きで作る）
* 置き方：物は、その語の部屋（rooms.py）で、目から下へ GAZE_DOWN_DEG（20度）の向きに置き、目もそこを向く（物が画面の真ん中）。
  1枚に物は1つ。切り出しの物のうち、画面の真ん中にいちばん近いものを、その物とする（真ん中から 40画素より遠ければ「拾えなかった」）
* 決まった種：描画は乱数を使わない。個体は instances.py の決まった種で作る。DINOv2 は GPU では半分の精度（floor_objects と同じ）

使い方（リポジトリの一番上で）：
  python -m core_v2.tools.make_appearance_bank --preview [--out G/logs/段階9_見た目の素_確かめ]   … 描くだけ（DINOv2 なし。クラウドでも回る）
  python -m core_v2.tools.make_appearance_bank --check   [--out G/logs/段階9_見た目の素_確かめ]   … 8個体×3向きを切り出しにかけ、拾った所を描いて保存（PC）
  python -m core_v2.tools.make_appearance_bank [--out core_v2/data/段階9_見た目の素.pt]               … 表を作る（PC、約2分の見込み）
  どれにも --objects real を足すと本物の物（world_eye/real_objects.py）で回す（表は core_v2/data/段階9_見た目の素_本物.pt）
出力（表）：torch.save の dict
  app[20, 4, 24, 384]（float16、拾えなかった所は 0）・found[20, 4, 24]（bool）・words・instances（個体の作り方）・views・meta
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from .. import world_eye  # noqa: F401  （描き方の決め方と、クラウドの読み込み順の対策を先に読む）
from ..world_eye import instances as ins
from ..world_eye import scene as scene_mod

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_OUT = os.path.join("core_v2", "data", "段階9_見た目の素.pt")
PX = 224
FOVY = 60.0
GAZE_DOWN_DEG = 20.0
CENTER_PX = 40.0
FIT_SIZE_M = 0.18   # これより大きい物は距離を遠ざける（20cm のわんわんが 25cm で収まりきらなかったので少し小さめの切りのよい値）
VIEWS = [{"yaw": 15.0 * k, "pitch": (-20.0, 0.0, 20.0)[k % 3], "dist": (0.25, 0.35, 0.45)[(k // 3) % 3]}
         for k in range(24)]


def _quat(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """上の軸のまわりに yaw、そのあと太郎の左右の軸（y）のまわりに pitch 回す（w, x, y, z）。"""
    a, b = math.radians(yaw_deg) / 2, math.radians(pitch_deg) / 2
    qz = np.array([math.cos(a), 0.0, 0.0, math.sin(a)])
    qy = np.array([math.cos(b), 0.0, math.sin(b), 0.0])
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qz
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Stage:
    """1つの部屋に1つの個体を置いて、目の画像を描く。"""

    def __init__(self, inst: dict) -> None:
        import mujoco
        self.mj = mujoco
        xml = scene_mod.build_xml({"px": PX, "fovy": FOVY, "room": inst["room"],
                                   "extra_assets": ins.assets_xml(inst), "extra_bodies": [ins.body_xml(inst, "obj")]})
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        j = self.model.body("obj").jntadr[0]
        self.qadr = int(self.model.jnt_qposadr[j])
        self.data.qpos[int(self.model.jnt_qposadr[self.model.joint("eye_tilt").id])] = -math.radians(GAZE_DOWN_DEG)
        self.renderer = mujoco.Renderer(self.model, PX, PX)
        # 大きい本物の物（いちばん長い辺が FIT_SIZE_M を越える物：くま・ぼうし・おさら・つみき）は、距離の段（25・35・45cm）を
        # 大きさに比例して遠ざける。25cm の向きで画面からはみ出して切り出しが拾えなかったため（依頼8：くま 54/96、ぼうし 63/96）。
        # 見かけの大きさは FIT_SIZE_M の物と同じになる。自作の物（size_m が無い）は今までどおり
        self.dist_scale = max(1.0, float(inst.get("size_m", 0.0)) / FIT_SIZE_M)

    def render(self, view: dict) -> np.ndarray:
        down = math.radians(GAZE_DOWN_DEG)
        d = np.cos(down) * scene_mod.FORWARD - np.sin(down) * scene_mod.UP
        self.data.qpos[self.qadr:self.qadr + 3] = scene_mod.EYE_LEFT + view["dist"] * self.dist_scale * d
        self.data.qpos[self.qadr + 3:self.qadr + 7] = _quat(view["yaw"], view["pitch"])
        self.mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="eye_left")
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


COLOR_EN = {"赤": "red", "青": "blue", "緑": "green", "黄": "yellow", "橙": "orange", "紫": "purple", "桃": "pink",
            "茶": "brown"}


def _label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((3, 3), text, fill=(255, 255, 255))
    ImageDraw.Draw(im).text((2, 2), text, fill=(0, 0, 0))
    return np.asarray(im)


def _sheet(rows: list[list[np.ndarray]], path: str) -> None:
    from PIL import Image
    w = max(len(r) for r in rows)
    blank = np.zeros((PX, PX, 3), dtype=np.uint8)
    big = np.concatenate([np.concatenate(r + [blank] * (w - len(r)), axis=1) for r in rows], axis=0)
    Image.fromarray(big).save(path)


def preview(out: str, views_for: tuple[str, ...] = ("わんわん", "こっぷ")) -> list[str]:
    """部屋ごとに 語×4個体（向き0番）の一覧と、2語の24の向きの一覧を描く（DINOv2 は使わない）。"""
    os.makedirs(out, exist_ok=True)
    paths = []
    show = VIEWS[4]   # 左右60度・上下0度・35cm（横向きの形がよく見える）
    for room in ("食事", "お風呂", "遊び"):
        rows = []
        for w in [w for w in ins.WORDS if ins.ROOM_OF[w] == room]:
            row = []
            for k in range(ins.N_INST):
                inst = ins.instance(w, k)
                st = Stage(inst)
                pat = {"格子": " checker", "点": " dots", None: ""}[inst["pattern"]]
                tag = f"{inst['kind']} {k} {COLOR_EN[inst['color']]} x{inst['scale']}{pat}"   # 絵の字は英字（日本語の字体が無い）
                row.append(_label(st.render(show), tag + ("" if inst["train"] else " TEST")))
                st.close()
            rows.append(row)
        p = os.path.join(out, f"個体_{room}.png")
        _sheet(rows, p)
        paths.append(p)
    for w in views_for:
        st = Stage(ins.instance(w, 0))
        imgs = [_label(st.render(v), f"{k} y{v['yaw']:.0f} p{v['pitch']:.0f} d{v['dist']}") for k, v in enumerate(VIEWS)]
        st.close()
        p = os.path.join(out, f"向き_{ins.KIND_OF[w]}.png")
        _sheet([imgs[i:i + 8] for i in range(0, 24, 8)], p)
        paths.append(p)
    return paths


def similarity_table(app: np.ndarray, found: np.ndarray) -> dict:
    """先行研究 5節の「まず測る」：荷札（tagging.appearance_tag、64）にしたときの、個体の平均どうしの cos。
    同じ種類の別の個体どうし・別の種類どうし・別の種類で同じ色どうし。"""
    import torch
    from ..tagging import appearance_tag
    W, K = app.shape[:2]
    tags = np.zeros((W, K, 64), dtype=np.float32)
    for i in range(W):
        for k in range(K):
            ok = found[i, k]
            if ok.any():
                v = torch.stack([appearance_tag(a) for a in app[i, k][ok].astype(np.float64)]).mean(0)
                tags[i, k] = (v / v.norm()).numpy()
    cols = [[ins.instance(w, k)["color"] for k in range(K)] for w in ins.WORDS]
    same_kind, diff_kind, same_color = [], [], []
    for i in range(W):
        for k in range(K):
            for j in range(W):
                for m in range(K):
                    if (i, k) >= (j, m):
                        continue
                    c = float(tags[i, k] @ tags[j, m])
                    if i == j:
                        same_kind.append(c)
                    else:
                        diff_kind.append(c)
                        if cols[i][k] == cols[j][m]:
                            same_color.append(c)
    f = lambda x: round(float(np.mean(x)), 3) if x else None  # noqa: E731
    return {"同じ種類の別の個体": f(same_kind), "別の種類": f(diff_kind), "別の種類で同じ色": f(same_color)}


CHECK_SET = [("わんわん", 0), ("にゃんにゃん", 1), ("こっぷ", 2), ("すぷうん", 3), ("ばなな", 0), ("たおる", 2),
             ("ぼうし", 1), ("つみき", 3)]
CHECK_VIEWS = (0, 4, 13)


def check(out: str) -> dict:
    """目で確かめる（表を作る前に1回）：CHECK_SET の8個体×3つの向きを描いて切り出しにかけ、拾った物の場所（緑の点、
    選んだ物は赤の丸）を画像に描いて保存する。拾えた数を返す。"""
    import torch
    from PIL import Image, ImageDraw
    from .. import floor_objects
    device = os.environ.get("TARO_VISION_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(device)
    os.makedirs(out, exist_ok=True)
    rows, n_ok, n = [], 0, 0
    for w, k in CHECK_SET:
        inst = ins.instance(w, k)
        st = Stage(inst)
        row = []
        for v in CHECK_VIEWS:
            img = st.render(VIEWS[v])
            dets = floor_objects.detect(model, device, img)
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            best = min(dets, key=lambda d: math.hypot(d["pos"][0] - PX / 2, d["pos"][1] - PX / 2)) if dets else None
            ok = best is not None and math.hypot(best["pos"][0] - PX / 2, best["pos"][1] - PX / 2) <= CENTER_PX
            for d in dets:
                x, y = d["pos"]
                dr.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0))
            if ok:
                x, y = best["pos"]
                dr.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(255, 0, 0), width=2)
            dr.text((3, 3), f"{inst['kind']} {k} v{v} dets{len(dets)} {'OK' if ok else 'MISS'}", fill=(0, 0, 0))
            row.append(np.asarray(im))
            n_ok += ok
            n += 1
        st.close()
        rows.append(row)
    p = os.path.join(out, "切り出しの確かめ.png")
    _sheet(rows, p)
    print(f"確かめ：拾えた {n_ok}/{n}。画像 {p}", flush=True)
    return {"found": n_ok, "n": n, "image": p}


def build(out_path: str, seed: int = 0) -> dict:
    import torch
    from .. import floor_objects
    from ..save import code_version
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = os.environ.get("TARO_VISION_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(device)
    W, K, V = len(ins.WORDS), ins.N_INST, len(VIEWS)
    app = np.zeros((W, K, V, 384), dtype=np.float32)
    found = np.zeros((W, K, V), dtype=bool)
    n_dets = np.zeros((W, K, V), dtype=np.int16)
    t0 = time.time()
    for i, w in enumerate(ins.WORDS):
        for k in range(K):
            st = Stage(ins.instance(w, k))
            for v, view in enumerate(VIEWS):
                img = st.render(view)
                dets = floor_objects.detect(model, device, img)
                n_dets[i, k, v] = len(dets)
                if dets:
                    d = min(dets, key=lambda d: math.hypot(d["pos"][0] - PX / 2, d["pos"][1] - PX / 2))
                    if math.hypot(d["pos"][0] - PX / 2, d["pos"][1] - PX / 2) <= CENTER_PX:
                        app[i, k, v] = np.asarray(d["appearance"], dtype=np.float32)
                        found[i, k, v] = True
            st.close()
        print(f"[{time.time() - t0:6.1f}s] {w} 拾えた {int(found[i].sum())}/{K * V}", flush=True)
    meta = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "device": device, "seed": seed, "eye": "dinov2_vits14+floor_objects",
            "px": PX, "fovy": FOVY, "gaze_down_deg": GAZE_DOWN_DEG, "center_px": CENTER_PX,
            "found_share": round(float(found.mean()), 4), "wall_s": round(time.time() - t0, 1), "objects": ins.OBJECTS,
            **code_version()}
    sim = similarity_table(app, found)
    meta["similarity"] = sim
    blob = {"app": torch.from_numpy(app).half(), "found": torch.from_numpy(found), "n_dets": torch.from_numpy(n_dets),
            "words": list(ins.WORDS), "instances": [[ins.instance(w, k) for k in range(K)] for w in ins.WORDS],
            "views": VIEWS, "meta": meta}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(blob, out_path)
    print("できた：", json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objects", default="made", choices=["made", "real"],
                    help="made＝自作の形（既定）、real＝本物の物（world_eye/real_objects.py。表は *_本物.pt、確かめは *_本物）")
    a = ap.parse_args(argv)
    ins.use_objects(a.objects)
    tail = "_本物" if a.objects == "real" else ""
    if a.preview:
        out = a.out or os.path.join("G", "logs", "段階9_見た目の素_確かめ" + tail)
        out = out if os.path.isabs(out) else os.path.join(REPO, out)
        for p in preview(out):
            print(p)
        return
    if a.check:
        out = a.out or os.path.join("G", "logs", "段階9_見た目の素_確かめ" + tail)
        check(out if os.path.isabs(out) else os.path.join(REPO, out))
        return
    out = a.out or (DEFAULT_OUT[:-3] + tail + ".pt")
    build(out if os.path.isabs(out) else os.path.join(REPO, out), a.seed)


if __name__ == "__main__":
    main()
