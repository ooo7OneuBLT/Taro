# -*- coding: utf-8 -*-
"""物体ファイルの検証（2026-09-03）：3つの場面で、追跡が期待どおり振る舞うか。

  A) ふつうの遮蔽    箱の後ろへ隠れて→同じ場所に戻る    → 追跡が続き、予測違反は出ない
  B) あり得ない再出現 箱の後ろへ隠れて→別の場所に現れる  → 予測違反として検出される
  C) 説明のない消失   遮蔽物なしで単に消える              → 追跡は自然に立ち消える（特別扱いしない）

検出は `senses/object_detector.py`（DINOv2のパッチ特徴・F2-59で成立確認）、
追跡は `brain/object_files.py`（カルマンフィルタ＋ハンガリアン法・場合分けなし）。

    .venv/Scripts/python.exe F/scripts/f60_object_files_probe.py
出力: F/logs/F2-60_物体ファイル/図_3場面.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.object_detector import patch_features, detect
from brain.object_files import ObjectFileSystem

OUT = "F/logs/F2-60_物体ファイル"
OCCLUDER_XML = ('<body name="occ" mocap="true" pos="0 0 -9">'
                '<geom type="box" size="0.028 0.028 0.028" rgba="0.30 0.55 0.30 1"/>'
                '</body>')


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    src = io.open(sc["world"]["xml"], encoding="utf-8").read()
    xml = "MIMo/mimoEnv/assets/f60_object_files.xml"
    io.open(xml, "w", encoding="utf-8", newline="\n").write(src.replace("</worldbody>", OCCLUDER_XML + "</worldbody>"))
    sc["name"] = "座位_12ヶ月_F2-60_物体ファイル_下見_%s" % G.DATE
    sc["note"] = "物体ファイルの検証。学習には使わない。"
    sc["world"]["xml"] = xml
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]; right = Rc[:, 0]; up = Rc[:, 1]
    slot = "toy5"; a = qadr[slot]
    ang = sc["world"]["parent_labeling"]["present_angles"][slot]
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "occ")
    obj_mi = int(m.body_mocapid[obj_bid])
    ren = mujoco.Renderer(m, height=224, width=224)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    def shot(obj_pos, occ_pos):
        d.qpos[a:a + 3] = obj_pos; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])
        d.mocap_pos[obj_mi] = occ_pos
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        return ren.render().copy()

    base = eye + fwd * 0.30; base[2] = max(base[2], 0.06)
    away = eye + fwd * 5.0                                   # 画面外へ（遮蔽物・物とも共通の「隠す」位置）
    far_pos = base + right * 0.16                            # B: あり得ない再出現の位置

    def frames_for(condition):
        """各コマの (物の位置, 遮蔽物の位置) を返す。"""
        out = []
        for _ in range(4):
            out.append((base, away))                          # 見えている
        for _ in range(5):
            out.append((base, eye + fwd * 0.20 + up * 0.006))   # 遮蔽物が前に来る（物より手前・物の実寸に近い大きさ）
        if condition == "A":
            for _ in range(4):
                out.append((base, away))                       # 遮蔽物がどく。物は同じ場所
        elif condition == "B":
            for _ in range(4):
                out.append((far_pos, away))                    # 遮蔽物がどく。物は違う場所（あり得ない）
        elif condition == "C":
            for _ in range(4):
                out.append((away, away))                       # 遮蔽物なしで物だけ消える
        return out

    results = {}
    strips = {}
    for cond, title in [("A", "A：ふつうの遮蔽（隠れて同じ場所へ）"),
                        ("B", "B：あり得ない再出現（隠れて別の場所へ）"),
                        ("C", "C：説明のない消失（遮蔽物なしで消える）")]:
        ofs = ObjectFileSystem(appearance_weight=0.6, max_dist=40.0, max_missed=20, reappear_gap=4, gate=0.75)
        log, imgs = [], []
        for t, (op, cp) in enumerate(frames_for(cond)):
            img = shot(op, cp)
            p, n = patch_features(model, img)
            dets = detect(p, n, thresh=0.55)
            r = ofs.step(dets)
            log.append((t, len(dets), r))
            imgs.append(img)
        results[cond] = log
        strips[cond] = imgs
        print("== 場面%s ==" % cond)
        for t, nd, r in log:
            tag = ""
            if r["created"]:
                tag += " 新規%s" % r["created"]
            if r["matched"]:
                tag += " 対応" + str([(fid, "%.1f" % res) for fid, j, res in r["matched"]])
            if r["prediction_violations"]:
                tag += " ★予測違反" + str([(fid, "%.1f" % res, gap) for fid, res, gap in r["prediction_violations"]])
            if r["lost"]:
                tag += " 消滅%s" % r["lost"]
            print("  t=%2d 検出%d件%s" % (t, nd, tag or "（対応なし）"))
    ren.close(); env.close()

    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 11)
    n_t = 13
    canvas = Image.new("RGB", (n_t * 96, 3 * 128 + 10), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    labels = {"A": "A：ふつうの遮蔽", "B": "B：あり得ない再出現", "C": "C：説明のない消失"}
    for row, cond in enumerate(("A", "B", "C")):
        y0 = row * 128
        dr.text((4, y0 + 2), labels[cond], fill=(0, 0, 120), font=font)
        for t, img in enumerate(strips[cond]):
            x = t * 96
            canvas.paste(Image.fromarray(img).resize((90, 90)), (x, y0 + 20))
            _, nd, r = results[cond][t]
            note = "違反!" if r["prediction_violations"] else ("消滅" if r["lost"] else ("新規" if r["created"] else ("対応" if r["matched"] else "-")))
            col = (200, 0, 0) if r["prediction_violations"] else (0, 0, 0)
            dr.text((x + 2, y0 + 112), "t%d %s" % (t, note), fill=col, font=f2)
    p = OUT + "/図_3場面.png"; canvas.save(p); print("図:", p)


if __name__ == "__main__":
    main()
