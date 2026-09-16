# -*- coding: utf-8 -*-
"""F2-49 産出テスト（2026-09-03）：初見個体を見せて、太郎（GRU）が自力で言う語で採点する。

物差しは F2-44 で決めた「視覚を渡してGRUが1音ずつ最尤に手繰る語の先頭が正解語と一致」
（研究日誌 2026-09-02「GRU自力」節。比の方法は分母の揺れで実力を過小評価するため不採用）。

刺激：本番と同じ環境（座位・親が持つ位置・物ごとの提示角度・太郎の左目224px）で描く。
  ・初見個体＝各語の4・5体目（バスは3体しかないので訓練個体1・2で代用＝「初見でない」と明記）
  ・角度＝提示角度を基準に横回転 −40／0／+40 の3つ → 8語×2個体×3角度＝48枚
採点対象：鎖の各ラウンド末モデル（成長曲線）＋任意のモデル。

    .venv/Scripts/python.exe F/scripts/f49_test.py                 # 鎖8本の全モデル
    .venv/Scripts/python.exe F/scripts/f49_test.py <model.pt> ...  # 指定モデルだけ
出力: F/logs/F2-49_実物スキャン/test/ テスト48_GRU自力_<モデル名>.csv・図_テスト48枚と答え_<モデル名>.png・成長曲線.png
"""
import os, sys, io, json, csv, warnings, glob
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src"))
sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco, torch
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from cerebral_cortex.recurrent_core import TaroBrain
from cerebral_cortex.visual_projection import VisualProjection
from hearing import Vocabulary, expand_long_vowel
from senses.vision_backends import get_backend

OUT = os.environ.get("F49_TEST_OUT", "F/logs/F2-49_実物スキャン/test")   # 別列の採点は環境変数で出力先を変える
DATE = G.DATE
TEST_IDX = (3, 4)                 # 4・5体目
YAW_OFFSETS = (-40, 0, 40)
FONT = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)


def test_individual(src, key, idx):
    """初見個体。足りない語（バス）は訓練個体で代用し、その旨を返す。"""
    lst = G.individuals(src, key)
    if idx < len(lst):
        return lst[idx], True
    return lst[idx - 3], False


def build_test_world(idx, xml_out):
    """初見個体 idx を8枠に差し込んだ世界XML（f_gen_f49.build_xml と同じ手順）。"""
    base_sc = json.load(io.open(G.BASE_SCENE, encoding="utf-8"))
    src = io.open(base_sc["world"]["xml"], encoding="utf-8").read()
    assets, info = [], []
    for k, (word, s, key) in enumerate(G.WORDS):
        name, unseen = test_individual(s, key, idx)
        a, g = G.asset_and_geom("w%d" % k, s, name)
        assets.append(a); info.append((word, name, unseen))
        src = G.replace_body(src, G.SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    for k in range(5, 13):
        b = "test_object%d" % k
        if b not in G.SLOT_BODIES:
            src = G.remove_body(src, b)
    io.open(xml_out, "w", encoding="utf-8", newline="\n").write(src)
    return info


def make_scene(idx):
    """テスト個体の世界を指すシーン（r1 のシーンから世界だけ差し替え）。"""
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    xml = "MIMo/mimoEnv/assets/f49_8way_test%d.xml" % (idx + 1)
    info = build_test_world(idx, xml)
    sc["name"] = "座位_12ヶ月_F2-49_実物8択_テスト個体%d_%s" % (idx + 1, DATE)
    sc["note"] = "%s F2-49 産出テスト用（初見個体%d）。学習には使わない。" % (DATE, idx + 1)
    sc["world"]["xml"] = xml
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    return sc["name"], info


def render_stimuli():
    """(語, 個体番号, 初見か, 角度, 画像) のリスト。太郎の左目で本番と同条件。"""
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % DATE, encoding="utf-8"))
    stim = []
    for idx in TEST_IDX:
        sname, info = make_scene(idx)
        env, sc, _ = scene_mod.build(sname, taro=spec["taro"], seed=0, verbose=False)
        env.reset(seed=0)
        u = env.unwrapped; m, d = u.model, u.data
        dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
        angles = sc["world"]["parent_labeling"]["present_angles"]
        qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
        qadr["toy1"] = int(u._toy_qadr)
        cid = int(m.camera("eye_left").id)
        ren = mujoco.Renderer(m, height=224, width=224)
        for k, (word, s, key) in enumerate(G.WORDS):
            slot = G.SLOT_KEYS[k]
            name, unseen = info[k][1], info[k][2]
            for off in YAW_OFFSETS:
                for a in qadr.values():
                    d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
                mujoco.mj_forward(m, d)
                eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
                goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
                a = qadr[slot]
                d.qpos[a:a + 3] = goal
                yaw, tilt = angles[slot]["yaw"] + off, angles[slot]["tilt"]
                d.qpos[a + 3:a + 7] = present_quat(u, yaw, tilt)
                mujoco.mj_forward(m, d)
                ren.update_scene(d, camera="eye_left")
                stim.append((word, idx + 1, unseen, yaw, ren.render().copy(), name))
        ren.close()
        env.close()
    return stim


def load_model(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pv = Vocabulary(); pv.char2idx = dict(blob["brain_vocab"]["char2idx"])
    pv.idx2char = {int(i): c for c, i in pv.char2idx.items()}; pv.size = max(pv.idx2char) + 1
    b = TaroBrain(vocab_size=3); b.resize_embedding(blob["brain"]["embedding.weight"].shape[0])
    b.load_state_dict({k: v for k, v in blob["brain"].items()
                       if k in b.state_dict() and b.state_dict()[k].shape == v.shape}, strict=False)
    vp = VisualProjection(384, 64); vp.load_state_dict(blob["visual_projection"])
    return b, vp, pv


def greedy(b, vp, pv, vec):
    """視覚を渡してGRUが1音ずつ最尤に手繰る（口の制約なし）。先頭音の自信も返す。"""
    PAR = pv.char2idx["<PARENT>"]
    with torch.no_grad():
        pfx = vp(torch.tensor(vec))
        out, hh = b.forward_hidden(torch.tensor([[PAR]]), prefix_vec=pfx)
        logits = b.perception_head(out)[0, -1]
        first_p = float(torch.softmax(logits, dim=-1).max())
        seq = []
        for _ in range(8):
            t = int(torch.argmax(logits))
            if t == 2:
                break
            seq.append(t)
            out, hh = b.forward_hidden(torch.tensor([[t]]), hidden=hh)
            logits = b.perception_head(out)[0, -1]
    return "".join(pv.idx2char.get(t, "?") for t in seq), first_p


def correct(said, word):
    cands = {word, word.replace("ー", ""), expand_long_vowel(word)}
    return any(said.startswith(c) for c in cands if c)


def sheet(stim, rows, path, title):
    T, L, per = 224, 30, 12
    n = len(stim); nrow = (n + per - 1) // per
    canvas = Image.new("RGB", (per * (T + 4), nrow * (T + L + 4) + 22), (255, 255, 255))
    dr = ImageDraw.Draw(canvas); dr.text((4, 2), title, fill=(0, 0, 0), font=FONT)
    for i, ((word, ind, unseen, yaw, img, name), r) in enumerate(zip(stim, rows)):
        x, y = (i % per) * (T + 4), 22 + (i // per) * (T + L + 4)
        canvas.paste(Image.fromarray(img), (x, y + L))
        col = (0, 120, 0) if r["正解?"] else (200, 0, 0)
        dr.text((x + 2, y + 1), "%s 個体%d%s 横%d" % (word, ind, "" if unseen else "(訓練)", yaw), fill=(0, 0, 0), font=FONT)
        dr.text((x + 2, y + 15), "→「%s」%.2f" % (r["GRUの発話"], r["先頭音の自信"]), fill=col, font=FONT)
    canvas.save(path)


def main(models):
    os.makedirs(OUT, exist_ok=True)
    stim = render_stimuli()
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    vecs = [np.asarray(be.encode(s[4], s[4]), dtype=np.float32) for s in stim]
    curve = []
    for mp in models:
        tag = os.path.splitext(os.path.basename(mp))[0]
        b, vp, pv = load_model(mp)
        rows = []
        for (word, ind, unseen, yaw, img, name), vec in zip(stim, vecs):
            said, conf = greedy(b, vp, pv, vec)
            rows.append({"正解": word, "個体": ind, "初見": unseen, "角度": yaw, "素材": name,
                         "GRUの発話": said, "先頭音の自信": round(conf, 2), "正解?": correct(said, word)})
        per_word = {w: sum(r["正解?"] for r in rows if r["正解"] == w) for w, _, _ in G.WORDS}
        total = sum(per_word.values())
        n_word_ok = sum(v >= 4 for v in per_word.values())     # 6枚中4枚以上で「言える語」
        curve.append((tag, total, n_word_ok, per_word))
        with io.open("%s/テスト48_GRU自力_%s.csv" % (OUT, tag), "w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        sheet(stim, rows, "%s/図_テスト48枚と答え_%s.png" % (OUT, tag),
              "%s  合計 %d/48  言える語(4/6以上) %d/8" % (tag, total, n_word_ok))
        print("%s: %d/48  言える語 %d/8  " % (tag, total, n_word_ok) + "  ".join("%s %d/6" % (w, per_word[w]) for w, _, _ in G.WORDS))
    if len(curve) > 1:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rcParams
        for c in ("Yu Gothic", "Meiryo", "MS Gothic"):
            if any(c in f.name for f in font_manager.fontManager.ttflist):
                rcParams["font.family"] = c; break
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = list(range(1, len(curve) + 1))
        ax.plot(xs, [c[1] for c in curve], "o-", label="正解枚数 /48")
        ax.plot(xs, [c[2] * 6 for c in curve], "s--", label="言える語 ×6 (/8語)")
        ax.set_xlabel("ラウンド"); ax.set_ylabel("枚"); ax.set_ylim(0, 48); ax.grid(alpha=.3); ax.legend()
        ax.set_title("F2-49 実物スキャン8語：初見個体48枚の産出テスト（GRU自力）")
        fig.tight_layout(); fig.savefig("%s/成長曲線.png" % OUT, dpi=120)
        print("図:", "%s/成長曲線.png" % OUT)
    return curve


if __name__ == "__main__":
    ms = sys.argv[1:] or sorted(glob.glob("F/models/F2-49_r*_seed*_%s.pt" % DATE),
                                key=lambda p: int(p.split("_r")[1].split("_")[0]))
    main(ms)
