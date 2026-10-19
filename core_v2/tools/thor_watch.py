"""家の場面（AI2-THOR）で、太郎が何をしているかを動画で見る（計画①：太郎の脳を家につなぐ、の確かめ）。

world_thor.taro.ThorGrower（41係の脳・目・見えない親の手）を --seconds 秒回し、0.1秒の1回ごとに1コマ描く
（--speed 1 で実時間＝10コマ/秒）。写しは作らない（意味の辞書と耳は、回している間だけ育つ。watch_scene と同じ）。
1コマ（watch_scene と同じ並び）：
* 左の大きな絵：太郎の目に入っている画像（AI2-THOR の床から 0.6m の目。--render-px で大きく描き、目の入口へは 224 に縮めて渡す）。
  桃色の丸＝視線の真ん中（半径5度）。黄色い四角＝物の名簿（objfile）の「今見ている物」。物の下の小さな字＝その物の本当の語
  （AI2-THOR の色分けの正解。太郎が知っているわけではない）
* 右の欄：秒、親が言ったこと、太郎が聞き取った語、太郎の声、見ている物（親から見た正解）と見ている秒数の棒、
  親の手（持ってきた物と持っている秒）、「覚えた！」、ここまでの持ってきた・名づけ・覚えた回数
終わりに、親の名づけの回数と、太郎が物を見続けた時間（回数・平均・真ん中・最長・1秒以上の回数・語ごと）を表示し、
動画と同じ名前の _まとめ.json に残す。

使い方（PC：WSL の Ubuntu の中。doc/PCへの依頼.md の依頼16）：
  python -m core_v2.tools.thor_watch [--seconds 20] [--scene FloorPlan1] [--eye dinov2|seg] [--from auto|birth|写しの名前]
                                     [--seed 0] [--stills 3] [--render-px 448] [--fov 60] [--out 動画の場所]
                                     [--place auto|start] [--spot x,z,yaw] [--toys 4] [--bring-in 5]
  python -m core_v2.tools.thor_watch --first-frame --scene FloorPlan1 [--spot x,z,yaw] [--out フォルダ]
    脳を回さず、最初の目の絵（視線の丸・おもちゃの印と語・名前の物の語）と、上から見た絵（歩ける所の点と座標の格子、
    座った所と向きの矢印・視野、おもちゃ、床の上の大きな物の箱）だけを残す。座る場所を手で選ぶための絵（すぐ終わる）
  python -m core_v2.tools.watch_scene --world thor [同じ引数]
  --fake：AI2-THOR の代わりの偽物の家（world_thor/fake.py。クラウドとテスト用。--eye seg と一緒に）
出力：--out（既定 G/logs/動画/家の場面_<部屋>_<目>_種<seed>.mp4。ffmpeg が無ければ .gif）、_1.png …（--stills）、_まとめ.json、
      動画と同じフォルダに events_<部屋>_seed<seed>.csv（親が持ってきた・言った・戻した、目の向き）
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import numpy as np

from ..world_thor import env as tp_env
from ..world_thor import parent as tp
from ..world_thor.taro import ThorGrower, make_env
from . import growth
from .watch_scene import (BG, CYAN, DIM, FG, GAZE, GREEN, H, LEARN_HOLD_S, LOOK_BAR_S, PAD, PINK, SAY_HOLD_S, W, YELLOW,
                          Fonts, Painter, Writer, pick_stills, word_of)

TICK_S = tp.TICK_S
EYE_PX = 672
ORANGE = (255, 170, 80)


class ThorRecorder(ThorGrower):
    """ThorGrower。1回ごとにコマを描く（frame_fn）。覚えた語を記録する（watch_scene.Recorder と同じ）。"""

    def __init__(self, *a, frame_fn=None, big_px: int = EYE_PX, **kw) -> None:
        super().__init__(*a, **kw)
        self.frame_fn = frame_fn
        self.big_px = big_px
        self.learned_word = ""
        sem = self.brain.boxes["semantic"]
        orig = sem._learn_pending

        def learn_pending(p, _orig=orig):
            _orig(p)
            if sem.last_event == "learn":
                self.learned_word = word_of(p["q"], tp.WORDS)
        sem._learn_pending = learn_pending

    def tick(self) -> dict:
        got = super().tick()
        if self.frame_fn is not None:
            self.frame_fn(self, got)
        return got

    def big_image(self) -> np.ndarray:
        from PIL import Image
        img = self.env.frame()
        if img.shape[0] != self.big_px:
            img = np.array(Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).resize((self.big_px, self.big_px),
                                                                                             Image.BILINEAR))
        return img


class ThorPainter(Painter):
    def update(self, g: ThorRecorder, got: dict) -> dict:
        n = g.n_tick
        sem = g.brain.boxes["semantic"]
        ear = g.brain.boxes["listen"]
        p = g.parent
        if got.get("heard"):
            self.said = (got["heard"], n)
            self.said_held = p.held
        elif self.said[0] and p.held != getattr(self, "said_held", None):
            self.said = ("", n)            # 親が物を戻したら、言ったことの表示も消す（手が空なのに名づけが出ていると見誤る。依頼17）
        if ear.last_word:
            self.heard = (ear.last_word, n)
        if g._echo:
            self.voice = (g._echo, n)
        if sem.last_event == "learn":
            self.learn = (g.learned_word, n)
        looking = g.env.word_of(p.looking)
        return {
            "n": n, "t": n * TICK_S, "in_scene": True,
            "said": self._hold(self.said, n, SAY_HOLD_S), "heard": self._hold(self.heard, n, SAY_HOLD_S),
            "voice": self._hold(self.voice, n, SAY_HOLD_S), "learn": self._hold(self.learn, n, LEARN_HOLD_S),
            "learn_now": sem.last_event == "learn", "looking": looking,
            "look_s": p.look_ticks * TICK_S if looking else 0.0,
            "holding": p.word if p.held else "", "hold_s": p.d * TICK_S if p.held else 0.0,
            "brings": p.stats["brings"], "names": p.stats["names"], "mode": p.mode if p.held else "",
            "b_follow": p.stats["brings_follow"], "b_in": p.stats["brings_bringin"], "learns": g.n_learn + (sem.last_event == "learn"),
        }

    def draw(self, g: ThorRecorder, st: dict) -> np.ndarray:
        from PIL import Image, ImageDraw
        env = g.env
        big = g.big_px
        img = Image.new("RGB", (W, H), BG)
        eye = Image.fromarray(g.big_image())
        d = ImageDraw.Draw(eye)
        # 物の正解の語（小さく。見えている所の真ん中の下）
        for oid, (x, y, npx) in env.object_centres(px=big).items():
            word = env.word_of(oid)
            r = math.sqrt(npx / math.pi) * big / env.render_px
            ty = min(y + r + 6, big - 26)
            tw = d.textlength(word, font=self.f.tiny)
            d.rectangle([x - tw / 2 - 4, ty - 2, x + tw / 2 + 4, ty + 22], fill=(0, 0, 0))
            d.text((x - tw / 2, ty), word, font=self.f.tiny, fill=(230, 230, 230))
        s = big / 224.0
        of = g.brain.boxes["objfile"]
        rw = getattr(of, "_held_now", None)
        if rw is not None:
            r_, c_ = rw.pos
            x, y = (8 * c_ + 4) * s, (8 * r_ + 4) * s
            h = 3.0 * 8 * s
            d.rectangle([x - h, y - h, x + h, y + h], outline=YELLOW, width=4)
        c = big / 2
        rr = 0.5 * big / math.tan(math.radians(env.fov) / 2) * math.tan(math.radians(5.0))
        for col, wd in (((0, 0, 0), 7), (GAZE, 3)):
            d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=col, width=wd)
            d.line([c - 10, c, c + 10, c], fill=col, width=wd - 1)
            d.line([c, c - 10, c, c + 10], fill=col, width=wd - 1)
        img.paste(eye, (PAD, PAD))
        D = ImageDraw.Draw(img)
        D.text((PAD, PAD + big + 2), "太郎の目（床から0.6m）　○（桃色）＝見ている所　□（黄色）＝太郎が「今見ている物」と思っている所",
               font=self.f.tiny, fill=DIM)
        x0 = PAD * 2 + big
        y = PAD
        D.text((x0, y), f"家の場面　{env.scene_name}", font=self.f.mid, fill=FG)
        y += 40
        D.text((x0, y), f"{st['t']:5.1f} 秒", font=self.f.small, fill=DIM)
        y += 40

        def row(label, val, color, y):
            D.text((x0, y), label, font=self.f.small, fill=DIM)
            D.text((x0 + 12, y + 30), val or "―", font=self.f.big if val else self.f.mid, fill=color if val else DIM)
            return y + 82

        y = row("親が言ったこと", "「" + st["said"] + "」" if st["said"] else "", CYAN, y)
        y = row("太郎が聞き取った語", st["heard"], GREEN, y)
        y = row("太郎の声", "「" + st["voice"] + "」" if st["voice"] else "", PINK, y)
        D.text((x0, y), "見ている物", font=self.f.small, fill=DIM)
        D.text((x0 + 12, y + 30), st["looking"] or "（名前のある物を見ていない）",
               font=self.f.big if st["looking"] else self.f.mid, fill=FG if st["looking"] else DIM)
        by = y + 80
        bw = W - x0 - PAD - 90
        D.rectangle([x0 + 12, by, x0 + 12 + bw, by + 16], outline=DIM, width=2)
        frac = min(1.0, st["look_s"] / LOOK_BAR_S)
        if frac > 0:
            D.rectangle([x0 + 12, by, x0 + 12 + int(bw * frac), by + 16], fill=YELLOW)
        D.text((x0 + 22 + bw, by - 6), f"{st['look_s']:.1f}秒", font=self.f.small, fill=FG)
        y = by + 30
        D.text((x0, y), "親の手", font=self.f.small, fill=DIM)
        tag = {"follow": "見ていた物", "bringin": "親から"}.get(st.get("mode", ""), "")
        hold = f"{st['holding']} を見せている（{tag}・{st['hold_s']:.1f}秒）" if st["holding"] else ""
        D.text((x0 + 12, y + 30), hold or "―", font=self.f.mid, fill=ORANGE if hold else DIM)
        y += 76
        if st["learn"]:
            D.rounded_rectangle([x0, y, W - PAD, y + 56], radius=12, fill=(60, 50, 10), outline=YELLOW, width=3)
            D.text((x0 + 16, y + 8), f"覚えた！「{st['learn']}」", font=self.f.big, fill=YELLOW)
            D.text((x0, y + 60), "（聞いてから約5秒の名残りのあと結び付ける。結ぶ物は名づけの前3秒と直後に見ていた物）",
                   font=self.f.tiny, fill=DIM)
        D.text((x0, H - PAD - 34), f"持ってきた {st['brings']}（見ていた {st['b_follow']}・親から {st['b_in']}）　"
                                   f"名づけ {st['names']}　覚えた {st['learns']}",
               font=self.f.mid, fill=FG)
        return np.asarray(img)


def fake_kw(a) -> dict:
    """--fake のとき：偽物の家は、台所に似せた部屋（world_thor/fake.DEMO_ROOM。物は調理台の上、始めは冷蔵庫の前）。"""
    if not a.fake:
        return {}
    from ..world_thor.fake import DEMO_ROOM
    return {"controller_kw": dict(DEMO_ROOM)}


def parse_spot(text: str) -> tuple:
    v = [float(x) for x in str(text).replace(" ", "").split(",")]
    if len(v) != 3:
        raise SystemExit(f"--spot は x,z,yaw：{text}")
    return tuple(v)


TOY_COL = (255, 150, 40)


def eye_overlay(env, fonts) -> "np.ndarray":
    """最初の目の絵に、視線の丸・おもちゃの印（幾何で写した真ん中）と語・写っている名前の物の語を重ねる。"""
    from PIL import Image, ImageDraw
    big = env.render_px
    img = Image.fromarray(np.ascontiguousarray(env.frame(), dtype=np.uint8))
    if big < 672:
        big = 672
        img = img.resize((big, big), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    for oid, (x, y, _n) in env.object_centres(px=big).items():
        w = env.word_of(oid)
        tw = d.textlength(w, font=fonts.tiny)
        d.rectangle([x - tw / 2 - 3, y + 8, x + tw / 2 + 3, y + 30], fill=(0, 0, 0))
        d.text((x - tw / 2, y + 9), w, font=fonts.tiny, fill=(230, 230, 230))
    for t in env.toy_info:
        if not t.get("placed") or t["id"] not in env.objects:
            continue
        pr = env.project(env._centre(env.objects[t["id"]]), px=big)
        if pr is None:
            continue
        x, y = pr[0], pr[1]
        d.line([x - 9, y - 9, x + 9, y + 9], fill=TOY_COL, width=3)
        d.line([x - 9, y + 9, x + 9, y - 9], fill=TOY_COL, width=3)
    c = big / 2
    rr = 0.5 * big / math.tan(math.radians(env.fov) / 2) * math.tan(math.radians(5.0))
    for col, wd in (((0, 0, 0), 6), (GAZE, 3)):
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=col, width=wd)
    pl = env.placement
    bar = Image.new("RGB", (big, 92), BG)
    D = ImageDraw.Draw(bar)
    D.text((8, 4), f"{env.scene_name}　場所 x={pl['position'][0]:.2f} z={pl['position'][2]:.2f} 向き {pl['yaw']:.0f}度（{pl['mode']}）",
           font=fonts.tiny, fill=FG)
    D.text((8, 32), f"おもちゃ 並べた{len(env.toys)}・絵に入る{pl['toys_in_view']}・写る{pl['toys_seen']}　"
                    f"{'・'.join(pl['toys']) or 'なし'}", font=fonts.tiny, fill=TOY_COL)
    D.text((8, 60), "×（橙）＝並べたおもちゃ　○（桃）＝視線の真ん中　字＝写っている名前の物（正解）", font=fonts.tiny, fill=DIM)
    out = Image.new("RGB", (big, big + 92), BG)
    out.paste(img, (0, 0))
    out.paste(bar, (0, big))
    return np.asarray(out)


def topdown(env, fonts) -> "np.ndarray":
    """上から見た絵（x が右、z が上。yaw は上から見て右回り＝0度が上・90度が右）。座る場所を手で選ぶための絵。"""
    from PIL import Image, ImageDraw
    reach = env.reach if env.reach is not None and len(env.reach) else np.zeros((0, 3))
    ag = env._event.metadata["agent"]
    pos = vec_(ag["position"])
    pts = [reach[:, [0, 2]]] if len(reach) else []
    pts.append(np.array([[pos[0], pos[2]]]))
    large = env._large_boxes(floor_only=False)
    for _i, (lo, hi) in large:
        pts.append(np.array([[lo[0], lo[2]], [hi[0], hi[2]]]))
    allp = np.concatenate(pts, axis=0)
    x0, z0 = np.floor(allp.min(axis=0) - 0.5)
    x1, z1 = np.ceil(allp.max(axis=0) + 0.5)
    s = min(160.0, 1400.0 / max(x1 - x0, z1 - z0))
    m = 70
    Wd, Hd = int((x1 - x0) * s) + 2 * m, int((z1 - z0) * s) + 2 * m + 60
    img = Image.new("RGB", (Wd, Hd), (250, 250, 247))
    d = ImageDraw.Draw(img)

    def P(x, z):
        return m + (x - x0) * s, m + (z1 - z) * s

    # 座標の格子（0.5m ごと、1m ごとに数字）
    for gx in np.arange(x0, x1 + 1e-9, 0.5):
        a, b = P(gx, z0), P(gx, z1)
        d.line([a, b], fill=(215, 215, 215) if gx % 1 else (170, 170, 170), width=1)
        if abs(gx - round(gx)) < 1e-6:
            for yy in (m - 26, Hd - 60 - m + 6):
                d.text((a[0] - 12, yy), f"{gx:.0f}", font=fonts.tiny, fill=(90, 90, 90))
    for gz in np.arange(z0, z1 + 1e-9, 0.5):
        a, b = P(x0, gz), P(x1, gz)
        d.line([a, b], fill=(215, 215, 215) if gz % 1 else (170, 170, 170), width=1)
        if abs(gz - round(gz)) < 1e-6:
            d.text((6, a[1] - 10), f"{gz:.0f}", font=fonts.tiny, fill=(90, 90, 90))
            d.text((Wd - m + 8, a[1] - 10), f"{gz:.0f}", font=fonts.tiny, fill=(90, 90, 90))
    d.text((m, 6), "x →（右）　z ↑（上）　向き：0度＝上、90度＝右、180度＝下、270度＝左", font=fonts.tiny, fill=(60, 60, 60))
    # 床の上の大きな物・吊った物の箱
    floor_ids = {i for i, _b in env._large_boxes(floor_only=True)}
    for i, (lo, hi) in large:
        a, b = P(lo[0], hi[2]), P(hi[0], lo[2])
        on_floor = i in floor_ids
        d.rectangle([a, b], outline=(120, 90, 60) if on_floor else (180, 180, 210), width=2 if on_floor else 1,
                    fill=(236, 226, 210) if on_floor else None)
        typ = env.objects[i]["objectType"]
        if (b[0] - a[0]) > 40:
            d.text((a[0] + 3, a[1] + 2), typ, font=fonts.tiny, fill=(110, 80, 50) if on_floor else (130, 130, 170))
    # 歩ける所（点）と、まばらな座標の札（x・z がどちらも 0.5 の倍数の点だけ）
    for x, _y, z in reach:
        c = P(x, z)
        d.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], fill=(60, 140, 220))
    for x, _y, z in reach:
        if abs(x * 2 - round(x * 2)) < 1e-3 and abs(z * 2 - round(z * 2)) < 1e-3 and \
                abs(x - round(x)) < 1e-3 and abs(z - round(z)) < 1e-3:
            c = P(x, z)
            d.text((c[0] + 4, c[1] - 18), f"{x:.1f},{z:.1f}", font=fonts.tiny, fill=(30, 90, 160))
    # 名前の物（小さな点）と、おもちゃ
    toyset = set(env.toys)
    for i in env.targets:
        if i in toyset:
            continue
        c3 = env._centre(env.objects[i])
        c = P(c3[0], c3[2])
        d.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], outline=(150, 150, 150))
    for t in env.toy_info:
        if not t.get("placed") or t["id"] not in env.objects:
            continue
        c3 = env._centre(env.objects[t["id"]])
        c = P(c3[0], c3[2])
        d.ellipse([c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8], fill=TOY_COL, outline=(0, 0, 0))
        d.text((c[0] + 10, c[1] - 10), t["word"], font=fonts.tiny, fill=(160, 80, 0))
    # 座った所・向き・目・視野
    yaw = math.radians(float(ag["rotation"]["y"]))
    fx, fz = math.sin(yaw), math.cos(yaw)
    c = P(pos[0], pos[2])
    d.ellipse([c[0] - 10, c[1] - 10, c[0] + 10, c[1] + 10], fill=(220, 40, 90))
    tip = P(pos[0] + 0.6 * fx, pos[2] + 0.6 * fz)
    d.line([c, tip], fill=(220, 40, 90), width=4)
    e = env.eye_pos
    ec = P(e[0], e[2])
    for half in (-env.fov / 2, env.fov / 2):
        h = yaw + math.radians(half)
        d.line([ec, P(e[0] + 2.5 * math.sin(h), e[2] + 2.5 * math.cos(h))], fill=(220, 40, 90), width=1)
    pl = env.placement
    d.text((m, Hd - 50), f"{env.scene_name}　座った所 x={pos[0]:.2f} z={pos[2]:.2f} 向き {pl['yaw']:.0f}度（{pl['mode']}）　"
                         f"前 {pl['ahead_m']}m　到達点 {len(reach)}　--spot {pos[0]:.2f},{pos[2]:.2f},{pl['yaw']:.0f}",
           font=fonts.tiny, fill=(30, 30, 30))
    d.text((m, Hd - 26), "青い点＝歩ける所（座れる所）　赤＝座った所と向き・視野60度　橙＝並べたおもちゃ　茶の箱＝床の上の大きな物　薄い箱＝高い所の物",
           font=fonts.tiny, fill=(80, 80, 80))
    return np.asarray(img)


def vec_(dct) -> np.ndarray:
    return np.array([float(dct["x"]), float(dct["y"]), float(dct["z"])])


def first_frame(a, place) -> dict:
    """--first-frame：脳を回さず、最初の目の絵と上から見た絵を残す。"""
    import time
    from PIL import Image
    t0 = time.time()
    folder = a.out or os.path.join(growth.REPO, "G", "logs", "動画")
    if os.path.splitext(folder)[1]:
        folder = os.path.dirname(os.path.abspath(folder))
    os.makedirs(folder, exist_ok=True)
    env = make_env(a.scene, fake=a.fake, render_px=a.render_px, platform=a.platform, fov=a.fov, log_dir=folder,
                   seed=a.seed, place=place, toys=a.toys, **fake_kw(a))
    fonts = Fonts()
    try:
        p1 = os.path.join(folder, f"家の場面3_最初_{a.scene}.png")
        p2 = os.path.join(folder, f"家の場面3_上から_{a.scene}.png")
        Image.fromarray(eye_overlay(env, fonts)).save(p1)
        Image.fromarray(topdown(env, fonts)).save(p2)
        pl = dict(env.placement)
        res = {"scene": a.scene, "placement": pl, "toys": env.toy_info, "eye_png": p1, "top_png": p2,
               "thor_calls": env.n_calls, "thor_fail": env.n_fail, "toy_error": getattr(env, "toy_error", ""),
               "n_reach": 0 if env.reach is None else int(len(env.reach)), "wall_s": round(time.time() - t0, 1),
               "fake": a.fake}
    finally:
        env.close()
    with open(os.path.join(folder, f"家の場面3_最初_{a.scene}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(f"[最初の絵] {a.scene}　座る場所（{pl['mode']}）：{pl['position']} 向き {pl['yaw']}度　前 {pl['ahead_m']}m　"
          f"候補 {pl['n_candidates']}（前が開いた所 {pl['n_open']}）　{pl.get('error', '')}", flush=True)
    print(f"[最初の絵] おもちゃ：並べた {sum(t['placed'] for t in env.toy_info)}/{len(env.toy_info)}（{'・'.join(pl['toys'])}）　"
          f"絵に入る {pl['toys_in_view']}　写る {pl['toys_seen']}　置き方 {[t['method'] for t in env.toy_info]}", flush=True)
    print(f"[最初の絵] {p1}\n[最初の絵] {p2}\n[最初の絵] AI2-THOR を {res['thor_calls']}回、失敗 {res['thor_fail']}、"
          f"{res['wall_s']}秒", flush=True)
    return res


def run(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--scene", default="FloorPlan1", help="AI2-THOR の部屋（台所 FloorPlan1〜30・居間 FloorPlan201〜230）")
    ap.add_argument("--eye", default="dinov2", choices=["dinov2", "seg"])
    ap.add_argument("--from", dest="start", default="auto", help="auto（目だけの場面のいちばん上の段の写し）／birth／写しの名前")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=tp.REPEAT, help="親が見続けている物の名前を言い直す回数")
    ap.add_argument("--speed", type=float, default=1.0, help="再生の速さ（1＝実時間＝10コマ/秒）")
    ap.add_argument("--stills", type=int, default=3)
    ap.add_argument("--render-px", type=int, default=448, help="AI2-THOR に描かせる大きさ（目の入口へは 224 に縮める）")
    ap.add_argument("--fov", type=float, default=60.0, help="目の画角（60＝目の入口の指令の写しの較正と同じ。変えると写しが外れる）")
    ap.add_argument("--platform", default=None, help="AI2-THOR の platform（既定なし＝Linux64・窓あり。WSL では CloudRendering は動かない）")
    ap.add_argument("--place", default="auto", choices=["auto", "start"],
                    help="太郎の座る場所：auto＝前の開いた所を選ぶ（控え）／start＝エージェントの始めの場所。--spot があればそちら")
    ap.add_argument("--spot", default=None, help="座る場所を手で：x,z,yaw（上から見た絵 --first-frame の座標と向き。x が負なら --spot=-1.5,2,90 と = でつなぐ）")
    ap.add_argument("--toys", type=int, default=tp_env.TOYS_N, help="親が太郎の前の床に並べるおもちゃの数（0＝並べない）")
    ap.add_argument("--first-frame", action="store_true",
                    help="脳を回さず、最初の目の絵（重ね書きつき）と上から見た絵だけを --out のフォルダへ残す（家の場面3_最初_<部屋>.png・家の場面3_上から_<部屋>.png）")
    ap.add_argument("--bring-in", type=float, default=tp.BRING_IN_AFTER_S,
                    help="名前の物をこの秒数見なければ親が物を持ってきて見せる（0＝しない）")
    ap.add_argument("--fake", action="store_true", help="偽物の家（world_thor/fake.py）。テスト・クラウド用")
    ap.add_argument("--max-ticks", type=int, default=0, help="この回数で止める（試し用）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    place = parse_spot(a.spot) if a.spot else a.place
    if a.first_frame:
        return first_frame(a, place)
    eye_tag = a.eye + ("_偽物" if a.fake else "")
    out = a.out or os.path.join(growth.REPO, "G", "logs", "動画", f"家の場面_{a.scene}_{eye_tag}_種{a.seed}.mp4")
    base = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if a.fov != 60.0:
        print(f"[家の動画] 画角 {a.fov}度：目の入口の指令の写しは60度で較正してある。目が動いた分の予想が外れる", flush=True)
    fonts = Fonts()
    painter = ThorPainter(fonts)
    writer = Writer(out, fps=a.speed / TICK_S)
    stats: list[dict] = []
    frames_kept: list = []
    ticks = int(round(a.seconds / TICK_S))
    if a.max_ticks:
        ticks = min(ticks, a.max_ticks)

    def on_tick(g, got):
        st = painter.update(g, got)
        frame = painter.draw(g, st)
        writer.add(frame)
        stats.append(st)
        if a.stills:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG", compress_level=1)
            frames_kept.append(buf.getvalue())

    env = make_env(a.scene, fake=a.fake, render_px=a.render_px, platform=a.platform, fov=a.fov,
                   log_dir=os.path.dirname(os.path.abspath(out)), seed=a.seed, place=place, toys=a.toys, **fake_kw(a))
    import time
    t0 = time.time()
    g = ThorRecorder(a.seed, a.eye, env=env, start=a.start, repeat=a.repeat, frame_fn=on_tick,
                     bring_in_after=a.bring_in if a.bring_in > 0 else None)
    pl = env.placement
    print(f"[家の動画] 座る場所（{pl['mode']}）：{pl['position']} 向き {pl['yaw']}度　目に入る名前の物 {pl['n_in_view']}"
          f"（{'・'.join(pl['words']) or 'なし'}）　点 {pl['score']}（引いた点 {pl['penalty']} {pl['why']}）　"
          f"始めの場所なら {pl['start_n_in_view']}　候補 {pl['n_candidates']}（前が開いた所 {pl['n_open']}）　前 {pl['ahead_m']}m", flush=True)
    print(f"[家の動画] おもちゃ：並べた {len(env.toys)}（{'・'.join(pl['toys']) or 'なし'}）　絵に入る {pl['toys_in_view']}　"
          f"写る {pl['toys_seen']}　置き方 {sorted({t['method'] for t in env.toy_info})}", flush=True)
    print(f"[家の動画] 始める太郎：{g.start_name()}　目：{g.eye_info.get('eye')}　部屋：{a.scene}　"
          f"名前を言える物 {len(env.targets)}（{sorted({env.word_of(i) for i in env.targets})}）　{ticks / 10:.0f}秒", flush=True)
    try:
        for k in range(ticks):
            g.step_once()
            if (k + 1) % 100 == 0:
                print(f"[家の動画] {(k + 1) * TICK_S:.0f}秒　実時間 {time.time() - t0:.0f}秒　持ってきた {g.parent.stats['brings']}　"
                      f"名づけ {g.parent.stats['names']}", flush=True)
        g.looks.flush()
    finally:
        g.close()
    path = writer.close()
    stills = []
    for j, i in enumerate(pick_stills(stats, a.stills), 1):
        p = base + f"_{j}.png"
        with open(p, "wb") as f:
            f.write(frames_kept[i])
        stills.append(p)
    res = g.summary()
    res.update({"out": path, "frames": len(stats), "wall_s": round(time.time() - t0, 1), "stills": stills,
                "bytes": os.path.getsize(path), "font": fonts.path, "fake": a.fake, "fov": a.fov})
    with open(base + "_まとめ.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    lk = res["looks"]
    print(f"[家の動画] 動画 {path}（{len(stats)}コマ、実時間 {res['wall_s']}秒、AI2-THOR を {res['thor_calls']}回、失敗 {res['thor_fail']}）",
          flush=True)
    ps = res["parent"]
    print(f"[家の動画] 親：持ってきた {ps['brings']}（follow {ps['brings_follow']}・bringin {ps['brings_bringin']}、"
          f"bringin を見た {ps['bringin_looked']}・見なかった {ps['bringin_unlooked']}）　名づけ {ps['names']}"
          f"（follow {ps['names_follow']}・bringin {ps['names_bringin']}、言い直し {ps['repeats']}）　語ごと {res['named_by_word']}",
          flush=True)
    print(f"[家の動画] 見た時間：{lk['looks']}回　平均 {lk['mean_s']}秒　真ん中 {lk['median_s']}秒　最長 {lk['max_s']}秒　"
          f"1秒以上 {lk['n_ge_1s']}回　語ごと {lk['by_word']}", flush=True)
    return res


if __name__ == "__main__":
    run(sys.argv[1:])
