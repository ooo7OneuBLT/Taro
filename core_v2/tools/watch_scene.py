"""目だけの場面で、太郎が何をしているかを動画で見る（段3の名づけ、grow_names と同じ場面・同じ親・同じ脳）。

grow_names.Grower をそのまま使い、0.1秒の1回ごとに1コマ描く（--speed 1 で実時間＝10コマ/秒）。写しは作らない。
1コマ：
* 左の大きな絵：太郎の目に入っている画像（目の入口 eyein が受け取るのと同じカメラ。大きく描き直しただけ）。
  真ん中の白い丸＝太郎が見ている所（視線の真ん中）。黄色い四角＝物の名簿（objfile）の「今見ている物」。
  物の下の小さな字＝その物の本当の語（場面に置いた物の正解。太郎が知っているわけではない）
* 右の欄：場面・部屋・秒、親が言ったこと（言ってから約2秒）、太郎が聞き取った語（耳 listen.last_word）、
  太郎の声、「覚えた！」（意味の辞書が覚えた回。語は覚えた語）、見ている物（親から見た正解）と見ている秒数の棒、
  ここまでの名づけ・覚えた回数

使い方：
  python -m core_v2.tools.watch_scene [--from 写しの名前|auto|birth] [--eye table|dinov2] [--objects made|real]
                                      [--scenes 2] [--repeat 2] [--seed 0] [--speed 1] [--stills 3] [--out 動画の場所]
出力：--out（既定 G/logs/動画/太郎の場面_<物>_<目>_種<seed>.mp4）。ffmpeg が無ければ .gif。
      --stills N なら、同じ名前に _1.png … _N.png の静止画を並べて残す
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from ..world_eye import instances as ins
from ..world_eye import noun_scene as ns
from . import growth
from .grow_names import Grower

TICK_S = ns.TICK_S
W, H = 1280, 720          # 1コマの大きさ
EYE_PX = 672              # 左の絵（目の画像 224 の3倍）
PAD = 24
SAY_HOLD_S = 2.0          # 親の言葉・聞き取った語・太郎の声を出しておく秒数
LEARN_HOLD_S = 1.5        # 「覚えた！」を出しておく秒数
LOOK_BAR_S = 6.0          # 見ている秒数の棒の長さ（これで満杯）

BG = (24, 26, 32)
FG = (240, 240, 240)
DIM = (150, 155, 165)
YELLOW = (255, 214, 64)
CYAN = (110, 210, 255)
GREEN = (120, 230, 140)
PINK = (255, 140, 180)
GAZE = (255, 60, 200)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]


def find_font() -> str | None:
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            return p
    for pat in ("/usr/share/fonts/**/*[Ii][Pp][Aa]*.tt*", "/usr/share/fonts/**/*CJK*.tt*", "/usr/share/fonts/**/*Takao*.tt*"):
        hit = sorted(glob.glob(pat, recursive=True))
        if hit:
            return hit[0]
    return None


class Fonts:
    def __init__(self) -> None:
        from PIL import ImageFont
        path = find_font()
        if path is None:
            print("[動画] 日本語の字体が見つからない（字が四角になる）。IPA ゴシックか Noto Sans CJK を入れること", flush=True)
        self.path = path

        def f(size):
            return ImageFont.truetype(path, size) if path else ImageFont.load_default()
        self.big, self.mid, self.small, self.tiny = f(40), f(30), f(24), f(18)


def encoder() -> str | None:
    """ffmpeg の場所（imageio-ffmpeg が入っていればそれ、無ければ PATH の ffmpeg）。無ければ None（GIF にする）。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class Recorder(Grower):
    """grow_names の Grower。1回ごとに太郎の様子を覚え、コマを描く。"""

    def __init__(self, *a, frame_fn=None, big_px: int = EYE_PX, **kw) -> None:
        super().__init__(*a, **kw)
        self.frame_fn = frame_fn
        self.big_px = big_px
        self._big = {}               # 部屋の環境 → 大きく描く renderer
        self.scene_tick = 0
        self.shown: dict = {}        # 最後に出した言葉など → (文字, 出した回)
        self.learned_word = ""
        sem = self.brain.boxes["semantic"]
        orig = sem._learn_pending

        def learn_pending(p, _orig=orig):     # 覚えた語を記録する（中身は変えない）
            _orig(p)
            if sem.last_event == "learn":
                self.learned_word = word_of(p["q"])
        sem._learn_pending = learn_pending

    def face_table(self, cause: str = "scene_start") -> None:
        super().face_table(cause)
        self.scene_tick = 0

    def big_image(self) -> np.ndarray:
        """太郎の目と同じカメラで、大きく描いた画像（中身は eyein が受け取る 224 の画像と同じ場面）。"""
        import mujoco
        env = self.env
        r = self._big.get(id(env))
        if r is None:
            g = env.model.vis.global_
            g.offwidth, g.offheight = max(g.offwidth, self.big_px), max(g.offheight, self.big_px)
            r = mujoco.Renderer(env.model, height=self.big_px, width=self.big_px)
            self._big[id(env)] = r
        r.update_scene(env.data, camera=env._cam_id, scene_option=env._vopt)
        return r.render().copy()

    def tick(self) -> dict:
        got = super().tick()
        self.scene_tick += 1
        if self.frame_fn is not None:
            self.frame_fn(self, got)
        return got

    def close(self) -> None:
        for r in self._big.values():
            r.close()
        self._big = {}
        super().close()


def word_of(q) -> str:
    """耳が意味の辞書へ渡した語の荷札から、いちばん近い語（20語）。"""
    import torch
    from ..tagging import word_tag
    best, bc = "", -2.0
    for w in ins.WORDS:
        c = float(word_tag(w) @ q / (torch.linalg.vector_norm(q) + 1e-9))
        if c > bc:
            best, bc = w, c
    return best


def room_of(g: Recorder) -> str:
    for room, e in g.envs.items():
        if e is g.env:
            return room
    return ""


class Painter:
    """1回ごとの太郎の様子から1コマ（W×H の RGB）を描く。"""

    def __init__(self, fonts: Fonts) -> None:
        self.f = fonts
        self.said = ("", -10 ** 9)
        self.heard = ("", -10 ** 9)
        self.voice = ("", -10 ** 9)
        self.learn = ("", -10 ** 9)
        self.n = 0

    def update(self, g: Recorder, got: dict) -> dict:
        """コマに出すことを集める（描かずに数えるだけ：テストと静止画の選び方にも使う）。"""
        n = g.n_tick
        sem = g.brain.boxes["semantic"]
        ear = g.brain.boxes["listen"]
        if got.get("heard"):
            self.said = (got["heard"], n)
        if ear.last_word:
            self.heard = (ear.last_word, n)
        if g._echo:
            self.voice = (g._echo, n)
        if sem.last_event == "learn":
            self.learn = (g.learned_word, n)
        p = g.parent.cur
        in_scene = bool(g.env.n9_view)
        st = {
            "n": n, "scene": g.n_scenes + 1, "room": room_of(g), "t": g.scene_tick * TICK_S, "in_scene": in_scene,
            "said": self._hold(self.said, n, SAY_HOLD_S), "heard": self._hold(self.heard, n, SAY_HOLD_S),
            "voice": self._hold(self.voice, n, SAY_HOLD_S), "learn": self._hold(self.learn, n, LEARN_HOLD_S),
            "learn_now": sem.last_event == "learn",
            "looking": p.word if p is not None else "", "look_s": (g.parent.d + 1) * TICK_S if p is not None else 0.0,
            "names": g.parent.stats["names"], "learns": g.n_learn + (sem.last_event == "learn"),
        }
        return st

    @staticmethod
    def _hold(v, n, s):
        txt, at = v
        return txt if txt and n - at < int(round(s / TICK_S)) else ""

    def draw(self, g: Recorder, st: dict) -> np.ndarray:
        from PIL import Image, ImageDraw
        from ..world_eye.eye_backend import project
        env = g.env
        img = Image.new("RGB", (W, H), BG)
        eye = Image.fromarray(g.big_image())
        d = ImageDraw.Draw(eye)
        s = g.big_px / 224.0
        # 物の正解の語（小さく）
        for nm in list(env.n9_view):
            pos = project(env, nm)
            if pos is None:
                continue
            x, y = pos[0] * s, pos[1] * s
            word = env.n9[nm]["word"]
            r = 0.5 * env.n9[nm]["size"] / ns.DIST[g.objects] / math.tan(math.radians(env.fovy) / 2) * g.big_px / 2
            ty = min(y + r + 6, g.big_px - 26)
            tw = d.textlength(word, font=self.f.tiny)
            d.rectangle([x - tw / 2 - 4, ty - 2, x + tw / 2 + 4, ty + 22], fill=(0, 0, 0))
            d.text((x - tw / 2, ty), word, font=self.f.tiny, fill=(230, 230, 230))
        # 名簿の今見ている物（黄色い四角）
        of = g.brain.boxes["objfile"]
        rw = getattr(of, "_held_now", None)
        if rw is not None:
            r_, c_ = rw.pos
            x, y = (8 * c_ + 4) * s, (8 * r_ + 4) * s
            h = 3.0 * 8 * s
            d.rectangle([x - h, y - h, x + h, y + h], outline=YELLOW, width=4)
        # 視線の真ん中（白い丸）
        c = g.big_px / 2
        rr = 5.0 / (env.fovy / 2) * c          # 半径 約5度
        for col, wd in (((0, 0, 0), 7), (GAZE, 3)):      # 黒い縁取りの上に色の丸（白い床でも見える）
            d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=col, width=wd)
            d.line([c - 10, c, c + 10, c], fill=col, width=wd - 1)
            d.line([c, c - 10, c, c + 10], fill=col, width=wd - 1)
        img.paste(eye, (PAD, PAD))
        D = ImageDraw.Draw(img)
        D.text((PAD, PAD + g.big_px + 2), "太郎の目に見えている画像　○（桃色）＝見ている所　□（黄色）＝太郎が「今見ている物」と思っている所",
               font=self.f.tiny, fill=DIM)
        # 右の欄
        x0 = PAD * 2 + g.big_px
        y = PAD
        sc = f"場面 {st['scene']}　{st['room']}の部屋" if st["room"] else f"場面 {st['scene']}"
        D.text((x0, y), sc, font=self.f.mid, fill=FG)
        y += 40
        D.text((x0, y), f"{st['t']:4.1f} 秒" + ("" if st["in_scene"] else "　（休み：物が無い）"), font=self.f.small, fill=DIM)
        y += 44

        def row(label, val, color, y):
            D.text((x0, y), label, font=self.f.small, fill=DIM)
            D.text((x0 + 12, y + 30), val or "―", font=self.f.big if val else self.f.mid, fill=color if val else DIM)
            return y + 86

        y = row("親が言ったこと", "「" + st["said"] + "」" if st["said"] else "", CYAN, y)
        y = row("太郎が聞き取った語", st["heard"], GREEN, y)
        y = row("太郎の声", "「" + st["voice"] + "」" if st["voice"] else "", PINK, y)
        # 見ている物と秒数の棒
        D.text((x0, y), "見ている物", font=self.f.small, fill=DIM)
        D.text((x0 + 12, y + 30), st["looking"] or "（物を見ていない）", font=self.f.big if st["looking"] else self.f.mid,
               fill=FG if st["looking"] else DIM)
        by = y + 82
        bw = W - x0 - PAD - 90
        D.rectangle([x0 + 12, by, x0 + 12 + bw, by + 18], outline=DIM, width=2)
        frac = min(1.0, st["look_s"] / LOOK_BAR_S)
        if frac > 0:
            D.rectangle([x0 + 12, by, x0 + 12 + int(bw * frac), by + 18], fill=YELLOW)
        D.text((x0 + 22 + bw, by - 6), f"{st['look_s']:.1f}秒", font=self.f.small, fill=FG)
        y = by + 38
        # 覚えた！
        if st["learn"]:
            D.rounded_rectangle([x0, y, W - PAD, y + 64], radius=12, fill=(60, 50, 10), outline=YELLOW, width=3)
            D.text((x0 + 16, y + 10), f"覚えた！「{st['learn']}」", font=self.f.big, fill=YELLOW)
            D.text((x0 + 4, y + 70), "少し前に聞いた語を、聞く前後に見ていた物と結び付けた", font=self.f.tiny, fill=DIM)
        y += 80
        D.text((x0, H - PAD - 34), f"ここまで　名づけ {st['names']} 回　覚えた {st['learns']} 回", font=self.f.mid, fill=FG)
        return np.asarray(img)


class Writer:
    """コマを MP4（ffmpeg）か GIF（PIL）へ書く。いったん手元の一時ファイルへ書き、最後に out へ写す。"""

    def __init__(self, out: str, fps: float) -> None:
        self.ff = encoder()
        if out.lower().endswith(".gif"):
            self.ff = None
        if self.ff is None and not out.lower().endswith(".gif"):
            out = os.path.splitext(out)[0] + ".gif"
            print("[動画] ffmpeg が無いので GIF にする", flush=True)
        self.out, self.fps = out, fps
        self.tmpdir = tempfile.mkdtemp(prefix="watch_scene_")
        self.tmp = os.path.join(self.tmpdir, "v" + os.path.splitext(out)[1])
        self.frames = []
        self.proc = None
        if self.ff:
            self.proc = subprocess.Popen(
                [self.ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                 "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
                 "-preset", "medium", "-movflags", "+faststart", self.tmp], stdin=subprocess.PIPE)

    def add(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        else:
            from PIL import Image
            self.frames.append(Image.fromarray(frame).resize((W // 2, H // 2)).convert("P", palette=Image.ADAPTIVE))

    def close(self) -> str:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg が失敗した")
        elif self.frames:
            self.frames[0].save(self.tmp, save_all=True, append_images=self.frames[1:],
                                duration=int(round(1000 / self.fps)), loop=0, optimize=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
        shutil.copyfile(self.tmp, self.out)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return self.out


def pick_stills(stats: list[dict], n: int) -> list[int]:
    """静止画にするコマ：覚えた！・親が言って聞き取れた・物を長く見ている、の順に、互いに離れたコマ。"""
    if n <= 0 or not stats:
        return []
    keys = [lambda s: s["learn_now"], lambda s: bool(s["said"] and s["heard"]),
            lambda s: s["in_scene"] and s["look_s"] >= 2.0, lambda s: s["in_scene"]]
    out: list[int] = []
    gap = max(10, len(stats) // (3 * n))
    for k in keys:
        for i, s in enumerate(stats):
            if len(out) >= n:
                break
            if k(s) and all(abs(i - j) >= gap for j in out):
                out.append(min(i + (3 if k is keys[0] else 0), len(stats) - 1))
    return sorted(out)


def run(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--objects", default="made", choices=["made", "real"])
    ap.add_argument("--eye", default="dinov2", choices=["dinov2", "table"])
    ap.add_argument("--table", default=None, help="代わりの目の見た目の素の表（grow_names と同じ）")
    ap.add_argument("--kind", default="off", choices=["off", "on"])
    ap.add_argument("--from", dest="start", default="auto", help="auto（段2→段1→生まれたまま）／birth／写しの名前")
    ap.add_argument("--repeat", type=int, default=ns.REPEAT_LABELS, help="親が名前を言い直す回数（grow_names と同じ）")
    ap.add_argument("--speed", type=float, default=1.0, help="再生の速さ（1＝実時間＝10コマ/秒、2＝2倍速）")
    ap.add_argument("--stills", type=int, default=0, help="静止画の数（動画と同じ名前に _1.png …）")
    ap.add_argument("--max-ticks", type=int, default=0, help="この回数で止める（試し用。0＝場面の数だけ）")
    ap.add_argument("--out", default=None, help="動画の場所（既定 G/logs/動画/太郎の場面_<物>_<目>_種<seed>.mp4）")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(growth.REPO, "G", "logs", "動画", f"太郎の場面_{a.objects}_{a.eye}_種{a.seed}.mp4")
    fonts = Fonts()
    painter = Painter(fonts)
    writer = Writer(out, fps=a.speed / TICK_S)
    stats: list[dict] = []
    frames_kept: list = []

    class Stop(Exception):
        pass

    def on_tick(g, got):
        st = painter.update(g, got)
        frame = painter.draw(g, st)
        writer.add(frame)
        stats.append(st)
        if a.stills:                      # 静止画の候補：PNG に縮めて覚えておく（全部を生で持つと重い）
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="PNG", compress_level=1)
            frames_kept.append(buf.getvalue())
        if a.max_ticks and len(stats) >= a.max_ticks:
            raise Stop

    g = Recorder(a.seed, a.objects, a.eye, a.table, log_root=None, start=a.start, kind=a.kind, repeat=a.repeat,
                 frame_fn=on_tick)
    g.save_images(False)
    print(f"[動画] 始める太郎：{g.start_name()}　目：{g.eye_info.get('eye')}　物：{a.objects}　場面 {a.scenes}", flush=True)
    try:
        for _ in range(a.scenes):
            g.scene()
    except Stop:
        pass
    finally:
        g.close()
    path = writer.close()
    stills = []
    for j, i in enumerate(pick_stills(stats, a.stills), 1):
        p = os.path.splitext(path)[0] + f"_{j}.png"
        with open(p, "wb") as f:
            f.write(frames_kept[i])
        stills.append(p)
    res = {"out": path, "frames": len(stats), "seconds": round(len(stats) / writer.fps, 1),
           "bytes": os.path.getsize(path), "names": g.parent.stats["names"], "learns": g.n_learn,
           "stills": stills, "start": g.start_name(), "font": fonts.path}
    print(f"[動画] {res}", flush=True)
    return res


if __name__ == "__main__":
    run(sys.argv[1:])
