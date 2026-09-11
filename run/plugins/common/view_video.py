# -*- coding: utf-8 -*-
"""見る道具：太郎の視界（左目の実入力）と第三者視点を、走行中にそのまま動画にする（2026-09-03）。

【なぜ】学習結果は数字だけでは騙される（指標は「行動の退化」を高評価しがち）。
  提示・見た目はユーザーの目視承認までは本走行しない、という運用のための道具。
  以前は trace の npy ダンプ→別スクリプトで変換だったが、「実験ファイルに1行」で済むようにした。

【使い方】実験ファイルの plugins に足す：
    "view_video": {"out": "F/logs/<実験>/動画_視界.mp4"}            # 全歩・既定設定
    "view_video": {"out": "...mp4", "until": 900, "third_person": true, "fps": 0}
  設定：
    out          動画の保存先（.mp4）。代表コマ PNG も同名で残す
    until        何歩まで撮るか（0=最後まで）。長い走行では最初の数百歩だけにする
    third_person 第三者視点を左に並べる（既定 true。描画1回ぶん遅くなる）
    fps          コマ/秒。0=実時間（1歩の秒数の逆数＝K=10なら10コマ/秒）
    eye          "left"/"right"（既定 left）
  下の字幕：親の発話（黄）と太郎の発話（水色）。字幕は1秒で消える。

【出るもの】左：第三者視点（走行中のその場描画＝本物の環境）、右：目の実入力（学習に使われた画像）。
  ffmpeg（PATH上）で mp4 化。ffmpeg が無ければ PNG 連番を残して終える。
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

from run.plugins.base import Plugin

_FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"


class ViewVideo(Plugin):
    name = "view_video"

    def setup(self, ctx):
        from PIL import ImageFont
        c = self.config
        self.out = c.get("out") or "F/logs/view_video.mp4"
        self.until = int(c.get("until", 0) or 0)
        self.third = bool(c.get("third_person", True))
        # 【2026-09-11】第三者視点を「頭と対象物が両方入る」画角にする。既定False＝従来不変。
        self.third_track = bool(c.get("third_track", False))
        # 【2026-09-11】目の映像に、上からの目的（赤○）と注意の行き先（水色＋）を
        #   重ねる。K1 のとき「動画にはK1が写っていない」とユーザーに指摘された
        #   （目的も注意も描いていなかったので、見えるはずがなかった）。既定False。
        self.overlay = bool(c.get("overlay", False))
        self.eye = "eye_left" if str(c.get("eye", "left")).startswith("l") else "eye_right"
        dt = float(getattr(ctx, "dt", 0.1) or 0.1)
        self.fps = int(c.get("fps", 0) or 0) or max(1, int(round(1.0 / dt)))
        self.font = ImageFont.truetype(_FONT_PATH, 18)
        self.font_s = ImageFont.truetype(_FONT_PATH, 14)
        self.tmp = tempfile.mkdtemp(prefix="view_video_")
        self.n = 0
        self.label_frames = []          # (歩, 親の発話, 目の画像) 最大48件
        self.parent_txt, self.parent_step = "", -10 ** 9
        self.taro_txt, self.taro_step = "", -10 ** 9
        self.ren = None
        self.cam = None
        if self.third:
            import mujoco
            u = ctx.env.unwrapped
            m = u.model
            m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
            m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)
            self.ren = mujoco.Renderer(m, height=448, width=560)
            self.cam = mujoco.MjvCamera()
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
        os.makedirs(os.path.dirname(self.out) or ".", exist_ok=True)

    def _third_person(self, ctx):
        u = ctx.env.unwrapped
        d = u.data
        look = d.xpos[self._head] if self._head >= 0 else d.qpos[:3]
        # 【2026-09-11・ユーザー指示「第三者視点は太郎の顔と対象物が映るように」】
        #   third_track=True（既定False＝従来と1ビットも変わらない）なら、
        #   固定の画角をやめて「頭と対象物の**中点**」を狙い、2つの距離に合わせて
        #   引く。見る方向も、頭→物の線に**直交**する側から見る（そうしないと
        #   物が頭の真後ろに隠れる）。対象物の位置は親が物を持つ場所（_rest_pos）。
        toy = getattr(u, "_rest_pos", None) if self.third_track else None
        if toy is not None:
            import math
            head = np.asarray(look, dtype=float)
            toy = np.asarray(toy, dtype=float)
            mid = (head + toy) / 2.0
            sep = float(np.linalg.norm(head - toy))
            self.cam.lookat[:] = mid
            self.cam.distance = max(0.55, sep * 2.4)
            self.cam.azimuth = math.degrees(math.atan2(toy[1] - head[1],
                                                        toy[0] - head[0])) + 90.0
            self.cam.elevation = -12
        else:
            self.cam.lookat[:] = [look[0] + 0.15, look[1], look[2] - 0.05]
            self.cam.distance = 1.1
            self.cam.azimuth = 150
            self.cam.elevation = -12
        self.ren.update_scene(d, camera=self.cam)
        return self.ren.render().copy()

    def on_step(self, ctx):
        if self.until and ctx.step > self.until:
            return
        last = getattr(ctx, "last", None)
        eye = None
        if last is not None:
            o = last.get("obs_out") or last.get("obs_in")
            if isinstance(o, dict) and self.eye in o:
                v = o[self.eye]
                eye = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        if eye is None:
            return          # 脳を通さない実行（measure）では目の画像が置かれない
        eye = np.asarray(eye)
        if eye.dtype != np.uint8:
            eye = np.clip(eye * (255.0 if eye.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if eye.ndim == 3 and eye.shape[0] in (1, 3) and eye.shape[-1] not in (1, 3):
            eye = np.transpose(eye, (1, 2, 0))
        if eye.ndim == 2 or eye.shape[-1] == 1:
            eye = np.repeat(eye.reshape(eye.shape[0], eye.shape[1], 1), 3, axis=2)
        # 字幕の材料（＋「語を聞いた瞬間の目」を一覧用に取っておく）
        for ev in (getattr(ctx, "last_parent_utterance", None) or []):
            self.parent_txt, self.parent_step = str(ev.get("text", "")), ctx.step
            if len(self.label_frames) < 48:
                self.label_frames.append((ctx.step, self.parent_txt, eye.copy()))
        pr = getattr(ctx, "last_produce", None)
        if isinstance(pr, dict):
            t = pr.get("generated_word") or pr.get("text") or pr.get("word")   # trainer._apply_word_production の形
            if t:
                self.taro_txt, self.taro_step = (t if isinstance(t, str) else "".join(map(str, t))), ctx.step
        if ctx.step - self.parent_step > self.fps:
            self.parent_txt = ""
        if ctx.step - self.taro_step > self.fps:
            self.taro_txt = ""
        self._frame(ctx, eye)

    def _frame(self, ctx, eye):
        from PIL import Image, ImageDraw
        E = 448
        W = (560 if self.third else 0) + E
        H = E + 60
        canvas = Image.new("RGB", (W, H), (20, 20, 20))
        x_eye = 0
        if self.third:
            canvas.paste(Image.fromarray(self._third_person(ctx)), (0, 0))
            x_eye = 560
        canvas.paste(Image.fromarray(eye).resize((E, E), Image.NEAREST), (x_eye, 0))
        dr = ImageDraw.Draw(canvas)
        # 【2026-09-11】目的（赤○）と注意の行き先（水色＋）を目の映像に重ねる。
        #   どちらも 224px 系の座標で置かれているので、表示の大きさへ拡大する。
        if self.overlay:
            k = E / float(eye.shape[0])
            g = getattr(ctx, "goal_point", None)
            if g is not None:
                gx, gy = x_eye + g[0] * k, g[1] * k
                dr.ellipse([gx - 26, gy - 26, gx + 26, gy + 26], outline=(255, 70, 70), width=4)
                dr.text((gx + 30, gy - 10), "探し物", fill=(255, 110, 110), font=self.font_s)
            a = getattr(ctx, "attention_point", None)
            if a is not None:
                ax, ay = x_eye + a[0] * k, a[1] * k
                dr.line([ax - 18, ay, ax + 18, ay], fill=(0, 230, 255), width=4)
                dr.line([ax, ay - 18, ax, ay + 18], fill=(0, 230, 255), width=4)
        if self.third:
            dr.text((6, 4), "第三者視点", fill=(255, 255, 255), font=self.font_s)
        dr.text((x_eye + 6, 4), "太郎の%s目の実入力 %dpx" % ("左" if self.eye == "eye_left" else "右", eye.shape[0]),
                fill=(255, 255, 255), font=self.font_s)
        dr.text((6, E + 6), "歩 %4d  %5.1f 秒" % (ctx.step, ctx.step / float(self.fps)),
                fill=(200, 200, 200), font=self.font)
        if self.parent_txt:
            dr.text((200, E + 6), "親「%s」" % self.parent_txt, fill=(255, 230, 120), font=self.font)
        if self.taro_txt:
            dr.text((200, E + 32), "太郎「%s」" % self.taro_txt, fill=(150, 220, 255), font=self.font)
        canvas.save(os.path.join(self.tmp, "f%06d.png" % self.n))
        self.n += 1

    def report(self, ctx):
        if self.ren is not None:
            self.ren.close()
        if self.n == 0:
            return {"コマ数": 0, "注意": "目の画像が置かれなかった（measure実行？）"}
        mid = os.path.join(self.tmp, "f%06d.png" % (self.n // 2))
        rep = self.out.replace(".mp4", "_代表コマ.png")
        shutil.copy(mid, rep)
        sheet = self._label_sheet()
        if shutil.which("ffmpeg") is None:
            keep = self.out.replace(".mp4", "_frames")
            shutil.move(self.tmp, keep)
            return {"コマ数": self.n, "注意": "ffmpeg が無いので PNG 連番のみ", "出力": keep}
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps),
                        "-i", os.path.join(self.tmp, "f%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", self.out], check=True)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return {"コマ数": self.n, "コマ/秒": self.fps, "出力": self.out, "代表コマ": rep, "発話時の目": sheet}

    def _label_sheet(self):
        """親が語を言った瞬間の目の画像を、時間順に並べた一覧（1行8枚）。"""
        if not self.label_frames:
            return None
        from PIL import Image, ImageDraw
        T, L, per = 224, 20, 8
        rows = (len(self.label_frames) + per - 1) // per
        canvas = Image.new("RGB", (per * (T + 4), rows * (T + L + 4)), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for i, (s, txt, eye) in enumerate(self.label_frames):
            x, y = (i % per) * (T + 4), (i // per) * (T + L + 4)
            canvas.paste(Image.fromarray(eye).resize((T, T), Image.NEAREST), (x, y + L))
            dr.text((x + 2, y + 2), "歩%d 親「%s」" % (s, txt), fill=(0, 0, 0), font=self.font_s)
        p = self.out.replace(".mp4", "_発話時の目.png")
        canvas.save(p)
        return p
