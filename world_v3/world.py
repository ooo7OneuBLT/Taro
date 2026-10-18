"""太郎の新しい世界（world_v3）：MuJoCo だけで作る、部屋と物と目（カメラ）の小さな世界。

【2026-10-18新設・ユーザー決定】MIMo（赤ちゃんの体）と、その上に積んだ環境の層（toy_env・scene_io・parent・
run_v2 の体の部分）は使わない。今の太郎は体を固定して目しか動かしていないので、体は持たない。
最優先は「AI（Claude）との相性」＝全部が Python の中にあって、物の位置・見えているか・絵を直接読める・変えられること。
次は「ユーザーが絵や動画で中を見られること」（record で1試行の絵を残し、save_video で動画にする）。

世界の中身（make_xml）：
* 部屋：床・奥と左右の壁（市松模様）・机。明かりは天井から
* 物：spec["objects"] の物を机の上に置く（形 sphere/cylinder/box/capsule/ellipsoid・大きさ・色・位置）。
  物は動かない台の上に置くだけ（物理で倒れたり転がったりはさせない。動かすのは予定表で位置を書き換える）
* 目：頭の位置に固定した目（カメラ、視野60度、224×224）。向き（左右 yaw・上下 pitch、度）だけを変える

1回（tick）＝0.1秒。脳の目を跳ばす反射が出す向き cmd=(h, v)（[-1,1]、視野の半分＝30度が1。右・上が正）を受けて、
目の向きを h×30度・v×30度だけ動かす（目標の向き）。動き方：
* 指令を受けてから SACC_LATENCY_S のあとに動き始め、その回のうちに目標へ着く（跳ぶ1回は 0.1秒より短い。
  大人のサッケードは数十ミリ秒［記憶ベース］）。目の可動域は左右上下 ±EYE_RANGE_DEG で止まる
* 跳んでいない間は、目は止まっている（ゆっくり追う動きは無い）
人間・前の太郎との違い（逸脱の候補）：首は動かさない（前の太郎の NECK_SHARE=0 と同じ）。筋肉の遅れは無い。

予定表（spec["events"]）：時刻 t の回に、物を show（見せる）・hide（見えなくする＝透明）・move（位置を移す）・
color（色を変える）、親が say（言う：耳へ届く文字）・sound（物の音：耳へ届く文字の印）。
時計は tick×0.1秒（0秒から始まる。MIMo の世界のような落ち着かせの 0.5秒は無い）。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

TICK_S = 0.1
IMG = 224
FOVY_DEG = 60.0
HALF_VIEW_DEG = FOVY_DEG / 2.0
EYE_RANGE_DEG = 45.0          # 目の可動域（前の太郎と同じ ±45度）
SACC_LATENCY_S = 0.0          # 指令から動き始めるまで［仮置き］。脳の側の遅れ（1回）は脳が持つ
EYE_POS = (0.0, 0.0, 0.35)    # 目の高さ（座った赤ちゃんの目の高さ相当）［仮置き］

SHAPE_SIZE = {
    "sphere": lambda r: f"{r}",
    "cylinder": lambda r: f"{r} {r * 1.3}",
    "box": lambda r: f"{r} {r} {r}",
    "capsule": lambda r: f"{r * 0.8} {r}",
    "ellipsoid": lambda r: f"{r} {r * 0.8} {r * 1.3}",
}


def _rgba(c):
    return " ".join(str(float(x)) for x in (list(c) + [1.0])[:4])


def make_xml(spec: dict) -> str:
    """spec から MuJoCo の設計図を作る。物は名前 obj_<name> の body（mocap＝位置を外から書く）。"""
    room = spec.get("room", {})
    wall1 = _rgba(room.get("wall_rgb1", [0.56, 0.58, 0.64]))[:-4]
    wall2 = _rgba(room.get("wall_rgb2", [0.49, 0.52, 0.60]))[:-4]
    floor1 = _rgba(room.get("floor_rgb1", [0.62, 0.55, 0.45]))[:-4]
    floor2 = _rgba(room.get("floor_rgb2", [0.55, 0.48, 0.40]))[:-4]
    table = room.get("table", {"pos": [0.32, 0.0, 0.25], "size": [0.12, 0.25, 0.01], "rgb": [0.75, 0.62, 0.45]})
    objs = []
    for o in spec.get("objects", []):
        shape = o.get("shape", "sphere")
        r = float(o.get("radius", 0.03))
        pos = " ".join(str(float(x)) for x in o["pos"])
        objs.append(
            f'<body name="obj_{o["name"]}" mocap="true" pos="{pos}">'
            f'<geom name="geom_{o["name"]}" type="{shape}" size="{SHAPE_SIZE[shape](r)}" '
            f'rgba="{_rgba(o.get("rgb", [0.8, 0.8, 0.8]))}" contype="0" conaffinity="0"/></body>')
    ex, ey, ez = EYE_POS
    return f"""
<mujoco model="taro_world_v3">
  <visual><global offwidth="{IMG}" offheight="{IMG}"/><quality shadowsize="0"/>
          <headlight ambient=".35 .35 .35" diffuse=".45 .45 .45" specular="0 0 0"/></visual>
  <asset>
    <texture name="wall" type="2d" builtin="checker" rgb1="{wall1}" rgb2="{wall2}" width="64" height="64"/>
    <material name="wall" texture="wall" texrepeat="6 4"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="{floor1}" rgb2="{floor2}" width="64" height="64"/>
    <material name="floor" texture="floor" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0.3 0 2.2" dir="0 0 -1" diffuse=".6 .6 .6" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 0" size="3 3 .1" material="floor" contype="0" conaffinity="0"/>
    <geom name="wall_front" type="box" pos="1.4 0 1" size=".02 3 1" material="wall" contype="0" conaffinity="0"/>
    <geom name="wall_left" type="box" pos="0 1.4 1" size="3 .02 1" material="wall" contype="0" conaffinity="0"/>
    <geom name="wall_right" type="box" pos="0 -1.4 1" size="3 .02 1" material="wall" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="{' '.join(map(str, table['pos']))}" size="{' '.join(map(str, table['size']))}"
          rgba="{_rgba(table['rgb'])}" contype="0" conaffinity="0"/>
    {''.join(objs)}
    <body name="eye" mocap="true" pos="{ex} {ey} {ez}"><camera name="eye" fovy="{FOVY_DEG}" xyaxes="0 -1 0 0 0 1"/></body>
  </worldbody>
</mujoco>"""


@dataclass
class World:
    """spec（辞書）で場面を決める。reset(seed) で試行の頭に戻し、step(cmd) で0.1秒進めて絵と出来事を返す。"""
    spec: dict
    record: bool = False
    frames: list = field(default_factory=list)

    def __post_init__(self):
        self.model = mujoco.MjModel.from_xml_string(make_xml(self.spec))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMG, IMG)
        self.names = [o["name"] for o in self.spec.get("objects", [])]
        self._mocap = {n: int(self.model.body(f"obj_{n}").mocapid[0]) for n in self.names}
        self._geom = {n: int(self.model.geom(f"geom_{n}").id) for n in self.names}
        self._eye_mocap = int(self.model.body("eye").mocapid[0])
        self._rgba0 = {n: self.model.geom_rgba[g].copy() for n, g in self._geom.items()}
        self._pos0 = {n: self.data.mocap_pos[i].copy() for n, i in self._mocap.items()}

    # ---- 状態 ----
    def reset(self, seed: int = 0, yaw: float | None = None, pitch: float | None = None):
        """試行の頭。物を最初の位置・色に戻し、目の向きを決める（既定は種から少しずらす。MIMo 版の start_jitter と同じ考え）。"""
        rng = np.random.default_rng(10_000 + int(seed))
        for n in self.names:
            self.data.mocap_pos[self._mocap[n]] = self._pos0[n]
            self.model.geom_rgba[self._geom[n]] = self._rgba0[n]
        self.tick = 0
        self.yaw = float(rng.uniform(-4.5, 4.5) if yaw is None else yaw)
        self.pitch = float(rng.uniform(-4.5, 4.5) if pitch is None else pitch)
        self._ev = sorted(self.spec.get("events", []), key=lambda e: float(e["t"]))
        self._ev_i = 0
        self.frames = []
        self._set_eye()
        return self.render()

    @property
    def t(self) -> float:
        return self.tick * TICK_S

    def _set_eye(self):
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.radians([0.0, -self.pitch, self.yaw]), "xyz")
        self.data.mocap_quat[self._eye_mocap] = q
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="eye")
        img = self.renderer.render().copy()
        if self.record:
            self.frames.append(img)
        return img

    # ---- 1回 ----
    def step(self, cmd=None):
        """0.1秒進める。cmd＝目を跳ばす向き (h, v)。戻り値 (絵, 出来事の辞書)。
        出来事：heard（親の声・物の音を、耳へ渡す文字にまとめたもの）・events（この回に起きた予定表の項目）"""
        if cmd is not None:
            h, v = float(cmd[0]), float(cmd[1])
            self.yaw = float(np.clip(self.yaw - h * HALF_VIEW_DEG, -EYE_RANGE_DEG, EYE_RANGE_DEG))   # 右が正＝yaw は負へ
            self.pitch = float(np.clip(self.pitch + v * HALF_VIEW_DEG, -EYE_RANGE_DEG, EYE_RANGE_DEG))
            self._set_eye()
        self.tick += 1
        heard, done = [], []
        while self._ev_i < len(self._ev) and float(self._ev[self._ev_i]["t"]) < self.t + 1e-9:
            e = self._ev[self._ev_i]
            self._ev_i += 1
            done.append(e)
            k, n = e["do"], e.get("obj")
            if k == "hide":
                self.model.geom_rgba[self._geom[n]][3] = 0.0
            elif k == "show":
                self.model.geom_rgba[self._geom[n]][3] = 1.0
            elif k == "move":
                self.data.mocap_pos[self._mocap[n]] = e["pos"]
            elif k == "color":
                self.model.geom_rgba[self._geom[n]][:3] = e["rgb"]
            elif k in ("say", "sound"):
                heard.append(str(e["text"]))
        mujoco.mj_forward(self.model, self.data)
        return self.render(), {"t": self.t, "heard": " ".join(heard), "events": done,
                               "yaw": self.yaw, "pitch": self.pitch}

    # ---- 確かめ用（脳には渡さない） ----
    def gaze_angle_to(self, name: str) -> float | None:
        """視線と物の中心のなす角（度）。見えなくしてある物は None。"""
        if self.model.geom_rgba[self._geom[name]][3] <= 0.0:
            return None
        cam = self.model.camera("eye").id
        p = self.data.cam_xpos[cam]
        fwd = -self.data.cam_xmat[cam].reshape(3, 3)[:, 2]
        v = self.data.geom_xpos[self._geom[name]] - p
        c = float(np.dot(fwd, v) / (np.linalg.norm(v) + 1e-12))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def angles(self) -> dict:
        return {n: self.gaze_angle_to(n) for n in self.names}

    def save_video(self, path: str, fps: int = 10) -> str:
        """record=True で残した絵を動画（mp4、無理なら gif）にする。"""
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            imageio.mimsave(path, self.frames, fps=fps)
        except Exception:  # noqa: BLE001
            path = os.path.splitext(path)[0] + ".gif"
            imageio.mimsave(path, self.frames, duration=1.0 / fps)
        return path


def stage2_spec(ball_left: bool = True) -> dict:
    """段階2b と同じ考えの場面：ボール・コップ・光って動く妨害物（ここでは点滅と移動は予定表で書く）。"""
    s = 1 if ball_left else -1
    return {"objects": [
        {"name": "ball", "shape": "sphere", "radius": 0.03, "rgb": [0.58, 0.46, 0.50], "pos": [0.30, 0.05 * s, 0.30]},
        {"name": "cup", "shape": "cylinder", "radius": 0.03, "rgb": [0.95, 0.95, 0.95], "pos": [0.30, -0.05 * s, 0.30]},
        {"name": "distractor", "shape": "box", "radius": 0.025, "rgb": [0.2, 0.4, 0.9], "pos": [0.30, 0.0, 0.29]},
    ], "events": []}
