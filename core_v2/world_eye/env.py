"""目だけの場面の環境本体。

run_v2（core_v2/run_v2.py）と道具が使う形をそのまま持つ：
* env.reset(seed=) → (obs, info)、env.step(action) → (obs, 0.0, False, False, info)
* obs["eye_left"]：左目の画像（px×px×3、uint8）。読んだときに描く（読まない歩は描かない）
* info["parent_utterance"]：親が話した歩だけ {"text", "target", ...}
* env.unwrapped._orienting：目の動き方（eye.EyeKinematics。set_map_target(h, v)・_map_target）
* env.unwrapped._gaze_angle_to(物の名前)・_gaze_dir()・model・data（カメラ "eye_left"、物の体の名前は MIMo と同じ
  test_object1〜3、段階6の sy_obj0〜5）・pairs_showing()
* env.action_space.shape（体が無いので長さ0）

物ははっきりした呼び出しでだけ変わる（勝手に置き直さない）：
* show(名前, 場所)・hide(名前)・move(名前, 場所)：1回きりの変化。そのたびに、変える前と後の左目の画像を記録の
  フォルダへ1枚ずつ残し、events.csv に1行書く
* parent_hold(名前, 場所)：親が手に持って動かす（毎歩、振る・差し出す）。持つ物が替わった歩だけ記録する
* 段階2の妨害物の揺れと点滅（場面の flash_move）：場面に書いてある決まった動き。毎歩は記録しない
* 隠すとは、物の形を「描かない組」へ移すこと（影も落とさない）。場所はそのまま。隠した物は見られない
  （_gaze_angle_to は None を返す）
* 段階9：場面の world.eye_world.stage9 = {"room": 部屋, "objects": made|real, "words": [...], "insts": [...]} で、
  部屋（rooms.py）と20語の個体（instances.py。体の名前は n9_<種類>_<個体>、本物は n9r_…）を作る（最初は全部隠してある）。
  並べ方・親は noun_scene.py。object_sound(名前, 波形) で物の音を鳴らす（次の歩の info["object_sound"]、events.csv に1行）
"""

from __future__ import annotations

import csv
import json
import os

import mujoco
import numpy as np

from . import scene as scene_mod
from .eye import EyeKinematics
from .objects import FLASH_RGBA, PRESENT_CATALOGUE, catalogue_rgba, unit

DT = 0.01


class _ActionSpace:
    shape = (0,)


class LazyObs(dict):
    """obs["eye_left"] を読んだときに描く。前の歩の obs を後から読むと、その歩の画像ではなくなるので止める。"""

    def __init__(self, env, n_step: int) -> None:
        super().__init__()
        self._env = env
        self._n = n_step

    def __missing__(self, key):
        if key != "eye_left":
            raise KeyError(key)
        if self._env.n_step != self._n:
            raise RuntimeError("古い歩の観測を読もうとした（目だけの場面は読んだときに描くので、その歩の画像はもう無い）")
        img = self._env.render_eye()
        self[key] = img
        return img


def _shape_kind(shape: str) -> str:
    s = str(shape or "sphere")
    if s in ("sphere", "cylinder", "box", "ellipsoid"):
        return s
    raise ValueError(f"目だけの場面では作れない形: {shape}（plate:… や asis は objects.py の種類で書く）")


class EyeWorldEnv:
    """sc：場面の辞書（run/scenes/*.json を scene_io.load で読んで world を重ねたもの）。"""

    metadata: dict = {}

    def __init__(self, sc: dict, seed: int = 0, log_dir: str | None = None,
                 save_images: bool = True, max_event_images: int = 4000) -> None:
        from run.world.parent import WordSchedule
        from .parent import EyeParentLabeling

        self.scene_name = sc.get("name", "")
        w = dict(sc.get("world") or {})
        body = dict(sc.get("body") or {})
        ew = dict(w.get("eye_world") or {})
        self.dt = DT
        self.px = int(body.get("vision_px") or 224)
        self.fovy = float(ew.get("fovy", 60.0))
        self.action_space = _ActionSpace()
        self.unwrapped = self
        self.taro = None

        # ---- 物の一覧 ----
        self._cfg = {k: dict(w.get(k) or {}) for k in ("toy", "toy2", "toy3", "toy4")}
        self._toy = bool(self._cfg["toy"].get("enabled"))
        self._toy2 = bool(self._cfg["toy2"].get("enabled"))
        self._toy3 = bool(self._cfg["toy3"].get("enabled"))
        self._toy4 = bool(self._cfg["toy4"].get("enabled"))
        objs = []
        self.toy_body = {"toy1": "test_object1", "toy2": "test_object2", "toy3": "test_object3",
                         "toy4": "test_object4"}
        for key, cfgk in (("toy1", "toy"), ("toy2", "toy2"), ("toy3", "toy3"), ("toy4", "toy4")):
            c = self._cfg[cfgk]
            if not c.get("enabled") and key != "toy1":
                continue
            rgba = c.get("rgba") or [0.9, 0.2, 0.15, 1.0]
            if key in ("toy1", "toy2") and ew.get("mimo_colors", True):
                rgba = list(scene_mod.MIMO_SEEN_RGBA.get(tuple(round(float(x), 4) for x in rgba), rgba))
            objs.append({"name": self.toy_body[key], "kind": _shape_kind(c.get("shape", "sphere")),
                         "radius": float(c.get("radius", 0.02)), "rgba": rgba})
        self._toy_shape = self._cfg["toy"].get("shape", "sphere")
        self._toy2_shape = self._cfg["toy2"].get("shape", "")
        self._toy3_rgba = list(self._cfg["toy3"].get("rgba") or [0.2, 0.4, 0.9, 1.0])
        fm = self._cfg["toy3"].get("flash_move")
        self._toy3_flash_move = ({"amp_m": float(fm.get("amp_m", 0.03)), "hz": float(fm.get("hz", 0.5)),
                                  "blink_hz": float(fm.get("blink_hz", 0.0))} if fm else None)
        pairs = dict(w.get("pairs") or {})
        self._pair_bodies = []
        for i, o in enumerate(pairs.get("objects") or []):
            nm = (pairs.get("bodies") or [f"sy_obj{k}" for k in range(len(pairs["objects"]))])[i]
            objs.append({"name": nm, "kind": _shape_kind(o.get("shape", "sphere")),
                         "radius": float(o.get("radius", 0.02)), "rgba": o.get("rgba", [0.8, 0.8, 0.8, 1.0])})
            self._pair_bodies.append(nm)
        self._pair_schedule = sorted(((float(e["t0"]), float(e["t1"]), e.get("left"), e.get("right"))
                                      for e in (pairs.get("schedule") or [])), key=lambda x: x[0])
        # 一覧の物（くつ・おさら・…）は全部作っておく（隠したまま）。親が名づけに使う的にするのは、場面の
        # world.eye_world.present に書いたキー（例 ["toy5", "toy12"]）だけ（MIMo の present_slots と同じく既定は無し）
        self.present_slots = {}
        use = set(ew.get("present") or [])
        for key, (nm, kind, r) in PRESENT_CATALOGUE.items():
            objs.append({"name": nm, "kind": kind, "radius": r, "rgba": catalogue_rgba(nm, kind)})
            if key in use:
                self.present_slots[key] = {"body": nm, "shape": "asis"}
        for o in ew.get("objects") or []:     # 場面ごとに足す物 {"name", "kind", "radius", "rgba"}
            objs.append({"name": o["name"], "kind": o["kind"], "radius": float(o.get("radius", 0.03)),
                         "rgba": o.get("rgba") or catalogue_rgba(o["name"], o["kind"])})
        self.object_names = [o["name"] for o in objs]
        # 段階9：部屋と20語の個体（world.eye_world.stage9 = {"room", "objects": made|real, "words", "insts"}）。
        # 個体の体の名前は instances の name（n9_cup_0／本物は n9r_cup_0）。最初は全部隠してある
        self.n9 = {}                 # 体の名前 → {"word", "i", "k", "size"}
        self.n9_view = {}            # 体の名前 → 今の向きの番号（見た目の素の表の向き。置く道具が書く）
        self._pending_sound = []
        s9 = dict(ew.get("stage9") or {})
        extra_assets, extra_bodies, room = [], [], None
        if s9:
            from . import instances as ins
            ins.use_objects(s9.get("objects", "made"))
            room = s9.get("room")
            for word in (s9.get("words") or ins.WORDS):
                for k in (s9.get("insts") or range(ins.N_INST)):
                    inst = ins.instance(word, int(k))
                    extra_assets += ins.assets_xml(inst)
                    extra_bodies.append(ins.body_xml(inst))
                    size = float(inst["size_m"]) if inst.get("src") == "real" else 4.0 * float(inst["r"])
                    self.n9[inst["name"]] = {"word": word, "i": ins.WORDS.index(word), "k": int(k), "size": size}
                    self.object_names.append(inst["name"])

        # ---- MuJoCo ----
        xml = scene_mod.build_xml({"px": self.px, "fovy": self.fovy, "floor_dim": w.get("floor_dim", 1.0),
                                   "backdrop": w.get("backdrop") or {"enabled": True},
                                   "table": bool(ew.get("table", False)),
                                   "parent_figure": bool(ew.get("parent_figure", False)),
                                   "objects": objs, "room": room,
                                   "extra_assets": extra_assets, "extra_bodies": extra_bodies})
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._cam_id = int(self.model.camera("eye_left").id)
        self._pan_q = int(self.model.jnt_qposadr[self.model.joint("eye_pan").id])
        self._tilt_q = int(self.model.jnt_qposadr[self.model.joint("eye_tilt").id])
        self._obj = {}
        for nm in self.object_names:
            b = self.model.body(nm)
            j = int(b.jntadr[0])
            adr, num = int(b.geomadr[0]), int(b.geomnum[0])
            self._obj[nm] = {"bid": int(b.id), "qadr": int(self.model.jnt_qposadr[j]),
                             "dadr": int(self.model.jnt_dofadr[j]), "geoms": list(range(adr, adr + num)),
                             "group0": [int(self.model.geom_group[g]) for g in range(adr, adr + num)],
                             "rgba0": [self.model.geom_rgba[g].copy() for g in range(adr, adr + num)]}
        self._renderer = None
        self._vopt = mujoco.MjvOption()
        self._vopt.geomgroup[scene_mod.HIDDEN_GROUP] = 0
        self._orienting = EyeKinematics()
        self._toy_swap_sign = 1.0
        self._present_slots = {}          # ParentLabeling の10択の退避（MIMo 用）は使わない
        self._taro_noticed_gone_time = None
        self._state_toy3 = None
        st = sc.get("state") or {}
        if self._cfg["toy3"].get("rest_from_state") and isinstance(st, dict) and len(st.get("qpos") or []) >= 17:
            self._state_toy3 = np.array(st["qpos"][14:17], dtype=float)   # MIMo の場面の保存（test_object3 の場所）

        # ---- 親 ----
        pl = dict(w.get("parent_labeling") or {})
        self._parent_labeling = EyeParentLabeling(**pl)
        self._word_schedule = WordSchedule(**dict(w.get("word_test") or {}))
        self._script = sorted([dict(e) for e in (ew.get("script") or [])], key=lambda e: float(e["t"]))

        # ---- 記録 ----
        self.log_root = log_dir
        self.save_images = bool(save_images)
        self.max_event_images = int(max_event_images)
        self.n_event_images = 0
        self._img_cap_warned = False
        self.n_episode = 0
        self._ep_dir = None
        self._ev_fp = None
        self._ev_w = None
        self.n_step = 0
        self.np_random = np.random.default_rng(seed)
        root, self.log_root = self.log_root, None     # 作ったときの reset は記録しない（試行は道具の reset から）
        self.reset(seed=seed)
        self.log_root = root
        self.n_episode = 0

    # ================================================================ 位置
    def _set_anchor(self) -> None:
        """toy1・toy2・toy3・toy4 の置き場所。toy_env._set_anchor と同じ式（両目の真ん中から、最初の視線を基準に
        左右 angle 度・上 elev 度・距離 dist）。toy3 は場面が rest_from_state なら MIMo の保存した場所。"""
        o = scene_mod.EYE_MID
        g, right, up = scene_mod.FORWARD, scene_mod.RIGHT, scene_mod.UP
        c1, c2, c3, c4 = (self._cfg[k] for k in ("toy", "toy2", "toy3", "toy4"))
        if self._toy2:
            ang = np.radians(float(c2.get("angle_deg", 12.0)))
            sign = float(self._toy_swap_sign)
            g1 = g * np.cos(ang) - sign * right * np.sin(ang)
            g2 = g * np.cos(ang) + sign * right * np.sin(ang)
            e2 = np.radians(float(c2.get("elev_deg", 0.0)))
            self._rest_pos2 = o + (g2 * np.cos(e2) + up * np.sin(e2)) * float(c2.get("dist", 0.3))
        else:
            g1 = g
            self._rest_pos2 = None
        e1 = np.radians(float(c1.get("elev_deg", 0.0)))
        self._rest_pos = o + (g1 * np.cos(e1) + up * np.sin(e1)) * float(c1.get("dist", 0.3))
        self._rest_pos3 = None
        if self._toy3:
            if self._state_toy3 is not None:
                self._rest_pos3 = self._state_toy3.copy()
            else:
                a3 = np.radians(float(c3.get("angle_deg", 0.0)))
                g3 = g * np.cos(a3) + up * np.sin(a3)
                e3 = np.radians(float(c3.get("elev_deg", 0.0)))
                self._rest_pos3 = o + (g3 * np.cos(e3) + up * np.sin(e3)) * float(c3.get("dist", 0.3))
        self._rest_pos4 = None
        if self._toy4:
            a4 = np.radians(float(c4.get("angle_deg", 0.0)))
            g4 = g * np.cos(a4) - up * np.sin(a4)
            e4 = np.radians(float(c4.get("elev_deg", 0.0)))
            self._rest_pos4 = o + (g4 * np.cos(e4) + up * np.sin(e4)) * float(c4.get("dist", 0.086))
        for p in (self._rest_pos, self._rest_pos2, self._rest_pos3, self._rest_pos4):
            if p is not None:
                p[2] = max(float(p[2]), 0.04)
        # 段階2・6の場面の物が、場面の置き方のとおりに並んでいるなら、ここで動かす（入れ替えのとき）
        if getattr(self, "_placed_once", False) and not self._pairs_active():
            for key, pos in (("toy1", self._rest_pos), ("toy2", self._rest_pos2)):
                nm = self.toy_body[key]
                if pos is not None and nm in self._obj and self.visible(nm):
                    self.move(nm, pos, cause="parent_swap")

    def _pairs_active(self) -> bool:
        return bool(self._pair_schedule)

    # ================================================================ 物を変える（はっきりした呼び出し）
    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        q = self._obj[name]["qadr"]
        return self.data.qpos[q:q + 3].copy(), self.data.qpos[q + 3:q + 7].copy()

    def visible(self, name: str) -> bool:
        o = self._obj[name]
        return int(self.model.geom_group[o["geoms"][0]]) != scene_mod.HIDDEN_GROUP

    def _write_pose(self, name, pos=None, quat=None) -> None:
        q = self._obj[name]["qadr"]
        if pos is not None:
            self.data.qpos[q:q + 3] = np.asarray(pos, dtype=float)
        if quat is not None:
            self.data.qpos[q + 3:q + 7] = np.asarray(quat, dtype=float)
        d = self._obj[name]["dadr"]
        self.data.qvel[d:d + 6] = 0.0

    def _set_visible(self, name, on: bool) -> None:
        o = self._obj[name]
        for g, g0 in zip(o["geoms"], o["group0"]):
            self.model.geom_group[g] = g0 if on else scene_mod.HIDDEN_GROUP

    def show(self, name: str, pos=None, quat=None, cause: str = "") -> None:
        if self.visible(name) and pos is None and quat is None:
            return
        self._event("show", name, cause, lambda: (self._write_pose(name, pos, quat), self._set_visible(name, True)))

    def hide(self, name: str, cause: str = "") -> None:
        if not self.visible(name):
            return
        self._event("hide", name, cause, lambda: self._set_visible(name, False))

    def move(self, name: str, pos, quat=None, cause: str = "") -> None:
        self._event("move", name, cause, lambda: self._write_pose(name, pos, quat))

    def parent_hold(self, name: str, pos, quat=None, cause: str = "parent") -> None:
        """親が手に持って動かす（毎歩呼ばれる）。持つ物が替わった歩・見えていなかった物を出した歩だけ記録する。"""
        if self._held != name or not self.visible(name):
            self._held = name
            self._event("parent_hold", name, cause,
                        lambda: (self._write_pose(name, pos, quat), self._set_visible(name, True)))
        else:
            self._write_pose(name, pos, quat)

    def set_eye(self, pan_deg: float, tilt_deg: float, cause: str = "tool") -> None:
        """道具用：目の向きを直接置く（太郎の跳びではない）。"""
        self._event("set_eye", "", f"{cause}:{pan_deg:.2f},{tilt_deg:.2f}",
                    lambda: (self._orienting.set_angles(pan_deg, tilt_deg), self._apply_eye()))

    def say(self, text: str, target=None, cause: str = "tool") -> None:
        """道具・台本から親に言わせる（次の歩の info に入る）。"""
        self._pending_say.append({"text": str(text), "target": target, "cause": cause})

    def object_sound(self, name: str, wave, cause: str = "parent") -> None:
        """段階9：物の音（波形、16kHz）を鳴らす。次の歩の info["object_sound"] に入る（run_v2.run_trial が読む）。
        絵は変えないので、鳴らした歩の画像を1枚だけ残す（話すのと同じ）。"""
        self._pending_sound.append(np.asarray(wave, dtype=np.float32))
        at = self._save_img(self.render_eye(), f"sound_{name}") if self._ep_dir and self.save_images else ""
        self._log_row("sound", name, cause, text=f"{len(self._pending_sound[-1]) / 16000.0:.2f}s", before=at, after=at)

    def taro_said(self, text: str) -> None:
        """太郎の声を親の耳へ（ParentLabeling の respond_to_voice が読む。MIMo では trainer が立てた合図）。"""
        if text:
            self._taro_voice_signal = True
            self._taro_voice_text = str(text)

    # ================================================================ gym の形
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(int(seed))
        self._close_episode()
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0
        self.n_step += 1
        self._orienting.reset()
        self._held = None
        self._pending_say = []
        self._pending_sound = []
        self._taro_voice_signal = False
        self._taro_voice_text = ""
        self._taro_noticed_gone_time = None
        self._pairs_now = (None, None)
        self._script_i = 0
        self._toy_swap_sign = 1.0
        self._placed_once = False
        self._set_anchor()
        for nm, o in self._obj.items():          # 色を場面の色に戻し、全部隠して遠くへ（見えない所から始める）
            for g, c in zip(o["geoms"], o["rgba0"]):
                self.model.geom_rgba[g] = c
            self._set_visible(nm, False)
            self._write_pose(nm, np.array([0.0, 0.0, -5.0]), np.array([1.0, 0, 0, 0]))
        if not self._pairs_active():
            for key, pos in (("toy1", self._rest_pos if self._toy else None), ("toy2", self._rest_pos2),
                             ("toy3", self._rest_pos3), ("toy4", self._rest_pos4)):
                if pos is not None:
                    self._write_pose(self.toy_body[key], pos)
                    self._set_visible(self.toy_body[key], True)
        self._placed_once = True
        self._apply_eye()
        self._parent_labeling.reset()
        self._word_schedule.reset()
        self.n_episode += 1
        self._open_episode(seed)
        self._update_pairs()
        self._flash_move_toy3()
        mujoco.mj_forward(self.model, self.data)
        self._event_image("reset", "", "", None)
        return LazyObs(self, self.n_step), {}

    def step(self, action=None):
        self.data.time += self.dt
        self.n_step += 1
        self._update_pairs()
        self._run_script()
        self._flash_move_toy3()
        utt = self._parent_labeling.update(self)
        if utt is None:
            utt = self._word_schedule.update(self)
        if utt is None and self._pending_say:
            utt = self._pending_say.pop(0)
        self._orienting.step(self.dt)
        self._apply_eye()
        mujoco.mj_forward(self.model, self.data)
        info = {}
        if self._pending_sound:
            info["object_sound"] = self._pending_sound.pop(0)
        if utt is not None:
            info["parent_utterance"] = utt
            # 話すことは絵を変えないので、話した歩の画像を1枚だけ残す（前と後の欄に同じ名前）
            at = self._save_img(self.render_eye(), "say") if self._ep_dir and self.save_images else ""
            self._log_row("say", str(utt.get("target") or ""), str(utt.get("cause") or "parent"),
                          text=str(utt.get("text", "")), before=at, after=at)
        return LazyObs(self, self.n_step), 0.0, False, False, info

    def close(self) -> None:
        self._close_episode()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ================================================================ 目
    def _apply_eye(self) -> None:
        # pan（右が正）→ 上向きの軸の回転は左が正なので符号を反転。tilt（上が正）→ 右向きの軸の回転で上が正
        self.data.qpos[self._pan_q] = -np.radians(self._orienting.pan)
        self.data.qpos[self._tilt_q] = np.radians(self._orienting.tilt)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def render_eye(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.px, width=self.px)
        self._renderer.update_scene(self.data, camera=self._cam_id, scene_option=self._vopt)
        return self._renderer.render().copy()

    def _gaze_dir(self, cam: str = "eye_left") -> np.ndarray:
        return -np.array(self.data.cam_xmat[self._cam_id], dtype=float).reshape(3, 3)[:, 2]

    def gaze_angle_raw(self, name: str):
        """見えるかどうかに関係なく、視線と物の中心のなす角（度）。"""
        eye = np.array(self.data.cam_xpos[self._cam_id], dtype=float)
        try:
            pos = np.array(self.data.body(name).xpos, dtype=float)
        except KeyError:
            return None
        v = pos - eye
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return float(np.degrees(np.arccos(np.clip(np.dot(self._gaze_dir(), v / n), -1.0, 1.0))))

    def _gaze_angle_to(self, body_name: str):
        """視線と物の中心のなす角（度）。隠した物・無い物は None（見えない物は見られない）。"""
        if body_name not in self._obj or not self.visible(body_name):
            return None
        return self.gaze_angle_raw(body_name)

    # ================================================================ 場面の決まった動き
    def _flash_move_toy3(self) -> None:
        """段階2の妨害物：左右に amp_m×sin(2π·hz·t) 揺れ、blink_hz で色が点滅する（toy_env._flash_move_toy3 と同じ式）。
        隠しているあいだは動かさない。"""
        fm = self._toy3_flash_move
        nm = self.toy_body["toy3"]
        if fm is None or self._rest_pos3 is None or nm not in self._obj or not self.visible(nm):
            return
        t = float(self.data.time)
        look = self._rest_pos3 - np.array(self.data.cam_xpos[self._cam_id], dtype=float)
        axis = unit(np.cross(look, np.array([0.0, 0.0, 1.0])))
        self._write_pose(nm, self._rest_pos3 + fm["amp_m"] * np.sin(2.0 * np.pi * fm["hz"] * t) * axis)
        if fm["blink_hz"] > 0.0:
            on = (int(np.floor(2.0 * fm["blink_hz"] * t)) % 2) == 0
            for g in self._obj[nm]["geoms"]:
                self.model.geom_rgba[g] = FLASH_RGBA if on else self._toy3_rgba

    def _update_pairs(self) -> None:
        """段階6：予定表どおりに物を2つずつ左右に出す（toy_env._hold_pairs と同じ時計・置き場所）。
        出す物が替わった歩だけ、見せる・隠すを記録する。"""
        if not self._pair_schedule:
            return
        t = float(self.data.time)
        left = right = None
        for t0, t1, lk, rk in self._pair_schedule:
            if t0 <= t < t1:
                left, right = lk, rk
                break
            if t < t0:
                break
        if (left, right) == self._pairs_now:
            return
        self._pairs_now = (left, right)
        want = {}
        if left is not None and self._rest_pos is not None:
            want[self._pair_bodies[int(left)]] = self._rest_pos
        if right is not None and self._rest_pos2 is not None:
            want[self._pair_bodies[int(right)]] = self._rest_pos2
        for nm in self._pair_bodies:
            if nm not in want:
                self.hide(nm, cause="pairs")
        for nm, pos in want.items():
            self.show(nm, pos, np.array([1.0, 0, 0, 0]), cause="pairs")

    def pairs_showing(self):
        return self._pairs_now

    def _run_script(self) -> None:
        """場面の world.eye_world.script：[{"t": 秒, "do": "show"|"hide"|"move"|"say", "name"|"text", "pos"}]"""
        t = float(self.data.time)
        while self._script_i < len(self._script) and t >= float(self._script[self._script_i]["t"]) - 1e-9:
            e = self._script[self._script_i]
            self._script_i += 1
            do = e["do"]
            if do == "show":
                self.show(e["name"], e.get("pos"), cause="script")
            elif do == "hide":
                self.hide(e["name"], cause="script")
            elif do == "move":
                self.move(e["name"], e["pos"], cause="script")
            elif do == "say":
                self.say(e["text"], e.get("target"), cause="script")
            else:
                raise ValueError(f"台本の知らない動き: {do}")

    # ================================================================ 記録（前後の画像）
    def _open_episode(self, seed) -> None:
        self._ep_dir = None
        if not self.log_root:
            return
        self._ep_dir = os.path.join(self.log_root, f"ep{self.n_episode:03d}_seed{seed}")
        os.makedirs(self._ep_dir, exist_ok=True)
        self._ev_fp = open(os.path.join(self._ep_dir, "events.csv"), "w", newline="", encoding="utf-8")
        self._ev_w = csv.writer(self._ev_fp)
        self._ev_w.writerow(["t", "step", "event", "object", "cause", "text", "pos", "visible",
                             "eye_pan", "eye_tilt", "before", "after"])
        self._n_ev = 0

    def _close_episode(self) -> None:
        if self._ev_fp is not None:
            self._ev_fp.close()
            self._ev_fp = None
            self._ev_w = None
            with open(os.path.join(self._ep_dir, "eye.json"), "w", encoding="utf-8") as f:
                json.dump({"n_saccades": self._orienting.n_saccades,
                           "blocked_target_n": self._orienting.blocked_target_n,
                           "fire_log": self._orienting.fire_log[:2000]}, f, ensure_ascii=False)

    def _save_img(self, img, tag: str):
        if not (self._ep_dir and self.save_images):
            return ""
        if self.n_event_images >= self.max_event_images:
            if not self._img_cap_warned:
                print(f"[目だけの場面] 前後の画像が上限 {self.max_event_images} 枚に達した。ここから先は events.csv だけ書く",
                      flush=True)
                self._img_cap_warned = True
            return ""
        from PIL import Image
        name = f"{self._n_ev:04d}_{tag}.png"
        Image.fromarray(img).save(os.path.join(self._ep_dir, name))
        self.n_event_images += 1
        return name

    def _event(self, event: str, name: str, cause: str, apply) -> None:
        """変える前の画像 → 変える → 変えた後の画像、を残す。"""
        before = ""
        if self._ep_dir and self.save_images:
            mujoco.mj_forward(self.model, self.data)
            before = self._save_img(self.render_eye(), f"{event}_{name}_before")
        apply()
        mujoco.mj_forward(self.model, self.data)
        after = ""
        if self._ep_dir and self.save_images:
            after = self._save_img(self.render_eye(), f"{event}_{name}_after")
        self._log_row(event, name, cause, before=before, after=after)

    def _event_image(self, event: str, name: str, cause: str, _unused) -> None:
        after = self._save_img(self.render_eye(), f"{event}_after") if self._ep_dir and self.save_images else ""
        self._log_row(event, name, cause, after=after)

    def _log_row(self, event, name, cause, text="", before="", after="") -> None:
        if self._ev_w is None:
            return
        pos = ""
        vis = ""
        if name in self._obj:
            pos = ",".join(f"{x:.4f}" for x in self.pose(name)[0])
            vis = int(self.visible(name))
        self._ev_w.writerow([f"{float(self.data.time):.2f}", self.n_step, event, name, cause, text, pos, vis,
                             f"{self._orienting.pan:.2f}", f"{self._orienting.tilt:.2f}", before, after])
        self._ev_fp.flush()
        self._n_ev += 1
