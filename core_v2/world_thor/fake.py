"""AI2-THOR の代わりの小さな偽物（テストとクラウドの確かめ用）。本物の Controller と同じ呼び方の所だけを持つ：
reset(scene=)・step(action=, ...)・stop()・last_event。返す Event は metadata（agent・objects・lastActionSuccess・
errorMessage）・third_party_camera_frames・third_party_instance_segmentation_frames・color_to_object_id を持つ。

絵の作り方：物は球（直径＝物の大きさ）。カメラから画素ごとに光線を飛ばし、いちばん近い球の色を塗る（重なりも正しい）。
後ろは、床（y=0、25cm の市松）と、方角で縞の壁。どちらも世界に貼りついている（目を回すと絵の中身がずれる。
目の入口の画像のずれ・指令の写しを確かめられる）。向きと写し方は env.py の cam_basis・project と同じ式。
エージェント：agent_pos・agent_yaw（既定 (0, 0.9, 0)・+z 向き）。GetReachablePositions は reachable（既定はその場所1つ）を
actionReturn で返す。Teleport・TeleportFull（position・rotation・horizon・standing）でエージェントを動かす（reachable に
無い場所は失敗。本物もそう）。物は (id, 種類, 真ん中, 大きさ, 色) に、6つめとして追加の metadata（openable・isOpen・
parentReceptacles・pickupable など）を渡せる。
物の位置（position）は物の底（本物の AI2-THOR も多くの物は底か中ほど）、見かけの真ん中は axisAlignedBoundingBox.center。
"""

from __future__ import annotations

import copy

import numpy as np

from .env import cam_basis, focal_px, vec

# (id, 種類, 見かけの真ん中 x,y,z, 大きさ m, 色)。エージェントは (0, 0.9, 0) で +z を向く → 目は (0, 0.6, 0.3)
DEFAULT_OBJECTS = [
    ("Apple|+00.00|+00.50|+01.00", "Apple", (0.0, 0.48, 1.0), 0.09, (200, 30, 30)),
    ("Cup|+00.35|+00.45|+01.10", "Cup", (0.35, 0.45, 1.1), 0.10, (240, 240, 250)),
    ("Book|-00.40|+00.30|+01.20", "Book", (-0.40, 0.30, 1.2), 0.22, (40, 90, 200)),
    ("Plate|+00.15|+00.20|+00.90", "Plate", (0.15, 0.20, 0.9), 0.20, (230, 220, 160)),
    ("Chair|-00.10|+00.40|+02.00", "Chair", (-0.10, 0.40, 2.0), 0.70, (110, 70, 40)),
]
LARGE_TYPES = ("Chair", "Fridge", "Sofa", "CounterTop", "DiningTable", "Cabinet")
FLOOR_ID = "Floor|+00.00|+00.00|+00.00"
WALL_ID = "Wall|+00.00|+00.00|+00.00"


class FakeEvent:
    def __init__(self, metadata, cams, color_to_object_id) -> None:
        self.metadata = metadata
        self.third_party_camera_frames = [c[0] for c in cams]
        self.third_party_instance_segmentation_frames = [c[1] for c in cams]
        self.color_to_object_id = color_to_object_id
        self.frame = cams[0][0] if cams else None


class FakeController:
    def __init__(self, scene: str = "FloorPlan1", width: int = 224, height: int = 224, objects=None,
                 agent_yaw: float = 0.0, agent_pos=(0.0, 0.9, 0.0), reachable=None, **kw) -> None:
        self.width, self.height = int(width), int(height)
        self.kw = dict(kw)
        self._objects_src = objects or DEFAULT_OBJECTS
        self.agent_yaw0 = float(agent_yaw)
        self.agent_pos0 = tuple(float(x) for x in agent_pos)
        self.reachable = [tuple(float(x) for x in p) for p in (reachable or [self.agent_pos0])]
        self.calls: list[tuple[str, dict]] = []
        self.stopped = False
        self.last_event = None
        self.reset(scene=scene)

    # ---------------------------------------------------------------- 本物と同じ呼び方
    def reset(self, scene: str = "FloorPlan1", **kw):
        self.scene = scene
        self.agent_yaw = self.agent_yaw0
        self.agent_pos = self.agent_pos0
        self.cams: list[dict] = []
        self.objects = []
        self.colors = {FLOOR_ID: (10, 10, 10), WALL_ID: (20, 20, 20)}
        for k, src in enumerate(self._objects_src):
            oid, typ, c, size, rgb = src[:5]
            extra = dict(src[5]) if len(src) > 5 else {}
            pos = (c[0], c[1] - size / 2.0, c[2])
            o = {"objectId": oid, "objectType": typ, "position": _d(pos),
                 "rotation": _d((0.0, 0.0, 0.0)), "pickupable": typ not in LARGE_TYPES,
                 "visible": True, "_size": float(size), "_rgb": tuple(rgb),
                 "axisAlignedBoundingBox": {"center": _d(c), "size": _d((size, size, size))}}
            o.update(extra)
            self.objects.append(o)
            self.colors[oid] = (40 + 20 * k, 200 - 15 * k, 100 + 30 * k)
        self.last_event = self._event(True, "")
        return self.last_event

    def step(self, action: str = "", **kw):
        self.calls.append((action, dict(kw)))
        ok, err = True, ""
        if action == "AddThirdPartyCamera":
            self.cams.append({"position": dict(kw["position"]), "rotation": dict(kw["rotation"]),
                              "fieldOfView": float(kw.get("fieldOfView", 90.0))})
        elif action == "UpdateThirdPartyCamera":
            i = int(kw.get("thirdPartyCameraId", 0))
            if i >= len(self.cams):
                ok, err = False, f"no third party camera {i}"
            else:
                for k in ("position", "rotation"):
                    if k in kw:
                        self.cams[i][k] = dict(kw[k])
                if "fieldOfView" in kw:
                    self.cams[i]["fieldOfView"] = float(kw["fieldOfView"])
        elif action == "TeleportObject":
            o = next((o for o in self.objects if o["objectId"] == kw.get("objectId")), None)
            if o is None:
                ok, err = False, f"no object {kw.get('objectId')}"
            else:
                p = vec(kw["position"])
                o["position"] = _d(p)
                if "rotation" in kw:
                    o["rotation"] = dict(kw["rotation"])
                o["axisAlignedBoundingBox"]["center"] = _d(p + np.array([0.0, o["_size"] / 2.0, 0.0]))
        elif action == "GetReachablePositions":
            self.last_event = self._event(True, "", action_return=[_d(p) for p in self.reachable])
            return self.last_event
        elif action in ("Teleport", "TeleportFull"):
            p = tuple(float(x) for x in vec(kw["position"]))
            if not any(abs(p[0] - q[0]) < 1e-3 and abs(p[2] - q[2]) < 1e-3 for q in self.reachable):
                ok, err = False, f"{action}: 歩けない場所 {p}"
            else:
                self.agent_pos = p
                if "rotation" in kw:
                    r = kw["rotation"]
                    self.agent_yaw = float(r["y"] if isinstance(r, dict) else r)
        elif action in ("Pass", "Done"):
            pass
        else:
            ok, err = False, f"FakeController は {action} を知らない"
        self.last_event = self._event(ok, err)
        return self.last_event

    def stop(self) -> None:
        self.stopped = True

    # ---------------------------------------------------------------- 描く
    def _event(self, ok: bool, err: str, action_return=None) -> FakeEvent:
        md = {"lastActionSuccess": ok, "errorMessage": err, "sceneName": self.scene, "actionReturn": action_return,
              "agent": {"position": _d(self.agent_pos), "rotation": _d((0.0, self.agent_yaw, 0.0))},
              "objects": [{k: copy.deepcopy(v) for k, v in o.items() if not k.startswith("_")} for o in self.objects],
              "thirdPartyCameras": copy.deepcopy(self.cams)}
        cams = [self._render(c) for c in self.cams]
        return FakeEvent(md, cams, {v: k for k, v in self.colors.items()})

    def _render(self, cam: dict):
        h, w = self.height, self.width
        eye = vec(cam["position"])
        fwd, right, up = cam_basis(float(cam["rotation"]["y"]), float(cam["rotation"]["x"]))
        f = focal_px(h, cam["fieldOfView"])
        cols = (np.arange(w) + 0.5 - 0.5 * w) / f
        rows = (np.arange(h) + 0.5 - 0.5 * h) / f
        d = fwd[None, None, :] + cols[None, :, None] * right[None, None, :] - rows[:, None, None] * up[None, None, :]
        d /= np.linalg.norm(d, axis=2, keepdims=True)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        seg = np.zeros((h, w, 3), dtype=np.uint8)
        # 床と壁（世界に貼りついた模様）
        dy = d[:, :, 1]
        t_floor = np.where(dy < -1e-6, -eye[1] / np.minimum(dy, -1e-6), np.inf)
        floor = t_floor < 8.0
        tf = np.where(floor, t_floor, 0.0)
        fx, fz = eye[0] + tf * d[:, :, 0], eye[2] + tf * d[:, :, 2]
        chk = ((np.floor(fx / 0.25) + np.floor(fz / 0.25)) % 2).astype(bool)
        rgb[floor] = np.where(chk[floor][:, None], np.array([150, 120, 90]), np.array([120, 95, 70]))
        seg[floor] = self.colors[FLOOR_ID]
        az = np.degrees(np.arctan2(d[:, :, 0], d[:, :, 2]))
        el = np.degrees(np.arcsin(np.clip(dy, -1, 1)))
        stripe = ((np.floor(az / 8.0) + np.floor(el / 8.0)) % 2).astype(bool)
        wall = ~floor
        rgb[wall] = np.where(stripe[wall][:, None], np.array([215, 210, 195]), np.array([185, 180, 170]))
        seg[wall] = self.colors[WALL_ID]
        depth = np.where(floor, t_floor, 8.0)
        # 物（球）
        for o in self.objects:
            c = vec(o["axisAlignedBoundingBox"]["center"]) - eye
            r = o["_size"] / 2.0
            b = d @ c
            disc = b * b - (float(c @ c) - r * r)
            hit = disc >= 0
            t = np.where(hit, b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
            near = hit & (t > 1e-3) & (t < depth)
            if not near.any():
                continue
            shade = np.clip(0.55 + 0.45 * (b - t) / max(r, 1e-6), 0.4, 1.0)    # 真ん中ほど明るい
            rgb[near] = (np.array(o["_rgb"])[None, :] * shade[near][:, None]).astype(np.uint8)
            seg[near] = self.colors[o["objectId"]]
            depth = np.where(near, t, depth)
        return rgb, seg


def _d(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def fake_env(scene: str = "FloorPlan1", px: int = 224, render_px: int | None = None, controller_kw=None, **kw):
    """偽物の AI2-THOR で家の場面を作る（テスト・クラウドの確かめ用）。controller_kw は FakeController へ。"""
    from .env import ThorEnv
    rp = int(render_px or px)
    ctl = FakeController(scene, width=rp, height=rp, **dict(controller_kw or {}))
    return ThorEnv(scene, controller=ctl, px=px, render_px=rp, **kw)


__all__ = ["FakeController", "FakeEvent", "fake_env", "DEFAULT_OBJECTS"]
