"""家の場面の環境本体（AI2-THOR）。目だけの場面（world_eye/env.py）と同じ形を持つ：
* env.reset(seed=, scene=) → (obs, info)、env.step(action) → (obs, 0.0, False, False, info)
* obs["eye_left"]：太郎の目の画像（px×px×3、uint8）。読んだときに AI2-THOR へ溜めた指令を送って描く（読まない歩は送らない）
* info["parent_utterance"]：親が話した歩だけ {"text", "target", "cause"}。info["object_sound"]：物の音（16kHz の波形）
* env._orienting：目の動き方（world_eye/eye.py の EyeKinematics をそのまま。撃てる間隔・1回で動く角度・主系列）
* say()・object_sound()・taro_said()・set_eye()：目だけの場面と同じ

目：AI2-THOR の「三人目のカメラ」（AddThirdPartyCamera）を、エージェントの EYE_AHEAD_M（0.3m）前・床から EYE_Y_M（0.6m）に置く。
目の向き（pan＝右が正、tilt＝上が正、度。目だけの場面と同じ決まり）を、そのカメラの向きに写す：
  rotation = {x: −tilt（AI2-THOR は下向きが正）, y: エージェントの向き＋pan（右回りが正）, z: 0}
目が動いても、カメラの場所は動かさない（首も体も無い。目だけの場面と同じ）。最初の向きは下へ BASE_TILT_DEG（10度）。

画角は FOV_DEG＝60度（目だけの場面と同じ）。PC の下見は 90度だったが、目の入口の指令の写し（eyein.command_shift）は
「指令1で視野の半分＝30度回る・224画素で画角60度の焦点距離」で、画像のずれを先に予想する（生まれつきの較正）。
画角を変えると予想と本当のずれが合わなくなるので、60度にそろえる（--fov で変えられるが、そのときは写しが外れる）［Tier3・モデル上の選択］

正解（記録・親・動画の字のためだけ。太郎には渡さない）：物ごとの色分けの絵（renderInstanceSegmentation）から、
「目の真ん中にある物」（centre_object）と、物の見えている所の真ん中（object_centres）を求める。

座る場所（place="auto"、既定。依頼16 の PC の絵で、エージェントの始めの場所は FloorPlan1 で冷蔵庫の扉の目の前、
FloorPlan201 でソファの後ろの壁ぎわだった。名前を言える物がほとんど目に入らず、親の「1秒見たら持ってくる」が1回も起きなかった）：
reset のとき、GetReachablePositions の場所から PLACE_MAX_CANDIDATES（60）か所を選び（種で決まる）、向き 0/90/180/270度ごとに、
「名前を言える拾える物で、目から PLACE_NEAR_M〜PLACE_FAR_M（0.5〜2.0m）にあり、最初の目の向き（下へ10度・画角60度）の
絵の中に入るもの」の数を、メタデータの物の場所から幾何で数える（描かない）。閉じた戸棚・冷蔵庫の中の物、目との間に
大きな物（PLACE_LARGE_M＝0.4m 以上）の箱がある物は数えない。目の真正面 PLACE_BLOCK_M（0.6m）以内に名前の無い大きな物が
あれば PLACE_BLOCK_PENALTY、目の 0.5m 先の床が歩けない（壁・家具がすぐ前）なら PLACE_WALL_PENALTY を引く。
いちばん点の高い所へ Teleport する（同点は種の乱数で選ぶ）。親が子を、物の見える所へ座らせる（旧太郎の親が
おもちゃを子の前に並べたのと同じ役）［Tier3・モデル上の選択。数は仮置き］。place="start" は今までどおり。

依頼17 の PC の結果（自動の座る場所でも、目に入る名前の物は FloorPlan1・FloorPlan201 とも1つだけ。台所の物は調理台の上にあり、
0.6m の目からは見えない）を受けて（2026-10-19）：
* 親がおもちゃを並べる（toys＝TOYS_N（4）、0 で切る）：座らせたあと、親が名前を言える拾える物を3〜4つ（語の表から、
  できるだけ違う語、目に近い順。閉じた戸棚の中の物は後回し）、太郎の前の床に並べる。向き（目の向き yaw から）
  TOY_ANGLES（−20・−7・+7・+20度）、目からの水平の距離 TOY_NEAR_M〜TOY_FAR_M（0.6〜1.2m）のうち、物の真ん中が最初の目の絵の
  下 TOY_ROW_MAX（85%）より上に入る、いちばん近い所。床が歩ける所の近く（TOY_FREE_M 以内に到達点）で、床の上の大きな物の箱に
  入らない所だけ。置き方は PlaceObjectAtPoint（物の底をその点に置く）、通らなければ TeleportObject（forceKinematic=False、
  真ん中の高さ＝床＋箱の高さの半分）。置いた所を物の「元の場所」（home）にする（親が持ってきて戻すとき、調理台ではなく床へ戻す）。
  置いたあと、目の絵に写すと何個入るか（幾何と色分けの絵の両方）を記録する。依頼の「目から 0.6〜1.0m」は、下へ10度・画角60度の
  目では床の 0.7m より手前が絵の下に外れるので、水平 0.6〜1.2m で絵に入るいちばん近い所にした［Tier3・モデル上の選択］
* 座る場所（place）：(x, z, yaw) を渡せば、そこへそのまま Teleport する（手で選ぶ。上から見た絵 thor_watch --first-frame を見て決める）。
  通らなければいちばん近い到達点へ。auto は控え：候補は「真正面 PLACE_AHEAD_MIN_M（1.5m）先まで歩ける」所だけ。
  おもちゃを並べるときは、前の開け具合（PLACE_AHEAD_CAP まで）＋おもちゃの所の左右の開け具合＋床の上の大きな物からの離れで点を付ける
  （物が目に入っているかは見ない。親が並べるので）

AI2-THOR は1回の呼び出しが重い（約55ミリ秒）ので、環境の1歩（0.01秒）ごとには呼ばない。物を動かす（hold・put_back）・
目の向きは控えておき、画像を読む（render_eye・centre_object）ときにまとめて送る（物を動かす → 最後にカメラの向き）。
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from ..world_eye.eye import EyeKinematics

DT = 0.01
EYE_Y_M = 0.6            # 目の高さ（床から。1歳前後の子が座った・はった目の高さ）［Tier2・記憶ベース］
EYE_AHEAD_M = 0.3        # エージェントの前へ出す距離（体の中から見ないため。依頼15）
BASE_TILT_DEG = -10.0    # 最初の目の向き（下へ10度。依頼15の絵）
FOV_DEG = 60.0           # 上の説明（目の入口の指令の写しの較正と同じ）
PX = 224
CENTRE_DEG = 5.0         # 「目の真ん中にある物」を数える円の半径（度。watch_scene の視線の丸と同じ）
CENTRE_MIN_FRAC = 0.15   # その円の画素のうち、これだけ以上を占める物だけを「真ん中の物」とする［仮置き］
PLACE_MAX_CANDIDATES = 60
PLACE_YAWS = (0.0, 90.0, 180.0, 270.0)
PLACE_NEAR_M, PLACE_FAR_M = 0.5, 2.0
PLACE_LARGE_M = 0.4        # これより大きい（箱のいちばん長い辺）名前の無い物を「大きな物」とする
PLACE_BLOCK_M = 0.6
PLACE_BLOCK_PENALTY = 3.0
PLACE_WALL_PENALTY = 1.0
PLACE_AHEAD_MIN_M = 1.5   # 真正面にこれだけ歩ける所が続くこと（座る場所の候補の条件）
PLACE_STEP_M = 0.25       # AI2-THOR の到達点の格子
PLACE_REACH_TOL_M = 0.1
PLACE_AHEAD_CAP_M = 3.0
PLACE_SIDE_CAP_M = 1.0
PLACE_CLEAR_CAP_M = 1.5
TOYS_N = 4
TOY_ANGLES = {1: (0.0,), 2: (-8.0, 8.0), 3: (-15.0, 0.0, 15.0), 4: (-20.0, -7.0, 7.0, 20.0)}
TOY_NEAR_M, TOY_FAR_M, TOY_STEP_M = 0.6, 1.2, 0.05
TOY_ROW_MAX = 0.85        # 物の真ん中が絵の下からこれより上（行の割合）
TOY_COL_MARGIN = 0.06
TOY_FREE_M = 0.35         # この距離以内に到達点がある床を「開いた床」とする
FLOOR_BOX_TOP_M = 0.5     # 箱の下がこれより低い大きな物を「床の上の大きな物」とする（吊り戸棚は数えない）
PLACE_IGNORE_TYPES = ("Floor", "Rug", "Window", "Painting", "Poster", "LightSwitch", "Curtains", "Blinds")
CONTROLLER_KW = {"server_timeout": 900, "makeAgentsVisible": False, "renderInstanceSegmentation": True}


class _ActionSpace:
    shape = (0,)


class LazyObs(dict):
    """obs["eye_left"] を読んだときに描く（world_eye/env.LazyObs と同じ）。"""

    def __init__(self, env, n_step: int) -> None:
        super().__init__()
        self._env = env
        self._n = n_step

    def __missing__(self, key):
        if key != "eye_left":
            raise KeyError(key)
        if self._env.n_step != self._n:
            raise RuntimeError("古い歩の観測を読もうとした（家の場面は読んだときに描くので、その歩の画像はもう無い）")
        img = self._env.render_eye()
        self[key] = img
        return img


# ---------------------------------------------------------------- 向きと写し方（Unity：x 右・y 上・z 前、左手系）
def vec(d) -> np.ndarray:
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])], dtype=float)


def xyz(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def cam_basis(yaw_deg: float, pitch_down_deg: float):
    """カメラの (前, 右, 上)。yaw は上から見て右回りが正、pitch は下向きが正（AI2-THOR の rotation の x・y）。"""
    y, p = math.radians(yaw_deg), math.radians(pitch_down_deg)
    fwd = np.array([math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
    right = np.array([math.cos(y), 0.0, -math.sin(y)])
    up = np.array([math.sin(y) * math.sin(p), math.cos(p), math.cos(y) * math.sin(p)])
    return fwd, right, up


def focal_px(px: int, fov_deg: float) -> float:
    return 0.5 * px / math.tan(math.radians(fov_deg) / 2.0)


def project(point, eye, yaw_deg, pitch_down_deg, px, fov_deg):
    """世界の点 → 画像の (列, 行, 奥行き)。カメラの後ろなら None。"""
    fwd, right, up = cam_basis(yaw_deg, pitch_down_deg)
    v = np.asarray(point, dtype=float) - np.asarray(eye, dtype=float)
    z = float(v @ fwd)
    if z <= 1e-6:
        return None
    f = focal_px(px, fov_deg)
    return 0.5 * px + f * float(v @ right) / z, 0.5 * px - f * float(v @ up) / z, z


def make_controller(scene: str, px: int, platform: str | None = None, **kw):
    """本物の AI2-THOR を起こす（ここで初めて ai2thor を読む）。platform=None は Linux64（窓あり。WSLg で見える）。"""
    from ai2thor.controller import Controller
    args = dict(CONTROLLER_KW, scene=scene, width=int(px), height=int(px))
    if platform:
        import ai2thor.platform as plat
        args["platform"] = getattr(plat, platform)
    args.update(kw)
    return Controller(**args)


class ThorEnv:
    """controller：AI2-THOR の Controller（か fake.FakeController）。None なら make_controller で起こす。
    px：太郎の目の画像の大きさ。render_px：AI2-THOR に描かせる大きさ（動画のために大きく描き、目には px へ縮めて渡す）。"""

    metadata: dict = {}

    def __init__(self, scene: str = "FloorPlan1", controller=None, px: int = PX, render_px: int | None = None,
                 fov: float = FOV_DEG, eye_y: float = EYE_Y_M, eye_ahead: float = EYE_AHEAD_M,
                 base_tilt: float = BASE_TILT_DEG, platform: str | None = None, log_dir: str | None = None,
                 seed: int = 0, controller_kw: dict | None = None, place="auto", toys: int = TOYS_N) -> None:
        from .parent import TYPE_WORD
        self.scene_name = scene
        self.px = int(px)
        self.render_px = int(render_px or px)
        self.fov = self.fovy = float(fov)
        self.eye_y, self.eye_ahead, self.base_tilt = float(eye_y), float(eye_ahead), float(base_tilt)
        self.type_word = dict(TYPE_WORD)
        self.dt = DT
        if isinstance(place, str) and "," in place:
            place = tuple(float(x) for x in place.split(","))
        if isinstance(place, (tuple, list)):
            if len(place) != 3:
                raise ValueError(f"place は (x, z, yaw)：{place}")
            place = tuple(float(x) for x in place)
        elif place not in ("auto", "start"):
            raise ValueError(f"place は auto・start・(x, z, yaw)：{place}")
        self.place = place
        self.n_toys = max(0, int(toys))
        self.placement: dict = {}
        self.toys: list[str] = []
        self.toy_info: list[dict] = []
        self.reach = None
        self.in_hand: set = set()        # 親が今持っている物（持っていない物の名づけは届けない）
        self.action_space = _ActionSpace()
        self.unwrapped = self
        self.controller = controller if controller is not None else make_controller(
            scene, self.render_px, platform, **dict(controller_kw or {}))
        self._orienting = EyeKinematics()
        self.log_dir = log_dir
        self._ev_fp = self._ev_w = None
        self.n_step = 0
        self._colors: dict[int, str] = {}     # 色分けの色（r<<16|g<<8|b）→ 物の id（届いた分を足していく）
        self.objects: dict[str, dict] = {}
        self.n_calls = 0                 # AI2-THOR を呼んだ回数（重さの目安）
        self.n_fail = 0                  # 失敗した呼び出しの数（物を動かせなかった等）
        self.last_error = ""
        self.np_random = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ================================================================ gym の形
    def reset(self, seed=None, scene: str | None = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(int(seed))
        if scene is not None:
            self.scene_name = scene
        self._close_log()
        ev = self.controller.reset(scene=self.scene_name)
        self.n_calls += 1
        self._take(ev)
        self.in_hand = set()
        self._place()
        self._set_out_toys()
        ag = self._event.metadata["agent"]
        self.agent_yaw = float(ag["rotation"]["y"])
        fwd0, _r, _u = cam_basis(self.agent_yaw, 0.0)
        p = vec(ag["position"]) + self.eye_ahead * fwd0
        p[1] = self.eye_y
        self.eye_pos = p
        self.time = 0.0
        self.n_step += 1
        self._orienting.reset()
        self._orienting.set_angles(0.0, self.base_tilt)
        self._pending_say: list[dict] = []
        self._pending_sound: list = []
        self._pending_tp: dict[str, tuple] = {}     # 物の id → (位置, 向き)：次に描くときに送る
        self._taro_voice_signal = False
        self._taro_voice_text = ""
        self._sent_cam = None
        ev = self._call("AddThirdPartyCamera", position=xyz(self.eye_pos), rotation=self._cam_rot(),
                        fieldOfView=self.fov)
        self._sent_cam = self._cam_key()
        if not self._ok(ev):
            raise RuntimeError(f"AI2-THOR に目（三人目のカメラ）を置けなかった：{self.last_error}")
        # 親が名前を言える物（語の表にある種類の、拾える物）
        self.home = {i: (vec(o["position"]), vec(o["rotation"])) for i, o in self.objects.items()}
        self.targets = [i for i, o in self.objects.items()
                        if o["objectType"] in self.type_word and o.get("pickupable", True)]
        if self.toys:
            seen = set(self.object_centres())
            for t in self.toy_info:
                t["seen"] = t["id"] in seen
        self.placement["toys_in_view"] = sum(t["in_view"] for t in self.toy_info)
        self.placement["toys_seen"] = sum(t.get("seen", False) for t in self.toy_info)
        self.placement["toys"] = [t["word"] for t in self.toy_info]
        self._open_log(seed)
        self._log("reset", "", f"{self.scene_name} 物{len(self.objects)} 名前を言える物{len(self.targets)}")
        pl = self.placement
        self._log("place", "", f"{pl.get('mode')} 場所{pl.get('position')} 向き{pl.get('yaw')} 点{pl.get('score')} "
                               f"前{pl.get('ahead_m')}m 目に入る名前の物{pl.get('n_in_view')} {'・'.join(pl.get('words', []))} "
                               f"候補{pl.get('n_candidates')}（前が開いた所{pl.get('n_open')}） {pl.get('error', '')}", cause="place")
        for t in self.toy_info:
            self._log("toy", t["id"], f"{t['word']} {t['angle']:+.0f}度 {t['dist']:.2f}m {t['method']} "
                                      f"絵に入る{int(t['in_view'])} 写る{int(t.get('seen', False))}", cause="parent_toys")
        self._log("toys", "", f"並べた{len(self.toys)} 絵に入る{pl['toys_in_view']} 写る{pl['toys_seen']} "
                              f"{'・'.join(pl['toys'])}", cause="parent_toys")
        return LazyObs(self, self.n_step), {"targets": list(self.targets)}

    # ================================================================ 座る場所
    def _nameable(self) -> list[str]:
        return [i for i, o in self.objects.items()
                if o["objectType"] in self.type_word and o.get("pickupable", True)]

    def _hidden(self, o) -> bool:
        """閉じた戸棚・冷蔵庫・引き出しなどの中にある物（外からは見えない）。"""
        for pid in o.get("parentReceptacles") or []:
            p = self.objects.get(pid)
            if p is not None and p.get("openable") and not p.get("isOpen", False):
                return True
        return False

    @staticmethod
    def _box(o):
        bb = o.get("axisAlignedBoundingBox") or {}
        if not bb.get("center") or not bb.get("size"):
            return None
        c, sz = vec(bb["center"]), vec(bb["size"])
        return c - sz / 2.0, c + sz / 2.0

    @staticmethod
    def _ray_box(p, d, lo, hi):
        """p から d の向き（単位でなくてよい）の光線が箱 [lo, hi] に入る t（箱の中からなら 0）。当たらなければ None。"""
        t0, t1 = -np.inf, np.inf
        for k in range(3):
            if abs(d[k]) < 1e-12:
                if p[k] < lo[k] or p[k] > hi[k]:
                    return None
                continue
            a, b = (lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t1 < max(t0, 0.0):
            return None
        return max(t0, 0.0)

    def score_spot(self, pos, yaw: float, reach=None) -> dict:
        """その場所・向きに座ったときの点（上の説明）。pos はエージェントの場所（GetReachablePositions の点）。"""
        fwd0 = cam_basis(yaw, 0.0)[0]
        eye = np.asarray(pos, dtype=float) + self.eye_ahead * fwd0
        eye[1] = self.eye_y
        pitch = -self.base_tilt
        targets = set(self._nameable())
        large = []
        for i, o in self.objects.items():
            if i in targets or o["objectType"] in PLACE_IGNORE_TYPES:
                continue
            bx = self._box(o)
            if bx is not None and float(np.max(bx[1] - bx[0])) >= PLACE_LARGE_M:
                large.append((i, bx))
        seen = []
        for i in sorted(targets):
            o = self.objects[i]
            if self._hidden(o):
                continue
            c = self._centre(o)
            v = c - eye
            dist = float(np.linalg.norm(v))
            if not (PLACE_NEAR_M <= dist <= PLACE_FAR_M):
                continue
            pr = project(c, eye, yaw, pitch, self.px, self.fov)
            if pr is None or not (0.0 <= pr[0] < self.px and 0.0 <= pr[1] < self.px):
                continue
            parents = set(o.get("parentReceptacles") or [])
            blocked = False
            for j, (lo, hi) in large:
                if j in parents:
                    continue
                t = self._ray_box(eye, v, lo, hi)
                if t is not None and 0.02 < t < 0.95:       # v の長さを1とした割合（物の手前で当たる）
                    blocked = True
                    break
            if not blocked:
                seen.append(i)
        penalty, why = 0.0, []
        hdir = np.array([fwd0[0], 0.0, fwd0[2]])
        for j, (lo, hi) in large:
            t = self._ray_box(eye, hdir, lo, hi)
            if t is not None and t < PLACE_BLOCK_M:
                penalty += PLACE_BLOCK_PENALTY
                why.append(self.objects[j]["objectType"])
                break
        if reach is not None and len(reach):
            ahead = np.asarray(pos, dtype=float) + 0.5 * fwd0
            dd = np.hypot(reach[:, 0] - ahead[0], reach[:, 2] - ahead[2])
            if float(dd.min()) > 0.15:
                penalty += PLACE_WALL_PENALTY
                why.append("前が歩けない")
        return {"score": len(seen) - penalty, "n_in_view": len(seen), "seen": seen, "penalty": penalty,
                "why": why, "eye": eye}

    def _large_boxes(self, floor_only: bool = False) -> list:
        """名前の無い大きな物の箱 [(id, (lo, hi))]。floor_only なら箱の下が床に近い物だけ（床の上の家具）。"""
        targets = set(self._nameable())
        out = []
        for i, o in self.objects.items():
            if i in targets or o["objectType"] in PLACE_IGNORE_TYPES:
                continue
            bx = self._box(o)
            if bx is None or float(np.max(bx[1] - bx[0])) < PLACE_LARGE_M:
                continue
            if floor_only and float(bx[0][1]) - self.floor_y > FLOOR_BOX_TOP_M:
                continue
            out.append((i, bx))
        return out

    def _reach_near(self, p, tol: float = PLACE_REACH_TOL_M) -> bool:
        if self.reach is None or not len(self.reach):
            return False
        d = np.hypot(self.reach[:, 0] - p[0], self.reach[:, 2] - p[2])
        return bool(d.min() <= tol)

    def _run_len(self, p, d, cap: float) -> float:
        """p から水平の向き d へ、到達点が続く長さ（PLACE_STEP_M きざみ、cap まで）。"""
        n = 0
        while (n + 1) * PLACE_STEP_M <= cap + 1e-9 and self._reach_near(p + (n + 1) * PLACE_STEP_M * d):
            n += 1
        return n * PLACE_STEP_M

    @staticmethod
    def _xz_dist_box(p, lo, hi) -> float:
        dx = max(lo[0] - p[0], 0.0, p[0] - hi[0])
        dz = max(lo[2] - p[2], 0.0, p[2] - hi[2])
        return float(math.hypot(dx, dz))

    def space_spot(self, pos, yaw: float) -> dict:
        """前の開け具合の点（おもちゃを並べるとき。上の説明）。ahead_m が PLACE_AHEAD_MIN_M 未満なら候補にしない。"""
        pos = np.asarray(pos, dtype=float)
        fwd = cam_basis(yaw, 0.0)[0]
        right = cam_basis(yaw, 0.0)[1]
        ahead = self._run_len(pos, fwd, PLACE_AHEAD_CAP_M)
        zone = pos + (self.eye_ahead + 1.0) * fwd                     # おもちゃを並べる辺り（目の 1m 先）
        zone_grid = pos + round((self.eye_ahead + 1.0) / PLACE_STEP_M) * PLACE_STEP_M * fwd
        side = min(self._run_len(zone_grid, right, PLACE_SIDE_CAP_M), self._run_len(zone_grid, -right, PLACE_SIDE_CAP_M))
        clear = PLACE_CLEAR_CAP_M
        for _j, (lo, hi) in self._large_boxes(floor_only=True):
            clear = min(clear, self._xz_dist_box(zone, lo, hi))
        return {"ahead_m": ahead, "side_m": side, "clear_m": round(clear, 3),
                "space": min(ahead, PLACE_AHEAD_CAP_M) + side + clear, "open": ahead >= PLACE_AHEAD_MIN_M - 1e-9}

    def _teleport_agent(self, pos, yaw: float) -> bool:
        p = xyz(pos)
        for act in ("Teleport", "TeleportFull"):
            ev = self._call(act, position=p, rotation={"x": 0.0, "y": float(yaw), "z": 0.0}, horizon=0.0, standing=True)
            if self._ok(ev):
                return True
        return False

    def _place(self) -> None:
        """座る場所を決めて、そこへ Teleport する（上の説明）。決めた所は self.placement に残す。"""
        ag = self._event.metadata["agent"]
        here, yaw0 = vec(ag["position"]), float(ag["rotation"]["y"])
        self.reach = None
        self.placement_error = ""
        self.floor_y = self._floor_y()
        if self.place != "start":
            ev = self._call("GetReachablePositions")
            pts = ev.metadata.get("actionReturn") if self._ok(ev) else None
            if not pts:
                self.placement_error = f"GetReachablePositions：{self.last_error}"
            if pts:
                self.reach = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in pts])
        reach = self.reach
        toys_on = self.n_toys > 0

        def cand(pos, yaw):
            c = {"pos": np.asarray(pos, dtype=float), "yaw": float(yaw), **self.score_spot(pos, yaw, reach)}
            if reach is not None:
                c.update(self.space_spot(pos, yaw))
            else:
                c.update({"ahead_m": None, "side_m": None, "clear_m": None, "space": 0.0, "open": False})
            c["rank"] = c["space"] if toys_on else c["score"]
            return c

        start = cand(here, yaw0)
        best = start
        n_cand = n_open = 0
        mode = self.place if isinstance(self.place, str) else "spot"
        if isinstance(self.place, tuple):
            x, z, yaw = self.place
            want = np.array([x, here[1], z])
            if not self._teleport_agent(want, yaw):
                err = self.last_error
                if reach is not None and len(reach):
                    k = int(np.argmin(np.hypot(reach[:, 0] - x, reach[:, 2] - z)))
                    want = reach[k]
                    if self._teleport_agent(want, yaw):
                        mode = "spot(近い到達点へ)"
                        self.placement_error = f"指定の所へ行けなかった：{err[:120]}"
                    else:
                        mode = "spot(失敗→始めの場所)"
                        self.placement_error = self.last_error
                else:
                    mode = "spot(失敗→始めの場所)"
                    self.placement_error = err
            ag = self._event.metadata["agent"]
            best = cand(vec(ag["position"]), float(ag["rotation"]["y"]))
        elif self.place == "auto" and reach is not None:
            idx = np.arange(len(reach))
            if len(idx) > PLACE_MAX_CANDIDATES:
                idx = np.sort(self.np_random.choice(idx, PLACE_MAX_CANDIDATES, replace=False))
            cands = [start] + [cand(reach[i], y) for i in idx for y in PLACE_YAWS]
            n_cand = len(cands)
            ok = [c for c in cands if c["open"]]
            n_open = len(ok)
            if not ok:
                self.placement_error = f"真正面 {PLACE_AHEAD_MIN_M}m 先まで歩ける所が無い（始めの場所のまま）"
                mode = "auto(前が開いた所なし→始めの場所)"
            else:
                top = max(c["rank"] for c in ok)
                tied = [c for c in ok if c["rank"] >= top - 1e-9]
                best = tied[int(self.np_random.integers(len(tied)))] if len(tied) > 1 else tied[0]
                moved = not (np.allclose(best["pos"], here, atol=1e-3) and abs((best["yaw"] - yaw0) % 360.0) < 1e-3)
                if moved and not self._teleport_agent(best["pos"], best["yaw"]):
                    best = start
                    self.placement_error = self.last_error
        elif self.place == "auto":
            mode = "start(到達点なし)"
        ag = self._event.metadata["agent"]
        self.placement = {
            "mode": mode, "position": [round(float(x), 3) for x in vec(ag["position"])],
            "yaw": round(float(ag["rotation"]["y"]), 1),
            "score": round(float(best["rank"]), 2), "n_in_view": int(best["n_in_view"]),
            "ahead_m": best["ahead_m"], "side_m": best["side_m"], "clear_m": best["clear_m"],
            "words": [self.type_word.get(self.objects[i]["objectType"], "") for i in best["seen"]],
            "penalty": best["penalty"], "why": list(best["why"]), "n_candidates": n_cand, "n_open": n_open,
            "start_n_in_view": int(start["n_in_view"]), "start_score": round(float(start["rank"]), 2),
            "start_ahead_m": start["ahead_m"], "error": self.placement_error[:200],
            "by": "space" if toys_on else "objects",
        }

    def _floor_y(self) -> float:
        """床の高さ（Floor の箱の上。無ければ 0）。"""
        for o in self.objects.values():
            if o["objectType"] == "Floor":
                bx = self._box(o)
                if bx is not None and -0.3 < float(bx[1][1]) < 0.3:
                    return float(bx[1][1])
        return 0.0

    # ================================================================ 親がおもちゃを並べる
    def _toy_choice(self, eye) -> list[str]:
        """並べる物：語の表の拾える物を、できるだけ違う語で、目に近い順（閉じた戸棚の中は後回し）。"""
        cand = sorted(self._nameable(), key=lambda i: (self._hidden(self.objects[i]),
                                                       round(float(np.linalg.norm(self._centre(self.objects[i]) - eye)), 4), i))
        out, words = [], set()
        for i in cand:
            w = self.type_word[self.objects[i]["objectType"]]
            if w not in words:
                out.append(i)
                words.add(w)
        out += [i for i in cand if i not in out]
        return out[:self.n_toys]

    def _toy_spot(self, oid, eye, yaw, ang):
        """ang の向きで、絵に入る・開いた床の、いちばん近い床の点。無ければ None。"""
        o = self.objects[oid]
        bx = self._box(o)
        size = (bx[1] - bx[0]) if bx is not None else np.array([0.1, 0.1, 0.1])
        r = 0.5 * float(max(size[0], size[2]))
        h = math.radians(yaw + ang)
        dvec = np.array([math.sin(h), 0.0, math.cos(h)])
        floor_boxes = self._large_boxes(floor_only=True)
        for d in np.arange(TOY_NEAR_M, TOY_FAR_M + 1e-9, TOY_STEP_M):
            p = np.array([eye[0], self.floor_y, eye[2]]) + d * dvec
            c = p + np.array([0.0, 0.5 * float(size[1]), 0.0])
            pr = project(c, eye, yaw, -self.base_tilt, self.px, self.fov)
            if pr is None or pr[1] > TOY_ROW_MAX * self.px or pr[1] < 0:
                continue
            if not (TOY_COL_MARGIN * self.px <= pr[0] <= (1 - TOY_COL_MARGIN) * self.px):
                continue
            if self.reach is not None and len(self.reach) > 1 and not self._reach_near(p, TOY_FREE_M):
                continue
            if any(self._xz_dist_box(p, lo, hi) < r for _j, (lo, hi) in floor_boxes):
                continue
            return p, c, float(d)
        return None

    def _set_out_toys(self) -> None:
        self.toys, self.toy_info = [], []
        if self.n_toys <= 0:
            return
        ag = self._event.metadata["agent"]
        yaw = float(ag["rotation"]["y"])
        eye = vec(ag["position"]) + self.eye_ahead * cam_basis(yaw, 0.0)[0]
        eye[1] = self.eye_y
        picks = self._toy_choice(eye)
        angles = TOY_ANGLES[max(1, min(4, len(picks)))] if picks else ()
        for oid, ang in zip(picks, angles):
            spot = self._toy_spot(oid, eye, yaw, ang)
            if spot is None:
                self._log("toy_skip", oid, f"{ang:+.0f}度に置ける床が無い", cause="parent_toys")
                self.toy_info.append({"id": oid, "word": self.word_of(oid), "angle": ang, "dist": 0.0, "method": "置けない",
                                      "spot": None, "in_view": False, "placed": False})
                continue
            p, c, d = spot
            method = "PlaceObjectAtPoint"
            try:
                ev = self._call("PlaceObjectAtPoint", objectId=oid, position=xyz(p))
                placed_ok = self._ok(ev)
            except Exception as ex:  # noqa: BLE001  版によっては知らない動作で例外になる（未確認）→ 控えへ
                placed_ok = False
                self.n_fail += 1
                self.last_error = f"{type(ex).__name__}: {ex}"
            if not placed_ok:
                err = self.last_error
                o = self.objects[oid]
                off = self._centre(o) - vec(o["position"])
                ev = self._call("TeleportObject", objectId=oid, position=xyz(c - off), rotation=o["rotation"],
                                forceAction=True, forceKinematic=False)
                method = "TeleportObject(控え)"
                if not self._ok(ev):
                    method = "失敗"
                self.toy_error = err[:120]
            o = self.objects.get(oid)
            got = self._centre(o) if o is not None else c
            pr = project(got, eye, yaw, -self.base_tilt, self.px, self.fov)
            inv = pr is not None and 0 <= pr[0] < self.px and 0 <= pr[1] < self.px
            placed = method != "失敗"
            if placed:
                self.toys.append(oid)
            self.toy_info.append({"id": oid, "word": self.word_of(oid), "angle": ang, "dist": round(d, 2),
                                  "method": method, "spot": [round(float(x), 3) for x in p],
                                  "at": [round(float(x), 3) for x in got], "in_view": bool(inv), "placed": placed})

    def step(self, action=None):
        self.time += self.dt
        self.n_step += 1
        self._orienting.step(self.dt)
        info = {}
        if self._pending_sound:
            info["object_sound"] = self._pending_sound.pop(0)
        while self._pending_say:
            utt = self._pending_say.pop(0)
            if str(utt.get("cause", "")).startswith("parent_name") and utt.get("target") not in self.in_hand:
                # 親の名づけは、その物を持っている間だけ届ける（戻したあとに残った名づけは捨てる。依頼17）
                self._log("say_drop", str(utt.get("target") or ""), str(utt.get("text", "")), cause=str(utt.get("cause", "")))
                continue
            info["parent_utterance"] = utt
            self._log("say", str(utt.get("target") or ""), str(utt.get("text", "")), cause=str(utt.get("cause", "")))
            break
        return LazyObs(self, self.n_step), 0.0, False, False, info

    def close(self) -> None:
        self._close_log()
        stop = getattr(self.controller, "stop", None)
        if stop is not None:
            try:
                stop()
            except Exception:  # noqa: BLE001  閉じるときの失敗は気にしない
                pass

    # ================================================================ 親・道具から
    def say(self, text: str, target=None, cause: str = "tool") -> None:
        """親に言わせる（次の歩の info["parent_utterance"] に入る）。"""
        self._pending_say.append({"text": str(text), "target": target, "cause": cause})

    def object_sound(self, name: str, wave, cause: str = "parent") -> None:
        """物の音（16kHz の波形）を鳴らす。次の歩の info["object_sound"] に入る。"""
        self._pending_sound.append(np.asarray(wave, dtype=np.float32))
        self._log("sound", name, f"{len(self._pending_sound[-1]) / 16000.0:.2f}s", cause=cause)

    def taro_said(self, text: str) -> None:
        if text:
            self._taro_voice_signal = True
            self._taro_voice_text = str(text)

    def set_eye(self, pan_deg: float, tilt_deg: float, cause: str = "tool") -> None:
        """道具用：目の向きを直接置く（太郎の跳びではない）。"""
        self._orienting.set_angles(pan_deg, tilt_deg)
        self._log("set_eye", "", f"{pan_deg:.2f},{tilt_deg:.2f}", cause=cause)

    def hold(self, oid: str, pos, rot=None, cause: str = "parent") -> None:
        """物を pos（世界の座標、物の見かけの真ん中）へ動かす。次に描くときに TeleportObject で送る。"""
        o = self.objects[oid]
        off = self._centre(o) - vec(o["position"])
        rot = vec(o["rotation"]) if rot is None else np.asarray(rot, dtype=float)
        self._pending_tp[oid] = (np.asarray(pos, dtype=float) - off, rot)
        self.in_hand.add(oid)

    def put_back(self, oid: str, cause: str = "parent") -> None:
        """物を最初の場所・向き（reset のとき）へ戻す。"""
        self._pending_tp[oid] = self.home[oid]
        self.in_hand.discard(oid)
        self._pending_say = [u for u in self._pending_say if u.get("target") != oid
                             or not str(u.get("cause", "")).startswith("parent_name")]
        self._log("put_back", oid, "", cause=cause)

    def word_of(self, oid: str | None) -> str:
        if not oid or oid not in self.objects:
            return ""
        return self.type_word.get(self.objects[oid]["objectType"], "")

    # ================================================================ 目
    @property
    def cam_yaw(self) -> float:
        return self.agent_yaw + float(self._orienting.pan)

    @property
    def cam_pitch(self) -> float:
        return -float(self._orienting.tilt)

    def _cam_rot(self) -> dict:
        return {"x": self.cam_pitch, "y": self.cam_yaw, "z": 0.0}

    def _cam_key(self):
        return (round(self.cam_yaw, 4), round(self.cam_pitch, 4))

    def gaze_dir(self) -> np.ndarray:
        return cam_basis(self.cam_yaw, self.cam_pitch)[0]

    _gaze_dir = gaze_dir

    def _gaze_angle_to(self, oid: str):
        """視線と物の見かけの真ん中のなす角（度。メタデータの正解から）。無い物は None。"""
        if oid not in self.objects:
            return None
        v = self._centre(self.objects[oid]) - self.eye_pos
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return float(np.degrees(np.arccos(np.clip(float(self.gaze_dir() @ (v / n)), -1.0, 1.0))))

    def project(self, point, px: int | None = None):
        return project(point, self.eye_pos, self.cam_yaw, self.cam_pitch, px or self.px, self.fov)

    def sync(self) -> None:
        """控えていた物の動きと目の向きを AI2-THOR へ送る（何も変わっていなければ呼ばない）。"""
        for oid, (pos, rot) in list(self._pending_tp.items()):
            ev = self._call("TeleportObject", objectId=oid, position=xyz(pos), rotation=xyz(rot),
                            forceAction=True, forceKinematic=True)
            if not self._ok(ev):
                self._log("teleport_fail", oid, self.last_error[:80])
        self._pending_tp = {}
        if self._cam_key() != self._sent_cam:
            ev = self._call("UpdateThirdPartyCamera", thirdPartyCameraId=0, position=xyz(self.eye_pos),
                            rotation=self._cam_rot(), fieldOfView=self.fov)
            self._sent_cam = self._cam_key()
            if not self._ok(ev):
                self._log("camera_fail", "", self.last_error[:80])

    def frame(self) -> np.ndarray:
        """今の目の絵（render_px のまま）。"""
        self.sync()
        return np.asarray(self._event.third_party_camera_frames[0])[:, :, :3]

    def render_eye(self) -> np.ndarray:
        img = self.frame()
        if img.shape[0] != self.px:
            from PIL import Image
            img = np.array(Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).resize((self.px, self.px),
                                                                                             Image.BILINEAR))
        return np.ascontiguousarray(img, dtype=np.uint8)

    def seg_ids(self) -> tuple[np.ndarray, list[str]]:
        """色分けの絵を、画素ごとの番号（render_px×render_px、−1＝知らない色）と、番号 → 物の id の一覧にする。"""
        self.sync()
        if self._seg_cache is not None:
            return self._seg_cache
        seg = np.asarray(self._event.third_party_instance_segmentation_frames[0])[:, :, :3].astype(np.int64)
        key = (seg[:, :, 0] << 16) | (seg[:, :, 1] << 8) | seg[:, :, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        ids, lut = [], np.full(len(uniq), -1, dtype=np.int64)
        for j, k in enumerate(uniq):
            oid = self._colors.get(int(k))
            if oid is not None:
                lut[j] = len(ids)
                ids.append(oid)
        self._seg_cache = (lut[inv.reshape(key.shape)], ids)
        return self._seg_cache

    def centre_object(self, radius_deg: float = CENTRE_DEG, only_targets: bool = True) -> str | None:
        """目の真ん中（半径 radius_deg の円）にいちばん多く写っている物の id（円の CENTRE_MIN_FRAC 以上のときだけ）。"""
        lab, ids = self.seg_ids()
        n = lab.shape[0]
        r = focal_px(n, self.fov) * math.tan(math.radians(radius_deg))
        ys, xs = np.ogrid[:n, :n]
        disc = (xs - (n - 1) / 2.0) ** 2 + (ys - (n - 1) / 2.0) ** 2 <= r * r
        vals = lab[disc]
        vals = vals[vals >= 0]
        if vals.size == 0:
            return None
        cnt = np.bincount(vals, minlength=len(ids))
        tset = set(self.targets) if only_targets else None
        order = np.argsort(-cnt)
        for j in order:
            if cnt[j] < CENTRE_MIN_FRAC * disc.sum():
                return None
            if tset is None or ids[j] in tset:
                return ids[j]
        return None

    def object_centres(self, px: int | None = None, only_targets: bool = True, min_pix: int = 12) -> dict:
        """写っている物ごとの {id: (列, 行, 画素の数)}（px の画像の座標。見えている所の真ん中）。"""
        lab, ids = self.seg_ids()
        n = lab.shape[0]
        s = float(px or self.px) / n
        tset = set(self.targets) if only_targets else None
        out = {}
        for j, oid in enumerate(ids):
            if tset is not None and oid not in tset:
                continue
            ys, xs = np.nonzero(lab == j)
            if len(xs) < min_pix:
                continue
            out[oid] = ((float(xs.mean()) + 0.5) * s, (float(ys.mean()) + 0.5) * s, int(len(xs)))
        return out

    # ================================================================ AI2-THOR を呼ぶ
    def _call(self, action: str, **kw):
        ev = self.controller.step(action=action, **kw)
        self.n_calls += 1
        self._take(ev)
        return ev

    def _take(self, ev) -> None:
        self._event = ev
        self._seg_cache = None
        for c, oid in (getattr(ev, "color_to_object_id", None) or {}).items():
            c = tuple(int(x) for x in c)[:3]
            self._colors[(c[0] << 16) | (c[1] << 8) | c[2]] = oid
        objs = ev.metadata.get("objects")
        if objs:
            self.objects = {o["objectId"]: o for o in objs}

    def _ok(self, ev) -> bool:
        ok = bool(ev.metadata.get("lastActionSuccess", True))
        if not ok:
            self.n_fail += 1
            self.last_error = str(ev.metadata.get("errorMessage", ""))
        return ok

    @staticmethod
    def _centre(o) -> np.ndarray:
        bb = o.get("axisAlignedBoundingBox") or {}
        return vec(bb["center"]) if bb.get("center") else vec(o["position"])

    def size_of(self, oid: str) -> float:
        bb = self.objects[oid].get("axisAlignedBoundingBox") or {}
        return float(max(vec(bb["size"]))) if bb.get("size") else 0.1

    # ================================================================ 記録（events.csv だけ。画像は動画の道具が残す）
    def _open_log(self, seed) -> None:
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self._ev_fp = open(os.path.join(self.log_dir, f"events_{self.scene_name}_seed{seed}.csv"), "w",
                           newline="", encoding="utf-8")
        self._ev_w = csv.writer(self._ev_fp)
        self._ev_w.writerow(["t", "step", "event", "object", "cause", "text", "eye_pan", "eye_tilt"])

    def _close_log(self) -> None:
        if self._ev_fp is not None:
            self._ev_fp.close()
            self._ev_fp = self._ev_w = None

    def _log(self, event: str, name: str, text: str = "", cause: str = "") -> None:
        if self._ev_w is None:
            return
        self._ev_w.writerow([f"{self.time:.2f}", self.n_step, event, name, cause, text,
                             f"{self._orienting.pan:.2f}", f"{self._orienting.tilt:.2f}"])
        self._ev_fp.flush()
