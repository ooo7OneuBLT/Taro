# -*- coding: utf-8 -*-
"""親の体の下見・段2（2026-09-03）：関節を計算で解く操り人形。

段1（f52）との違いは、肘や手の座標を人が書かないこと。
  ・命令  「コップを指さして、太郎を見て」
  ・翻訳  右手=指さす(的=コップ)／頭=向ける(的=太郎)／左手=持つ(コップ)
  ・計算  手の目標位置 → 肩・肘の角度を逆運動学（2リンクの解析解）で逆算
  ・指    角度の組（指さし=人差し指0度・他90度／持つ=全部70度／開く=10度）

親の関節は MuJoCo の物理には入れない。Python 側で骨格を解いて、
見た目だけ mocap（動かせる飾り）で置く。太郎の姿勢データの長さは変わらないので、
保存済みの姿勢・過去の実験はそのまま読める。

    .venv/Scripts/python.exe F/scripts/f53_parent_ik_probe.py
出力: F/logs/F2-53_親のIK/図_親のIK_下見.png
"""
import os, sys, io, json
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod

OUT = "F/logs/F2-53_親のIK"
SKIN, SHIRT = ".95 .80 .70 1", ".35 .55 .80 1"
FING = ["index", "middle", "ring", "little"]
# 体の寸法[m]（成人の目安。厳密な文献値ではない）
D = dict(upper=0.27, fore=0.24, palm=0.078, seg1=0.042, seg2=0.036, thumb=0.055,
         sh_half=0.16, head_r=0.09, hip_z=0.035, torso=0.45, head_up=0.60, lean=38.0)
# 手の形＝指の曲げ角の組[度]（0=まっすぐ / 90=握り込む）。親指は別枠。
SHAPES = {"point": ([2, 78, 82, 84], 62), "hold": ([58, 62, 64, 64], 50), "open": ([6, 6, 6, 8], 20)}


def body_xml():
    """親の体を mocap（動かせる飾り）として書き出す。接触なし・物理に参加しない。"""
    parts = []
    # 胴は腰（原点）から肩まで。ローカル+z が背骨の向き＝前傾ぶん傾けて置く
    parts.append('<body name="p_torso" mocap="true" pos="0 0 -9">'
                 '<geom type="capsule" size="0.100" fromto="0 0 0.10 0 0 0.45" rgba="{sh}" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" size="0.120" fromto="0 0 0.10 0 0 0.13" rgba="{sh}" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" size="0.046" fromto="0 0 0.45 0 0 0.53" rgba="{sk}" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" size="0.050" fromto="0 -0.16 0.45 0 0.16 0.45" rgba="{sh}" contype="0" conaffinity="0"/>'
                 '</body>'.format(sh=SHIRT, sk=SKIN))
    parts.append('<body name="p_legs" mocap="true" pos="0 0 -9">'
                 '<geom type="capsule" size="0.075" fromto="0.03 -0.12 0 0.34 -0.10 0" rgba="{sh}" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" size="0.075" fromto="0.03 0.12 0 0.34 0.10 0" rgba="{sh}" contype="0" conaffinity="0"/>'
                 '</body>'.format(sh=SHIRT))
    parts.append('<body name="p_head" mocap="true" pos="0 0 -9">'
                 '<geom type="sphere" size="{r:.3f}" rgba="{sk}" contype="0" conaffinity="0"/>'
                 '<geom type="sphere" size="0.013" pos="0.083 0.032 0.02" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>'
                 '<geom type="sphere" size="0.013" pos="0.083 -0.032 0.02" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" size="0.009" fromto="0.085 0 -0.005 0.098 0 -0.025" rgba=".92 .72 .62 1" contype="0" conaffinity="0"/>'
                 '<geom type="ellipsoid" size="0.095 0.098 0.06" pos="-0.02 0 0.045" rgba="0.25 0.15 0.1 1" contype="0" conaffinity="0"/>'
                 '</body>'.format(r=D["head_r"], sk=SKIN))
    for sd in ("l", "r"):
        parts.append('<body name="p_upper_{s}" mocap="true" pos="0 0 -9">'
                     '<geom type="capsule" size="0.035" fromto="0 0 -{h:.3f} 0 0 {h:.3f}" rgba="{c}" contype="0" conaffinity="0"/>'
                     '</body>'.format(s=sd, h=D["upper"] / 2, c=SHIRT))
        parts.append('<body name="p_fore_{s}" mocap="true" pos="0 0 -9">'
                     '<geom type="capsule" size="0.030" fromto="0 0 -{h:.3f} 0 0 {h:.3f}" rgba="{c}" contype="0" conaffinity="0"/>'
                     '</body>'.format(s=sd, h=D["fore"] / 2, c=SKIN))
        parts.append('<body name="p_palm_{s}" mocap="true" pos="0 0 -9">'
                     '<geom type="box" size="{h:.3f} 0.036 0.011" rgba="{c}" contype="0" conaffinity="0"/>'
                     '</body>'.format(s=sd, h=D["palm"] / 2, c=SKIN))
        for f in FING:
            for seg, ln in (("1", D["seg1"]), ("2", D["seg2"])):
                parts.append('<body name="p_{f}{g}_{s}" mocap="true" pos="0 0 -9">'
                             '<geom type="capsule" size="0.0095" fromto="0 0 -{h:.4f} 0 0 {h:.4f}" rgba="{c}" contype="0" conaffinity="0"/>'
                             '</body>'.format(f=f, g=seg, s=sd, h=ln / 2, c=SKIN))
        parts.append('<body name="p_thumb_{s}" mocap="true" pos="0 0 -9">'
                     '<geom type="capsule" size="0.0105" fromto="0 0 -{h:.4f} 0 0 {h:.4f}" rgba="{c}" contype="0" conaffinity="0"/>'
                     '</body>'.format(s=sd, h=D["thumb"] / 2, c=SKIN))
    return "\n".join(parts)


def quat_from_z(z_dir, ref=None):
    z = np.asarray(z_dir, float); z = z / np.linalg.norm(z)
    r = np.asarray(ref if ref is not None else ([1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0]), float)
    x = r - z * (r @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0, 0, 1.0]) - z * z[2]
    x = x / np.linalg.norm(x); y = np.cross(z, x)
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).ravel()); return q


def mat_from_xz(x_dir, z_hint):
    """ローカル+x を x_dir に、+z をできるだけ z_hint に向ける回転行列。"""
    x = np.asarray(x_dir, float); x = x / np.linalg.norm(x)
    z = np.asarray(z_hint, float); z = z - x * (z @ x)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0, 0, 1.0]) - x * x[2]
    z = z / np.linalg.norm(z); y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def quat_of(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


class ParentBody:
    """親の骨格。関節角は持たず、手の目標位置から肘を逆算する（2リンクの解析解）。"""

    def __init__(self, hip_xy, fwd, right, lean_deg=None):
        """hip_xy＝腰の水平位置。親は床に座り、前傾して子に顔を近づける。"""
        self.fwd = np.asarray(fwd, float) / np.linalg.norm(fwd)
        self.right = np.asarray(right, float) / np.linalg.norm(right)
        self.hip = np.array([hip_xy[0], hip_xy[1], D["hip_z"]], float)
        la = np.radians(D["lean"] if lean_deg is None else lean_deg)
        self.axis = np.sin(la) * self.fwd + np.cos(la) * np.array([0, 0, 1.0])
        self.torso = self.hip
        sh_h = self.hip + self.axis * D["torso"]
        self.head = self.hip + self.axis * D["head_up"] + self.fwd * 0.02
        self.base = self.head
        self.shoulder = {"r": sh_h - self.right * D["sh_half"], "l": sh_h + self.right * D["sh_half"]}

    def solve_arm(self, side, hand_pos):
        """肩から hand_pos（手首）へ。肘は下に垂れる側を選ぶ。届かなければ縮める。"""
        sh = self.shoulder[side]; L1, L2 = D["upper"], D["fore"]
        v = np.asarray(hand_pos, float) - sh; dist = float(np.linalg.norm(v))
        far = dist > L1 + L2 - 1e-4
        if far:
            v = v / dist * (L1 + L2 - 1e-3); dist = L1 + L2 - 1e-3
        if dist < abs(L1 - L2) + 1e-4:
            v = v / max(dist, 1e-9) * (abs(L1 - L2) + 1e-3); dist = abs(L1 - L2) + 1e-3
        u = v / dist
        a = (dist ** 2 + L1 ** 2 - L2 ** 2) / (2 * dist)
        h = float(np.sqrt(max(L1 ** 2 - a ** 2, 0.0)))
        down = np.array([0, 0, -1.0]); perp = down - u * (down @ u)
        if np.linalg.norm(perp) < 1e-6:
            perp = (self.right if side == "r" else -self.right)
            perp = perp - u * (perp @ u)
        perp = perp / np.linalg.norm(perp)
        return sh + u * a + perp * h, sh + v, far

    def parts(self, arms, head_at):
        """arms = {"r": (手首の目標, 指の向き先, 手の形), ...} → mocap に書く一覧。"""
        out = {"p_torso": (self.torso, quat_from_z(self.axis, ref=self.fwd)),
               "p_legs": (np.array([self.hip[0], self.hip[1], 0.075]),
                          quat_of(mat_from_xz(self.fwd, [0, 0, 1.0]))),
               "p_head": (self.head, quat_of(mat_from_xz(np.asarray(head_at, float) - self.head, [0, 0, 1.0])))}
        for side, (wrist_goal, aim, shape) in arms.items():
            elbow, wrist, _ = self.solve_arm(side, wrist_goal)
            sh = self.shoulder[side]
            out["p_upper_%s" % side] = ((sh + elbow) / 2, quat_from_z(elbow - sh))
            out["p_fore_%s" % side] = ((elbow + wrist) / 2, quat_from_z(wrist - elbow))
            aimv = np.asarray(aim, float) - wrist
            up_hint = np.array([0, 0, 1.0]) if shape != "hold" else (wrist - elbow)
            out.update(hand_parts(wrist, mat_from_xz(aimv, up_hint), shape, side))
        return out


def hand_parts(wrist, R, shape, side):
    """手首の位置と手の姿勢 R（+x=指の向き、+z=手の甲）から、手のひらと指の置き場所を出す。
    指は2節で、曲げ角が大きいほど手のひら側（-z）へ折り畳まれる。"""
    out = {}
    palm_c = wrist + R[:, 0] * (D["palm"] / 2)
    out["p_palm_%s" % side] = (palm_c, quat_of(R))
    bends, th = SHAPES[shape]
    for f, y, b in zip(FING, [0.028, 0.009, -0.009, -0.028], bends):
        root = palm_c + R[:, 0] * (D["palm"] / 2) + R[:, 1] * y
        for seg, (frac, ln) in enumerate(((0.60, D["seg1"]), (1.45, D["seg2"]))):
            a_ = np.radians(b * frac)
            dirv = np.cos(a_) * R[:, 0] - np.sin(a_) * R[:, 2]
            out["p_%s%d_%s" % (f, seg + 1, side)] = (root + dirv * (ln / 2), quat_from_z(dirv))
            root = root + dirv * ln
    sgn = 1.0 if side == "r" else -1.0
    root = palm_c + R[:, 1] * (0.040 * sgn) - R[:, 0] * 0.010
    a_ = np.radians(th)
    dirv = np.cos(a_) * R[:, 0] + np.sin(a_) * (R[:, 1] * sgn * 0.35 - R[:, 2] * 0.85)
    dirv = dirv / np.linalg.norm(dirv)
    out["p_thumb_%s" % side] = (root + dirv * (D["thumb"] / 2), quat_from_z(dirv))
    return out


def parent_geom_ids(m):
    """親の体（p_ で始まる body）に属する geom の一覧。"""
    out = []
    for gi in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        if nm.startswith("p_"):
            out.append(gi)
    return out


def fit_hand(m, d, body, arms, head_at, hidden, side, targets):
    """side の手が物を突き抜けていたら、当たらない位置へ置き直す。
    後ろに引く・上げる・横にずらす、の組み合わせを試し、ずれの小さいものを採る。
    「持つ」手は物に触れてよい（1.5cmまでのめり込みは掴んでいるとみなす）。"""
    wrist0, aim, shape = arms[side]
    aimv = np.asarray(aim, float) - wrist0
    fwdv = aimv / np.linalg.norm(aimv)
    upv = np.array([0, 0, 1.0]) - fwdv * fwdv[2]
    upv = upv / max(np.linalg.norm(upv), 1e-9)
    sidev = np.cross(fwdv, upv)
    best = None
    backs = (0.0, -0.03, 0.03, -0.06, 0.06, 0.10, 0.15) if shape == "hold" else (0.0, 0.03, 0.06, 0.10, 0.15)
    for back in backs:
        for up in (0.0, 0.04, -0.04, 0.09, 0.15):
            for off in (0.0, 0.07, -0.07, 0.14, -0.14):
                cost = abs(back) + abs(up) * 1.2 + abs(off) * 1.5
                if best is not None and cost >= best[0]:
                    continue
                trial = dict(arms)
                trial[side] = (wrist0 - fwdv * back + upv * up + sidev * off, aim, shape)
                apply(m, d, body.parts(trial, head_at), hidden=hidden)
                mujoco.mj_forward(m, d)
                near = [t[2] for t in check_penetration(m, d, targets=targets, margin=0.03)
                        if t[0].endswith("_%s" % side)]
                mind = min(near) if near else 9.9
                # 持つ手は「触れている」こと。指さす手は「どこにも触れない」こと
                ok = (-0.022 <= mind <= 0.008) if shape == "hold" else (mind >= 0.0)
                if ok:
                    best = (cost, trial[side])
    if best is None:
        return arms, False
    out = dict(arms); out[side] = best[1]
    return out, True


def check_penetration(m, d, pg=None, margin=0.0, targets=None):
    """親の体が他の物にめり込んでいないか測る。物理に入れていない体でも距離は測れる。
    返り値: [(親の部位, 相手, 符号付き距離[m]), ...] めり込み（距離 < margin）のみ。"""
    pg = parent_geom_ids(m) if pg is None else pg
    ps = set(pg); bad = []
    for a in pg:
        if d.geom_xpos[a][2] < -1.0:
            continue                                      # 地下に隠してある部位は見ない
        na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[a])) or "?"
        for b in (range(m.ngeom) if targets is None else targets):
            if b in ps or d.geom_xpos[b][2] < -1.0:
                continue                                  # 親の体どうし・地下の物は見ない
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.05, None)
            if dist < margin:
                nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or                      (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[b])) or "?")
                bad.append((na, nb, float(dist)))
    return bad


def apply(m, d, parts, hidden=()):
    for name, (pos, quat) in parts.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); mi = int(m.body_mocapid[bid])
        d.mocap_pos[mi] = pos; d.mocap_quat[mi] = quat
    for name in hidden:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); mi = int(m.body_mocapid[bid])
        d.mocap_pos[mi] = [0, 0, -9]


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    src = io.open(sc["world"]["xml"], encoding="utf-8").read()
    xml = "MIMo/mimoEnv/assets/f53_parent_ik.xml"
    io.open(xml, "w", encoding="utf-8", newline="\n").write(src.replace("</worldbody>", body_xml() + "\n</worldbody>"))
    sc["name"] = "座位_12ヶ月_F2-53_親のIK_下見_%s" % G.DATE
    sc["note"] = "親の骨格を計算で解く下見。学習には使わない。"
    sc["world"]["xml"] = xml
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 900)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 700)
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]
    fwd_h = fwd.copy(); fwd_h[2] = 0; fwd_h = fwd_h / np.linalg.norm(fwd_h)
    right_h = Rc[:, 0].copy(); right_h[2] = 0; right_h = right_h / np.linalg.norm(right_h)
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    obj = eye + fwd * dist; obj[2] = max(obj[2], 0.06)
    slot = "toy5"; ang = sc["world"]["parent_labeling"]["present_angles"][slot]
    body = ParentBody(eye + fwd_h * 0.80 + right_h * 0.20, -fwd_h, right_h)
    # 当たりを見る相手＝提示中の物・太郎の体・床（親の体どうしは見ない）
    TARGETS = [g for g in range(m.ngeom)
               if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").startswith("p_")]

    # ---- 命令 → 腕の割り当て（この翻訳を将来LLMが書く）----
    def scene_point():          # 「コップを指さして、太郎を見て」
        left = obj - np.array([0, 0, 0.03]) - body.fwd * 0.10
        # 指は太郎の視線を遮らない位置から。物の斜め上・親側の横から差す
        wrist_r = obj + right_h * 0.16 + np.array([0, 0, 0.13]) + fwd_h * 0.10
        return {"l": (left, obj, "hold"), "r": (wrist_r, obj, "point")}, eye

    def scene_show():           # 「コップを見せて」
        return {"r": (obj - np.array([0, 0, 0.03]) - body.fwd * 0.10, obj, "hold")}, obj

    def scene_point_taro():     # 「コップを持って、太郎を指さして」
        left = obj - np.array([0, 0, 0.03]) - body.fwd * 0.10
        v = eye - body.shoulder["r"]; v = v / np.linalg.norm(v)
        wrist_r = body.shoulder["r"] + v * 0.26
        return {"r": (wrist_r, eye, "point"), "l": (left, obj, "hold")}, eye

    LEFT_ALL = ["p_upper_l", "p_fore_l", "p_palm_l", "p_thumb_l"] + ["p_%s%d_l" % (f, g) for f in FING for g in (1, 2)]
    tiles = []
    for title, fn in [("コップを指さして、太郎を見て", scene_point),
                      ("コップを見せて", scene_show),
                      ("コップを持って、太郎を指さして", scene_point_taro)]:
        arms, head_at = fn()
        a = qadr[slot]
        d.qpos[a:a + 3] = obj; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])
        hidden = () if "l" in arms else LEFT_ALL
        for sd in ("l", "r"):                      # 手を置き直す（持つ手を先に決めてから指さす手）
            if sd in arms:
                arms, ok = fit_hand(m, d, body, arms, head_at, hidden, sd, TARGETS)
                if not ok:
                    print("   ! %s手は当たらない置き方が見つからなかった" % ("左" if sd == "l" else "右"))
        apply(m, d, body.parts(arms, head_at), hidden=hidden)
        mujoco.mj_forward(m, d)
        holds = {"_%s" % sd for sd, v in arms.items() if v[2] == "hold"}
        bad = [t for t in check_penetration(m, d, targets=TARGETS)
               if not (any(t[0].endswith(h) for h in holds) and t[2] > -0.015)]
        obj_bad = [t for t in bad if "floor" not in t[1] and "ground" not in t[1]]
        flr_bad = [t for t in bad if t not in obj_bad]
        if obj_bad:
            w = min(obj_bad, key=lambda t: t[2])
            note = "×物とめり込み %d 箇所（最悪 %s↔%s %.1fcm）" % (len(obj_bad), w[0], w[1], w[2] * 100)
        else:
            note = "○物とのめり込みなし"
        if flr_bad:
            note += " ／ 床に %.0fcm 沈み" % (-100 * min(t[2] for t in flr_bad))
        print(note, "|", title)
        for t in sorted(obj_bad, key=lambda t: t[2])[:6]:
            print("    %s ↔ %s  %.1fcm" % t[:2] + "" if False else "    %s ↔ %s  %+.1fcm" % (t[0], t[1], t[2] * 100))
        ren = mujoco.Renderer(m, height=224, width=224); ren.update_scene(d, camera="eye_left")
        im_eye = ren.render().copy(); ren.close()
        ren = mujoco.Renderer(m, height=430, width=430)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (eye + body.head) / 2 + np.array([0, 0, -0.06])
        cam.distance = 1.60
        cam.azimuth = np.degrees(np.arctan2(fwd_h[1], fwd_h[0])) + 68.0   # 二人を横から
        cam.elevation = -10
        ren.update_scene(d, camera=cam); im3 = ren.render().copy(); ren.close()
        tiles.append((title, im3, im_eye, note))
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 15)
    f2 = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 13)
    W = 430 + 236
    canvas = Image.new("RGB", (W * 3, 300), (255, 255, 255)); dr = ImageDraw.Draw(canvas)
    for i, (title, im3, im_eye, note) in enumerate(tiles):
        x = i * W
        dr.text((x + 6, 4), "「%s」" % title, fill=(0, 0, 120), font=font)
        canvas.paste(Image.fromarray(im3), (x + 2, 28))
        canvas.paste(Image.fromarray(im_eye).resize((224, 224)), (x + 436, 28))
        dr.text((x + 436, 278), "太郎の左目（学習に使う画像）", fill=(0, 0, 0), font=f2)
        dr.text((x + 6, 278), note, fill=(180, 0, 0) if note.startswith("×") else (0, 120, 0), font=f2)
    p = OUT + "/図_親のIK_下見.png"
    canvas.save(p); print("図:", p)
    env.close()


if __name__ == "__main__":
    main()
