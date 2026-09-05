# -*- coding: utf-8 -*-
"""親の体（関節つき）のモデル定義（2026-09-03）。

段2（f53）までは手首の座標を人が書いていた。段3では書かない。
親を「関節を持つ体」として作り、mink（MuJoCo用の逆運動学）に

  ・人差し指の先の向きが対象を通る（LookAtTask）
  ・顔の前方が対象を通る（LookAtTask）
  ・手のひらの把持点が物の表面にある（FrameTask）
  ・関節は人間の可動域に収まる（ConfigurationLimit）
  ・どの部位も他の物とぶつからない（CollisionAvoidanceLimit）

を条件として渡し、関節角度を解かせる。手や肘の位置は解の側から出てくる。

このモジュールは体の形と関節だけを定義する（解くのは f54_parent_pose.py）。
"""
import numpy as np

SKIN, SHIRT, HAIR, DARK = ".95 .80 .70 1", ".35 .55 .80 1", "0.25 0.15 0.1 1", "0.1 0.1 0.1 1"

# 体の寸法[m]（成人の目安）
D = dict(upper=0.27, fore=0.24, palm=0.078, seg1=0.042, seg2=0.036, thumb=0.055,
         sh_half=0.16, head_r=0.09, hip_z=0.035, torso=0.45, head_up=0.60)
FING = ["index", "middle", "ring", "little"]
# 手の形＝指の曲げ角[度]（0=まっすぐ / 90=握り込む）。親指は別枠
SHAPES = {"point": ([2, 78, 82, 84], 62), "hold": ([58, 62, 64, 64], 50), "open": ([6, 6, 6, 8], 20)}

# 人間の可動域[度]（成人の目安。左右対称に使う）
RANGE = dict(spine=(-5, 55), head_z=(-70, 70), head_y=(-40, 45),
             sh_y=(-45, 170), sh_x=(-150, 25), sh_z=(-70, 70),
             elbow=(0, 145), wrist_y=(-60, 60), wrist_z=(-25, 25))


def _finger_geoms(side):
    """手のひら body の中に置く指。関節は持たせず、形（角度）はモデル生成時に固定する。
    IKで動かすのは腕と手首だけ。指はその形のまま運ばれる。"""
    return ""      # 形ごとの指は build() 側で shape を見て入れる


def _hand_geoms(shape, side):
    """手のひらと指の geom。手のローカル系は +x=指の向き、+z=手の甲。"""
    g = ['<geom name="palm_%s" type="box" size="%.3f 0.036 0.011" pos="%.3f 0 0" rgba="%s"/>'
         % (side, D["palm"] / 2, D["palm"] / 2, SKIN)]
    bends, th = SHAPES[shape]
    for f, y, b in zip(FING, [0.028, 0.009, -0.009, -0.028], bends):
        root = np.array([D["palm"], y, 0.0])
        for seg, (frac, ln) in enumerate(((0.60, D["seg1"]), (1.45, D["seg2"]))):
            a = np.radians(b * frac)
            dv = np.array([np.cos(a), 0.0, -np.sin(a)])
            tip = root + dv * ln
            g.append('<geom name="%s%d_%s" type="capsule" size="0.0095" fromto="%.4f %.4f %.4f %.4f %.4f %.4f" rgba="%s"/>'
                     % (f, seg + 1, side, root[0], root[1], root[2], tip[0], tip[1], tip[2], SKIN))
            root = tip
    sgn = 1.0 if side == "r" else -1.0
    a = np.radians(th)
    root = np.array([D["palm"] / 2 - 0.010, 0.040 * sgn, 0.0])
    dv = np.array([np.cos(a), np.sin(a) * sgn * 0.35, -np.sin(a) * 0.85])
    dv = dv / np.linalg.norm(dv)
    tip = root + dv * D["thumb"]
    g.append('<geom name="thumb_%s" type="capsule" size="0.0105" fromto="%.4f %.4f %.4f %.4f %.4f %.4f" rgba="%s"/>'
             % (side, root[0], root[1], root[2], tip[0], tip[1], tip[2], SKIN))
    return "\n      ".join(g)


def _arm(side, shape):
    s = side
    sgn = -1.0 if side == "r" else 1.0            # 右手は太郎から見て左（-y 側）
    r = RANGE
    return """
    <body name="p_upper_{s}" pos="0 {sy:.3f} {sz:.3f}">
      <joint name="sh_{s}_y" type="hinge" axis="0 -1 0" range="{shy0} {shy1}"/>
      <joint name="sh_{s}_x" type="hinge" axis="1 0 0" range="{shx0} {shx1}"/>
      <joint name="sh_{s}_z" type="hinge" axis="0 0 1" range="{shz0} {shz1}"/>
      <geom name="upper_{s}" type="capsule" size="0.035" fromto="0 0 0 0 0 -{up:.3f}" rgba="{shirt}"/>
      <body name="p_fore_{s}" pos="0 0 -{up:.3f}">
        <joint name="el_{s}" type="hinge" axis="0 -1 0" range="{el0} {el1}"/>
        <geom name="fore_{s}" type="capsule" size="0.030" fromto="0 0 0 0 0 -{fo:.3f}" rgba="{skin}"/>
        <body name="p_palm_{s}" pos="0 0 -{fo:.3f}" euler="0 -90 0">
          <joint name="wr_{s}_y" type="hinge" axis="0 -1 0" range="{wy0} {wy1}"/>
          <joint name="wr_{s}_z" type="hinge" axis="0 0 1" range="{wz0} {wz1}"/>
          {hand}
          <site name="tip_{s}" pos="{tipx:.3f} 0.028 0" size="0.004"/>
          <site name="grip_{s}" pos="{gripx:.3f} 0 -0.020" size="0.004"/>
        </body>
      </body>
    </body>""".format(
        s=s, sy=sgn * D["sh_half"], sz=D["torso"], up=D["upper"], fo=D["fore"],
        shirt=SHIRT, skin=SKIN, hand=_hand_geoms(shape, s),
        tipx=D["palm"] + D["seg1"] + D["seg2"] * 0.5, gripx=D["palm"] * 0.6,
        shy0=r["sh_y"][0], shy1=r["sh_y"][1], shx0=r["sh_x"][0], shx1=r["sh_x"][1],
        shz0=r["sh_z"][0], shz1=r["sh_z"][1], el0=r["elbow"][0], el1=r["elbow"][1],
        wy0=r["wrist_y"][0], wy1=r["wrist_y"][1], wz0=r["wrist_z"][0], wz1=r["wrist_z"][1])


def build_xml(shape_r="point", shape_l="hold", obstacles=()):
    """親の体（腰は世界に固定）＋よけたい相手の代理形状。
    obstacles = [(名前, 形, 大きさ), ...] 位置は実行時に mocap で入れる。"""
    obs = "\n  ".join(
        '<body name="obs_%s" mocap="true" pos="0 0 -9"><geom name="obs_%s" type="%s" size="%s" rgba="0.8 0.3 0.3 0.35"/></body>'
        % (nm, nm, tp, sz) for nm, tp, sz in obstacles)
    return """<mujoco model="parent">
 <compiler angle="degree"/>
 <option gravity="0 0 0"/>
 <visual><global offwidth="1200" offheight="900"/><headlight diffuse="0.65 0.65 0.65"/></visual>
 <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0.9 0.94 1" rgb2="0.75 0.8 0.9" width="8" height="8"/>
  <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.82" rgb2="0.72 0.72 0.75" width="80" height="80"/>
  <material name="matgrid" texture="grid" texrepeat="6 6"/></asset>
 <worldbody>
  <light pos="0.5 -1 1.6" dir="-0.2 0.5 -1"/>
  <geom name="floor" type="plane" size="4 4 0.1" material="matgrid"/>
  <body name="p_pelvis" pos="0 0 {hip:.3f}">
   <geom name="pelvis" type="capsule" size="0.120" fromto="0 0 0 0 0 0.04" rgba="{shirt}"/>
   <geom name="leg_r" type="capsule" size="0.075" fromto="0.03 -0.12 -0.005 0.34 -0.10 -0.005" rgba="{shirt}"/>
   <geom name="leg_l" type="capsule" size="0.075" fromto="0.03 0.12 -0.005 0.34 0.10 -0.005" rgba="{shirt}"/>
   <body name="p_torso" pos="0 0 0.02">
    <joint name="spine" type="hinge" axis="0 -1 0" range="{sp0} {sp1}"/>
    <geom name="torso" type="capsule" size="0.100" fromto="0 0 0.06 0 0 {top:.3f}" rgba="{shirt}"/>
    <geom name="shoulders" type="capsule" size="0.050" fromto="0 -0.16 {top:.3f} 0 0.16 {top:.3f}" rgba="{shirt}"/>
    <geom name="neck" type="capsule" size="0.046" fromto="0 0 {top:.3f} 0 0 {neck:.3f}" rgba="{skin}"/>
    <body name="p_head" pos="0 0 {hu:.3f}">
     <joint name="head_z" type="hinge" axis="0 0 1" range="{hz0} {hz1}"/>
     <joint name="head_y" type="hinge" axis="0 -1 0" range="{hy0} {hy1}"/>
     <geom name="head" type="sphere" size="{hr:.3f}" rgba="{skin}"/>
     <geom name="eye_r" type="sphere" size="0.013" pos="0.083 -0.032 0.02" rgba="{dark}"/>
     <geom name="eye_l" type="sphere" size="0.013" pos="0.083 0.032 0.02" rgba="{dark}"/>
     <geom name="nose" type="capsule" size="0.009" fromto="0.085 0 -0.005 0.098 0 -0.025" rgba=".92 .72 .62 1"/>
     <geom name="hair" type="ellipsoid" size="0.095 0.098 0.06" pos="-0.02 0 0.045" rgba="{hair}"/>
     <site name="gaze" pos="0.09 0 0" size="0.004"/>
    </body>{arm_r}{arm_l}
   </body>
  </body>
  {obs}
 </worldbody>
</mujoco>""".format(hip=D["hip_z"], shirt=SHIRT, skin=SKIN, hair=HAIR, dark=DARK,
                    top=D["torso"], neck=D["torso"] + 0.08, hu=D["head_up"] - 0.02, hr=D["head_r"],
                    sp0=RANGE["spine"][0], sp1=RANGE["spine"][1],
                    hz0=RANGE["head_z"][0], hz1=RANGE["head_z"][1],
                    hy0=RANGE["head_y"][0], hy1=RANGE["head_y"][1],
                    arm_r=_arm("r", shape_r), arm_l=_arm("l", shape_l), obs=obs)

# 部位ごとの geom（IK用モデルと見た目用で同じ定義を使う。ずれると絵と当たり判定が食い違う）
def part_geoms(shape_r="point", shape_l="hold"):
    return {
        "p_pelvis": ('<geom name="pelvis" type="capsule" size="0.120" fromto="0 0 0 0 0 0.04" rgba="{sh}"/>'
                     '<geom name="leg_r" type="capsule" size="0.075" fromto="0.03 -0.12 -0.005 0.34 -0.10 -0.005" rgba="{sh}"/>'
                     '<geom name="leg_l" type="capsule" size="0.075" fromto="0.03 0.12 -0.005 0.34 0.10 -0.005" rgba="{sh}"/>'
                     ).format(sh=SHIRT),
        "p_torso": ('<geom name="torso" type="capsule" size="0.100" fromto="0 0 0.06 0 0 {t:.3f}" rgba="{sh}"/>'
                    '<geom name="shoulders" type="capsule" size="0.050" fromto="0 -0.16 {t:.3f} 0 0.16 {t:.3f}" rgba="{sh}"/>'
                    '<geom name="neck" type="capsule" size="0.046" fromto="0 0 {t:.3f} 0 0 {n:.3f}" rgba="{sk}"/>'
                    ).format(t=D["torso"], n=D["torso"] + 0.08, sh=SHIRT, sk=SKIN),
        "p_head": ('<geom name="head" type="sphere" size="{r:.3f}" rgba="{sk}"/>'
                   '<geom name="eye_r" type="sphere" size="0.013" pos="0.083 -0.032 0.02" rgba="{dk}"/>'
                   '<geom name="eye_l" type="sphere" size="0.013" pos="0.083 0.032 0.02" rgba="{dk}"/>'
                   '<geom name="nose" type="capsule" size="0.009" fromto="0.085 0 -0.005 0.098 0 -0.025" rgba=".92 .72 .62 1"/>'
                   '<geom name="hair" type="ellipsoid" size="0.095 0.098 0.06" pos="-0.02 0 0.045" rgba="{ha}"/>'
                   ).format(r=D["head_r"], sk=SKIN, dk=DARK, ha=HAIR),
        "p_upper_r": '<geom name="upper_r" type="capsule" size="0.035" fromto="0 0 0 0 0 -%.3f" rgba="%s"/>' % (D["upper"], SHIRT),
        "p_upper_l": '<geom name="upper_l" type="capsule" size="0.035" fromto="0 0 0 0 0 -%.3f" rgba="%s"/>' % (D["upper"], SHIRT),
        "p_fore_r": '<geom name="fore_r" type="capsule" size="0.030" fromto="0 0 0 0 0 -%.3f" rgba="%s"/>' % (D["fore"], SKIN),
        "p_fore_l": '<geom name="fore_l" type="capsule" size="0.030" fromto="0 0 0 0 0 -%.3f" rgba="%s"/>' % (D["fore"], SKIN),
        "p_palm_r": _hand_geoms(shape_r, "r"),
        "p_palm_l": _hand_geoms(shape_l, "l"),
    }


def mocap_xml(shape_r="point", shape_l="hold", prefix="q"):
    """見た目だけの親（太郎の世界へ入れる用）。IK用モデルと同じ geom を mocap body で置く。
    太郎の姿勢データ（qpos）の長さは変わらないので、保存済みモデル・過去の実験に影響しない。"""
    out = []
    for body, geoms in part_geoms(shape_r, shape_l).items():
        g = geoms.replace('name="', 'name="%s_' % prefix)
        out.append('<body name="%s_%s" mocap="true" pos="0 0 -9">%s</body>' % (prefix, body, g))
    return chr(10).join(out)
