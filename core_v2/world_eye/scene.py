"""目だけの場面の XML を作る。

寸法と置き方は MIMo の段階2〜6の場面（run/scenes/段階2b_*・段階6_*、run/world/toy_env.py）と同じにする：
* 座標：太郎の体の真ん中が原点、太郎は -x の向きを見る（右が +y、上が +z）
* 目：両目の真ん中 EYE_MID。MIMo の場面の保存した物の場所（state.toy_pos）と、場面の置き方（目から 30cm、
  上へ elev 度・左右へ angle 度）から逆算した（ボール・コップ・妨害物の3つで、ずれ 1.2cm 以内）。
  描くのは左目（MIMo と同じ eye_left）。左目は真ん中から左へ EYE_HALF_IPD
* 床：無地（toy_env の PLAIN_RGBA × floor_dim）。壁：太郎のまわりに4枚、市松模様（toy_env._add_backdrop と
  同じ式：距離 dist、高さ height、下端 z0=-0.2、幅の半分 dist×1.6、模様の繰り返し repeat、texuniform）
* 照明：MIMo の場面の XML（benchmarkv2_scene_*.xml）はこのリポジトリに無いので、頭の明かり（カメラから）＋上からの
  平行光 1つ（影あり）。明るさは、MIMo の左目の画像（G/data/目のテスト/段階2b_*、物の切り抜きつき）と、物の無い所
  （壁と床）の平均の色が近くなるように1度だけ決めた（2026-10-18。MIMo 111,119,134 ／ここ 117,123,140）
* MIMo で見えていた色：ボール（test_object1）とコップ（test_object2）は、MIMo では物に材質が付いていて、場面の
  rgba ではなく材質の色で描かれていた（段階8の nai_body の注意書き。画像でもボールは桃色ではなく茶色）。段階2〜6の
  見え方をそろえるため、場面の rgba がこの2つの既定の値のときは、MIMo の画像で測った色に置き換える（MIMO_SEEN_RGBA。
  MIMo ボール 61,39,30 ／ここ 64,43,36、コップ 115 ／ここ 121）
* 机・親の体：MIMo の場面には無かった（物は宙に浮いて見えた）。段階2〜6の確かめを同じ見え方で回すため、
  既定は出さない。場面の world.eye_world.table / parent_figure を true にすると出る
"""

from __future__ import annotations

import numpy as np

from .objects import body_xml

EYE_MID = np.array([-0.375, 0.009, 0.4306])   # 両目の真ん中（上の説明。MIMo の場面から逆算）
EYE_HALF_IPD = 0.02                           # 両目の間の半分［Tier3・12か月児の瞳の間 約4cm（記憶ベース）］
FORWARD = np.array([-1.0, 0.0, 0.0])          # 最初の視線（水平。逆算で上下のずれは1度以内）
RIGHT = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
EYE_LEFT = EYE_MID - RIGHT * EYE_HALF_IPD

PLAIN_RGBA = (0.55, 0.62, 0.70, 1.0)          # toy_env.PLAIN_RGBA
LIGHT_DIFFUSE = 0.6                           # 上からの平行光の強さ（上の説明）
HEAD_AMBIENT, HEAD_DIFFUSE = 0.3, 0.5         # 頭の明かり（上の説明）
# 場面の rgba → MIMo で実際に見えていた色（上の説明。test_object1・2 だけに使う）
MIMO_SEEN_RGBA = {(0.58, 0.46, 0.5, 1.0): (0.31, 0.19, 0.15, 1.0),
                  (0.95, 0.95, 0.95, 1.0): (0.62, 0.62, 0.62, 1.0)}
HIDDEN_GROUP = 3                              # 隠した物の形を移す組。目のカメラはこの組を描かない


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def build_xml(cfg: dict) -> str:
    """cfg:
      px, fovy           : 目の画像の大きさ（画素）と画角（度）
      floor_dim          : 床の明るさの倍率（toy_env.FLOOR_DIM）
      backdrop           : {"enabled", "dist", "height", "z0", "repeat", "rgb1", "rgb2"}（場面の world.backdrop）
      table, parent_figure : bool
      objects            : [{"name", "kind", "radius", "rgba"}, ...]
      room               : 段階9の部屋（"食事"|"お風呂"|"遊び"、rooms.py）。あれば壁と床をこれに替える
      extra_assets, extra_bodies : 段階9の個体の材質と体の XML（instances.assets_xml・body_xml）
    """
    px = int(cfg.get("px", 224))
    fovy = float(cfg.get("fovy", 60.0))
    fd = float(cfg.get("floor_dim", 1.0))
    floor = list(PLAIN_RGBA)
    floor[:3] = [c * fd for c in floor[:3]]
    b = dict(cfg.get("backdrop") or {})
    parts_asset, parts_world = list(cfg.get("extra_assets") or []), []
    floor_xml = f'<geom name="floor" type="plane" size="5 5 0.1" rgba="{_f(floor)}" contype="0" conaffinity="0"/>'
    if cfg.get("room"):
        # 段階9：部屋（rooms.py）を、市松模様の壁と無地の床の代わりに作る
        from .rooms import room_parts
        ra, rw = room_parts(cfg["room"])
        parts_asset += ra
        parts_world += rw
        floor_xml = ""
    elif b.get("enabled", True):
        dist = float(b.get("dist", 1.2))
        height = float(b.get("height", 2.0))
        z0 = float(b.get("z0", -0.2))
        rep = b.get("repeat", [6, 5])
        rgb1 = b.get("rgb1", [0.52, 0.56, 0.64])
        rgb2 = b.get("rgb2", [0.44, 0.48, 0.58])
        parts_asset.append(f'<texture name="backdrop_tex" type="2d" builtin="checker" width="128" height="128" '
                           f'rgb1="{_f(rgb1)}" rgb2="{_f(rgb2)}"/>')
        parts_asset.append(f'<material name="backdrop" texture="backdrop_tex" texuniform="true" '
                           f'texrepeat="{float(rep[0]):g} {float(rep[1]):g}"/>')
        half_w, half_h = dist * 1.6, height / 2.0
        cz = z0 + half_h
        r2 = 0.70710678
        for nm, pos, quat in (("backdrop_nx", [-dist, 0, cz], [r2, 0, r2, 0]),
                              ("backdrop_px", [dist, 0, cz], [r2, 0, -r2, 0]),
                              ("backdrop_ny", [0, -dist, cz], [r2, -r2, 0, 0]),
                              ("backdrop_py", [0, dist, cz], [r2, r2, 0, 0])):
            parts_world.append(f'<geom name="{nm}" type="plane" size="{half_w:.6g} {half_h:.6g} 0.1" pos="{_f(pos)}" '
                               f'quat="{_f(quat)}" material="backdrop" contype="0" conaffinity="0"/>')
    if cfg.get("table"):
        # 机：物の置き場の下（目から 30cm 先）。天板の上面 0.33m（座った12か月児の胸の高さくらい）［Tier3］
        cx = EYE_MID[0] - 0.30
        top = float(cfg.get("table_top_z", 0.33))
        parts_world.append(f'<geom name="table_top" type="box" pos="{cx:.4g} 0 {top - 0.012:.4g}" '
                           f'size="0.2 0.4 0.012" rgba="0.72 0.55 0.38 1" contype="0" conaffinity="0"/>')
        for sx in (-0.17, 0.17):
            for sy in (-0.37, 0.37):
                parts_world.append(f'<geom type="box" pos="{cx + sx:.4g} {sy:.4g} {(top - 0.024) / 2:.4g}" '
                                   f'size="0.015 0.015 {(top - 0.024) / 2:.4g}" rgba="0.6 0.45 0.3 1" '
                                   f'contype="0" conaffinity="0"/>')
    if cfg.get("parent_figure"):
        # 親：机の向こうに座っている大人（胴・頭・腕）。顔は太郎の方を向く［Tier3・形はモデル上の選択］
        px0 = EYE_MID[0] - 0.75
        skin = "0.93 0.78 0.66 1"
        parts_world += [
            f'<geom name="parent_torso" type="capsule" fromto="{px0:.4g} 0 0.25 {px0:.4g} 0 0.62" size="0.15" '
            f'rgba="0.35 0.5 0.7 1" contype="0" conaffinity="0"/>',
            f'<geom name="parent_head" type="sphere" pos="{px0 + 0.02:.4g} 0 0.9" size="0.1" rgba="{skin}" '
            f'contype="0" conaffinity="0"/>',
            f'<geom name="parent_hair" type="sphere" pos="{px0 - 0.02:.4g} 0 0.93" size="0.1" '
            f'rgba="0.15 0.1 0.08 1" contype="0" conaffinity="0"/>',
            f'<geom name="parent_arm_l" type="capsule" fromto="{px0:.4g} -0.17 0.62 {px0 + 0.25:.4g} -0.2 0.4" '
            f'size="0.045" rgba="0.35 0.5 0.7 1" contype="0" conaffinity="0"/>',
            f'<geom name="parent_arm_r" type="capsule" fromto="{px0:.4g} 0.17 0.62 {px0 + 0.25:.4g} 0.2 0.4" '
            f'size="0.045" rgba="0.35 0.5 0.7 1" contype="0" conaffinity="0"/>',
        ]
    objs = "\n".join([body_xml(o["name"], o["kind"], o["radius"], o["rgba"]) for o in cfg.get("objects", [])]
                     + list(cfg.get("extra_bodies") or []))   # 段階9：個体（instances.body_xml）
    # 目：頭の位置（左目）に、左右の回転（pan、上向きの軸）→上下の回転（tilt、太郎の右向きの軸）の順の関節。
    # カメラは -z を見るので、カメラの z＝太郎の後ろ（+x）、x＝右（+y）、y＝上（+z）
    eye = (f'<body name="eye_head" pos="{_f(EYE_LEFT)}">\n'
           f'      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>\n'
           f'      <joint name="eye_pan" type="hinge" axis="0 0 1"/>\n'
           f'      <body name="eye_tilt_body">\n'
           f'        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>\n'
           f'        <joint name="eye_tilt" type="hinge" axis="0 1 0"/>\n'
           f'        <camera name="eye_left" pos="0 0 0" xyaxes="0 1 0 0 0 1" fovy="{fovy:g}"/>\n'
           f'      </body>\n    </body>')
    xml = f"""<mujoco model="taro_eye_world">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.01"/>
  <visual>
    <headlight ambient="{HEAD_AMBIENT} {HEAD_AMBIENT} {HEAD_AMBIENT}" diffuse="{HEAD_DIFFUSE} {HEAD_DIFFUSE} {HEAD_DIFFUSE}" specular="0.5 0.5 0.5"/>
    <quality shadowsize="2048"/>
    <global offwidth="{max(px, 640)}" offheight="{max(px, 480)}"/>
  </visual>
  <asset>
    {chr(10).join(parts_asset)}
  </asset>
  <worldbody>
    <light name="top" directional="true" pos="0 0 3" dir="0 0 -1" diffuse="{LIGHT_DIFFUSE} {LIGHT_DIFFUSE} {LIGHT_DIFFUSE}" castshadow="true"/>
    {floor_xml}
    {chr(10).join(parts_world)}
    {eye}
{objs}
  </worldbody>
</mujoco>
"""
    return xml
