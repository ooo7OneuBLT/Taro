# -*- coding: utf-8 -*-
"""benchmarkv2_scene_sy6.xml（スミスとユーの6物体入りの舞台）を作る。

【2026-10-17新設・段階6】スミスとユー（Smith & Yu 2008）の再現の場面用。
make_f4toys_xml.py と同じ流儀で、**機械で**作る（MIMo/ はリポジトリ外の共有物なので、
手で置いたXMLは消えうる。2026-10-15 に f4toys が消えていた）。中身は
benchmarkv2_scene_f4toys.xml に、6つの物 sy_obj0〜sy_obj5（球・遠方・地下）を足すだけ。
形・大きさ・色・位置は環境側（toy_env の pairs、_hold_pairs）が上書きする。

使い方:  python run/scene_tools/make_sy6_xml.py      （既にあれば上書きしない。--force で上書き）
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run.scene_tools.make_f4toys_xml import ANCHOR, DST as F4TOYS, main as make_f4toys  # noqa: E402

DST = os.path.join(os.path.dirname(F4TOYS), "benchmarkv2_scene_sy6.xml")
N_OBJECTS = 6
BODY_NAMES = [f"sy_obj{i}" for i in range(N_OBJECTS)]


def _extra():
    out = []
    for i, nm in enumerate(BODY_NAMES):
        out.append(f"""<body name="{nm}" pos="{6.0 + 0.5 * i} 4 -2">
			<freejoint/>
			<geom type="sphere" material="matgeom" size="0.03" pos="0 0 0" condim="6" solimp="0.99 0.995 0.001 0.5 2" solref="-20000 -20"/>
		</body>

		""")
    return "".join(out)


def main(force=False):
    if os.path.exists(DST) and not force:
        print(f"[sy6] 既にある（上書きしない）: {DST}")
        return DST
    make_f4toys()
    with open(F4TOYS, encoding="utf-8") as f:
        s = f.read()
    if s.count(ANCHOR) != 1:
        raise RuntimeError(f"f4toys XMLの目印が1つではない（{s.count(ANCHOR)}個）: {F4TOYS}")
    if "sy_obj0" in s:
        raise RuntimeError("f4toys XMLに既に sy_obj0 がある（前提が変わった）")
    s = s.replace(ANCHOR, _extra() + ANCHOR)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"[sy6] 書いた: {DST}")
    return DST


if __name__ == "__main__":
    main(force="--force" in sys.argv)
