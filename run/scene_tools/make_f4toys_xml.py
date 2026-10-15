# -*- coding: utf-8 -*-
"""benchmarkv2_scene_f4toys.xml（test_object3/4 入りの舞台）を作り直す。

【2026-10-15新設・段階2】toy3/toy4 を使う場面は `world.xml` にこのファイルを指定する
（scene_io.py の "xml" キーのコメント参照）。ところが MIMo/ はリポジトリ外の共有物
（ジャンクション・gitignore）で、このXMLは手元から消えていた（2026-10-15に確認、
参照している場面 `座位_12ヶ月_1おもちゃ_F2-7_くつだけ_2026-08-25.json` も開けない状態）。
共有XML（benchmarkv2_scene.xml）は変えない方針なので、そこから**機械で**作り直せる
ようにこの道具を置く。中身は共有XMLに test_object3/4（toy2と同じ球・遠方）を足すだけ。
形・大きさ・色・位置は環境側（toy_env._configure_toy_geom・_set_anchor）が上書きする。

使い方:  python run/scene_tools/make_f4toys_xml.py      （既にあれば上書きしない。--force で上書き）
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ASSETS = os.path.join(ROOT, "MIMo", "mimoEnv", "assets")
SRC = os.path.join(ASSETS, "benchmarkv2_scene.xml")
DST = os.path.join(ASSETS, "benchmarkv2_scene_f4toys.xml")

ANCHOR = '<!-- The location and orientation of the base model can be set using this body -->'
EXTRA = """<body name="test_object3" pos="5 3 -2">
			<freejoint/>
			<geom type="sphere" material="matgeom" size="0.1" pos="0 0 0" condim="6" solimp="0.99 0.995 0.001 0.5 2" solref="-20000 -20"/>
		</body>

		<body name="test_object4" pos="5.5 3 -2">
			<freejoint/>
			<geom type="sphere" material="matgeom" size="0.1" pos="0 0 0" condim="6" solimp="0.99 0.995 0.001 0.5 2" solref="-20000 -20"/>
		</body>

		"""


def main(force=False):
    if os.path.exists(DST) and not force:
        print(f"[f4toys] 既にある（上書きしない）: {DST}")
        return DST
    with open(SRC, encoding="utf-8") as f:
        s = f.read()
    if s.count(ANCHOR) != 1:
        raise RuntimeError(f"共有XMLの目印が1つではない（{s.count(ANCHOR)}個）: {SRC}")
    if "test_object3" in s:
        raise RuntimeError("共有XMLに既に test_object3 がある（前提が変わった）")
    s = s.replace(ANCHOR, EXTRA + ANCHOR)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(s)
    print(f"[f4toys] 書いた: {DST}")
    return DST


if __name__ == "__main__":
    main(force="--force" in sys.argv)
