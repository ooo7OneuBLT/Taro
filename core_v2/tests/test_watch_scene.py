"""watch_scene（太郎の場面の動画）：grow_names と同じ場面を数回だけ回し、画面なしでコマを描いて動画にできる。"""

import os

from core_v2.tools import watch_scene


def test_renders_a_few_ticks(tmp_path):
    out = str(tmp_path / "v.mp4")
    res = watch_scene.run(["--eye", "table", "--objects", "made", "--from", "birth", "--scenes", "1",
                           "--max-ticks", "5", "--stills", "1", "--out", out])
    assert res["frames"] == 5
    assert os.path.isfile(res["out"]) and os.path.getsize(res["out"]) > 0
    assert len(res["stills"]) == 1 and os.path.isfile(res["stills"][0])
