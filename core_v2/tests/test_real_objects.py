"""段階9 本物の物：一覧の形・切り替え・色の名前・XML（ファイルが無いクラウドでは XML の所だけ飛ばす）。"""

import os

import pytest

from core_v2.world_eye import instances as ins
from core_v2.world_eye import real_objects as ro


@pytest.fixture(autouse=True)
def _back_to_made():
    yield
    ins.use_objects("made")


def test_default_is_made():
    assert ins.OBJECTS == "made"
    assert "src" not in ins.instance("こっぷ", 0)
    with pytest.raises(ValueError):
        ins.use_objects("plastic")


def test_list_covers_20x4_with_allowed_licenses():
    data = ro.load_list()
    keys = {(it["word"], int(it["k"])) for it in data["items"]}
    assert keys == {(w, k) for w in ins.WORDS for k in range(ins.N_INST)}
    for it in data["items"]:
        assert it["source"] in ("gso", "objaverse")
        assert it["license_slug"] in ("cc0", "by", "by-sa"), it
        assert it["author"]
        assert 0.05 <= it["size_m"] <= 0.35
        assert it["color"] in ro.COLOR_NAMES


def test_real_instances_and_colors():
    ins.use_objects("real")
    inst = ins.instance("ばなな", ins.TEST_INST)
    assert inst["src"] == "real" and inst["k"] == ins.TEST_INST and not inst["train"]
    assert len(ins.kind_colors("ばなな")) == 3


def test_color_name():
    assert ro.color_name([[0.9, 0.1, 0.1]] * 5) == "赤"
    assert ro.color_name([[0.1, 0.2, 0.9]] * 5) == "青"
    assert ro.color_name([[0.95, 0.95, 0.95]] * 5) == "白"
    assert ro.color_name([[0.05, 0.05, 0.05]] * 5) == "黒"
    assert ro.color_name([[0.4, 0.25, 0.1]] * 5) == "茶"


def test_xml_loads_in_mujoco():
    ins.use_objects("real")
    inst = ins.instance("こっぷ", 0)
    if not os.path.exists(os.path.join(ro.obj_dir(inst["kind"], 0), "parts.json")):
        pytest.skip("本物の物のファイルが無い（PC で make_real_objects を回す）")
    mujoco = pytest.importorskip("mujoco")
    xml = ("<mujoco><asset>" + "".join(ins.assets_xml(inst)) + "</asset><worldbody>"
           + ins.body_xml(inst, "obj") + "</worldbody></mujoco>")
    assert 'inertia="shell"' in xml
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nmesh >= 1
