"""育ちの段の写し（tools/growth.py）：保存して読むと係の中身が戻る・条件が無いと止まる・作り直しの判定。"""
import pytest

from core_v2.tools import growth
from core_v2.tools.voice_bed import make_boxes


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(growth, "LEDGER", str(tmp_path / "育ち帳.json"))
    monkeypatch.setattr(growth, "SNAP_DIR", str(tmp_path / "育ち"))
    return tmp_path


def test_save_load_roundtrip(ledger):
    a = make_boxes(0)
    a["listen"].alone.update({"ぼおる": 5})
    a["voice"].lexicon.update({"ぼおる": 3})
    with pytest.raises(ValueError):
        growth.save(a, 2, "x", conditions={"world": "bed"}, out_dir=growth.SNAP_DIR)
    meta = growth.save(a, 2, "段2_種0_24か月", conditions={"world": "bed", "eye": "bed_random_tags", "seed": 0},
                       month=24.0, end=True, out_dir=growth.SNAP_DIR)
    assert meta["git_rev"] is not None and meta["wiring_hash"]
    b = make_boxes(1)
    growth.load(b, "段2_種0_24か月")
    assert b["listen"].alone["ぼおる"] == 5 and b["voice"].lexicon["ぼおる"] == 3
    assert growth.latest(2, seed=0)["name"] == "段2_種0_24か月"
    assert growth.stale("段2_種0_24か月")["regrow_from"] is None


def test_stale_points_to_the_phase(ledger):
    growth.save(make_boxes(0), 3, "段3", conditions={"world": "eye", "eye": "dinov2", "seed": 0}, end=True,
                out_dir=growth.SNAP_DIR)
    rows = growth.read_ledger()
    rows[0]["fingerprints"]["core_v2/boxes/listen.py"] = "old"       # 耳が変わったことにする
    rows[0]["fingerprints"]["core_v2/world_eye"] = "old"             # 世界も
    growth._write_ledger(rows)
    s = growth.stale("段3")
    assert s["regrow_from"] == 1 and "core_v2/boxes/listen.py" in s["changed"]
    with pytest.raises(LookupError):
        growth.latest(3)
