"""41係の箱。1係1ファイル（ファイル名＝係id）。"""

from __future__ import annotations

import importlib

from .. import wiring
from .base import ENV, Box

__all__ = ["ENV", "Box", "box_class", "make_all"]


def box_class(box_id: str) -> type[Box]:
    module = importlib.import_module(f"{__name__}.{box_id}")
    found = [
        obj for obj in module.__dict__.values()  # 係 id "vars" が組み込みの vars を隠すため
        if isinstance(obj, type) and issubclass(obj, Box) and obj.BOX_ID == box_id
    ]
    if len(found) != 1:
        raise ImportError(f"boxes/{box_id}.py に BOX_ID={box_id!r} の箱がちょうど1つ無い")
    return found[0]


def make_all() -> dict[str, Box]:
    return {box_id: box_class(box_id)() for box_id in wiring.BOX_IDS}
