"""脳の保存と読み込み。

旧太郎では重みだけを保存していたので、どのコードで読めばよいか分からなくなった。
core_v2 では重みと一緒に「コードの版」「未コミットの変更があったか」「配線表の指紋」
「荷札の長さ」を同じファイルに書く。配線表の指紋か荷札の長さが今のコードと違えば、
読み込みをエラーで止める（同じミスを2回したら機械で防ぐ）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch

from .brain import Brain
from .formats import TAG_DIM
from .wiring import wiring_hash

REPO = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1


class WiringMismatch(RuntimeError):
    """保存したときと配線・書式が違う。古い重みを今の配線に読むと黙って壊れる。"""


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def code_version() -> dict[str, Any]:
    rev = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"git_rev": rev, "git_dirty": None if status is None else bool(status)}


def save(brain: Brain, path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "format_version": FORMAT_VERSION,
        **code_version(),
        "wiring_hash": wiring_hash(),
        "tag_dim": TAG_DIM,
        "t": brain.t,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "state_dict": brain.state_dict()}, path)
    return meta


def load(brain: Brain, path: str | Path) -> dict[str, Any]:
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = blob["meta"]
    if meta.get("tag_dim") != TAG_DIM:
        raise WiringMismatch(f"荷札の長さが違う（保存 {meta.get('tag_dim')}・今 {TAG_DIM}）")
    if meta.get("wiring_hash") != wiring_hash():
        raise WiringMismatch(
            "配線表が保存したときと違う。"
            f"保存時のコード {meta.get('git_rev')} を checkout して読むか、配線を戻すこと"
        )
    brain.load_state_dict(blob["state_dict"])
    brain.t = meta.get("t", 0)
    return meta
