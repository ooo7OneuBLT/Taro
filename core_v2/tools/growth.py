"""育ちの段の写し（スナップショット）と育ち帳。0から育て直さず、前の段の終わりの太郎から続けるための道具。

設計：`太郎/2026-10-18_育ちの段_設計.md`（ユーザーが読んでから中身を決める。ここは保存と読み込みの汎用部分だけ）。

* 段（PHASES）：0 生まれつき → 1 耳 → 2 喃語（声、0〜24か月）→ 3 名前（名づけ・スミスとユー）→ 4 名前を言う → 5 ない
* 写し1つ＝ある段の終わり（または途中の月齢）の、育つ係（GROWN_BOXES）の中身を全部。係の state_dict（声の係などの
  数え上げは get_extra_state で中に入る）をそのまま保存する。試験場の係にも、41係の脳（Brain.boxes）にも読める
* 写しには必ず書く：git の版（と未コミットの変更の有無）、配線の指紋、係ごとのソースの指紋、作った条件
  （世界 eye|mimo|bed、目の作り方、種、月齢、元にした写し）
* 育ち帳（core_v2/data/育ち帳.json）：写しの一覧。実験は「段3の終わりの太郎」を名前で引く（latest()）
* 作り直しの判定（stale()）：写しを作ったときと今とで、係のソース・世界のソースの指紋を比べ、変わった物から
  「どの段から育て直すか」を出す（耳の係が変わった→段1から）

試験場（voice_bed の Bed）の続きから回すための丸ごとの写し（乱数・親の状態を含む pickle）は、voice_bed の
_save_ckpt の形で別に置ける（save(..., resume=bed)）。voice_bed の中身は変えない。

使い方：
  python -m core_v2.tools.growth list            … 育ち帳を表示（どの写しが今のコードで古くなったかも）
  python -m core_v2.tools.growth stale 名前       … その写しのあと、どの段から育て直すか
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LEDGER = os.path.join(REPO, "core_v2", "data", "育ち帳.json")
SNAP_DIR = os.path.join(REPO, "core_v2", "data", "育ち")

PHASES = {
    0: "生まれつき（目・反射・物の名簿。育たない）",
    1: "耳（聞き分ける係が親の話しかけで育つ）",
    2: "喃語（声の係・言う語の係が 0〜24か月で育つ）",
    3: "名前（意味の辞書が名づけ・スミスとユーで育つ）",
    4: "名前を言う（声と意味の辞書のつながり）",
    5: "ない（驚きと「ない」）",
}
# 育つ係 → その係が最初に育つ段。この係のソースが変わったら、この段から育て直す
GROWN_BOXES = {"listen": 1, "voice": 2, "wordplan": 2, "semantic": 3, "surprise": 5}
# 係のほかに、変わったら育て直す物 → 段（目と世界は名前の段から使う。口の表は喃語から）
OTHER_SOURCES = {
    "core_v2/tools/parent_speech.py": 1,
    "core_v2/mouth.py": 2,
    "core_v2/tools/voice_bed.py": 2,
    "core_v2/tagging.py": 1,
    "core_v2/formats.py": 1,
    "core_v2/wiring.py": 1,
    "core_v2/boxes/eyein.py": 3,
    "core_v2/floor_objects.py": 3,
    "core_v2/boxes/objfile.py": 3,
    "core_v2/world_eye": 3,
    "run/world": 3,
}


# ---------------------------------------------------------------- 指紋
def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _hash_path(rel: str) -> str | None:
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        return None
    h = hashlib.sha1()
    files = [p] if os.path.isfile(p) else sorted(
        os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs if f.endswith(".py"))
    for f in files:
        h.update(os.path.relpath(f, REPO).encode())
        with open(f, "rb") as fp:
            h.update(fp.read())
    return h.hexdigest()[:12]


def fingerprints() -> dict:
    out = {f"core_v2/boxes/{b}.py": _hash_path(f"core_v2/boxes/{b}.py") for b in GROWN_BOXES}
    out.update({k: _hash_path(k) for k in OTHER_SOURCES})
    return out


def source_phase(rel: str) -> int:
    if rel.startswith("core_v2/boxes/") and rel[len("core_v2/boxes/"):-3] in GROWN_BOXES:
        return GROWN_BOXES[rel[len("core_v2/boxes/"):-3]]
    return OTHER_SOURCES[rel]


# ---------------------------------------------------------------- 育ち帳
def read_ledger() -> list[dict]:
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return json.load(f)


def _write_ledger(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    tmp = LEDGER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LEDGER)


def find(name: str) -> dict:
    for r in read_ledger():
        if r["name"] == name:
            return r
    raise KeyError(f"育ち帳に無い写し：{name}")


def latest(phase: int, seed: int | None = None, world: str | None = None, fresh_only: bool = True) -> dict:
    """段 phase の終わりの写しのうち、いちばん新しいもの（end=True）。fresh_only なら今のコードで古くなっていないものだけ。"""
    rows = [r for r in read_ledger() if r["phase"] == phase and r.get("end")
            and (seed is None or r["conditions"].get("seed") == seed)
            and (world is None or r["conditions"].get("world") == world)]
    if fresh_only:
        rows = [r for r in rows if stale(r)["regrow_from"] is None]
    if not rows:
        raise LookupError(f"段{phase}の終わりの写しが無い（種={seed}、世界={world}、古いものを除く={fresh_only}）")
    return max(rows, key=lambda r: r["created"])


# ---------------------------------------------------------------- 保存と読み込み
def save(boxes, phase: int, name: str, *, conditions: dict, month: float | None = None, end: bool = False,
         parent: str | None = None, note: str = "", resume=None, out_dir: str = SNAP_DIR) -> dict:
    """育つ係（boxes にあるものだけ）の中身を写し、育ち帳に1行足す。boxes は Brain.boxes でも係の dict でもよい。
    conditions には少なくとも world（eye|mimo|bed）・eye（目の作り方、例 dinov2 / bed_random_tags）・seed を書く。"""
    for k in ("world", "eye", "seed"):
        if k not in conditions:
            raise ValueError(f"写しの条件に {k} が無い（world・eye・seed は必ず書く）")
    if any(r["name"] == name for r in read_ledger()):
        raise ValueError(f"同じ名前の写しがもう育ち帳にある：{name}")
    from ..save import code_version
    from ..wiring import wiring_hash
    states = {b: boxes[b].state_dict() for b in GROWN_BOXES if b in boxes}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pt")
    meta = {"name": name, "phase": phase, "phase_name": PHASES[phase], "month": month, "end": bool(end),
            "parent": parent, "note": note, "conditions": dict(conditions), "boxes": sorted(states),
            "created": datetime.datetime.now().isoformat(timespec="seconds"), **code_version(),
            "wiring_hash": wiring_hash(), "fingerprints": fingerprints(),
            "file": os.path.relpath(path, REPO)}
    torch.save({"meta": meta, "states": states}, path)
    if resume is not None:          # 試験場の続きから回すための丸ごと（voice_bed の _save_ckpt と同じ形）
        from .voice_bed import _save_ckpt
        rp = os.path.join(out_dir, f"{name}_続き.pkl")
        _save_ckpt(rp, {"bed": resume, "first": {}, "last_probe": getattr(resume, "day", 0.0), "wall_s": 0.0,
                        "torch_rng": torch.get_rng_state()})
        meta["resume_file"] = os.path.relpath(rp, REPO)
    meta["size_kb"] = round(os.path.getsize(path) / 1024, 1)
    rows = read_ledger()
    rows.append(meta)
    _write_ledger(rows)
    return meta


def load(boxes, name_or_row, only: list[str] | None = None) -> dict:
    """写しを係へ読む（boxes にある係だけ。only で絞れる）。配線の指紋が違えば止める。"""
    from ..wiring import wiring_hash
    row = find(name_or_row) if isinstance(name_or_row, str) else name_or_row
    blob = torch.load(os.path.join(REPO, row["file"]), map_location="cpu", weights_only=False)
    if blob["meta"]["wiring_hash"] != wiring_hash():
        raise RuntimeError(f"写し {row['name']} は配線が違うときに作られた（作り直しの判定：stale）")
    for b, sd in blob["states"].items():
        if b in boxes and (only is None or b in only):
            boxes[b].load_state_dict(sd)
    return blob["meta"]


def stale(row) -> dict:
    """写しを作ったあとに変わったソースと、どの段から育て直すか（None なら作り直し不要）。"""
    row = find(row) if isinstance(row, str) else row
    now = fingerprints()
    changed = [k for k, v in row.get("fingerprints", {}).items() if now.get(k) != v]
    phases = [source_phase(k) for k in changed if source_phase(k) <= row["phase"]]
    return {"changed": changed, "regrow_from": min(phases) if phases else None}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "list":
        for r in read_ledger():
            s = stale(r)
            flag = "使える" if s["regrow_from"] is None else f"古い（段{s['regrow_from']}から育て直す）"
            print(f"{r['name']}\t段{r['phase']}{'の終わり' if r.get('end') else ''}\t{r.get('month')}か月\t"
                  f"{r['conditions']}\t{r.get('git_rev', '')[:8]}\t{flag}")
    elif argv[0] == "stale":
        print(json.dumps(stale(argv[1]), ensure_ascii=False))


if __name__ == "__main__":
    main()
