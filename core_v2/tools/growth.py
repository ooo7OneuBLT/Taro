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
  python -m core_v2.tools.growth since 版         … git の版（例 93329a2）で作った写しがあったとしたら、段ごとに使えるか・どの段から育て直すか
                                                   （育ち帳に写しが無くても、コードの変更がどの段に響くかを見られる）
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
# 環境変数 TARO_GROWTH_DIR を置くと、育ち帳と写しをその下に置く（クラウドの短い確かめで、本物の育ち帳を汚さないため）
_ROOT = os.environ.get("TARO_GROWTH_DIR")
LEDGER = os.path.join(_ROOT, "育ち帳.json") if _ROOT else os.path.join(REPO, "core_v2", "data", "育ち帳.json")
SNAP_DIR = os.path.join(_ROOT, "育ち") if _ROOT else os.path.join(REPO, "core_v2", "data", "育ち")

PHASES = {
    0: "生まれつき（目・反射・物の名簿。育たない）",
    1: "耳（聞き分ける係が親の話しかけで育つ）",
    2: "喃語（声の係・言う語の係が 0〜24か月で育つ）",
    3: "名前（意味の辞書が名づけ・スミスとユーで育つ）",
    4: "名前を言う（声と意味の辞書のつながり）",
    5: "ない（驚きと「ない」）",
}
# 育つ係 → その係が最初に育つ段。この係のソースが変わったら、この段から育て直す
GROWN_BOXES = {"listen": 1, "voice": 2, "wordplan": 2, "semantic": 3, "kind": 3, "surprise": 5}
# 係のほかに、変わったら育て直す物 → 段（目と世界は名前の段から使う。口の表は喃語から）
OTHER_SOURCES = {
    "core_v2/tools/parent_speech.py": 1,
    "core_v2/mouth.py": 2,
    "core_v2/tools/voice_bed.py": 2,
    "core_v2/tagging.py": 1,
    "core_v2/formats.py": 1,
    "core_v2/boxes/eyein.py": 3,
    "core_v2/floor_objects.py": 3,
    "core_v2/boxes/objfile.py": 3,
    "core_v2/boxes/soundin.py": 3,
    "core_v2/world_eye": 3,
    "run/world": 3,
}


# 配線の指紋は段ごとに分ける（段階9、仕様書 付録「配線の変更は、新しい係が初めて育つ段から」）：矢印1本の段＝両端の係の段の
# 大きい方。係の段は GROWN_BOXES、それ以外は BOX_PHASE（耳の入口は段1、声の出口は段2、目・名簿・音の入口などは段3）。
# だから、段3で初めて使う係（種類の層）へ耳から矢印を足しても、段1・2の写しはそのまま使える
BOX_PHASE_OTHER = {"earin": 1, "voiceout": 2}
DEFAULT_BOX_PHASE = 3


def box_phase(bid: str) -> int:
    return GROWN_BOXES.get(bid, BOX_PHASE_OTHER.get(bid, DEFAULT_BOX_PHASE))


def wiring_fingerprints() -> dict:
    from ..wiring import EDGES
    by = {}
    for e in EDGES:
        by.setdefault(max(box_phase(e.src), box_phase(e.dst)), []).append([e.src, e.dst, e.fmt.value, e.size])
    out = {}
    for p in sorted(PHASES):
        blob = json.dumps(sorted(by.get(p, [])), ensure_ascii=False).encode("utf-8")
        out[f"wiring@{p}"] = hashlib.sha1(blob).hexdigest()[:12]
    return out


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
        # 区切りは「\」にそろえる（今までの写しは全部 Windows で作られた。Linux（WSL）で数えても同じ指紋になるように）
        h.update(os.path.relpath(f, REPO).replace("/", "\\").encode())
        with open(f, "rb") as fp:
            h.update(fp.read())
    return h.hexdigest()[:12]


def fingerprints() -> dict:
    out = {f"core_v2/boxes/{b}.py": _hash_path(f"core_v2/boxes/{b}.py") for b in GROWN_BOXES}
    out.update({k: _hash_path(k) for k in OTHER_SOURCES})
    out.update(wiring_fingerprints())
    return out


def source_phase(rel: str) -> int:
    if rel.startswith("wiring@"):
        return int(rel[len("wiring@"):])
    if rel == "core_v2/wiring.py":          # 段階9より前の写し（配線を1つの指紋で持っていた）
        return 1
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


def start_point(phase: int, seed: int | None = None, world: str | None = None) -> dict | None:
    """段 phase を育て始める元：段 phase−1, phase−2, … の終わりの写しのうち、いちばん上の段の、いちばん新しい物
    （今のコードで古くなっていない物だけ）。world を渡すと、その段が段3以上のときだけ世界も合わせる。無ければ None（生まれたままから）。"""
    for p in range(phase - 1, 0, -1):
        try:
            return latest(p, seed=seed, world=world if p >= 3 else None)
        except LookupError:
            continue
    return None


def free_name(base: str) -> str:
    """育ち帳にまだ無い名前（base、あれば base_2, base_3 …）。"""
    names = {r["name"] for r in read_ledger()}
    if base not in names:
        return base
    k = 2
    while f"{base}_{k}" in names:
        k += 1
    return f"{base}_{k}"


def month_of(row: dict) -> float | None:
    """写しの月齢（無ければ元にした写しをさかのぼる）。"""
    while row is not None:
        if row.get("month") is not None:
            return float(row["month"])
        row = find(row["parent"]) if row.get("parent") else None
    return None


def open_start(boxes, spec: str | None, phase: int, seed: int | None = None) -> dict | None:
    """道具の --from：None なら何もしない。"auto" なら段 phase の元（start_point、世界は eye）、それ以外は写しの名前。
    読んだ写しの行を返す（無ければ止める）。"""
    if not spec:
        return None
    row = start_point(phase, seed=seed, world="eye") if spec == "auto" else find(spec)
    if row is None:
        raise LookupError(f"段{phase}の元にする写しが育ち帳に無い（種={seed}）")
    load(boxes, row)
    return row


# ---------------------------------------------------------------- 保存と読み込み
def save(boxes, phase: int, name: str, *, conditions: dict, month: float | None = None, end: bool = False,
         parent: str | None = None, note: str = "", resume=None, out_dir: str | None = None) -> dict:
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
    out_dir = out_dir or SNAP_DIR
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
    # 育ち帳の場所は Windows で書くと「\」区切り。Linux（WSL）でも読めるように区切りをそろえる
    blob = torch.load(os.path.join(REPO, *row["file"].replace("\\", "/").split("/")), map_location="cpu",
                      weights_only=False)
    fp = blob["meta"].get("fingerprints", {})
    if any(k.startswith("wiring@") for k in fp):
        now = wiring_fingerprints()
        bad = [k for k, v in fp.items() if k.startswith("wiring@") and source_phase(k) <= row["phase"] and now.get(k) != v]
    else:
        bad = [] if blob["meta"]["wiring_hash"] == wiring_hash() else ["wiring_hash"]
    if bad:
        raise RuntimeError(f"写し {row['name']} は、その段までの配線が違うときに作られた（{bad}。作り直しの判定：stale）")
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


def fingerprints_at(rev: str) -> dict:
    """git の版 rev のソースの指紋（今の growth.py の決め方で数える）。rev の core_v2 と run/world を一時の場所へ出して数える。"""
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = [p for p in ("core_v2", "run/world") if _git("cat-file", "-e", f"{rev}:{p}") is not None]
        arch = subprocess.run(["git", "archive", rev, *paths], cwd=REPO, capture_output=True, check=True).stdout
        # Python の tarfile で開く（Windows の tar は日本語のファイル名を展開できずに止まる。ソースの指紋に要るのは .py だけ）
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(arch)) as tf:
            tf.extractall(tmp, members=[m for m in tf.getmembers() if m.isdir() or m.name.endswith(".py")], filter="data")
        shutil.copy(__file__, os.path.join(tmp, "core_v2", "tools", "growth.py"))
        code = "import json; from core_v2.tools import growth; print(json.dumps(growth.fingerprints()))"
        env = {**os.environ, "PYTHONPATH": tmp}
        r = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, capture_output=True, text=True, check=True)
        return json.loads(r.stdout.strip().splitlines()[-1])


def since(rev: str) -> dict:
    """版 rev で段ごとの写しを作っていたとしたら、今のコードで、段ごとにどこから育て直すか（None＝そのまま使える）。"""
    old = fingerprints_at(rev)
    now = fingerprints()
    changed = sorted(k for k in set(old) | set(now) if old.get(k) != now.get(k))
    out = {}
    for p in sorted(PHASES):
        phases = [source_phase(k) for k in changed if source_phase(k) <= p]
        out[p] = min(phases) if phases else None
    return {"rev": rev, "changed": changed, "regrow_from_by_phase": out}


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
    elif argv[0] == "since":
        r = since(argv[1])
        print("変わったソース：", ", ".join(r["changed"]) or "なし")
        for p, f in r["regrow_from_by_phase"].items():
            print(f"段{p}（{PHASES[p].split('（')[0]}）の写し：" + ("そのまま使える" if f is None else f"段{f}から育て直す"))


if __name__ == "__main__":
    main()
