# -*- coding: utf-8 -*-
"""太郎のコード全部の索引を自動で作り、コマンドラインから探せるようにする。

【なぜ・2026-09-11】「実はこの機能もう作られていた」という事故が繰り返し起きる。
同じ日に3回起きた：
  ・`f105_eye_force_probe.py` を新しく書いた → `e_eye_power_probe.py` が同じことを測っていた
  ・`action` を直接読んで計器を壊した → `read_joint_command` が既にあった
  ・「advisor は存在しません」と回答した → 最初から存在していた
共通点は「探す前に作った／探す前に『無い』と言った」。

`doc/道具一覧.md`（2026-08-05作成）が**まさにこの事故のために**作られていて、
「作ったら1項目追記する」という運用ルールまで書いてある。それでも守られていない。
実測した網羅率（2026-09-11）：**418本中149本＝36%**。F/scripts は 98本中0本。

`.git/hooks/pre-commit` の冒頭にあるとおり「手作業の約束は守られない。機械にやらせる」。
**人が書き足す欄をゼロにして、腐らない索引にする。**

`doc/道具一覧.md` は捨てない。役割を分ける：
  ・この索引     … 全部載る。浅い。腐らない
  ・道具一覧.md  … 一部だけ。深い（動作確認の記録つき）。腐ってもよい

設計：`doc/設計_道具索引の自動生成_2026-09-11.md`

使い方:
    python run/tools/index_tools.py                 索引を作り直す
    python run/tools/index_tools.py --check         索引が最新かだけ返す（0=最新）
    python run/tools/index_tools.py --find 眼球 力  既存の道具を探す
    python run/tools/index_tools.py --find-file F/scripts/new.py
                                                    これから足すファイルに似たものを探す
"""
import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
SCAN_DIRS = ["run/tools", "run/plugins", "run/scene_tools", "run/viewer_tools",
             "taro_core/src", "E/scripts", "F/scripts"]
SKIP_PARTS = ("__pycache__", ".venv", "MIMo", ".git", "node_modules")
OUT_JSON = "doc/道具索引.json"
OUT_MD = "doc/道具索引.md"


def _iter_py():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in SKIP_PARTS]
            if any(p in r for p in SKIP_PARTS):
                continue
            for f in files:
                if f.endswith(".py") and not f.startswith("__"):
                    yield os.path.relpath(os.path.join(r, f), ROOT).replace("\\", "/")


def _first_line(text):
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            return ln[:160]
    return ""


def _entry(rel):
    p = os.path.join(ROOT, rel)
    try:
        src = io.open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return None
    doc, defs = "", []
    try:
        tree = ast.parse(src)
        doc = _first_line(ast.get_docstring(tree) or "")
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.append(n.name)
    except SyntaxError:
        pass
    if not doc:
        # docstring が無ければ先頭のコメント行を使う
        for ln in src.splitlines()[:15]:
            t = ln.strip()
            if t.startswith("#") and not t.startswith("#!") and "coding" not in t:
                doc = t.lstrip("# ").strip()[:160]
                break
    goal = rel.split("/")[0] if rel.split("/")[0] in ("E", "F", "M") else ""
    return {"path": rel, "doc": doc, "defs": defs,
            "lines": src.count("\n") + 1, "goal": goal}


def _last_commits(paths):
    """まとめて1回の git log で最終更新日を取る（1本ずつ呼ぶと遅い）。"""
    out = {}
    try:
        r = subprocess.run(["git", "log", "--name-only", "--format=%x00%ad",
                            "--date=short", "--", *paths],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
        cur = ""
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("\x00"):
                cur = ln[1:].strip()
            elif ln.strip() and cur:
                out.setdefault(ln.strip().replace("\\", "/"), cur)
    except Exception:
        pass
    return out


def build():
    rels = sorted(_iter_py())
    entries = [e for e in (_entry(r) for r in rels) if e]
    dates = _last_commits([e["path"] for e in entries])
    for e in entries:
        e["last_commit"] = dates.get(e["path"], "")
    return entries


def write(entries):
    io.open(os.path.join(ROOT, OUT_JSON), "w", encoding="utf-8").write(
        json.dumps(entries, ensure_ascii=False, indent=1))
    by = {}
    for e in entries:
        by.setdefault(os.path.dirname(e["path"]), []).append(e)
    L = ["# 道具索引（自動生成・手で書き足さないこと）", "",
         "`python run/tools/index_tools.py` で作り直す。",
         "探すときは `python run/tools/index_tools.py --find <語> <語>`。", "",
         "詳しい取扱説明書は `doc/道具一覧.md`（手書き・一部だけ・深い）。",
         "こちらは**網羅**が役目（浅いが腐らない）。なぜ要るかは "
         "`doc/設計_道具索引の自動生成_2026-09-11.md`。", "",
         "**合計 %d 本**" % len(entries), ""]
    for d in sorted(by):
        L.append("## `%s`（%d本）" % (d, len(by[d])))
        L.append("")
        L.append("| ファイル | 1行説明 | 行数 | 最終更新 |")
        L.append("|---|---|---|---|")
        for e in sorted(by[d], key=lambda x: x["path"]):
            doc = (e["doc"] or "—").replace("|", "\\|")
            L.append("| `%s` | %s | %d | %s |"
                     % (os.path.basename(e["path"]), doc, e["lines"],
                        e["last_commit"] or "—"))
        L.append("")
    io.open(os.path.join(ROOT, OUT_MD), "w", encoding="utf-8").write("\n".join(L))


def _tokens(s):
    """語に割る。

    【2026-09-11・直し】最初の版は2つ壊れていた：
      ・`f105_eye_force_probe` を**1語のまま**扱っていた（`_` で割っていない）。
        そのため `e_eye_power_probe` と共通語が0個になり、似た道具を見つけられなかった。
      ・日本語で助詞まで拾っていた（`眼球の`・`を直接測る`）。意味のある語と
        一致しなくなる。形態素解析器は使わず、**漢字とカタカナの連続だけ**を取る
        （ひらがなはほぼ文法なので落とす）。
    """
    s = (s or "").lower()
    out = set()
    for w in re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", s):
        for part in w.split("_"):
            if len(part) >= 3 and not part.isdigit():
                out.add(part)
    out |= {w for w in re.findall(r"[一-龥]{2,}", s)}
    out |= {w for w in re.findall(r"[ァ-ヴー]{3,}", s)}
    return out


def _score(entry, words):
    hay = " ".join([entry["path"], entry["doc"], " ".join(entry["defs"])]).lower()
    hit = 0.0
    for w in words:
        w = w.lower()
        if not w:
            continue
        if w in hay:
            hit += 1.0
        elif len(w) >= 3 and any(w[i:i + 2] in hay for i in range(len(w) - 1)):
            # 日本語は2文字の部分一致も弱く数える
            hit += 0.3
    return hit / max(len(words), 1)


def load():
    p = os.path.join(ROOT, OUT_JSON)
    if not os.path.exists(p):
        e = build()
        write(e)
        return e
    return json.load(io.open(p, encoding="utf-8"))


def find(words, limit=8, threshold=0.0, exclude=None):
    out = []
    for e in load():
        if exclude and e["path"] == exclude:
            continue
        s = _score(e, words)
        if s > threshold:
            out.append((s, e))
    out.sort(key=lambda x: -x[0])
    return out[:limit]


STOP = {"py", "def", "self", "main", "import", "する", "ための", "これ", "もの",
        "ように", "ことを", "ている", "という", "とき", "など"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="太郎のコードの索引を作る・探す")
    ap.add_argument("--check", action="store_true", help="索引が最新かだけ返す（0=最新）")
    ap.add_argument("--find", nargs="+", help="語で既存の道具を探す")
    ap.add_argument("--find-file", help="これから足すファイルに似たものを探す")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()

    if a.check:
        cur = build()
        try:
            old = json.load(io.open(os.path.join(ROOT, OUT_JSON), encoding="utf-8"))
        except Exception:
            old = None
        same = (old is not None and
                {x["path"] for x in old} == {x["path"] for x in cur})
        print("最新（%d本）" % len(cur) if same
              else "古い（%d本 → %d本）" % (len(old) if old else 0, len(cur)))
        return 0 if same else 1

    if a.find:
        for s, e in find(a.find, a.limit, a.threshold):
            print("  %.2f  %-46s %s" % (s, e["path"], e["doc"][:80]))
        return 0

    if a.find_file:
        rel = a.find_file.replace("\\", "/")
        full = os.path.join(ROOT, rel)
        e = _entry(rel) if os.path.exists(full) else None
        words = _tokens(os.path.basename(rel))
        if e:
            words |= _tokens(e["doc"]) | _tokens(" ".join(e["defs"]))
        words = sorted(w for w in words if w not in STOP)
        if not words:
            return 0
        for s, x in find(words, a.limit, a.threshold, exclude=rel):
            print("  %.2f  %-46s %s" % (s, x["path"], x["doc"][:80]))
        return 0

    e = build()
    write(e)
    print("索引を作り直した: %d 本 → %s / %s" % (len(e), OUT_JSON, OUT_MD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
