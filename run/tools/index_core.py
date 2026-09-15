# -*- coding: utf-8 -*-
"""太郎の本体（taro_core/src）の構造を、コードから自動で書き起こす。

【なぜ・2026-09-15】ユーザーの指摘：
    「あなたは太郎のすべての構造を把握できていなくて、一部のもの
      （しかも過去の古い記録の場合もある）で判断している」

実際そうだった。手で書いた構造の文書は、例外なく腐っていた：
    コード構成.md          最終更新 2026-07-08（69日前）。目標Bの旧コード B/src/taro/ を
                           説明していて、名指しした instincts/ も brocas_area.py も存在しない
    部位の入出力一覧.md    作成 2026-09-03（12日前）。2026-09-15に消した lexicon.observe()・
                           observe_view・proto を指していた

**手で書いた構造の文書は 2/2 で腐った。**だから人が書き足す欄をゼロにする。

【この道具が出すもの】
    doc/太郎の構造.md        毎回読む（約8千トークン）。筋道（手書き・印で囲む）＋部位ごとの索引
    doc/部位の入出力一覧.md  引く（約2万3千トークン）。公開関数の 引数→返り値→何をするか

【腐るもの・腐らないものの線引き（2026-09-15・ユーザーと確認）】
    骨（引数・返り値・存在・行数）… 毎回コードから取り直す。**腐らない**
    肉（日本語の説明文）          … 隣にあるぶん腐りにくいだけ。**腐る**
    実測：本体143本中31本（22%）が、説明文を書いた後にコードが変わっている

    「なぜそうしたか」はこの文書に**載せない**。コードのコメントに置いたままにする。
    二重に持つと片方が必ず腐るため（上の2件がまさにそれ）。

【腐りを見つける検査（完全ではない。--unknown と --stale は別のものを見つける）】
    --unknown  説明文の中の名前が、コードに実在しないもの      （消えた名前を捕まえる）
    --stale    説明文を書いた後にコードが変わったファイル      （中身のズレを捕まえる）
    --commit   **判断材料を並べて、コミットを止める**（pre-commit が呼ぶ）
    実測（2026-09-15）：Aが9本・Bが31本で、両方に出るのは4本だけ。
    そして読んで見つけた58件のうち、**機械が自力で名指しできたのは4件（7%）**。
    どちらでも見つからない第3類（最初から間違い・コードも不変）は、読むしかない。

    だから --commit は**数えるのをやめた**。`.py` を触ったら毎回、その説明文と
    変更箇所（前後8行のコードつき）を並べて出し、コミットを止める。判定は読む人がする。
    機械には「この日本語がこのコードと合っているか」は判定できない
    （できるなら説明文を機械が書ける）。詳しくは review_commit() の説明。
    A・Bは後追いで知らせるだけ。**直せる瞬間に止めるのは --commit だけ**。

呼称は doc/脳の地図.md §0 が唯一の決まり。ここでもその名前を使う。

使い方:
    python run/tools/index_core.py              2つの文書を作り直す
    python run/tools/index_core.py --check      最新かだけ返す（0=最新、1=古い）
    python run/tools/index_core.py --io <語>    その部位の入出力をその場で出す
    python run/tools/index_core.py --full <名>  説明文の全文をその場で出す（mdには残さない）
    python run/tools/index_core.py --unknown    検査A
    python run/tools/index_core.py --stale      検査B
    python run/tools/index_core.py --commit     コミット前の検査（pre-commit が呼ぶ）
"""
import argparse
import ast
import io
import os
import re
import subprocess
import sys
import tokenize

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
SCAN_DIRS = ["taro_core/src"]
SKIP = ("__pycache__", ".venv", "MIMo", ".git", "node_modules")

OUT_STRUCT = "doc/太郎の構造.md"
OUT_IO = "doc/部位の入出力一覧.md"

MARK_A = "<!-- 手書きここから（自動生成はこの間を書き換えません） -->"
MARK_Z = "<!-- 手書きここまで -->"

# 部位の日本語名と並び順（感覚 → 脳 → からだ の流れ）
REGIONS = [
    ("senses",                                   "感覚（目・耳・触覚）"),
    ("brain/cerebral_cortex",                    "大脳皮質"),
    ("brain/cerebral_cortex/parietal_lobe",      "大脳皮質 / 頭頂葉"),
    ("brain/cerebral_cortex/temporal_lobe",      "大脳皮質 / 側頭葉"),
    ("brain/cerebral_cortex/frontal_lobe",       "大脳皮質 / 前頭葉"),
    ("brain/left_frontal_lobe",                  "左前頭葉（ブローカ野）"),
    ("brain/language_hippocampus",               "言語海馬"),
    ("brain/midbrain",                           "中脳"),
    ("brain/brainstem",                          "脳幹"),
    ("brain/cerebellum_lobe",                    "小脳"),
    ("brain/spinal_cord",                        "脊髄"),
    ("brain/subcortical_nuclei",                 "皮質下核"),
    ("brain/limbic",                             "大脳辺縁系"),
    ("brain/drives",                             "欲求"),
    ("brain/neuromodulator",                     "神経修飾"),
    ("brain/goal_babbling",                      "目標喃語"),
    ("brain/maturation",                         "成熟"),
    ("brain/principles",                         "原則"),
    ("brain",                                    "脳の土台（1歩を回す）"),
    ("body",                                     "からだ"),
    ("wrapper",                                  "外枠"),
]
REGION_NAME = dict(REGIONS)
REGION_ORDER = {k: i for i, (k, _) in enumerate(REGIONS)}


def _iter_py():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for f in sorted(files):
                if f.endswith(".py") and f != "__init__.py":
                    yield os.path.relpath(os.path.join(r, f), ROOT).replace("\\", "/")


def _region(rel):
    d = os.path.dirname(rel[len("taro_core/src/"):])
    return d if d else "(直下)"


def _first(text):
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""


def _sig(fn):
    """引数を人が読める形に並べる。self は落とす。"""
    a = [x.arg for x in fn.args.args if x.arg != "self"]
    if fn.args.vararg:
        a.append("*" + fn.args.vararg.arg)
    if fn.args.kwarg:
        a.append("**" + fn.args.kwarg.arg)
    return ", ".join(a)


def _returns(fn):
    """値を返すか（return か yield があるか）を調べる。"""
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            return True
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            return True
    return False


def collect():
    """全ファイルを構文解析して、骨（事実）と肉（説明文）を集めて返す。"""
    out = []
    for rel in _iter_py():
        p = os.path.join(ROOT, rel)
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
            tree = ast.parse(src)
        except Exception as e:
            out.append({"path": rel, "region": _region(rel),
                        "doc": "（読めません: %s）" % e, "fulldoc": "",
                        "lines": 0, "items": []})
            continue
        doc = ast.get_docstring(tree) or ""
        items = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                ms = []
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and not m.name.startswith("_"):
                        ms.append({"name": m.name, "args": _sig(m), "ret": _returns(m),
                                   "doc": _first(ast.get_docstring(m) or "")})
                items.append({"kind": "class", "name": node.name,
                              "doc": _first(ast.get_docstring(node) or ""),
                              "methods": ms})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not node.name.startswith("_"):
                items.append({"kind": "func", "name": node.name, "args": _sig(node),
                              "ret": _returns(node),
                              "doc": _first(ast.get_docstring(node) or "")})
        out.append({"path": rel, "region": _region(rel), "doc": _first(doc),
                    "fulldoc": doc, "lines": src.count("\n") + 1, "items": items})
    out.sort(key=lambda e: (REGION_ORDER.get(e["region"], 99), e["path"]))
    return out


def _by_region(entries):
    """部位ごとにまとめ直す（並び順は REGIONS のまま）。"""
    g = []
    for e in entries:
        if not g or g[-1][0] != e["region"]:
            g.append((e["region"], []))
        g[-1][1].append(e)
    return g


def _n_funcs(entries):
    """公開関数・メソッドの総数を数える。"""
    n = 0
    for e in entries:
        for i in e["items"]:
            n += len(i["methods"]) if i["kind"] == "class" else 1
    return n


def _keep_handwritten(path, default):
    """既にある手書き部分（印の間）を拾う。無ければ default を返す。"""
    p = os.path.join(ROOT, path)
    if os.path.exists(p):
        s = io.open(p, encoding="utf-8", errors="replace").read()
        if MARK_A in s and MARK_Z in s:
            return s.split(MARK_A, 1)[1].split(MARK_Z, 1)[0].strip("\n")
    return default


DEFAULT_SUJIMICHI = "## 実際に通る筋道\n\n（ここは手書き。自動生成は書き換えません。まだ書かれていません）"
DEFAULT_ANA = "## 実測でわかっている接続の穴\n\n（ここは手書き。自動生成は書き換えません。まだ書かれていません）"


def render_struct(entries):
    """doc/太郎の構造.md の中身を組み立てる（毎回読むほう）。"""
    L = []
    L.append("# 太郎の構造")
    L.append("")
    L.append("**自動生成です。表を手で書き足さないでください**"
             "（`python run/tools/index_core.py` で作り直す）。")
    L.append("手で書いてよいのは下の印で囲まれた節だけです。")
    L.append("")
    L.append("**%d ファイル・%d 行・公開関数 %d 個**"
             "（`taro_core/src` のみ。`run/` は未収録）"
             % (len(entries), sum(e["lines"] for e in entries), _n_funcs(entries)))
    L.append("")
    L.append("- 呼称の決まり … `doc/脳の地図.md` §0（**これ以外の呼び方をしない**）")
    L.append("- 入力と出力 … `doc/部位の入出力一覧.md`（Grepで必要な節だけ引く）")
    L.append("- 「**なぜ**そうしたか」は**ここには書きません**。コードのコメントにあります"
             "（例：`taro_core/src/body/infant_body.py` の 640〜653行目に、"
             "`NECK_TONE_STIFFNESS = 0.40` の文献と Tier が書いてあります）")
    L.append("- 測定器・ビューアなどの道具は `doc/道具索引.md`")
    L.append("")
    L.append(MARK_A)
    L.append(_keep_handwritten(OUT_STRUCT, DEFAULT_SUJIMICHI))
    L.append(MARK_Z)
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 部位ごとの索引")
    L.append("")
    for reg, es in _by_region(entries):
        L.append("### %s （%d本）" % (REGION_NAME.get(reg, reg), len(es)))
        L.append("")
        L.append("| ファイル | 何をするか | 行 |")
        L.append("|---|---|---|")
        for e in es:
            L.append("| `%s` | %s | %d |"
                     % (os.path.basename(e["path"]), e["doc"] or "—", e["lines"]))
        L.append("")
    return "\n".join(L) + "\n"


def _func_table(rows):
    """関数の表を組み立てる。rows は dict の並び。"""
    L = ["| 関数 | 受け取る | 返す | 何をするか |", "|---|---|---|---|"]
    for m in rows:
        L.append("| `%s` | %s | %s | %s |"
                 % (m["name"], m["args"] or "—", "あり" if m["ret"] else "なし",
                    m["doc"] or "**（説明文なし）**"))
    L.append("")
    return L


def render_io(entries):
    """doc/部位の入出力一覧.md の中身を組み立てる（Grepで引くほう）。"""
    L = []
    L.append("# 部位の入力と出力")
    L.append("")
    L.append("**自動生成です。表を手で書き足さないでください**"
             "（`python run/tools/index_core.py` で作り直す）。")
    L.append("手で書いてよいのは下の印で囲まれた節だけです。")
    L.append("")
    L.append("公開している関数が、**何を受け取り・何を返し・何をするか**。"
             "コードの引数と返り値から機械的に書き起こしたもの。")
    L.append("**「なぜ」は載せません**（コードのコメントにあります）。")
    L.append("全体像は `doc/太郎の構造.md`、呼称は `doc/脳の地図.md` §0。")
    L.append("")
    L.append("## 目次")
    L.append("")
    L.append("| 部位 | ファイル | 公開関数 |")
    L.append("|---|---|---|")
    for reg, es in _by_region(entries):
        L.append("| %s | %d | %d |"
                 % (REGION_NAME.get(reg, reg), len(es), _n_funcs(es)))
    L.append("")
    L.append(MARK_A)
    L.append(_keep_handwritten(OUT_IO, DEFAULT_ANA))
    L.append(MARK_Z)
    L.append("")
    L.append("---")
    L.append("")
    for reg, es in _by_region(entries):
        L.append("## %s" % REGION_NAME.get(reg, reg))
        L.append("")
        for e in es:
            L.append("### `%s`" % e["path"])
            L.append("")
            L.append(e["doc"] or "—")
            L.append("")
            if not e["items"]:
                L.append("公開している関数・クラスはありません。")
                L.append("")
                continue
            loose = [i for i in e["items"] if i["kind"] == "func"]
            for it in e["items"]:
                if it["kind"] != "class":
                    continue
                L.append("**class `%s`** %s" % (it["name"], it["doc"] or ""))
                L.append("")
                if it["methods"]:
                    L += _func_table(it["methods"])
                else:
                    L.append("公開しているメソッドはありません。")
                    L.append("")
            if loose:
                L.append("**クラスに属さない関数**")
                L.append("")
                L += _func_table(loose)
    return "\n".join(L) + "\n"


# ---------------- 検査 ----------------

# 照合の範囲。**2種類に分けてある**。
# 【なぜ分けるか・2026-09-15】1種類でやると、どちらかに倒れて使い物にならなかった：
#   狭すぎ（taro_core/src と run の NAME だけ）→ 22件中14件が誤検知。
#     MuJoCo の関節名 head_swivel・センサー名 vestibular_acc は**文字列**なので
#     NAME トークンにならず、「実在しない」と誤って言う
#   広すぎ（E/F の使い捨てスクリプトの文字列まで含める）→ 0件。
#     lexicon が保存ファイルのキー文字列として残っているだけで「実在する」ことになり、
#     **本当に消えた名前まで見逃す**
# そこで「関数・クラスとして生きている名前」と「設定の文字列として使われている名前」を
# 別々に集め、**前者に無ければ知らせる**（後者は誤検知を抑える補助にだけ使う）。
# 注意：テストは taro_core/tests にある。ルートの tests/ は存在しない。
# （2026-09-15、"tests" と書いていて SensoryFusion を誤検知した）
CODE_DIRS = ["taro_core", "run", "E/scripts", "F/scripts"]
STRING_DIRS = ["taro_core", "run", "MIMo"]   # MuJoCoの関節名・設定のキーなど
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# 「もう無い」と書いてあるのは腐りではなく履歴。知らせない。
_HISTORY = re.compile(
    r"元(は|の)|以前|かつて|削除|廃止|だった|やめた|旧|→|変えた|移した|使い捨て"
    r"|一般化|置き換え|差し替え|改名|存在しない|もう無い|訂正")


def _string_names(src):
    """コードの中の文字列リテラルに出てくる語を集める。**説明文は数えない**。

    【なぜ説明文を外すか・2026-09-15】外さないと検査Aが自分で自分を無効にする。
    説明文も文字列リテラルなので、「説明文に書いてある」ことが
    「コードに実在する」の証拠になってしまい、嘘を1件も検出できない。
    実際、わざと存在しない名前を説明文に入れても素通りした。
    （長い説明文だけは「200字未満」という別の条件でたまたま除外されていて、
      短い説明文の嘘だけが見逃されるという分かりにくい形になっていた）
    """
    out = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return out
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                          ast.AsyncFunctionDef)):
            b = getattr(n, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant):
                docs.add(id(b[0].value))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and id(n) not in docs and len(n.value) < 200:
            out.update(_WORD.findall(n.value))
    return out


def _all_names():
    """コード中に実在する名前を集める。識別子と、設定の文字列の両方。"""
    names = set()
    for d in CODE_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for f in files:
                if not f.endswith(".py"):
                    continue
                names.add(f[:-3])
                try:
                    with io.open(os.path.join(r, f), encoding="utf-8",
                                 errors="replace") as fh:
                        for t in tokenize.generate_tokens(fh.readline):
                            if t.type == tokenize.NAME:
                                names.add(t.string)
                except Exception:
                    pass
    for d in STRING_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs
                       if x not in (".git", "__pycache__", ".venv", "node_modules")]
            for f in files:
                p = os.path.join(r, f)
                try:
                    if f.endswith(".xml"):
                        names.update(_WORD.findall(
                            io.open(p, encoding="utf-8", errors="replace").read()))
                    elif f.endswith(".py"):
                        names.update(_string_names(
                            io.open(p, encoding="utf-8", errors="replace").read()))
                except Exception:
                    pass
    # 実在するファイル名（`e_body_measure.py` のような引用のため）。
    # F/ は7万ファイル・6.8GBあるので全体は歩かない。道具の置き場だけを見る。
    for d in ["taro_core", "run", "tests", "E/scripts", "F/scripts", "doc", "MIMo"]:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [x for x in dirs
                       if x not in (".git", "__pycache__", ".venv", "node_modules")]
            for f in files:
                names.add(f)
                if "." in f:
                    names.add(f.rsplit(".", 1)[0])
    return names


_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def check_unknown(entries=None):
    """検査A：説明文の中でバッククォートされた名前が、コードに実在しないもの。"""
    entries = entries if entries is not None else collect()
    names = _all_names()
    hits = []
    for e in entries:
        lines = (e.get("fulldoc") or "").splitlines()
        for ln_no, ln in enumerate(lines, 1):
            # 履歴の言い回しは行をまたぐ（「`touched_name` を」で改行して次の行に
            # 「一般化した」が来る）ので、前後1行まで見る。2026-09-15に3件誤検知した。
            near = "".join(lines[max(0, ln_no - 2):ln_no + 1])
            if _HISTORY.search(near):
                continue        # 「もう無い」と書いてある行は履歴。腐りではない
            for tokn in re.findall(r"`([^`]+)`", ln):
                if " " in tokn or "=" in tokn:
                    continue    # 数式や文。名前ではない
                base = tokn.split("(")[0].split(".")[0].strip()
                if not _ID.match(base) or base in names:
                    continue
                hits.append((e["path"], ln_no, tokn, ln.strip()))
    return hits


def _git(args):
    try:
        return subprocess.run(["git"] + args, cwd=ROOT, capture_output=True,
                              text=True, encoding="utf-8",
                              errors="replace").stdout.strip()
    except Exception:
        return ""


REVIEW_DIRS = ("taro_core/src", "run")
REVIEW_CONTEXT = 8        # 変更箇所の前後に何行のコードを添えるか


def review_commit():
    """コミット直前に、**判断材料を並べて止める**。文字列を返す（空なら止めない）。

    【なぜ「数える」のをやめたか・2026-09-15】
    はじめは「コードを10行以上変えたのに説明文を1行も書いていないか」を数えていた。
    これは形式しか見ていない。実際、`retina.py` の事故はこの数え方では捕まらない：
      あのコミットが変えたのは**コメント3行だけ**（コード変更0行）で、
      「truncate=2.0 にした」と書きながら、すぐ下のコード行には truncate が無かった。
      問題の行は変わっていないので **diff にすら写らない**。

    なので数えるのをやめて、**変更箇所の前後のコードごと並べて見せる**ことにした。
    上の例なら、足したコメントのすぐ下に `gaussian_filter(a, sigma_px)` が並ぶので
    一目で食い違いが分かる。判定は人（読む側）がする。機械には日本語とコードが
    合っているかを判定できない。

    【なぜ全ファイルを見せるか】ファイル数で絞ろうとしたが、実測すると
    `.py` を触ったコミット181件のうち **62%が2本以下**、20本超えは6.1%しかない。
    まれな大量変更のために設計を歪めない。長くなったら理由を書いて --no-verify で通す。

    【止める理由】表示だけだとコミットが通って流れて消える。止めれば必ず目に入る。
    `.py` は git が全部追跡しているので、.md と違ってここでの検査はちゃんと効く。
    """
    files = [f for f in _git(["diff", "--cached", "--name-only", "--diff-filter=ACM",
                              "--"] + list(REVIEW_DIRS)).splitlines()
             if f.endswith(".py")]
    if not files:
        return ""
    L = ["", "=" * 70,
         " 説明文が今のコードと合っているか見てください（%d ファイル）" % len(files),
         "=" * 70]
    for rel in files:
        L.append("")
        L.append("─" * 70)
        L.append("  %s" % rel)
        p = os.path.join(ROOT, rel)
        doc = ""
        if os.path.exists(p):
            try:
                doc = ast.get_docstring(ast.parse(
                    io.open(p, encoding="utf-8", errors="replace").read())) or ""
            except Exception:
                doc = ""
        if doc:
            d = doc.splitlines()
            L.append("  ── このファイルの説明 " + "─" * 44)
            for ln in d[:12]:
                L.append("    " + ln)
            if len(d) > 12:
                L.append("    …（あと %d 行。全文は "
                         "python run/tools/index_core.py --full %s）"
                         % (len(d) - 12, os.path.basename(rel)))
        L.append("  ── 今回の変更（前後%d行つき） %s" % (REVIEW_CONTEXT, "─" * 36))
        diff = _git(["diff", "--cached", "-U%d" % REVIEW_CONTEXT, "--", rel])
        for ln in diff.splitlines():
            if ln.startswith(("diff --git", "index ", "--- ", "+++ ")):
                continue
            L.append("    " + ln)
    L += ["", "=" * 70,
          "  合っていれば     git commit --no-verify（理由を1行考えてから）",
          "  違っていれば     説明文を直してからコミットし直す",
          "=" * 70, ""]
    return "\n".join(L)


def check_commit(min_lines=10):
    """【旧】コードを大きく変えたのに説明文を1行も触っていないものを数える。

    2026-09-15に review_commit() へ置き換えた。形式（行数）しか見ておらず、
    retina.py の事故（コメントだけ足して嘘を書く）を原理的に捕まえられなかったため。
    比較や過去の再現のために残してある。pre-commit はもう呼んでいない。
    """
    staged = _git(["diff", "--cached", "--name-only", "--diff-filter=M",
                   "--", "taro_core/src", "run"]).splitlines()
    out = []
    for rel in staged:
        if not rel.endswith(".py") or not os.path.exists(os.path.join(ROOT, rel)):
            continue
        try:
            tree = ast.parse(io.open(os.path.join(ROOT, rel),
                                     encoding="utf-8", errors="replace").read())
        except Exception:
            continue
        if not (tree.body and ast.get_docstring(tree)):
            continue
        # 説明文の行の範囲を全部集める（モジュール・クラス・関数）。
        # 【なぜ全部か・2026-09-15】最初はモジュール冒頭だけを見ていたが、
        #   **関数の説明文を64個足しただけのコミットで誤発火した**。
        #   知りたいのは「コードを変えたのに説明を一切書かなかったか」なので、
        #   どの階層の説明文でも触っていれば触ったと数える。
        doc_lines = set()
        doc_end = tree.body[0].end_lineno
        for n in ast.walk(tree):
            if not isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                  ast.AsyncFunctionDef)):
                continue
            b = getattr(n, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                doc_lines.update(range(b[0].lineno, b[0].end_lineno + 1))
        # 【数え方・2026-09-15】「説明を一切書かずにコードだけ変えたか」を見たい。
        #   だから **# のコメントも「説明を書いた」に数える**（説明文と同じ扱い）。
        #   コメントだけを直したコミットで誤発火したため（3度目の調整）。
        #   逆に、コメントでも説明文でもない行が増減したぶんだけを「コードの変更」と数える。
        diff = _git(["diff", "--cached", "-U0", "--", rel])
        changed = 0
        touched_doc = False
        cur = 0
        for ln in diff.splitlines():
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
            if m:
                cur = int(m.group(1))
                continue
            if ln.startswith("+") and not ln.startswith("+++"):
                body, advance = ln[1:], True
            elif ln.startswith("-") and not ln.startswith("---"):
                body, advance = ln[1:], False   # 消えた行は今のファイルに無い
            else:
                continue
            is_doc = body.lstrip().startswith("#") or (advance and cur in doc_lines)
            if is_doc:
                touched_doc = True
            elif body.strip():
                changed += 1
            if advance:
                cur += 1
        if changed >= min_lines and not touched_doc:
            out.append((rel, changed, doc_end))
    return out


def check_stale():
    """検査B：説明文を最後に書いた後に、そのファイルのコードが変わったもの。"""
    rows = []
    for rel in _iter_py():
        p = os.path.join(ROOT, rel)
        try:
            tree = ast.parse(io.open(p, encoding="utf-8", errors="replace").read())
        except Exception:
            continue
        if not (tree.body and ast.get_docstring(tree)):
            continue
        dd = _git(["log", "-1", "--format=%ad", "--date=short",
                   "-L", "1,%d:%s" % (tree.body[0].end_lineno, rel)]).splitlines()
        dd = dd[0] if dd else ""
        fd = _git(["log", "-1", "--format=%ad", "--date=short", "--", rel])
        if not dd or not fd or fd <= dd:
            continue
        n = _git(["rev-list", "--count", "HEAD", "--since=%s" % dd, "--", rel])
        rows.append((rel, dd, fd, int(n) if n.isdigit() else 0))
    rows.sort(key=lambda r: -r[3])
    return rows


# ---------------- 入口 ----------------

def _write(path, text):
    """中身が変わったときだけ書く。書いたら True。"""
    p = os.path.join(ROOT, path)
    old = io.open(p, encoding="utf-8", errors="replace").read() \
        if os.path.exists(p) else None
    if old == text:
        return False
    io.open(p, "w", encoding="utf-8", newline="\n").write(text)
    return True


def main():
    """コマンドラインの入口。"""
    ap = argparse.ArgumentParser(description="太郎の本体の構造をコードから書き起こす")
    ap.add_argument("--check", action="store_true", help="最新かだけ返す（0=最新）")
    ap.add_argument("--io", metavar="語", help="その部位／ファイルの入出力を出す")
    ap.add_argument("--full", metavar="名", help="説明文の全文を出す")
    ap.add_argument("--unknown", action="store_true", help="検査A")
    ap.add_argument("--stale", action="store_true", help="検査B")
    ap.add_argument("--commit", action="store_true",
                    help="コミット前の検査（本体を変えたのに説明文を触っていない）")
    a = ap.parse_args()

    if a.commit:
        text = review_commit()
        if not text:
            return 0
        print(text)
        return 1

    if a.unknown:
        hits = check_unknown()
        if not hits:
            print("検査A：説明文の中の名前は全部コードに実在します。")
            return 0
        print("検査A：説明文の中の名前が、コードに実在しません（%d件）" % len(hits))
        for path, ln, tokn, line in hits:
            print("  %s（説明文 %d行目） %s" % (path, ln, tokn))
            print("      %s" % line[:110])
        return 1

    if a.stale:
        rows = check_stale()
        if not rows:
            print("検査B：説明文より後にコードが変わったファイルはありません。")
            return 0
        print("検査B：説明文を書いた後にコードが変わったファイル（%d本）" % len(rows))
        for rel, dd, fd, n in rows:
            print("  %-58s 説明文 %s / コード %s / その後 %d回" % (rel, dd, fd, n))
        return 1

    entries = collect()

    if a.full:
        for e in entries:
            if a.full in e["path"]:
                print("=== %s （%d行） ===" % (e["path"], e["lines"]))
                print(e["fulldoc"] or "（説明文なし）")
                print()
        return 0

    if a.io:
        want = [e for e in entries
                if a.io in e["path"] or a.io in REGION_NAME.get(e["region"], "")]
        if not want:
            print("見つかりません: %s" % a.io)
            return 1
        print(render_io(want))
        return 0

    s, t = render_struct(entries), render_io(entries)
    if a.check:
        ok = True
        for path, text in ((OUT_STRUCT, s), (OUT_IO, t)):
            p = os.path.join(ROOT, path)
            cur = io.open(p, encoding="utf-8", errors="replace").read() \
                if os.path.exists(p) else None
            if cur != text:
                print("古い: %s" % path)
                ok = False
        if ok:
            print("最新です。")
        return 0 if ok else 1

    for path, text in ((OUT_STRUCT, s), (OUT_IO, t)):
        print(("書き直しました: %s" if _write(path, text) else "変更なし: %s") % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
