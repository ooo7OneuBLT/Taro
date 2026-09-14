# -*- coding: utf-8 -*-
"""プラグインが太郎を書き換えていないかを調べる（2026-09-13）。

【なぜ作ったか】`run/plugins/base.py` の規約はこう書いてある：

    プラグインは ctx を**読むだけ**。太郎も環境も変えない
    （測る道具が対象を変えたら測定にならない）

しかし 2026-09-13 に調べたところ、`object_files.py` が

  ・眼球に命令を出していた（`_or.set_map_target(...)`）
  ・親に「太郎が気づいた」と伝えていた（`env.unwrapped._taro_noticed_gone_time = ...`）

規約はファイルの冒頭に書いてあったが、**誰も見ていなかった**。
そして同じファイルのヘッダ自身が「太郎の脳の入力に繋ぐことは範囲外」と
約束していたのに、後から破られ、ヘッダは更新されないままだった。

⇒ **文書で守らせるのをやめて、機械で見る。**（同じ型の失敗が3回目なので）

【2026-09-13・段Aで強くした3点】
  (1) **掲示板（ctx）経由の介入**を見る。道具が `ctx.X = ` で置いた値を
      `run/trainer.py`（太郎の脳）や `E/scripts/`（世界）が読んでいるなら、
      それは「読むだけ」ではなく**太郎への入力を作っている**。
      実際 `object_files.py` は眼球への直接命令を core へ移した後も、
      `ctx.attended_object` 経由で発話の生成を左右し続けていた。
  (2) 見落としていた書き込みの形を足した（`u.X =`・`backend.X =`・`p.X =` など、
      環境や太郎の部品を別名で受けてから書くもの）。
  (3) 例外の登録を**行番号（腐る）から、クラス属性 `intervenes`（腐らない）へ**変えた。
      `run/plugins/base.py` の「道具は2種類ある」を参照。

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_plugins
    .venv/Scripts/python.exe -m run.tools.check_plugins --quiet   # 違反があるときだけ出す

【何を違反とみなすか】`run/plugins/` の下のコードで、下の `_RULES` に当たる書き込みや
呼び出しをしていて、かつそのファイルのプラグインが `intervenes` を名乗っていないもの。
読むだけ（`getattr`・比較・print）は違反にしない。
"""
import argparse
import ast
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
PLUGIN_DIR = os.path.join(_ROOT, "run", "plugins")

# 掲示板に置いた値を「太郎の脳」「世界」が読んでいたら、それは介入。
#   ここに挙げた場所を読み手として探す。
_READER_FILES = [
    os.path.join(_ROOT, "run", "trainer.py"),
    os.path.join(_ROOT, "run", "taro_setup.py"),
]
_READER_DIRS = [os.path.join(_ROOT, "E", "scripts")]

# 太郎の状態を変えるメソッド（呼んだら違反）。増えたらここに足す
_MUTATORS = (
    "set_map_target", "set_target", "reset", "apply",
    "step_brain", "learn_action", "update_weights",
)

# 環境・太郎を受けるのによく使われる別名。`X.attr = ` を書いていたら疑う。
#   （2026-09-13：`u.`（scene.py）と `backend.`（produce_snapshot.py）を
#     見落としていた。実害はないが「見えていなかった」こと自体が問題）
_ALIASES = r"(?:env|_env|_u|u|taro|_taro|backend|brain|_or|orienting|_orienting|lp|_lp)"

_RULES = (
    # (正規表現, 何が悪いか)
    (re.compile(r"\.unwrapped\.[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)"),
     "環境（env.unwrapped）の属性に書き込んでいる"),
    (re.compile(r"\b" + _ALIASES + r"\.[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)"),
     "太郎か環境の属性に書き込んでいる"),
    (re.compile(r"\b(?:_or|orienting|_orienting)\.(?:" + "|".join(_MUTATORS) + r")\s*\("),
     "太郎の反射を動かしている（読むだけではない）"),
)

_CTX_WRITE = re.compile(r"\bctx\.([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")


def _code_line_numbers(src_text):
    """実際に動くコードがある行番号の集合。文字列・docstring の中は含めない。

    【2026-09-13】説明文に書いた例（「taro.goal_babbling=true が前提」等）を
      違反として拾ってしまったため。文字列リテラルが占める行を除く。
    """
    try:
        tree = ast.parse(src_text)
    except SyntaxError:
        return set(range(1, len(src_text.splitlines()) + 1))   # 読めなければ全部見る
    in_str = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            a = getattr(node, "lineno", None)
            b = getattr(node, "end_lineno", a)
            if a:
                in_str.update(range(a, (b or a) + 1))
    return set(range(1, len(src_text.splitlines()) + 1)) - in_str


def _is_plugin_file(src_text):
    """このファイルが本当にプラグイン（Plugin を継承したクラス）を定義しているか。

    【2026-09-13】`run/plugins/` の下には、プラグインではないものが混ざっている
      （`scene.py`＝環境を組み立てる関数、`body_geoms.py`・`probe_defaults.py`＝
      共有部品）。これらを「規約違反」として数えると、本当の違反が埋もれる。
      違反ではなく**置き場の誤り**として別に出す。
    """
    try:
        tree = ast.parse(src_text)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 規約そのものを定める base.py（class Plugin）もプラグイン側
            if node.name == "Plugin":
                return True
            for b in node.bases:
                nm = b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                if nm == "Plugin":
                    return True
    return False


def _declared_intervention(src_text):
    """このファイルのプラグインが `intervenes = "..."` を名乗っているか。

    行番号ではなくクラス属性で持つ（行番号は編集のたびに腐るため）。
    """
    try:
        tree = ast.parse(src_text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for st in node.body:
            if not isinstance(st, ast.Assign):
                continue
            for tgt in st.targets:
                if (isinstance(tgt, ast.Name) and tgt.id == "intervenes"
                        and isinstance(st.value, ast.Constant)
                        and isinstance(st.value.value, str)):
                    return st.value.value
    return None


def _plugin_sources():
    out = []
    for root, _dirs, files in os.walk(PLUGIN_DIR):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, _ROOT).replace("\\", "/")
            try:
                out.append((rel, path, open(path, encoding="utf-8").read()))
            except Exception:
                continue
    return out


def _reader_texts():
    texts = []
    for p in _READER_FILES:
        if os.path.exists(p):
            texts.append((os.path.relpath(p, _ROOT).replace("\\", "/"),
                          open(p, encoding="utf-8", errors="replace").read()))
    for d in _READER_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py"):
                p = os.path.join(d, fn)
                texts.append((os.path.relpath(p, _ROOT).replace("\\", "/"),
                              open(p, encoding="utf-8", errors="replace").read()))
    return texts


def _find_ctx_readers(name, readers):
    """掲示板の `name` を読んでいる場所（file:行）を返す。"""
    pat = re.compile(r"(?:ctx|self\.ctx)\s*,\s*[\"']" + re.escape(name) + r"[\"']"
                     r"|(?:ctx|self\.ctx)\." + re.escape(name) + r"\b")
    hits = []
    for rel, text in readers:
        for i, line in enumerate(text.splitlines(), 1):
            s = line.split("#", 1)[0]
            if pat.search(s):
                hits.append("%s:%d" % (rel, i))
    return hits


def _scan():
    """(直接の書き込み, 掲示板経由の介入, 名乗り済みの道具, 場所違い) を返す。"""
    direct, via_ctx, declared, misplaced = [], [], {}, []
    readers = _reader_texts()
    for rel, _path, src_text in _plugin_sources():
        # プラグインが共有して使う部品（道具の一部）は置き場の誤りではない
        _HELPERS = ("run/plugins/common/body_geoms.py",
                    "run/plugins/common/probe_defaults.py")
        if not _is_plugin_file(src_text):
            if rel not in _HELPERS and re.search(r"^\s*(?:def |class )", src_text, re.M):
                misplaced.append(rel)
            continue
        decl = _declared_intervention(src_text)
        if decl:
            declared[rel] = decl
        code_lines = _code_line_numbers(src_text)
        for i, line in enumerate(src_text.splitlines(), 1):
            if i not in code_lines:
                continue
            s = line.split("#", 1)[0]          # 行コメントは見ない
            if not s.strip():
                continue
            for rx, why in _RULES:
                if rx.search(s):
                    direct.append((rel, i, why, line.strip()[:110], decl))
                    break
            m = _CTX_WRITE.search(s)
            if m:
                who = _find_ctx_readers(m.group(1), readers)
                if who:
                    via_ctx.append((rel, i, m.group(1), who, decl))
    return direct, via_ctx, declared, misplaced


def _print_block(title, rows, kind):
    print("")
    print("  **%s（%d 件）**" % (title, len(rows)))
    for r in rows:
        if kind == "direct":
            rel, i, why, src, _d = r
            print("    %s:%d" % (rel, i))
            print("      %s" % why)
            print("      > %s" % src)
        else:
            rel, i, name, who, _d = r
            print("    %s:%d  ctx.%s を置いている" % (rel, i, name))
            print("      読んでいるのは: %s%s"
                  % (", ".join(who[:3]), (" ほか%d" % (len(who) - 3)) if len(who) > 3 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="違反があるときだけ出力する")
    a = ap.parse_args()

    direct, via_ctx, declared, misplaced = _scan()
    bad_direct = [r for r in direct if not r[4]]
    bad_ctx = [r for r in via_ctx if not r[4]]
    ok_direct = [r for r in direct if r[4]]
    ok_ctx = [r for r in via_ctx if r[4]]

    if not a.quiet or bad_direct or bad_ctx:
        print("プラグインの規約検査（run/plugins/base.py：測る道具は読むだけ）")
        print("  調べた場所: run/plugins/")

    if misplaced and not a.quiet:
        print("")
        print("  【プラグインではないのに run/plugins/ に置かれているファイル】")
        print("    （違反ではなく**置き場の誤り**。検査の対象外にしている）")
        for rel in misplaced:
            print("    %s" % rel)

    if declared and not a.quiet:
        print("")
        print("  【介入すると名乗っている道具】この実験は「測るだけ」ではない：")
        for rel, why in sorted(declared.items()):
            print("    %-42s %s" % (rel, why))

    if not bad_direct and not bad_ctx:
        if not a.quiet:
            print("")
            print("  名乗らない違反なし（名乗り済みの書き込み %d 件・掲示板経由 %d 件）"
                  % (len(ok_direct), len(ok_ctx)))
        return 0

    if bad_direct:
        _print_block("太郎か環境を直接書き換えている", bad_direct, "direct")
    if bad_ctx:
        _print_block("掲示板(ctx)を通じて太郎の入力を作っている", bad_ctx, "ctx")

    print("")
    print("  直し方は2つ：")
    print("   (a) その処理を太郎の脳（taro_core）へ移す ← 能力に関わるなら必ずこちら")
    print("   (b) わざとの介入なら、台帳（doc/人間模倣からの逸脱リスト.md）に登録し、")
    print("       そのプラグインのクラスに `intervenes = \"台帳その◯◯（理由）\"` と書く")
    print("")
    print("  判定の基準：**その処理を消したら太郎の能力が落ちるか？**")
    print("   落ちる → 脳（core）へ移す。落ちない（記録が減るだけ）→ 道具のままでよい")
    return 1


if __name__ == "__main__":
    sys.exit(main())
