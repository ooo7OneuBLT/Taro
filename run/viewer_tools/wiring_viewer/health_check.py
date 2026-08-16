"""TaroMap（⑥健康診断ページ）用：2つの独立したヒューリスティック検査。

【この2つの検査は「確定判定」ではない、仕様より】
  (a) 未配線検出：taro_core内に実在するがtaro_setup.pyから一度も呼ばれていない
      クラスを一覧化する。動的な構築（getattr等）や、意図的にヘルパー/基底クラス
      として使う場合は見逃す・誤検出することがある。
  (b) config.pyのブール設定の配線カバレッジ：2026-08-13に実際に起きた
      「touch_adaptationがwiring_map.NODESに反映されていなかった」欠落を、機械的に
      検出できるようにしたもの。サブパラメータ的なキー（例：本体機能に付随する
      閾値・倍率など）は正当に配線図へ単独ノードを持たないことがあるため、これも
      確定的な「エラー」ではない。
⇒ どちらも戻り値・UI表示は「要確認（自動判定）」という中立的な言葉で扱うこと。
  「エラー」「壊れている」等の断定語は使わない（仕様の禁止事項）。

【scan_taro_setup.py への依存について】
(a)は `run/viewer_tools/wiring_viewer/scan_taro_setup.py` の
`scan_taro_setup()`（{"constructed_classes": set[str], "self_attrs": dict}を返す）を
使う。着手時点で既に他の実装担当により作成済みだったため、それを import して使う
（自己完結コードの二重実装はしない）。万一importに失敗した場合（並列作業中の
一時的な欠落等を想定）だけ、このファイル内に持つ最小限のフォールバック実装
（_fallback_scan_taro_setup）で自己完結させる。
"""
from __future__ import annotations

import ast
import inspect
import os

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
_TARO_CORE_SRC = os.path.join(_ROOT, "taro_core", "src")
_TARO_SETUP_PATH = os.path.join(_ROOT, "run", "taro_setup.py")


# ---------------------------------------------------------------- (a) 未配線検出
def _fallback_scan_taro_setup(path=None) -> dict:
    """scan_taro_setup.pyが import できないときの最小限フォールバック。

    ロジックはscan_taro_setup.scan_taro_setup()と同じ契約
    （{"constructed_classes": set[str], "self_attrs": dict[str, str]}）を守るが、
    実装は独立して持つ（依存が解決するまで手を止めないため。仕様の指示通り）。
    """
    target = path or _TARO_SETUP_PATH
    with open(target, encoding="utf-8") as fp:
        source = fp.read()
    tree = ast.parse(source, filename=target)

    class_node = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Taro":
            class_node = node
            break
    if class_node is None:
        return {"constructed_classes": set(), "self_attrs": {}}

    init_node = None
    for node in ast.iter_child_nodes(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_node = node
            break
    if init_node is None:
        return {"constructed_classes": set(), "self_attrs": {}}

    def _class_name_of_call(call_node):
        node = call_node
        while True:
            func = node.func
            if isinstance(func, ast.Name) and func.id[:1].isupper():
                return func.id
            if isinstance(func, ast.Attribute) and func.attr[:1].isupper():
                return func.attr
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
                node = func.value
                continue
            return None

    constructed_classes = set()
    self_attrs = {}
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        cls_name = _class_name_of_call(node.value)
        if cls_name is None:
            continue
        for target_expr in node.targets:
            if (isinstance(target_expr, ast.Attribute)
                    and isinstance(target_expr.value, ast.Name)
                    and target_expr.value.id == "self"):
                constructed_classes.add(cls_name)
                self_attrs[target_expr.attr] = cls_name
    return {"constructed_classes": constructed_classes, "self_attrs": self_attrs}


def _scan_taro_setup(path=None) -> dict:
    try:
        from run.viewer_tools.wiring_viewer.scan_taro_setup import scan_taro_setup
        return scan_taro_setup(path)
    except Exception:
        # importできない（並列作業中でまだ無い、循環import等）場合だけ自己完結する。
        return _fallback_scan_taro_setup(path)


def _find_taro_core_classes(src_dir=None) -> list[dict]:
    """taro_core/src配下で`class X` として定義されている全クラスを列挙する。

    astで各.pyファイルのトップレベル（モジュール直下）のClassDefだけを拾う
    （ネストしたクラス・関数内で定義されるクラスは対象外＝通常はモジュール直下に
    主要クラスが1つ、という太郎のコードの流儀に合わせた）。
    """
    src_dir = src_dir or _TARO_CORE_SRC
    classes = []
    for dirpath, _dirs, files in os.walk(src_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8") as fp:
                    source = fp.read()
                tree = ast.parse(source, filename=fpath)
            except Exception:
                continue
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "file": os.path.relpath(fpath, _ROOT).replace(os.sep, "/"),
                    })
    return classes


def find_unwired_classes(src_dir=None, taro_setup_path=None) -> dict:
    """taro_core内に実在するがtaro_setup.pyから一度も構築されていないクラスを探す。

    【ヒューリスティックであることの明記、仕様より】これは完全な保証ができない
    判定である。動的な構築（getattr・辞書経由の呼び出し等）や、他のクラスの内部で
    構築される補助クラス、テスト専用クラス等は「未配線」として誤検出される
    ことがある。「エラー」ではなく「要確認（自動判定）」として扱うこと。

    戻り値：
        {
          "heuristic": True,   # 常にTrue。呼び出し側に必ず明示させるための印
          "total_classes": int,          # taro_core/src内の全クラス数
          "constructed_classes": int,    # taro_setup.pyから構築されている数
          "unwired": [{"name": str, "file": str}, ...],  # 未配線とみなした一覧
        }
    """
    all_classes = _find_taro_core_classes(src_dir)
    setup_info = _scan_taro_setup(taro_setup_path)
    constructed = setup_info["constructed_classes"]

    unwired = [c for c in all_classes if c["name"] not in constructed]
    unwired.sort(key=lambda c: (c["file"], c["name"]))
    return {
        "heuristic": True,
        "total_classes": len(all_classes),
        "constructed_classes": len(constructed),
        "unwired": unwired,
    }


# ------------------------------------------------------ (b) bool設定の配線カバレッジ
def find_uncovered_bool_settings(nodes=None, taro_defaults=None, bools=None) -> dict:
    """TARO_DEFAULTSのbool設定のうち、wiring_map.NODESのis_on関数のソース中に
    `c.<キー名>` という参照が見当たらないものを一覧化する。

    【なぜソースコード文字列を見るだけか】NODESの各要素[6]はConfigを受け取る
    関数（lambda等）。実際にConfigを渡して実行してもよいが、副作用の心配が
    無い最も単純な方法として、`inspect.getsource(fn)` の文字列に
    `c.<キー名>`（またはlambdaの引数名が別でも同じ形の属性アクセス）が
    含まれるかをテキストとして確認する。引数名がcでない関数（外部から渡された
    ヘルパー関数等）は拾えない可能性があるが、wiring_map.py内の全is_on関数は
    実測で全て `lambda c: ...` の形（引数名 c で統一）なので実用上問題ない。

    【なぜ「要確認」であって「エラー」でないか、仕様より】サブパラメータ的な
    キー（例：閾値・倍率など、本体機能に付随し単独ノードを持たない設定）は
    正当に配線図へ現れないことがある。ここでの判定は機械的な文字列照合に
    すぎず、意味的な正しさまでは判定できない。

    戻り値：
        {
          "heuristic": True,
          "total_bool_settings": int,
          "uncovered": [{"key": str, "doc": str}, ...],
        }
    """
    if nodes is None or taro_defaults is None or bools is None:
        # 【なぜ遅延importか】このモジュール自体をwiring_map/config非依存の
        #   まま保ち、検証コードから合成データ（before/after）を渡して
        #   ロジックだけをテストできるようにするため（仕様の合成検証のため）。
        import run.wiring_map as _wm
        import run.config as _cfg
        if nodes is None:
            nodes = _wm.NODES
        if taro_defaults is None:
            taro_defaults = _cfg.TARO_DEFAULTS
        if bools is None:
            bools = _cfg._BOOLS

    # 全is_on関数のソースコードを1つの文字列にまとめて照合する（どのノードが
    # 参照しているかまでは区別しないが、健康診断としてはキー単位の有無で十分）。
    sources = []
    for node in nodes:
        fn = node[6]
        if fn is None:
            continue
        try:
            sources.append(inspect.getsource(fn))
        except (OSError, TypeError):
            # 動的に作られた関数等でソースが取れない場合はスキップ（無視して継続）
            continue
    combined_source = "\n".join(sources)

    uncovered = []
    for key, (default, doc, _envname) in taro_defaults.items():
        if key not in bools:
            continue
        ref = f"c.{key}"
        if ref not in combined_source:
            uncovered.append({"key": key, "doc": doc})

    uncovered.sort(key=lambda x: x["key"])
    return {
        "heuristic": True,
        "total_bool_settings": len(bools),
        "uncovered": uncovered,
    }


if __name__ == "__main__":
    unwired_result = find_unwired_classes()
    print(f"[未配線検出・要確認（自動判定）] taro_core全クラス数="
          f"{unwired_result['total_classes']} / "
          f"taro_setup.pyから構築確認できた数={unwired_result['constructed_classes']} / "
          f"未配線とみなした数={len(unwired_result['unwired'])}")
    for c in unwired_result["unwired"][:10]:
        print(f"  {c['name']}  ({c['file']})")

    bool_result = find_uncovered_bool_settings()
    print(f"\n[bool設定カバレッジ・要確認（自動判定）] bool設定総数="
          f"{bool_result['total_bool_settings']} / "
          f"要確認とみなした数={len(bool_result['uncovered'])}")
    for u in bool_result["uncovered"]:
        print(f"  {u['key']}: {u['doc']}")
