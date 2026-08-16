"""taro_core/src 配下の全 .py ファイルから、モジュールdocstringを抽出する道具。

【なぜ都度スキャンか、仕様より】
一覧をハードコードすると、ファイルが増えたときに静かに古びる（「手で書く一覧は
必ず腐る」問題）。このアプリはキャッシュを作らず、呼ばれるたびに taro_core/src を
rglobで読み直し、その場の実際のファイル一覧からdocstringを取り直す
（body/brain/senses/wrapper・深さ無制限・新しいフォルダが増えても自動で追従する）。

【暫定閾値について、仕様より】
`THIN_DOCSTRING_THRESHOLD`（既定200字）を下回るdocstringを「要加筆」候補として
扱っているが、200という数値そのものに強い根拠は無い（Tier3・ユーザーの暫定判断）。
根拠が無いことを画面側（page_organs.py）にも明記すること。閾値を変えるときは
この1箇所と表示文言の両方を直す。
"""
import ast
from pathlib import Path

# 【なぜこの値か】根拠なし、Tier3の暫定値。ユーザーから「暫定値であることを画面に
#   明記する」よう指定されている（このファイルのdocstring・page_organs.py両方に明記）。
THIN_DOCSTRING_THRESHOLD = 200


def _taro_root() -> Path:
    """このファイルの位置（run/viewer_tools/wiring_viewer/scan_docstrings.py）から
    Taroリポジトリのルートを逆算する。"""
    return Path(__file__).resolve().parents[3]


def default_src_root() -> Path:
    """taro_core/src の絶対パスを返す（scan_docstrings()の既定rootと同じ計算）。"""
    return _taro_root() / "taro_core" / "src"


def scan_docstrings(root=None) -> list:
    """taro_core/src 配下（body/brain/senses/wrapper、深さ無制限、rglobで自動追従）の
    全 .py ファイルから、モジュールdocstring（ast.get_docstring相当）を抽出する。

    キャッシュはしない。呼ばれるたびにファイルシステムを読み直す
    （新規ファイルの追加・docstringの加筆が、アプリを再起動しなくても次に開いたときに
    反映される設計）。

    Args:
        root: スキャン対象のルート。Noneなら taro_core/src（既定）。
              検証コードから任意のディレクトリを渡せるように引数化してある。

    Returns:
        list[dict]  各要素は例えば：
            {"path": "taro_core/src/brain/basal_ganglia.py"（Taroルートからの相対パス、
                      スラッシュ区切り）,
             "rel_dir": "brain"（rootから見た相対ディレクトリ。直下なら空文字列。
                      深いフォルダなら "brain/spinal_cord" のようにスラッシュ区切り）,
             "filename": "basal_ganglia.py",
             "docstring": "...モジュールdocstring全文...",
             "char_count": 175,
             "thin": True}  # char_count < THIN_DOCSTRING_THRESHOLD の暫定閾値で
                            # 「要加筆」候補と判定したことを示す

        ファイル一覧はソート済み（rglobの結果をPathでソートしてから処理する）。
        構文エラーで ast.parse に失敗したファイル・読み込めなかったファイルは
        黙って結果から除外する（このアプリはdocstring閲覧が目的で、壊れた
        Pythonファイルの構文検査は目的ではないため）。
        docstringを持たない __init__.py（多くが空）は表示対象から除外する。
        それ以外のdocstring無しファイルは docstring="" (char_count=0, thin=True) で
        一覧に含める（「加筆されていない」という事実そのものが情報のため）。
    """
    src_root = _taro_root()
    if root is None:
        root = default_src_root()
    else:
        root = Path(root)

    results = []
    if not root.exists():
        return results

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # 読めないファイルは一覧から静かに除外する（docstring閲覧が目的であり
            #   壊れたファイルの検出はこのツールの役目ではない）。
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        docstring = ast.get_docstring(tree) or ""

        if path.name == "__init__.py" and not docstring:
            # 中身の無い __init__.py は表示対象から外す（仕様：「中身があれば
            #   表示対象にする」の裏返し。空のパッケージ宣言だけのファイルを
            #   一覧に混ぜても情報が無い）。
            continue

        char_count = len(docstring)
        try:
            rel_path = path.relative_to(src_root).as_posix()
        except ValueError:
            # root が taro_root の外を指す検証用呼び出し等の保険。相対化できなければ
            #   絶対パス文字列をそのまま使う。
            rel_path = path.as_posix()

        rel_dir = path.relative_to(root).parent.as_posix()
        if rel_dir == ".":
            rel_dir = ""

        results.append({
            "path": rel_path,
            "rel_dir": rel_dir,
            "filename": path.name,
            "docstring": docstring,
            "char_count": char_count,
            "thin": char_count < THIN_DOCSTRING_THRESHOLD,
        })

    return results
