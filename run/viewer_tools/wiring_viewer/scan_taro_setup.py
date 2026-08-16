"""taro_setup.py を静的解析して「実際に組み立てられているクラス」を洗い出す。

【なぜ ast か、仕様より】taro_setup.py を import すると、MuJoCo環境の構築や
torch のロードなど重い副作用が走る（このアプリは配線図を眺めるためだけの
デスクトップアプリで、太郎本体を起動する必要はない）。
⇒ ソースコードをテキストとして構文木（ast）に変換し、**実行せずに**
  「class Taro の __init__ の中で self.属性 = ClassName(...) の形になっている行」
  だけを拾う。

【契約、仕様より】この関数の戻り値の形は、他の実装担当（④⑥ページ担当）が
  「taro_core内に実在するクラスのうち、taro_setup.pyから一度も呼ばれていないもの」
  を検出するために前提として使う。戻り値の形を変えるときは呼び出し側と合わせること。

使い方：
    from run.viewer_tools.wiring_viewer.scan_taro_setup import scan_taro_setup
    info = scan_taro_setup()
    info["constructed_classes"]   # {"TouchAdaptation", "MinimalFusion", ...}
    info["self_attrs"]            # {"touch_adaptation": "TouchAdaptation", ...}
"""
import ast
import os

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
_DEFAULT_PATH = os.path.join(_ROOT, "run", "taro_setup.py")


def _find_class_taro(tree):
    """モジュール直下（トップレベル）にある `class Taro` のASTノードを返す。"""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Taro":
            return node
    return None


def _find_init(class_node):
    for node in ast.iter_child_nodes(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    return None


def _class_name_of_call(call_node):
    """`ClassName(...)` の呼び出しから、大文字始まりのクラス名だけを取り出す。

    対象：`self.x = ClassName(...)`（`=` の直後がそのままクラス名の呼び出し）。
    対象外：
      ・`self.x = some_module.ClassName(...)`（モジュール越しの属性アクセス呼び出し）
      ・`self.x = helper(...)`（小文字始まりの関数呼び出し＝クラスではない）
    判定基準は、この関数の検証に使う手検算の grep
    （`self\\.\\w+\\s*=\\s*[A-Z]\\w*\\(`＝`=` の直後が直接大文字識別子＋`(`）と
    **揃えている**。`torch.optim.Adam(...)` のような属性越しの呼び出しは、
    最終セグメントが大文字でも `=` の直後は小文字（`torch`）なので grep は拾わない
    ＝ここでも対象外にする。

    【なぜチェーン呼び出しは剥がすか、実装時に発見】taro_setup.py には
    `self.target_fusion = MinimalFusion(...).freeze()` のように、構築した
    直後にメソッドチェーンする書き方がある。素直に一番外側の呼び出しだけを
    見ると `.freeze()`（Attribute, 小文字）を見てしまい、MinimalFusion を拾えない。
    ただしこの行は `=` の直後が `MinimalFusion(` なので grep は拾う。
    ⇒ 外側が「小文字始まりの属性呼び出し（.freeze() 等）」かつ、その`value`が
    さらに Call なら、内側の Call まで剥がして見る（これは対象に含める）。
    """
    node = call_node
    while True:
        func = node.func
        if isinstance(func, ast.Name) and func.id[:1].isupper():
            return func.id
        if (isinstance(func, ast.Attribute) and not func.attr[:1].isupper()
                and isinstance(func.value, ast.Call)):
            node = func.value
            continue
        return None


def scan_taro_setup(path=None) -> dict:
    """taro_setup.py を静的解析する（importせず、astだけで読む）。

    class Taro の __init__ 内にある「self.属性 = ClassName(...)」という形の
    代入文だけを拾う。対象は **トップレベルの代入文**（if/forの中にある
    条件付き構築も拾う。ast.walk で __init__ 配下を再帰的に見るため）。

    戻り値：
      {
        "constructed_classes": set[str],   # 構築されているクラス名の集合
        "self_attrs": dict[str, str],      # self.属性名 -> クラス名
      }
    """
    target = path or _DEFAULT_PATH
    with open(target, encoding="utf-8") as fp:
        source = fp.read()
    tree = ast.parse(source, filename=target)

    class_node = _find_class_taro(tree)
    if class_node is None:
        # class Taro が見つからないのは taro_setup.py の構造が変わったということ。
        #   黙って空を返すと「未配線が0件」に見えて誤解を招くので、空を明示して返す
        #   （呼び出し側が taro_setup.py の構造変化に気づけるよう、例外にはしない
        #   ＝GUI側が落ちるのを避ける。中身が空なのは呼び出し側で判断できる）。
        return {"constructed_classes": set(), "self_attrs": {}}

    init_node = _find_init(class_node)
    if init_node is None:
        return {"constructed_classes": set(), "self_attrs": {}}

    constructed_classes = set()
    self_attrs = {}

    for node in ast.walk(init_node):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        cls_name = _class_name_of_call(node.value)
        if cls_name is None:
            continue
        for target_expr in node.targets:
            # self.属性 = ClassName(...) の形だけを拾う（タプル代入等は対象外）
            if (isinstance(target_expr, ast.Attribute)
                    and isinstance(target_expr.value, ast.Name)
                    and target_expr.value.id == "self"):
                constructed_classes.add(cls_name)
                self_attrs[target_expr.attr] = cls_name

    return {"constructed_classes": constructed_classes, "self_attrs": self_attrs}


if __name__ == "__main__":
    info = scan_taro_setup()
    print(f"構築されているクラス数: {len(info['constructed_classes'])}")
    for attr, cls in sorted(info["self_attrs"].items()):
        print(f"  self.{attr} = {cls}(...)")
