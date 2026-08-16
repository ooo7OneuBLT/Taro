"""体の部位分類（「器官の説明」ページの小さな絞り込みフィルタ用）。

【なぜ独自の一覧を作らないか、仕様より】
体の部位一覧は `taro_core/src/senses/somatosensory_cortex.py` の `_BODY_GROUPS` に
既に定義されている（触覚グループ→実body名→皮質割り当てカテゴリの対応表）。
ここで新しい台帳を増やすと二重管理になり、どちらかが古びる。⇒ `_BODY_GROUPS` を
そのままimportして使い、このファイルは読み取り専用のヘルパーだけを持つ。

【この分類の使いどころ】
「②器官の説明」ページの中で、感覚器系のファイルを体の部位でざっくり絞り込む
**小さなフィルタ**として使う（独立ページにはしない、というユーザーの確定判断）。

【フィルタの限界（正直に書く）】
`_BODY_GROUPS` はもともと触覚（体性感覚）のためのグループ定義であり、視覚・
前庭感覚（三半規管・耳石器）などのファイルはそもそもこの定義に含まれない。
`CATEGORY_FILE_KEYWORDS` によるカテゴリ→ファイルの対応づけは、taro_core/src配下の
ファイル名からの**簡易的な推測**であり、実際の配線（どのニューロンがどの体部位の
入力を受けるか）を機械的に裏取りしたものではない（Tier3・簡易フィルタ）。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from taro_core.src.senses.somatosensory_cortex import _BODY_GROUPS  # noqa: E402


def body_region_names() -> list:
    """_BODY_GROUPS のキー（グループ名。例: "right_palm"・"head"・"chest"）一覧を返す。

    件数は _BODY_GROUPS の要素数と常に一致する（この関数は検索するだけで、
    _BODY_GROUPS 自体は書き換えない）。
    """
    return [gname for gname, _body_names, _category in _BODY_GROUPS]


def region_categories() -> list:
    """_BODY_GROUPS の3要素目（皮質割り当てカテゴリ）を、重複を除いた出現順で返す。

    somatosensory_cortex.py 時点では
    ["head", "trunk", "upper_limb", "lower_limb"] の4種類。
    ページ側のフィルタ用ドロップダウンは、26件ある細かいグループ名より、この
    粗いカテゴリの方が扱いやすいため、UIにはこちらを主に使う想定。
    """
    seen = []
    for _gname, _body_names, category in _BODY_GROUPS:
        if category not in seen:
            seen.append(category)
    return seen


def region_of(group_name: str) -> str:
    """触覚グループ名（例: "right_palm"）が属するカテゴリ（例: "upper_limb"）を返す。

    Raises:
        ValueError: group_name が _BODY_GROUPS に存在しないとき。
    """
    for gname, _body_names, category in _BODY_GROUPS:
        if gname == group_name:
            return category
    known = body_region_names()
    raise ValueError(
        f"region_of: 未知のグループ名 '{group_name}'。\n"
        f"  既知のグループ名: {known}")


# カテゴリ→関連ファイル名キーワードの簡易対応表（モジュールdocstring参照：厳密対応ではない）。
# taro_core/src配下のファイル名・相対パスに、ここに挙げたキーワードのいずれかが
# 部分一致すれば「関連度が高い」とみなす、という単純なルール。
CATEGORY_FILE_KEYWORDS = {
    "head": [
        "retina", "vision", "otolith", "semicircular", "insula",
        "superior_colliculus", "vocal_tract", "atnr",
    ],
    "trunk": [
        "somatosensory", "internal_state", "homeostasis", "stomach",
        "blood_vessel", "insula", "adenosine",
    ],
    "upper_limb": [
        "somatosensory", "proprioceptive", "infant_limbs",
        "double_touch", "grasp_reflex", "motor_cortex", "touch_adaptation",
    ],
    "lower_limb": [
        "somatosensory", "proprioceptive", "infant_limbs", "motor_cortex",
    ],
}


def files_related_to_category(category: str, entries: list) -> list:
    """scan_docstrings()が返す entries のうち、category に簡易的に関連すると
    思われるものだけを返す（"path" または "filename" にキーワードが部分一致するか）。

    厳密な配線対応ではない簡易フィルタ（モジュールdocstring参照）。
    未知の category（CATEGORY_FILE_KEYWORDSに無いキー）を渡すと空リストを返す
    （例外にはしない。UIの「全て」表示等で使い回しやすくするため）。
    """
    keywords = CATEGORY_FILE_KEYWORDS.get(category, [])
    if not keywords:
        return []
    out = []
    for entry in entries:
        haystack = entry.get("path", "") + "/" + entry.get("filename", "")
        if any(k in haystack for k in keywords):
            out.append(entry)
    return out
