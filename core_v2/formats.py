"""係どうしのやり取りの書式（段階2・2026-10-15決定）。

人間の脳には決まった書式は無い（逸脱リスト行き）。係を1つずつNNに替えても
配線をいじらずに済むよう、矢印ごとに書式を1つ決めて、毎回その形だけを渡す。

| 書式        | 中身                          | 型                             |
|-------------|-------------------------------|--------------------------------|
| TAG         | 何か＝荷札                    | float32[TAG_DIM]、L2正規化（ゼロは例外） |
| MAP         | どこか＝地図                  | float32[GRID, GRID]、0〜1      |
| AMOUNT      | どれくらい＝数1つ             | float32 スカラー、0〜1         |
| MOTOR       | 筋への指令                    | float32[n]（n は矢印ごと）     |
| TAG_AND_MAP | 物ごとの荷札と場所（組）      | list[(Tag, Map)]、最大 MAX_OBJECTS 個 |
| TEXT        | 例外（耳の入口→聞き分け、聞き分け→声の係）| str（normalize_kana 済み）|

TEXT の2つめ（段階7、2026-10-17）：聞き分け→声の係。声の係は、親の語と自分の声の似かたを口の形で測り、
語を1音ずつ口の指令にするので、かなの並びが要る。代案の「音の荷札」（64個の数）では、かなの並びを
声の係が取り出し直せない（語を1音ずつ言えない）ので選ばなかった。声がかなである間（逸脱リスト その55）の例外。

値はすべて torch.Tensor（TEXT は str、TAG_AND_MAP は tuple の list）。
NNに置き換えたときにそのまま勾配が流れるようにするため。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch

TAG_DIM = 64
GRID = 28
MAX_OBJECTS = 8
DTYPE = torch.float32


class Fmt(str, Enum):
    TAG = "tag"
    MAP = "map"
    AMOUNT = "amount"
    MOTOR = "motor"
    TAG_AND_MAP = "tag_and_map"
    TEXT = "text"


class FormatError(ValueError):
    """書式違反。配線か係の中身の間違いなので、黙って直さず止める。"""


def make_tag(v) -> torch.Tensor:
    """任意の64個の数を L2 正規化して荷札にする。長さ0ならゼロのまま。"""
    t = torch.as_tensor(v, dtype=DTYPE).reshape(TAG_DIM)
    n = torch.linalg.vector_norm(t)
    return t / n if n > 0 else t


def zero(fmt: Fmt, size: int | None = None) -> Any:
    """その書式の「空っぽ」。素通しの係はこれを渡す。"""
    if fmt is Fmt.TAG:
        return torch.zeros(TAG_DIM, dtype=DTYPE)
    if fmt is Fmt.MAP:
        return torch.zeros(GRID, GRID, dtype=DTYPE)
    if fmt is Fmt.AMOUNT:
        return torch.zeros((), dtype=DTYPE)
    if fmt is Fmt.MOTOR:
        if not size:
            raise FormatError("MOTOR には size（筋の数）が要る")
        return torch.zeros(size, dtype=DTYPE)
    if fmt is Fmt.TAG_AND_MAP:
        return []
    if fmt is Fmt.TEXT:
        return ""
    raise FormatError(f"知らない書式: {fmt}")


def _check_tensor(name, value, shape, lo=None, hi=None):
    if not isinstance(value, torch.Tensor):
        raise FormatError(f"{name}: Tensor でない（{type(value).__name__}）")
    if value.dtype != DTYPE:
        raise FormatError(f"{name}: dtype {value.dtype}（{DTYPE} のはず）")
    if tuple(value.shape) != tuple(shape):
        raise FormatError(f"{name}: 形 {tuple(value.shape)}（{tuple(shape)} のはず）")
    if not torch.isfinite(value).all():
        raise FormatError(f"{name}: NaN か無限大が入っている")
    if lo is not None and (value < lo - 1e-5).any():
        raise FormatError(f"{name}: {lo} 未満の値がある")
    if hi is not None and (value > hi + 1e-5).any():
        raise FormatError(f"{name}: {hi} を超える値がある")


def _check_tag(name, value):
    _check_tensor(name, value, (TAG_DIM,))
    n = float(torch.linalg.vector_norm(value))
    if n != 0.0 and abs(n - 1.0) > 1e-3:
        raise FormatError(f"{name}: 荷札が正規化されていない（長さ {n:.4f}）")


def check(fmt: Fmt, value: Any, size: int | None = None, name: str = "値") -> None:
    """value が書式 fmt に合っているか確かめる。合わなければ FormatError。"""
    if fmt is Fmt.TAG:
        _check_tag(name, value)
    elif fmt is Fmt.MAP:
        _check_tensor(name, value, (GRID, GRID), 0.0, 1.0)
    elif fmt is Fmt.AMOUNT:
        _check_tensor(name, value, (), 0.0, 1.0)
    elif fmt is Fmt.MOTOR:
        _check_tensor(name, value, (size,))
    elif fmt is Fmt.TAG_AND_MAP:
        if not isinstance(value, list):
            raise FormatError(f"{name}: list でない（{type(value).__name__}）")
        if len(value) > MAX_OBJECTS:
            raise FormatError(f"{name}: 物が {len(value)} 個（最大 {MAX_OBJECTS}）")
        for i, pair in enumerate(value):
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise FormatError(f"{name}[{i}]: (荷札, 地図) の組でない")
            _check_tag(f"{name}[{i}].荷札", pair[0])
            _check_tensor(f"{name}[{i}].地図", pair[1], (GRID, GRID), 0.0, 1.0)
    elif fmt is Fmt.TEXT:
        if not isinstance(value, str):
            raise FormatError(f"{name}: str でない（{type(value).__name__}）")
    else:
        raise FormatError(f"知らない書式: {fmt}")
