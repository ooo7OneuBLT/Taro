"""係の箱の共通の形。

各係は step(inbox) -> outbox だけを持つ。
* inbox  : {出した係のid: 前の回にその矢印で届いた値}。入口の係には ENV キーで環境からの入力も入る
* outbox : {受ける係のid: 次に渡す値}。出ていく矢印すべてに1つずつ。
           出口の係は ENV キーで環境への指令を返してよい（Brain.tick の戻り値になる）

係どうしは inbox/outbox 以外で値をやり取りしない（ほかの係の中身を直接読まない）。
これが崩れると「全員同時に動く」が成り立たず、実行順で結果が変わる。
inbox の Tensor は他の係と共有しているので、その場で書き換えない。

既定の step は素通し：出ていく矢印それぞれに、その書式のゼロを渡す。
中身を入れるときはサブクラスで step を上書きする（配線は触らない）。
NN に置き換えるとき学習する重みは nn.Module として持てば、保存（save.py）に自動で入る。
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .. import wiring
from ..formats import zero

ENV = "env"  # 環境とのやり取りに使う特別なキー（係の id とぶつからない）


class Box(nn.Module):
    BOX_ID: str = ""
    # True の係は、出すものが届いた値だけで決まる（中に前の回から持ち越す状態が無い）。brain.tick は、届いた値が
    # 前の回とすべて同じ物なら step を呼ばず前の outbox を使う。状態を持つ係は False のまま
    PURE: bool = False

    def __init__(self) -> None:
        super().__init__()
        if self.BOX_ID not in wiring.BOX_IDS:
            raise ValueError(f"{type(self).__name__}: 配線表に無い係 id {self.BOX_ID!r}")
        self.inputs = wiring.incoming(self.BOX_ID)
        self.outputs = wiring.outgoing(self.BOX_ID)

    def __setattr__(self, name: str, value: Any) -> None:
        # 速くするため：もう普通の属性としてある名前に、重み・係でない値を入れ直すときは nn.Module の手続きを飛ばす
        # （毎回の中の状態の書き換えが多い係で、nn.Module.__setattr__ が回の時間の2割を食っていた）
        if name in self.__dict__ and not isinstance(value, (nn.Parameter, nn.Module)):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def step(self, inbox: dict[str, Any]) -> dict[str, Any]:
        return self.passthrough()

    def passthrough(self) -> dict[str, Any]:
        # ゼロの Tensor は毎回作らず使い回す（受け取った値はその場で書き換えない約束なので安全。段階7の試験場で速くするため）。
        # 物ごとの組（list）だけは毎回新しく作る
        zeros = self.__dict__.get("_zeros")
        if zeros is None:
            zeros = {e.dst: zero(e.fmt, e.size) for e in self.outputs}
            self.__dict__["_zeros"] = zeros
        return {k: ([] if isinstance(v, list) else v) for k, v in zeros.items()}

    def reset(self) -> None:
        """場面の切り替わりで中の状態を空にする。素通しの係は何もしない。"""
