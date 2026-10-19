"""脳全体：41係を全員同時に1回ずつ進める。

二重バッファ：毎回、すべての係が「前の回に届いた値」だけを読んで outbox を書く。
全員の outbox を集め終わってから、次の回の「届いた値」として一度に入れ替える。
だから係を呼ぶ順番に結果が依存せず、図の中の輪（目的↔門番など）もそのまま回る。
代わりに、矢印1本ごとに1回の遅れが出る（声から目まで7回ほど）。
"""

from __future__ import annotations

from typing import Any, Iterable

from torch import nn

from . import wiring
from .boxes import ENV, Box, make_all
from .formats import FormatError, check, zero


class Brain(nn.Module):
    def __init__(self, boxes: dict[str, Box] | None = None, check_formats: bool = True) -> None:
        super().__init__()
        boxes = make_all() if boxes is None else boxes
        if sorted(boxes) != sorted(wiring.BOX_IDS):
            raise ValueError("箱の組が配線表の41係と合わない")
        self.boxes = nn.ModuleDict({bid: boxes[bid] for bid in wiring.BOX_IDS})
        self.edges = {e.key: e for e in wiring.EDGES}
        self._incoming = {bid: wiring.incoming(bid) for bid in wiring.BOX_IDS}
        # 速くするため：矢印の鍵（出す係, 受ける係）を先に作っておく（Edge.key を毎回呼ぶと回の時間の1割を食う）
        self._in_keys = {bid: [(e.src, e.key) for e in self._incoming[bid]] for bid in wiring.BOX_IDS}
        self._out_dsts = {bid: {e.dst for e in self.boxes[bid].outputs} for bid in wiring.BOX_IDS}
        self.check_formats = check_formats
        self.skip: set[str] = set()   # 動かさない係（出すものはゼロのまま）。盤面の世界で使わない係を飛ばして速くする
        self.reset()

    def reset(self) -> None:
        """全部の矢印をゼロにし、各係の中身も空にする。"""
        self.buffer: dict[tuple[str, str], Any] = {
            k: zero(e.fmt, e.size) for k, e in self.edges.items()
        }
        self.delivered: set[tuple[str, str]] = set()
        self.t = 0
        for box in self.boxes.values():
            box.reset()

    def tick(
        self,
        external: dict[str, Any] | None = None,
        order: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """全係を同時に1回進める。

        external : {係id: 環境からの入力}。その係の inbox に ENV キーで入る
        order    : 係を呼ぶ順番（テスト用。どの順でも結果は同じになるはず）
        戻り値   : {係id: 環境への指令}。outbox に ENV キーを返した係だけ
        """
        external = external or {}
        unknown = set(external) - set(self.boxes)
        if unknown:
            raise KeyError(f"知らない係への入力: {sorted(unknown)}")

        order = list(wiring.BOX_IDS if order is None else order)
        if sorted(order) != sorted(wiring.BOX_IDS):
            raise ValueError("order は41係をちょうど1回ずつ含むこと")

        outboxes = {}
        skip = getattr(self, "skip", ())
        for bid in order:
            if bid in skip:
                outboxes[bid] = self.boxes[bid].passthrough()
                continue
            buf = self.buffer
            inbox = {src: buf[key] for src, key in self._in_keys[bid]}
            if bid in external:
                inbox[ENV] = external[bid]
            outboxes[bid] = self.boxes[bid].step(inbox)

        new_buffer: dict[tuple[str, str], Any] = {}
        to_env: dict[str, Any] = {}
        for bid in wiring.BOX_IDS:
            out = dict(outboxes[bid])
            if ENV in out:
                to_env[bid] = out.pop(ENV)
            expected = self._out_dsts[bid]
            if out.keys() != expected:
                missing, extra = expected - set(out), set(out) - expected
                raise FormatError(f"{bid} の outbox が配線と合わない（不足 {sorted(missing)}・余分 {sorted(extra)}）")
            for dst, value in out.items():
                if self.check_formats:
                    edge = self.edges[(bid, dst)]
                    check(edge.fmt, value, edge.size, name=f"{bid}→{dst}")
                new_buffer[(bid, dst)] = value

        self.buffer = new_buffer
        self.delivered = set(new_buffer)
        self.t += 1
        return to_env
