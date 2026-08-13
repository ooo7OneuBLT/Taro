"""Viewer(PySide6版)で使い回す小さなUI部品。

【これは何か】`run/viewer_tools/e_viewer_qt.py` の第2段階移植（体・環境タブの
残り4区画）で必要になった、Qt標準部品では足りない2つの部品をまとめたもの。
将来の区画（駆動タブ・感覚と報酬タブ等）でも同じ部品を使い回す想定
（仕様 作業記録（非公開）
2026-08-13_Viewerのモダン化_第2段階_体環境タブと状態表示.md 2-1節）。

【入っているもの】
    FloatSlider        浮動小数点値を扱うスライダー（QSliderは整数しか
                        扱えないため、実際の物理量[度・秒・倍率など]と
                        スライダーの整数値を相互変換する薄いラッパー）
    CollapsibleSection  クリックで開閉できる区画（旧版 e_viewer.py の
                        `Section` クラスと同じ役割。43関節のスライダーを
                        部位ごとに折りたたむのに使う）

【設計判断】
    ・FloatSlider は「実際の値」だけを外部に公開する（内部の整数ステップ数は
      隠す）。呼び出し側が`res`ぶんの丸め誤差を意識しなくて済むようにするため。
    ・値の変換は四捨五入だけの単純な式にとどめ、対数スケール等の特殊な
      変換は持たない（旧版のスライダーは全て線形だったため。将来もし
      対数スケールが要る区画が出てきたら、そのときに別クラスを足す）。
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class FloatSlider(QtWidgets.QWidget):
    """浮動小数点の値を扱うスライダー（QSliderの整数値を裏で相互変換する）。

    使い方は素朴：`FloatSlider("肩（開き）", -90, 90, 1.0, init=0.0)` のように
    ラベル・下限・上限・刻み幅・初期値を渡す。`value()` で今の実値（float）を
    読み、`valueChanged`（float型のQtシグナル）で変化を検知できる。
    """

    valueChanged = QtCore.Signal(float)

    def __init__(self, label, lo, hi, res, init=None, unit="", decimals=None,
                 width_label=140, length=220, note=None, parent=None):
        super().__init__(parent)
        self._lo = float(lo)
        self._hi = float(hi)
        self._res = float(res) if res else 1.0
        self._unit = unit
        # 表示桁数：res が 1 以上なら整数寄り、細かいほど桁を増やす。
        if decimals is None:
            decimals = 0 if self._res >= 1.0 else (1 if self._res >= 0.1 else 3)
        self._decimals = decimals
        self._steps = max(1, int(round((self._hi - self._lo) / self._res)))

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        row = QtWidgets.QHBoxLayout()
        self.name_label = QtWidgets.QLabel(label)
        self.name_label.setFixedWidth(width_label)
        row.addWidget(self.name_label)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self._steps)
        self.slider.setFixedWidth(length)
        row.addWidget(self.slider)

        self.value_label = QtWidgets.QLabel("")
        self.value_label.setFixedWidth(70)
        row.addWidget(self.value_label)
        row.addStretch(1)
        outer.addLayout(row)

        if note:
            note_label = QtWidgets.QLabel(note)
            note_label.setStyleSheet("color: #666;")
            note_label.setWordWrap(True)
            outer.addWidget(note_label)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.setValue(self._lo if init is None else init)

    # ---- 変換 -----------------------------------------------------------
    def _int_to_value(self, i):
        return self._lo + i * self._res

    def _value_to_int(self, v):
        v = max(self._lo, min(self._hi, float(v)))
        return int(round((v - self._lo) / self._res))

    # ---- 公開API ----------------------------------------------------------
    def value(self):
        """今の実値（float）。"""
        return self._int_to_value(self.slider.value())

    def setValue(self, v, block_signal=False):
        """実値（float）でスライダーを設定する。

        block_signal=True のときは valueChanged を発火させない
        （プログラムからの初期化・同期で無限ループを避けるために使う。
        旧版 tkinter の `_syncing` フラグと同じ役割）。
        """
        i = self._value_to_int(v)
        if block_signal:
            self.slider.blockSignals(True)
        self.slider.setValue(i)
        if block_signal:
            self.slider.blockSignals(False)
        self._update_value_label()

    def _on_slider_changed(self, _i):
        self._update_value_label()
        self.valueChanged.emit(self.value())

    def _update_value_label(self):
        v = self.value()
        self.value_label.setText(f"{v:.{self._decimals}f}{self._unit}")


class CollapsibleSection(QtWidgets.QWidget):
    """クリックで開閉できる区画（旧版 e_viewer.py の `Section` クラスと同じ役割）。

    43個の関節スライダーを部位ごとにまとめて畳めるようにするために使う。
    """

    def __init__(self, title, opened=True, parent=None):
        super().__init__(parent)
        self._title = title
        self._opened = bool(opened)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.toggle_btn = QtWidgets.QPushButton()
        self.toggle_btn.setStyleSheet(
            "text-align: left; background: #dde3ee; font-weight: bold;")
        self.toggle_btn.clicked.connect(self._on_toggle)
        outer.addWidget(self.toggle_btn)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(14, 4, 4, 4)
        outer.addWidget(self.body)

        self.body.setVisible(self._opened)
        self._refresh_label()

    def _refresh_label(self):
        self.toggle_btn.setText(("▼ " if self._opened else "▶ ") + self._title)

    def _on_toggle(self):
        self._opened = not self._opened
        self.body.setVisible(self._opened)
        self._refresh_label()
