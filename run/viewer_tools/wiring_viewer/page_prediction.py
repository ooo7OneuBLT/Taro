"""TaroMap ⑦「予測のしくみ（実測）」ページ。

【なぜ要るか、仕様より】ユーザーが既存の①配線図ページ（38機構の概念図）を見て
こう評価した：

    正直配線図見ても何も情報を得られなかった。現状何がどこに情報がわたってるのか、
    例えば何をもとに何を予測してるのかみたいなのが全く分からない。

概念上の箱と矢印だけでは「量」が見えない。このページは
`scan_dimensions.measure_dimensions()`（工程1の実測モジュール）が実際に太郎の層を
組み立てて測った値をそのまま使い、「感覚→融合→脳→予測」という一直線の流れと、
その核心である「学習側と正解側の2経路に分かれて予測誤差を作る仕組み」を、
数字と図とやさしい日本語の説明で見せる。

【このページが触ってよいもの・触らないもの】
  ・scan_dimensions.py はimportして使うだけ（変更しない・自分で再計測ロジックを作らない）。
  ・page_wiring.py（既存38機構の概念図）は変更しない。役割が違う（あちらは「配線の一覧」、
    このページは「予測という1つの流れを深掘りする」）。

【感覚のON/OFF判定について】
`run/wiring_map.py` の NODES id（insula/prop/vest/touch/vision）を使って
`is_on(node, cfg)` を呼ぶ。id の意味づけについては scan_dimensions.py 側の
コメントにある通り、"insula" は wiring_map.py上は「まとめる(fuse)」列に分類されて
いるが、常にON（lambdaが無いnode）であり、コード上の実体（Insula()クラス）は
scan_dimensions.py の "insula" と同一なので、id参照としてそのまま使ってよい
（工程1の報告に明記済みの想定外）。

【対数スケールについて】
視覚(約213万パラメータ)と内受容(320パラメータ)の比は約6,600倍あるため、線形の
棒グラフでは内受容・前庭感覚の棒が消えてしまう。ここでは常用対数(log10)で棒の
長さを決め、凡例に「対数目盛りである」ことを明記する（仕様の必須要件）。

【QScrollAreaについて、main_window.pyの注意を踏まえて】
このページは内容が縦に長い（5感覚＋融合層＋脳＋予測の2経路＋遠心性コピー＋注記）
ため、page_health.py と同じ考え方で「このページ専用に1個だけ」QScrollAreaを持つ。
QStackedWidget全体を包む2個目は作らない（main_window.py側は変更しない）。
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer import scan_dimensions  # noqa: E402
from run.viewer_tools.wiring_viewer.theme import get_theme  # noqa: E402
from run.wiring_map import is_on, node_index  # noqa: E402

# 感覚を並べる順序。scan_dimensions.py の _SENSE_SPECS と同じ順（融合層で連結する
# 順そのもの）。ここにハードコードせず measured["融合層"]["統合する感覚id"] から
# 毎回取る（実測モジュール側の仕様変更に追従するため）。

# 棒グラフ（対数スケール）の最小・最大ピクセル幅。
_BAR_MIN_PX = 24
_BAR_MAX_PX = 340


def _bar_width_px(value: float, vmin: float, vmax: float) -> int:
    """パラメータ数を対数スケールで棒の幅(px)に変換する。

    【なぜ対数か、仕様より】視覚(約213万)と内受容(320)の比は約6,600倍あり、
    線形だと小さい箱が消えてしまう。
    """
    value = max(value, 1)
    vmin = max(vmin, 1)
    vmax = max(vmax, 1)
    if vmax <= vmin:
        return _BAR_MAX_PX
    lo, hi = math.log10(vmin), math.log10(vmax)
    frac = (math.log10(value) - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    return int(round(_BAR_MIN_PX + frac * (_BAR_MAX_PX - _BAR_MIN_PX)))


def _box(text: str, color: str, bg: str = "transparent", bold: bool = False) -> QLabel:
    """予測経路図に使う「箱」ラベル。"""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    weight = "700" if bold else "500"
    lbl.setStyleSheet(
        f"border: 1.5px solid {color}; border-radius: 8px; padding: 8px 10px; "
        f"background: {bg}; font-weight: {weight};"
    )
    return lbl


def _arrow(text: str = "→") -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    f = lbl.font()
    f.setPointSize(f.pointSize() + 3)
    lbl.setFont(f)
    return lbl


def _note(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setObjectName("predNote")
    return lbl


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("predSectionTitle")
    return lbl


class PagePrediction(QWidget):
    """⑦予測のしくみ（実測）ページ。1ページ＝1つのQScrollArea。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pagePrediction")
        self._theme_name = "windows11"
        self._theme = get_theme(self._theme_name)
        self._cfg = None  # None = （実装したもの全部）

        # 初回の実測（キャッシュがあれば使う。無ければここで作られる）。
        self._measured = scan_dimensions.measure_dimensions(cfg=self._cfg, use_cache=True)
        self._node_idx = node_index()

        self._sense_rows = {}       # nid -> {container, effect, off_label}
        self._efference_effect = None
        self._efference_off_label = None

        self._build_ui()
        self.apply_theme(self._theme_name)

    # ---------------------------------------------------------------- UI
    # 【なぜここまで詰めるか、2026-08-15】1920x1080だけでなく1280x720でも
    #   このページ専用のQScrollAreaにスクロールを残してはいけない（仕様の検証項目）。
    #   実測したところ、素直に書くと中身の高さが約1130pxになり720pxウィンドウでは
    #   大きくはみ出した（`run/tools/check_wiring_viewer_prediction_page.py`で実測・
    #   修正前は1280x720でmaximum=446だった）。そこで、長い説明文の多くは
    #   常時表示ではなく**ホバー時のツールチップ**に変えた（page_wiring.pyの
    #   ノードホバーで詳細を出す設計と同じ「ホバーで詳細」という考え方の流用）。
    #   数字（次元・パラメータ数）と、④の「なぜ2つに分けるか」だけは常時表示のまま
    #   残す（★最重要の核心説明はホバーに隠さない）。
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- 実験選択（⑥）。page_wiring.py の _exp_combo と同じ考え方 ------
        top = QHBoxLayout()
        top.setContentsMargins(10, 6, 10, 4)
        top.addWidget(QLabel("表示する実験:"))
        self._exp_combo = QComboBox()
        self._exp_combo.setObjectName("predExpCombo")
        self._exp_combo.addItem("（実装したもの全部・どの実験にも縛られない）", None)
        exp_dir = os.path.join(_ROOT, "E", "experiments")
        try:
            names = sorted(f for f in os.listdir(exp_dir) if f.endswith(".json"))
        except OSError:
            names = []
        for name in names:
            self._exp_combo.addItem(name, os.path.join(exp_dir, name))
        self._exp_combo.currentIndexChanged.connect(self._on_experiment_changed)
        top.addWidget(self._exp_combo, 1)
        outer.addLayout(top)

        scroll = QScrollArea()
        scroll.setObjectName("predScroll")
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner.setObjectName("predInner")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 4, 10, 8)
        lay.setSpacing(4)

        lay.addWidget(self._build_senses_section())
        lay.addWidget(self._build_fusion_section())
        lay.addWidget(self._build_brain_section())
        lay.addWidget(self._build_prediction_section())
        lay.addWidget(self._build_efference_section())
        lay.addWidget(self._build_footer())

        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        self._refresh_on_off()
        self._refresh_footer()

    # ------------------------------------------------------ ① 感覚エンコーダ
    def _build_senses_section(self) -> QWidget:
        box = QFrame()
        box.setObjectName("predSectionBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(2)
        lay.addWidget(_section_title("① 感覚エンコーダ（5つ） ── 生の感覚を脳向けの短い数値に変換"))
        lay.addWidget(_note(
            "棒の長さ＝パラメータ数（対数目盛り。線形だと視覚以外の棒が消えるため）。"
            "各行にマウスを乗せると、その感覚が何を捉えるかの説明が出ます。"))

        senses = self._measured["感覚"]
        order = self._measured["融合層"]["統合する感覚id"]
        values = [senses[nid]["パラメータ数"] for nid in order]
        vmin, vmax = min(values), max(values)

        for nid in order:
            info = senses[nid]
            row = self._build_sense_row(nid, info, vmin, vmax)
            lay.addWidget(row)
        return box

    def _build_sense_row(self, nid: str, info: dict, vmin: float, vmax: float) -> QWidget:
        container = QFrame()
        container.setObjectName("senseRow")
        container.setToolTip(info["入力の説明"])
        h = QHBoxLayout(container)
        h.setContentsMargins(3, 1, 3, 1)

        name_lbl = QLabel(info["日本語"])
        name_lbl.setFixedWidth(96)
        name_lbl.setStyleSheet("font-weight: 600;")
        h.addWidget(name_lbl)

        bar_w = _bar_width_px(info["パラメータ数"], vmin, vmax)
        bar = QFrame()
        bar.setObjectName("senseBar")
        bar.setFixedSize(bar_w, 12)
        h.addWidget(bar, 0, Qt.AlignmentFlag.AlignVCenter)

        num_lbl = QLabel(
            f"出力{info['出力次元']}次元／パラメータ{info['パラメータ数']:,}")
        h.addWidget(num_lbl)

        off_label = QLabel("OFF（この実験では未使用）")
        off_label.setObjectName("senseOffLabel")
        off_label.setVisible(False)
        h.addWidget(off_label)

        h.addStretch(1)

        effect = QGraphicsOpacityEffect(container)
        effect.setOpacity(1.0)
        container.setGraphicsEffect(effect)

        self._sense_rows[nid] = {
            "container": container, "bar": bar, "off_label": off_label, "effect": effect,
        }
        return container

    # ------------------------------------------------------------- ② 融合層
    def _build_fusion_section(self) -> QWidget:
        fusion = self._measured["融合層"]
        box = QFrame()
        box.setObjectName("predSectionBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(2)
        lay.addWidget(_section_title("② 融合層（fusion） ── ①の5つを1本のベクトルにまとめる"))
        note = _note(
            f"統合する感覚：{'、'.join(fusion['統合する感覚id'])}　／　"
            f"出力{fusion['出力次元']}次元　／　"
            f"パラメータ数：学習側{fusion['パラメータ数_学習側']:,}　"
            f"凍結・正解側{fusion['パラメータ数_凍結正解側']:,}")
        lay.addWidget(note)
        return box

    # --------------------------------------------------------------- ③ 脳
    def _build_brain_section(self) -> QWidget:
        brain = self._measured["脳"]
        box = QFrame()
        box.setObjectName("predSectionBox")
        box.setToolTip(brain["代表値の前提"])
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(2)
        lay.addWidget(_section_title("③ 脳（TaroBrainWithMotor） ── ②を受け取り行動を作る"))
        lay.addWidget(_note(
            f"入力{brain['入力次元']}次元（＝②の出力）　／　"
            f"出力{brain['出力次元']}次元（＝行動）　／　"
            f"パラメータ数{brain['パラメータ数']:,}　"
            "（マウスを乗せると代表値の前提を表示）"))
        return box

    # ------------------------------------------------------- ④ 予測の仕組み
    def _build_prediction_section(self) -> QWidget:
        pred = self._measured["予測構造"]
        c = self._theme
        box = QFrame()
        box.setObjectName("predSectionBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(2)
        lay.addWidget(_section_title(
            "④ 予測の仕組み ── ここが今回いちばん見せたかった部分"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)

        accent = c["accent"]
        gray = c["sub"]

        a_label = QLabel("経路A（学習側）")
        grid.addWidget(a_label, 0, 0)
        grid.addWidget(_box("いまの感覚", accent), 0, 1)
        grid.addWidget(_arrow(), 0, 2)
        fusion_a = _box("fusion（学習中）", accent)
        fusion_a.setToolTip(pred["学習側"])
        grid.addWidget(fusion_a, 0, 3)
        grid.addWidget(_arrow(), 0, 4)
        grid.addWidget(_box("脳の中の予測", accent, bold=True), 0, 5)

        b_label = QLabel("経路B（正解側）")
        grid.addWidget(b_label, 1, 0)
        grid.addWidget(_box("いまの感覚", gray), 1, 1)
        grid.addWidget(_arrow(), 1, 2)
        fusion_b = _box("target_fusion（凍結）", gray)
        fusion_b.setToolTip(pred["正解側"])
        grid.addWidget(fusion_b, 1, 3)
        grid.addWidget(_arrow(), 1, 4)
        grid.addWidget(_box("予測の正解", gray, bold=True), 1, 5)

        merge_note = QLabel("↓↓　上の2つ（脳の中の予測／予測の正解）を比べた差")
        merge_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        merge_note.setObjectName("predNote")
        grid.addWidget(merge_note, 2, 1, 1, 5)

        error_box = _box("予測誤差", c["role_deviation"], bold=True)
        grid.addWidget(error_box, 3, 3)

        lay.addLayout(grid)

        why_lbl = QLabel(f"なぜ2つに分けるのか：{pred['なぜ分けるか']}")
        why_lbl.setWordWrap(True)
        why_lbl.setObjectName("predWhyLabel")
        lay.addWidget(why_lbl)
        return box

    # ------------------------------------------------------- ⑤ 遠心性コピー
    def _build_efference_section(self) -> QWidget:
        eff = self._measured["遠心性コピー"]
        box = QFrame()
        box.setObjectName("predSectionBox")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(2)
        lay.addWidget(_section_title("⑤ 遠心性コピー ── 自分が出した命令を覚えておく"))

        diagram = QHBoxLayout()
        b1 = _box("脳が出した行動", self._theme["accent"])
        b2 = _box("次の瞬間の「直前の行動」として脳に戻る", self._theme["accent"])
        b1.setToolTip(eff["説明"])
        b2.setToolTip(eff["説明"])
        diagram.addWidget(b1)
        diagram.addWidget(_arrow())
        diagram.addWidget(b2)
        lay.addLayout(diagram)

        off_label = QLabel("この実験では遠心性コピーはOFFです（直前の行動は常にゼロ扱い）")
        off_label.setObjectName("efferenceOffLabel")
        off_label.setVisible(False)
        lay.addWidget(off_label)

        effect = QGraphicsOpacityEffect(box)
        effect.setOpacity(1.0)
        box.setGraphicsEffect(effect)

        self._efference_effect = effect
        self._efference_off_label = off_label
        return box

    # --------------------------------------------------------------- ⑦ 注記
    def _build_footer(self) -> QWidget:
        box = QFrame()
        box.setObjectName("predFooter")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(8, 4, 8, 4)
        note = _note(
            "注意：寸法は新生児相当の体を仮定した代表値です（実際の実験の体格とは"
            "多少ずれます）。")
        lay.addWidget(note)
        self._gen_time_label = QLabel("")
        self._gen_time_label.setObjectName("predGenTimeLabel")
        lay.addWidget(self._gen_time_label)
        lay.addStretch(1)
        return box

    # ---------------------------------------------------------- 実験の選択
    def _on_experiment_changed(self, _index):
        path = self._exp_combo.currentData()
        if not path:
            self._cfg = None
        else:
            try:
                from run.config import Config
                from run.main import load_spec
                spec = load_spec(path)
                self._cfg = Config.from_spec(spec)
            except Exception:
                # 実験ファイルが壊れていても地図自体は出し続ける（page_wiring.pyと同じ考え方）。
                self._cfg = None
        self._measured = scan_dimensions.measure_dimensions(cfg=self._cfg, use_cache=True)
        self._refresh_on_off()
        self._refresh_footer()

    def _on(self, nid: str) -> bool:
        if self._cfg is None:
            return True
        node = self._node_idx.get(nid)
        if node is None:
            return True
        return is_on(node, self._cfg)

    def _refresh_on_off(self):
        for nid, row in self._sense_rows.items():
            on = self._on(nid)
            row["effect"].setOpacity(1.0 if on else 0.35)
            row["off_label"].setVisible(not on)

        enabled = bool(self._measured["遠心性コピー"]["有効か"])
        if self._efference_effect is not None:
            self._efference_effect.setOpacity(1.0 if enabled else 0.35)
        if self._efference_off_label is not None:
            self._efference_off_label.setVisible(not enabled)

    def _refresh_footer(self):
        gen_time = self._measured.get("生成時刻", "不明")
        from_cache = self._measured.get("キャッシュから", False)
        cache_txt = "キャッシュから読みました" if from_cache else "いま実測しました"
        self._gen_time_label.setText(
            f"実測モジュールの生成時刻：{gen_time}　（{cache_txt}）")

    # ------------------------------------------------------------- テーマ
    def apply_theme(self, theme_name: str):
        self._theme_name = theme_name
        c = get_theme(theme_name)
        self._theme = c
        self.setStyleSheet(f"""
            QWidget#pagePrediction {{ background: {c['bg']}; }}
            QWidget#predInner {{ background: {c['bg']}; }}
            QLabel {{ color: {c['ink']}; }}
            QLabel#predNote {{ color: {c['sub']}; font-size: 12px; }}
            QLabel#predSectionTitle {{
                font-size: 15px; font-weight: 700; color: {c['ink']}; padding-top: 4px;
            }}
            QLabel#predWhyLabel {{
                color: {c['role_deviation']}; font-weight: 600;
                border: 1px dashed {c['role_deviation']}; border-radius: 6px; padding: 6px;
            }}
            QLabel#senseDesc {{ color: {c['sub']}; font-size: 11px; }}
            QLabel#senseOffLabel, QLabel#efferenceOffLabel {{
                color: {c['role_deviation']}; font-weight: 600;
            }}
            QLabel#predGenTimeLabel {{ color: {c['sub']}; font-size: 11px; }}
            QFrame#predSectionBox {{
                background: {c['card']}; border: 1px solid {c['line']}; border-radius: 8px;
            }}
            QFrame#predFooter {{
                background: {c['card']}; border: 1px dashed {c['line']}; border-radius: 8px;
            }}
            QFrame#senseRow {{ border-bottom: 1px solid {c['line']}; }}
            QFrame#senseBar {{ background: {c['accent']}; border-radius: 3px; }}
            QComboBox#predExpCombo {{
                background: {c['card']}; color: {c['ink']};
                border: 1px solid {c['line']}; border-radius: 6px; padding: 4px 8px;
            }}
            QScrollArea#predScroll {{ border: none; background: {c['bg']}; }}
        """)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    w = PagePrediction()
    w.resize(1200, 900)
    w.show()
    sys.exit(app.exec())
