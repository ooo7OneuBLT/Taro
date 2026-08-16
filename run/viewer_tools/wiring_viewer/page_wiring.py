"""TaroMap ①配線図ページ。太郎の中身38機構・40本の依存関係を1枚の絵で見せる。

【なぜQGraphicsView/QGraphicsSceneか、仕様より】NodeGraphQtは不採用が確定済み
（ユーザーの判断）。標準のQt Graphics Viewフレームワークだけでノードグラフを組む。

【位置計算・線の引き方について、仕様より】`run/tools/wiring.py` の
`layout()` / `_edge_path()` / `_fit()` と**同じ式**を使う（車輪の再発明をしない）。
ただし wiring.py はSVG文字列を組み立てる関数であり、アンダースコア始まりの
実装詳細（`_edge_path` 等）を外部からimportして使う想定のAPIではない。
仕様も「移植してよい」と書いている＝**同じ式をこのファイルにも書く**
（コピーではなく、意味は完全に同じ・数値は完全に同じにする）。
片方だけ直すと形が食い違うので、位置計算の式を直すときは wiring.py 側にも
同じ変更が要ることを踏まえること（今回はそちらを触らない）。

【primary_axis / deviation_keywordsの扱い、仕様より】
現状 wiring_map.NODES は全38行が primary_axis=None, deviation_keywords=() の
ままである（ユーザーの確定判断待ち＝「宿題が見える状態」が正しい）。
⇒ 全ノードに「未分類」バッジが出るのが**正しい結果**。バッジを出さない実装や、
  None を別の値に読み替える実装は誤り。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from run.viewer_tools.wiring_viewer.theme import get_theme  # noqa: E402
from run.wiring_map import (  # noqa: E402
    COLUMNS,
    EDGES,
    LOWER,
    NODES,
    POSITIONS,
    TIER_STYLE,
    is_on,
)

# ---------------------------------------------------------------- 絵の寸法
# 【なぜ run/tools/wiring.py と同じ数値か】仕様の「同じ式を使う」に従う。
#   ここだけ値が違うと、HTML版とQt版で見た目の縮尺が食い違い、比較できなくなる。
BOX_W, BOX_H = 176, 46
COL_GAP, ROW_GAP = 34, 16
PAD_X, PAD_Y = 18, 64
LOWER_GAP = 54

# primary_axis の値 → (バッジの文字, テーマのroleキー)
#   None（未確定）は必ず「未分類」になる。将来 innate/emergent/tool が入ったら
#   対応する行に自動で切り替わる（if/elif の分岐がその担保）。
_AXIS_BADGE = {
    None: ("未分類", "role_unclassified"),
    "innate": ("生得", "role_innate"),
    "emergent": ("創発", "role_emergent"),
    "tool": ("道具", "role_tool"),
}


def _layout():
    """機構の置き場所 (x, y) を決める。run/tools/wiring.py の layout() と同じ式。

    注意：wiring.py 側にある「自動並べ替え（バリセンター法）を試したが悪化した」
    という調査結果はそのまま踏襲する＝ここでも並べ替えはしない（定義順のまま）。
    POSITIONS（人が決めた位置）があれば上書きするのも同じ。
    """
    pos, col_title = {}, []
    order = {}
    for key, jp, eng in list(COLUMNS) + list(LOWER):
        order[key] = [n[0] for n in NODES if n[3] == key]

    x = PAD_X
    upper_h = 0
    for key, jp, eng in COLUMNS:
        col_title.append((x, PAD_Y - 30, jp, eng))
        y = PAD_Y
        for nid in order[key]:
            pos[nid] = (x, y)
            y += BOX_H + ROW_GAP
        upper_h = max(upper_h, y)
        x += BOX_W + COL_GAP
    total_w = x - COL_GAP + PAD_X
    y0 = upper_h + LOWER_GAP
    x = PAD_X
    lower_h = y0
    for key, jp, eng in LOWER:
        col_title.append((x, y0 - 30, jp, eng))
        y = y0
        for nid in order[key]:
            pos[nid] = (x, y)
            y += BOX_H + ROW_GAP
        lower_h = max(lower_h, y)
        x += BOX_W + COL_GAP
    if POSITIONS:
        for nid, xy in POSITIONS.items():
            if nid in pos:
                pos[nid] = (float(xy[0]), float(xy[1]))
        total_w = max(total_w, max(p[0] for p in pos.values()) + BOX_W + PAD_X)
        lower_h = max(lower_h, max(p[1] for p in pos.values()) + BOX_H)
    return pos, col_title, total_w, lower_h + PAD_Y


def _edge_geom(p1, p2):
    """箱と箱を結ぶベジエ曲線の始点・制御点2つ・終点・ラベル位置。

    run/tools/wiring.py の `_edge_path()` と同じ式（戻る線は下に回すS字曲線）。
    戻り値の形だけQt向けに変えている（SVGパス文字列ではなく座標タプル）。
    """
    x1, y1 = p1[0] + BOX_W, p1[1] + BOX_H / 2
    x2, y2 = p2[0], p2[1] + BOX_H / 2
    if x2 < x1:        # 戻る線（体→感覚など）は下に回す
        x1, y1 = p1[0] + BOX_W / 2, p1[1] + BOX_H
        x2, y2 = p2[0] + BOX_W / 2, p2[1] + BOX_H
        dy = 26 + abs(x2 - x1) * 0.03
        c1 = (x1, y1 + dy)
        c2 = (x2, y2 + dy)
        lx, ly = (x1 + x2) / 2, max(y1, y2) + dy * 0.75
        return (x1, y1), c1, c2, (x2, y2), (lx, ly)
    mx = (x1 + x2) / 2
    c1 = (mx, y1)
    c2 = (mx, y2)
    return (x1, y1), c1, c2, (x2, y2), (mx, (y1 + y2) / 2 - 5)


def _fit_font_px(text, width, size):
    """字が箱から出ないよう、長ければ文字を小さくする。_fit() と同じ式。"""
    w = sum(size if ord(c) > 0x2000 else size * 0.55 for c in text)
    return size if w <= width else max(7.5, size * width / w)


class NodeItem(QGraphicsRectItem):
    """1つの機構を表す箱。ドラッグで動かせる（このページの中だけのローカル状態）。"""

    def __init__(self, node, on_move):
        super().__init__(0, 0, BOX_W, BOX_H)
        self.node = node          # wiring_map.NODES の1タプル
        self.nid = node[0]
        self._on_move = on_move
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self._on_move is not None:
                self._on_move(self.nid)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        scene = self.scene()
        if scene is not None and hasattr(scene, "page") and scene.page is not None:
            scene.page._show_detail(self.node)
        super().hoverEnterEvent(event)

    def mousePressEvent(self, event):
        scene = self.scene()
        if scene is not None and hasattr(scene, "page") and scene.page is not None:
            scene.page._show_detail(self.node)
        super().mousePressEvent(event)


def _node_item_of(item):
    """クリック／ホバーされたitem（子の文字・バッジのことがある）から NodeItem を辿る。"""
    while item is not None and not isinstance(item, NodeItem):
        item = item.parentItem()
    return item


class WiringView(QGraphicsView):
    """ホイールでズーム（マウス位置中心）。背景ドラッグで全体移動、ノードはそのまま動く。"""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._scale = 1.0

    def wheelEvent(self, event):
        # 【なぜ数行で足りるか】QGraphicsViewのtransformを使うだけで、
        #   AnchorUnderMouseがマウス位置を中心にした拡大縮小を面倒見てくれる。
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.12 if delta > 0 else (1 / 1.12)
        new_scale = self._scale * factor
        if new_scale < 0.25 or new_scale > 4.0:
            return
        self._scale = new_scale
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        # 背景（何もない場所）をドラッグしたときだけ画面全体を動かす。
        #   ノードの上ではノード自体のドラッグ（ItemIsMovable）に任せる。
        item = self.itemAt(event.pos())
        node_item = _node_item_of(item)
        if node_item is None:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)


class WiringScene(QGraphicsScene):
    """page（PageWiring）への参照を持たせるためだけの薄いサブクラス。"""

    def __init__(self, page):
        super().__init__()
        self.page = page


class PageWiring(QWidget):
    """①配線図ページ。画面の主役はグラフ本体。情報パネルはその下に小さく添える。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme_name = "windows11"
        self._theme = get_theme(self._theme_name)
        self._cfg = None                 # None = 全部ONの地図
        self._node_items = {}            # nid -> NodeItem
        self._edge_items = []            # (a, b, path_item, arrow_item)
        self._edges_by_node = {}         # nid -> [index in _edge_items]

        self._build_ui()
        self._build_scene()
        self.apply_theme(self._theme_name)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        top = QHBoxLayout()
        self._exp_label = QLabel("表示する実験:")
        self._exp_combo = QComboBox()
        self._exp_combo.addItem("（実装したもの全部・どの実験にも縛られない地図）", None)
        exp_dir = os.path.join(_ROOT, "E", "experiments")
        try:
            names = sorted(f for f in os.listdir(exp_dir) if f.endswith(".json"))
        except OSError:
            names = []
        for name in names:
            self._exp_combo.addItem(name, os.path.join(exp_dir, name))
        self._exp_combo.currentIndexChanged.connect(self._on_experiment_changed)
        top.addWidget(self._exp_label)
        top.addWidget(self._exp_combo, 1)
        outer.addLayout(top)

        self._scene = WiringScene(self)
        self._view = WiringView(self._scene)
        self._view.setMinimumHeight(360)
        outer.addWidget(self._view, 1)          # グラフが主役＝伸縮の主体

        # 情報パネル（画面の脇役。小さく1つだけ）
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(120)
        self._detail.setPlaceholderText("機構にマウスを乗せる、またはクリックすると説明が出ます。")
        outer.addWidget(self._detail, 0)

    # ------------------------------------------------------------- シーン
    def _build_scene(self):
        pos, col_titles, total_w, total_h = _layout()
        self._pos = pos

        # ---- 列見出し ----------------------------------------------------
        for x, y, jp, eng in col_titles:
            t1 = self._scene.addSimpleText(jp)
            t1.setPos(x, y - 14)
            t1.setFont(QFont("Yu Gothic", 10, QFont.Weight.Bold))
            t1.setData(0, "col_jp")
            t2 = self._scene.addSimpleText(eng)
            t2.setPos(x, y)
            f = QFont("Yu Gothic", 8)
            f.setItalic(True)
            t2.setFont(f)
            t2.setData(0, "col_eng")

        # ---- 線（先に描いて箱の下に置く） ----------------------------------
        for a, b, label, ckey in EDGES:
            if a not in pos or b not in pos:
                continue
            path_item = QGraphicsPathItem()
            path_item.setZValue(0)
            self._scene.addItem(path_item)
            arrow_item = QGraphicsPolygonItem()
            arrow_item.setZValue(0)
            self._scene.addItem(arrow_item)
            idx = len(self._edge_items)
            self._edge_items.append((a, b, path_item, arrow_item))
            self._edges_by_node.setdefault(a, []).append(idx)
            self._edges_by_node.setdefault(b, []).append(idx)

        # ---- 箱 ------------------------------------------------------------
        for node in NODES:
            nid = node[0]
            if nid not in pos:
                continue
            x, y = pos[nid]
            item = NodeItem(node, on_move=self._on_node_moved)
            item.setPos(x, y)
            self._scene.addItem(item)
            self._node_items[nid] = item
            self._decorate_node(item, node)

        self._redraw_all_edges()
        self._scene.setSceneRect(-40, -40, total_w + 80, total_h + 80)

    def _decorate_node(self, item, node):
        """箱の中身（日本語名・英語名・Tier色帯・未分類/生得等バッジ）を作る。

        バッジ・文字はすべて item（NodeItem）の子にする。子にしておくことで
        QGraphicsView.itemAt()がバッジをクリックしても _node_item_of() で
        親のNodeItemまで辿れる（ホバー/クリックの判定を1箇所にまとめるため）。
        """
        nid, jp, eng, col, tier, desc, _fn, axis, _dev = node
        color_hex, _tier_jp, _tier_en = TIER_STYLE.get(tier, TIER_STYLE[None])

        # 左端の根拠色帯
        bar = QGraphicsRectItem(0, 0, 5, BOX_H, item)
        bar.setBrush(QBrush(QColor(color_hex)))
        bar.setPen(QPen(Qt.PenStyle.NoPen))
        bar.setData(0, "tierbar")

        fs = _fit_font_px(jp, BOX_W - 22, 12.5)
        t_jp = QGraphicsSimpleTextItem(jp, item)
        t_jp.setPos(12, 4)
        t_jp.setFont(QFont("Yu Gothic", int(round(fs))))
        t_jp.setData(0, "njp")

        fs2 = _fit_font_px(eng, BOX_W - 22, 9.5)
        t_en = QGraphicsSimpleTextItem(eng, item)
        t_en.setPos(12, 20)
        f_en = QFont("Yu Gothic", max(7, int(round(fs2))))
        f_en.setItalic(True)
        t_en.setFont(f_en)
        t_en.setData(0, "neng")

        # ---- primary_axis バッジ（現状は全て None＝未分類が正しい）---------
        badge_text, role_key = _AXIS_BADGE.get(axis, _AXIS_BADGE[None])
        badge = QGraphicsSimpleTextItem(badge_text, item)
        badge.setFont(QFont("Yu Gothic", 7))
        badge.setPos(12, BOX_H - 15)
        badge.setData(0, "badge")
        badge.setData(1, role_key)

    def _on_node_moved(self, nid):
        for idx in self._edges_by_node.get(nid, ()):
            self._redraw_edge(idx)

    def _redraw_all_edges(self):
        for idx in range(len(self._edge_items)):
            self._redraw_edge(idx)

    def _redraw_edge(self, idx):
        a, b, path_item, arrow_item = self._edge_items[idx]
        item_a, item_b = self._node_items.get(a), self._node_items.get(b)
        if item_a is None or item_b is None:
            return
        pa = (item_a.pos().x(), item_a.pos().y())
        pb = (item_b.pos().x(), item_b.pos().y())
        (x1, y1), (cx1, cy1), (cx2, cy2), (x2, y2), _label_pos = _edge_geom(pa, pb)
        path = QPainterPath(QPointF(x1, y1))
        path.cubicTo(QPointF(cx1, cy1), QPointF(cx2, cy2), QPointF(x2, y2))
        path_item.setPath(path)

        # 矢印の先端：終点への接線方向（制御点2→終点）を向く小さな三角形
        import math
        dx, dy = x2 - cx2, y2 - cy2
        ang = math.atan2(dy, dx) if (dx or dy) else 0.0
        size = 7.0
        p_tip = QPointF(x2, y2)
        left = QPointF(x2 - size * math.cos(ang - 0.4), y2 - size * math.sin(ang - 0.4))
        right = QPointF(x2 - size * math.cos(ang + 0.4), y2 - size * math.sin(ang + 0.4))
        arrow_item.setPolygon(QPolygonF([p_tip, left, right]))

    # ------------------------------------------------------------- テーマ
    def apply_theme(self, theme_name: str):
        """テーマに合わせて色を塗り直す（QApplication全体へは当てない・局所QSSのみ）。"""
        self._theme_name = theme_name
        c = get_theme(theme_name)
        self._theme = c

        self.setStyleSheet(f"""
            QWidget {{ background: {c['bg']}; color: {c['ink']}; }}
            QComboBox {{
                background: {c['card']}; color: {c['ink']};
                border: 1px solid {c['line']}; border-radius: 6px; padding: 4px 8px;
            }}
            QTextEdit {{
                background: {c['card']}; color: {c['ink']};
                border: 1px solid {c['line']}; border-radius: 6px;
            }}
        """)
        self._view.setBackgroundBrush(QBrush(QColor(c["bg"])))

        for nid, item in self._node_items.items():
            self._paint_node_theme(item)
        for a, b, path_item, arrow_item in self._edge_items:
            self._paint_edge_theme(a, b, path_item, arrow_item)
        self._repaint_off_state()

    def _paint_node_theme(self, item):
        c = self._theme
        item.setBrush(QBrush(QColor(c["card"])))
        node = item.node
        tier = node[4]
        color_hex, _jp, _en = TIER_STYLE.get(tier, TIER_STYLE[None])
        pen = QPen(QColor(color_hex))
        pen.setWidthF(1.6)
        item.setPen(pen)
        for child in item.childItems():
            kind = child.data(0)
            if kind == "njp":
                child.setBrush(QBrush(QColor(c["ink"])))
            elif kind == "neng":
                child.setBrush(QBrush(QColor(c["sub"])))
            elif kind == "badge":
                role_key = child.data(1) or "role_unclassified"
                child.setBrush(QBrush(QColor(c.get(role_key, c["sub"]))))

    def _paint_edge_theme(self, a, b, path_item, arrow_item):
        c = self._theme
        pen = QPen(QColor(c["sub"]))
        pen.setWidthF(1.9)
        path_item.setPen(pen)
        arrow_item.setBrush(QBrush(QColor(c["sub"])))
        arrow_item.setPen(QPen(Qt.PenStyle.NoPen))

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
            except Exception as exc:
                # 実験ファイルが壊れていても地図自体は出し続ける（is_onと同じ考え方）。
                self._cfg = None
                self._detail.setPlainText(f"（この実験ファイルは読み込めなかった: {exc}）")
        self._repaint_off_state()

    def _repaint_off_state(self):
        c = self._theme
        for node in NODES:
            nid = node[0]
            item = self._node_items.get(nid)
            if item is None:
                continue
            on = True if self._cfg is None else is_on(node, self._cfg)
            item.setOpacity(1.0 if on else 0.32)
            pen = item.pen()
            pen.setStyle(Qt.PenStyle.SolidLine if on else Qt.PenStyle.DashLine)
            item.setPen(pen)
        for a, b, path_item, arrow_item in self._edge_items:
            node_a, node_b = self._node_lookup(a), self._node_lookup(b)
            on_a = True if self._cfg is None else is_on(node_a, self._cfg)
            on_b = True if self._cfg is None else is_on(node_b, self._cfg)
            live = on_a and on_b
            path_item.setOpacity(1.0 if live else 0.35)
            arrow_item.setOpacity(1.0 if live else 0.35)
            pen = path_item.pen()
            pen.setStyle(Qt.PenStyle.SolidLine if live else Qt.PenStyle.DashLine)
            path_item.setPen(pen)

    def _node_lookup(self, nid):
        for n in NODES:
            if n[0] == nid:
                return n
        return None

    # ---------------------------------------------------------- 情報パネル
    def _show_detail(self, node):
        nid, jp, eng, col, tier, desc, _fn, axis, dev_keywords = node
        axis_label = _AXIS_BADGE.get(axis, _AXIS_BADGE[None])[0]
        if axis is None:
            axis_line = f"区分: 未分類（primary_axis未確定）"
        else:
            axis_line = f"区分: {axis_label}（primary_axis={axis}）"
        dev_line = ("逸脱キーワード: " + "、".join(dev_keywords)
                    if dev_keywords else "逸脱キーワード: なし")
        tier_line = f"根拠(Tier): {tier or '未記載'}"
        text = f"{jp}（{eng}）\n{tier_line}　{axis_line}\n{dev_line}\n\n{desc}"
        self._detail.setPlainText(text)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    w = PageWiring()
    w.resize(1400, 860)
    w.show()
    sys.exit(app.exec())
