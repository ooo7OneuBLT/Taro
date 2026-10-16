"""意味の辞書（3段目 計画）。

**段階3（2026-10-15）：経験から学ぶ形に替えた（初めてのNN化、仕様書 付録B）。**
中身は 64×64 の結び付きの表 W（見た目×語）＝1層の線形のNN。重みはヘッブ則で足し込む。
* 覚える（learning=True のときだけ）：語の荷札 q が届いた回に、目の入口から前の回に届いた
  物のうち、山の頂点が視野の真ん中から LEARN_RADIUS（3マス≒8度。2026-10-16 に 4×4 マスから広げた）以内にある物の
  荷札 a を探し、W += ETA × a qᵀ。真ん中に物が無ければ覚えない。1つの発話で1回だけ
  （q は発話のあった1回だけ届く。続けて届いたら最初の回だけ覚える）
* **覚える物は「聞く直前に一番長く真ん中にあった物」（2026-10-16）**：毎回、真ん中の物の荷札を
  直前 LEARN_WINDOW 回分（1秒）覚えておき、語が届いた回に、その中で一番多く真ん中にあった物
  （同じ物かは荷札の cos ≥ SAME_OBJ_COS でまとめる）の荷札の平均で覚える。親の決まり（3秒見たら言う）と
  太郎が見続けて離れる時間（約3秒）が重なり、聞いた次の回にはもう別の物へ跳んでいて、「こっぷ」を
  ボールで2回覚えた（5回中）。語は耳から2回遅れで届くので、窓（10回）はその遅れも含む。
  名前は、聞いた瞬間ではなく少し前から注意していた物に結び付く（記憶ベース・要確認）［Tier3］
* 思い出す：r = W q。‖r‖ ≥ RECALL_MIN なら目的の係へ noisy(r/‖r‖) を出す。未満ならゼロ
  （聞いた回数が少ないうちは何も出さない）
学ぶこと自体のやり方（ヘッブ則・窓・ETA・RECALL_MIN）は生まれつき＝手で決める。
W の中身だけが経験で変わる。W は buffer なので save.py の保存に入り、reset() では消えない。

段階2の手で登録した表（register / load_table、`core_v2/tools/register_appearance.py`）も残す。
表に1件でも登録してあれば、段階2と同じく表を引く（段階3の合格テスト②で使う）。
"""

import math

import torch

from ..formats import GRID, TAG_DIM, DTYPE, make_tag
from ..tagging import noisy, word_tag
from .base import Box
from collections import deque

MATCH_COS = 0.9          # 段階2の表引き
LEARN_WINDOW = 10        # 覚えるとき振り返る回数（1秒）［Tier3・仮置き］
SAME_OBJ_COS = 0.7       # 窓の中で同じ物とみなす荷札の cos。0.9 だと、同じ物でも目の入口の雑音だけで
                         # cos 平均0.86・最小0.78（実測、200組）になり別の物に分かれる。別の物どうしは0.2〜0.5
ETA = 1.0                # 1回に足す強さ［仮置き］
RECALL_MIN = 2.5         # 同じ組み合わせ3回で ‖r‖≈3、2回では出ない［仮置き・合格テスト③④で確かめる］
C0, C1 = GRID // 2 - 2, GRID // 2 + 2     # 真ん中 4×4 マス（行・列 12〜15）。今は使わない（下の LEARN_RADIUS）
CENTER = (GRID - 1) / 2.0
LEARN_RADIUS = 3.0   # 覚えるとき「見ている物」とみなす、山の頂点の真ん中からの距離（マス、≒8度）。
                     # 2026-10-16：4×4 マス（約±5度）だと、太郎の目は跳びが足りず4〜8度ずれて止まることが多く、
                     # 親が名前を言っても（10度以内を3秒）、名づけ1回分で6回中5回見送った。親の条件に合わせて広げた


def centered_tag(objs):
    """物ごとの (荷札, 地図) から、山の頂点が真ん中から LEARN_RADIUS マス以内にある物の荷札を1つ返す。
    2つ以上あれば、山がいちばん高い物。荷札ゼロの組（目の動きの地図）は除く。無ければ None。"""
    best, best_v = None, 0.0
    for tag, m in objs or []:
        if float(torch.linalg.vector_norm(tag)) == 0:
            continue
        r, c = divmod(int(torch.argmax(m)), GRID)
        if math.hypot(r - CENTER, c - CENTER) <= LEARN_RADIUS:
            v = float(m.max())
            if v > best_v:
                best, best_v = tag, v
    return best


class Semantic(Box):
    BOX_ID = "semantic"

    def __init__(self, learning: bool = False) -> None:
        super().__init__()
        self.table: dict[str, torch.Tensor] = {}   # 段階2：{語: 見た目の荷札}
        self.register_buffer("W", torch.zeros(TAG_DIM, TAG_DIM, dtype=DTYPE))
        self.register_buffer("n_learn", torch.zeros((), dtype=torch.int64))
        self.learning = bool(learning)
        self.reset()

    def reset(self) -> None:
        self._prev_heard = False
        self._recent = deque(maxlen=LEARN_WINDOW)   # 直前の回ごとの「真ん中の物の荷札」（無ければ None）
        self.last_event = ""      # 記録用："learn" / "learn_skip"（真ん中に物が無い） / "recall" / ""
        self.last_recall_norm = 0.0

    # ---- 段階2の表 ----
    def register(self, word: str, appearance_tag) -> None:
        self.table[word] = make_tag(appearance_tag)

    def load_table(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for w, tag in blob["table"].items():
            self.register(w, tag)
        return blob.get("meta", {})

    # ---- 段階3の学習 ----
    def learn(self, appearance_tag, word_q) -> None:
        a, q = make_tag(appearance_tag), make_tag(word_q)
        self.W += ETA * torch.outer(a, q)
        self.n_learn += 1

    def recall(self, q):
        r = self.W @ q
        n = float(torch.linalg.vector_norm(r))
        self.last_recall_norm = n
        return make_tag(r) if n >= RECALL_MIN else None

    def save_learned(self, path, meta: dict | None = None) -> None:
        torch.save({"W": self.W.clone(), "n_learn": int(self.n_learn), "meta": meta or {}}, path)

    def load_learned(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.W.copy_(blob["W"])
        self.n_learn.fill_(int(blob.get("n_learn", 0)))
        return blob.get("meta", {})

    def attended_tag(self):
        """窓の中で一番多く真ん中にあった物の荷札（その物の荷札の平均）。無ければ None。"""
        tags = [t for t in self._recent if t is not None]
        if not tags:
            return None
        best = max(range(len(tags)),
                   key=lambda i: (sum(float(tags[i] @ u) >= SAME_OBJ_COS for u in tags), i))
        group = [u for u in tags if float(tags[best] @ u) >= SAME_OBJ_COS]
        return make_tag(torch.stack(group).mean(0))

    def step(self, inbox):
        out = self.passthrough()
        self.last_event = ""
        if self.learning:
            self._recent.append(centered_tag(inbox.get("eyein")))
        q = inbox.get("listen")
        heard = q is not None and float(torch.linalg.vector_norm(q)) > 0.0
        first = heard and not self._prev_heard
        self._prev_heard = heard
        if not heard:
            return out
        if self.table:
            best, best_cos = None, MATCH_COS
            for w, tag in self.table.items():
                c = float(word_tag(w) @ q)
                if c >= best_cos:
                    best, best_cos = w, c
            if best is not None:
                out["goal"] = noisy(self.table[best])
            return out
        r = self.recall(q)
        if r is not None:
            out["goal"] = noisy(r)
            self.last_event = "recall"
        if self.learning and first:
            a = self.attended_tag()
            if a is None:
                self.last_event = "learn_skip"
            else:
                self.learn(a, q)
                self.last_event = "learn"
        return out
