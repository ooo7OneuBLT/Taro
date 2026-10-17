"""意味の辞書（3段目 計画）。

**段階3（2026-10-15）：経験から学ぶ形に替えた（初めてのNN化、仕様書 付録B）。**
中身は 64×64 の結び付きの表 W（見た目×語）＝1層の線形のNN。重みはヘッブ則で足し込む。
* 覚える（learning=True のときだけ）：語の荷札 q が届いた回に、名簿から届いた物の荷札 a で
  W += ETA × a qᵀ。名簿に結び付ける行が無ければ覚えない。1つの発話で1回だけ
  （q は発話のあった1回だけ届く。続けて届いたら最初の回だけ覚える）
* **覚える物は物の名簿が決める（段階4、2026-10-16）**：名簿（objfile）が「最後に MIN_LOOK_S（1秒）以上
  続けて見た行」の荷札（その物のかけらの見た目を、見えた回数で平均したもの）を毎回送ってくる。語が届いた回に
  それで覚える。段階3の「直前1秒の窓」と「同じ物とみなす似かた0.7」は名簿へ移り、要らなくなった。
  名前は、聞いた瞬間ではなく少し前から注意していた物に結び付く（記憶ベース・要確認）［Tier3］
* 思い出す：r = W q。‖r‖ ≥ RECALL_MIN なら目的の係へ noisy(r/‖r‖) を出す。未満ならゼロ
  （聞いた回数が少ないうちは何も出さない）
* **覚え方B（段階6、2026-10-17、ユーザー承認の仕様書）**：名簿の「名前を結び付ける行」1つではなく、
  名簿に載っている全部の行と結び付ける。強さは、聞く前 LOOK_WINDOW_S（3秒）にその行を見ていた割合 s_i：
  W += ETA × Σ s_i a_i qᵀ。1回ごとには違う物も混ざるが、いつも一緒に出てくる組だけが積み上がる
  （場面をまたいだ学習。Smith・Yu 2008）。learn_mode="A" で段階4の覚え方（名前を結び付ける行だけ）に戻せる
* **聞いた語の名残り（2026-10-17、ユーザー承認）**：覚え方Bは、語が届いた回にすぐ覚えず、名残りが薄れるまで
  （5×WORD_TRACE_S）待ってから1回で覚える。名簿から毎回届く「今見ている行」の荷札を自分で覚えておき、
  届く前 LOOK_WINDOW_S（3秒）に見ていた回は重み1、届いたあとに見ていた回は重み exp(-経過/WORD_TRACE_S)
  で足し、重みの合計で割る（物を見ていなかった回は数えない。合わせた強さは今までと同じ1）。語が画面の出た直後に来て
  まだ物を見ていない回も、すぐあとに見た物と結び付く。1歳には頭の中で語を唱え直す力が無く、聞いた音は短く薄れるだけ
  （名づけのあと0.5〜0.9秒まで結び付く）。段階6の試験場（太郎5人×6語）で、捨てる語 53%→約0%、正しい物が一番の語
  15→25/30。τ＝0.5〜2秒のどれでも 23〜25/30 で、値のちょうどには頼らない。唱え直す力は後の「目的を持ち続ける係」で足す
* **逆向きに引く（段階7、2026-10-17、ユーザー承認の仕様書）**：名簿の「今見ている行」の荷札 a から w = Wᵀa を作り、
  ‖w‖ ≥ RECALL_MIN なら言う語の係（wordplan）へ noisy(w/‖w‖) を出す（見ている物 → 語）。表 W は同じ。
  思い出す強さの線も、語→物と同じ RECALL_MIN（同じ組み合わせ3回で出る）
学ぶこと自体のやり方（ヘッブ則・ETA・RECALL_MIN・覚え方）は生まれつき＝手で決める。
W の中身だけが経験で変わる。W は buffer なので save.py の保存に入り、reset() では消えない。

段階2の手で登録した表（register / load_table、`core_v2/tools/register_appearance.py`）も残す。
表に1件でも登録してあれば、段階2と同じく表を引く（段階3の合格テスト②で使う）。
"""

import math

import torch

from ..formats import TAG_DIM, DTYPE, make_tag
from ..tagging import NOISE_SIGMA, noisy, word_tag
from .base import Box

MATCH_COS = 0.9          # 段階2の表引き
ETA = 1.0                # 1回に足す強さ［仮置き］
WORD_TRACE_S = 1.0       # 聞いた語の名残りが薄れる速さ（生まれつき・1歳の値。育つと伸びる。逸脱リスト登録）［Tier2］
LOOK_WINDOW_S = 3.0      # 語が届く前、見ていた物を数える秒数（物の名簿と同じ。Yu・Smith 2012）［Tier2］
TICK_S = 0.1
TRACE_TICKS = int(round(5 * WORD_TRACE_S / TICK_S))   # 名残りがほぼ消える（1%）まで
RECALL_MIN = 2.5         # 同じ組み合わせ3回で ‖r‖≈3、2回では出ない［仮置き・合格テスト③④で確かめる］


class Semantic(Box):
    BOX_ID = "semantic"

    def __init__(self, learning: bool = False, learn_mode: str = "B") -> None:
        super().__init__()
        self.table: dict[str, torch.Tensor] = {}   # 段階2：{語: 見た目の荷札}
        self.register_buffer("W", torch.zeros(TAG_DIM, TAG_DIM, dtype=DTYPE))
        self.register_buffer("n_learn", torch.zeros((), dtype=torch.int64))
        self.learning = bool(learning)
        self.learn_mode = learn_mode
        self._gen = torch.Generator().manual_seed(0)
        self.reset()

    def reset(self) -> None:
        self._prev_heard = False
        self._held_hist: list = []       # 直前 LOOK_WINDOW_S に毎回見ていた行の荷札（見ていない回は None）
        self._pending: list[dict] = []   # 名残りの残っている語 {q, 経過回数, 足した見た目, 足した重み}
        self.last_event = ""      # 記録用："learn" / "learn_skip"（名簿に結び付ける行が無い） / "recall" / ""
        self.last_recall_norm = 0.0
        self.last_word_norm = 0.0

    # ---- 段階2の表 ----
    def register(self, word: str, appearance_tag) -> None:
        self.table[word] = make_tag(appearance_tag)

    def load_table(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        for w, tag in blob["table"].items():
            self.register(w, tag)
        return blob.get("meta", {})

    # ---- 段階3の学習 ----
    def learn(self, appearance_tag, word_q, strength: float = 1.0) -> None:
        a, q = make_tag(appearance_tag), make_tag(word_q)
        self.W += ETA * strength * torch.outer(a, q)

    def targets(self, lst) -> list:
        """名簿の列から、結び付ける (見た目の荷札, 強さ) の組を選ぶ。A：名前を結び付ける行（最後の組）だけ、強さ1。
        B：最後の組より前の行それぞれ、強さ＝見ていた割合（山の高さ）。"""
        if not lst:
            return []
        if self.learn_mode == "A":
            a = lst[-1][0]
            return [(a, 1.0)] if float(torch.linalg.vector_norm(a)) > 0.0 else []
        return [(a, float(m.max())) for a, m in lst[:-2]
                if float(m.max()) > 0.0 and float(torch.linalg.vector_norm(a)) > 0.0]

    def recall(self, q):
        r = self.W @ q
        n = float(torch.linalg.vector_norm(r))
        self.last_recall_norm = n
        return make_tag(r) if n >= RECALL_MIN else None

    def recall_word(self, lst):
        """段階7：逆向きに引く。名簿の「今見ている行」の荷札 a から語 Wᵀa を思い出す（同じ表を逆に引くだけ）。"""
        self.last_word_norm = 0.0
        if not lst or len(lst) < 2 or float(torch.linalg.vector_norm(lst[-2][0])) == 0.0:
            return None
        r = self.W.T @ make_tag(lst[-2][0])
        n = float(torch.linalg.vector_norm(r))
        self.last_word_norm = n
        return make_tag(r) if n >= RECALL_MIN else None

    def save_learned(self, path, meta: dict | None = None) -> None:
        torch.save({"W": self.W.clone(), "n_learn": int(self.n_learn), "meta": meta or {}}, path)

    def load_learned(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.W.copy_(blob["W"])
        self.n_learn.fill_(int(blob.get("n_learn", 0)))
        return blob.get("meta", {})

    def step(self, inbox):
        out = self.passthrough()
        self.last_event = ""
        q = inbox.get("listen")
        heard = q is not None and float(torch.linalg.vector_norm(q)) > 0.0
        first = heard and not self._prev_heard
        self._prev_heard = heard
        self._trace_step(inbox.get("objfile"), q if (self.learning and first and self.learn_mode == "B"
                                                     and not self.table) else None)
        if not self.table:
            w = self.recall_word(inbox.get("objfile"))
            if w is not None:   # 雑音は自分専用の乱数で（足したことで段階3〜6の乱数の並び＝結果が変わらないように）
                out["wordplan"] = make_tag(w + NOISE_SIGMA * torch.randn(TAG_DIM, generator=self._gen))
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
        if self.learning and first and self.learn_mode == "A":
            pairs = self.targets(inbox.get("objfile"))
            if not pairs:
                self.last_event = "learn_skip"
            else:
                for a, s_i in pairs:
                    self.learn(a, q, s_i)
                self.n_learn += 1
                self.last_event = "learn"
        return out

    def skip(self, n: int, lst=None) -> None:
        """何も聞かない n 回を飛ばす（段階7の試験場用）。見ている物（名簿の列 lst）はその間ずっと同じとする。
        1回ずつ _trace_step を回したのと同じ結果になる（名残りの重みは等比の和でまとめて足す）。"""
        held = None
        if lst and len(lst) >= 2 and float(torch.linalg.vector_norm(lst[-2][0])) > 0.0:
            held = make_tag(lst[-2][0])
        keep_n = int(round(LOOK_WINDOW_S / TICK_S))
        self._held_hist = (self._held_hist + [held] * min(n, keep_n))[-keep_n:]
        self._prev_heard = False
        keep = []
        for p in self._pending:
            m = min(n, TRACE_TICKS - p["k"])
            if held is not None:
                g = sum(math.exp(-(p["k"] + j) * TICK_S / WORD_TRACE_S) for j in range(1, m + 1))
                p["v"] += g * held
                p["w"] += g
            p["k"] += m
            if p["k"] < TRACE_TICKS:
                keep.append(p)
            else:
                self._learn_pending(p)
        self._pending = keep

    def _trace_step(self, lst, new_q) -> None:
        """覚え方B：今見ている行を覚えておき、名残りの残っている語ごとに足す。名残りが消えたら覚える。"""
        held = None
        if lst and len(lst) >= 2 and float(torch.linalg.vector_norm(lst[-2][0])) > 0.0:
            held = make_tag(lst[-2][0])
        self._held_hist.append(held)
        del self._held_hist[:-int(round(LOOK_WINDOW_S / TICK_S))]
        keep = []
        for p in self._pending:
            p["k"] += 1
            if held is not None:
                g = math.exp(-p["k"] * TICK_S / WORD_TRACE_S)
                p["v"] += g * held
                p["w"] += g
            if p["k"] < TRACE_TICKS:
                keep.append(p)
            else:
                self._learn_pending(p)
        self._pending = keep
        if new_q is not None:
            seen = [h for h in self._held_hist if h is not None]
            v = torch.stack(seen).sum(0) if seen else torch.zeros(TAG_DIM, dtype=DTYPE)
            self._pending.append({"q": new_q.clone(), "k": 0, "v": v, "w": float(len(seen))})

    def _learn_pending(self, p) -> None:
        if p["w"] == 0.0:
            self.last_event = "learn_skip"
            return
        self.W += ETA * torch.outer(p["v"] / p["w"], make_tag(p["q"]))
        self.n_learn += 1
        self.last_event = "learn"
