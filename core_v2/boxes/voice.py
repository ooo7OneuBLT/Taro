"""声と口の動きを組み立てる係（2段目 習慣）。

**段階7（2026-10-17、ユーザー承認の仕様書 `2026-10-17_段階7_仕様書_言える.md` §2-2）**。前の太郎の部品を移し
（移し方の文書 `2026-10-17_段階7_旧太郎の声の部品_移し方.md` 付録 C）、つながり M を足した。

* **口の帳面**（前の太郎の発話の小脳 `cerebellum_lobe/speech.py`）：「この口の指令で、この音が出た」を数える。
  音は耳から戻ってきた自分の声（聞き分ける係→この係、TEXT）で知る。逆引き（音→口の指令）は、その音が出た回数が
  いちばん多い指令。前の太郎は指令ごとの回数で比べていたが、ここは（指令, 音）の組ごとに数え、同じ指令で別の音が
  出たら古い組の数えは消す（順モデルは最新の音で上書き、前の太郎と同じ）。月齢で口が育つと、同じ指令から違う音が
  出るようになる（6か月までは何を指令しても母音、12か月までは声帯がいつも有声）。1回目の育ち（2026-10-17）では
  古い組が残り、12か月から「ぼおる」の計画が「ぽおす」になって M が止まった
* **喃語の出し方**（生まれつき）：1回に1〜2音。口の4つの数は一様にばらばらに選び、2音目は直前と同じ形に戻りやすい
  （前の太郎 B の反復の傾き +2.0。顎の上げ下げがくり返し、舌と唇が据え置かれる＝MacNeilage 1998）。
  出せる形かどうかは声の出口（体）が決める（月齢の段階・連動）。前の太郎の口の選び方の NN（ほうびでしか学ばない）は移さない
* **言う順番の箱**（前の太郎のブローカ野 `brocas_area.py`）：言う語を1音ずつ帳面の逆引きで口の指令にし、
  SYLL_TICKS 回に1音ずつ声の出口へ送る。帳面に無い音があれば言わない（大人の正解表で埋めない）
* **語の選び方**（前の太郎 B6-2、`B/src/environment/core_b.py:520-562`）：喃語を出すたびに、聞いた語を言うか自由な喃語かを選ぶ。
  語の重み＝聞いた回数、自由な喃語の重み＝**これまでに自分で出した自由な喃語の回数**。候補は全部の音が帳面にある語だけ。
  状態の一致（B6-4）は外した。つまり、ある語を言う確率＝その語を聞いた回数 ÷（聞いた語の回数の合計＋自分の喃語の回数）＝
  「耳に入った声（親の語と自分の喃語）のうち、その語が占める割合」。新しく決めた数は無い。
  前の太郎 B の自由な喃語の重み 1000（K_BABBLE）は、親が世話のときにしか話さない B に合わせた値で、親が1日約800回話す
  試験場では数日で語の回数に負け、12か月で喃語が消えた（新しい音が帳面に入らなくなった。2026-10-17 の1回目・2回目の育ち）。
  割合にすると、喃語をたくさん出してきた口ほど喃語の癖が残り（Vihman の「口に慣れた形」）、親の話す量を変えても
  釣り合いは聞く量と出す量の比で決まる
* **つながり M（新）**：親の語 w を聞いてから M_WINDOW_S 秒以内に自分が声 v を出したら、M[w][v] += 似かた(w, v)。
  似かたが SIM_MIN 以下（半分も同じでない）なら数えない。**数えるだけ。ほうびは使わない**
* **言う**：言う語の係（wordplan）から語の荷札が届き始めたら、M の行のうちその荷札の語（似かた WORD_COS 以上）を選び、
  一番強い声 v の強さが M_MIN 以上なら v を言う。弱ければ黙る（わかるけれど言えない時期）。
  M が強いかどうかは M を持っているこの係しか分からないので、言うか黙るかはここで決める（wordplan は語を持ち続けるだけ）
* **親が応えると、また声を出したくなる**（やる気の側の生まれつきのほうび、仕様書 §2-3。Goldstein ら 2003）：
  自分の声のあと M_WINDOW_S 秒以内に親の語を聞いたら、次の喃語までの待ちを RESP_LATENCY_S 秒の平均で引き直す。
  ふだんは平均 BABBLE_INTERVAL_S 秒ごと（前の太郎 B の穏やかな時間の喃語の間隔）
* **自分の声の写し**（聞き分ける係へ。段階9から TEXT）：口の指令を出してから、その音が耳から戻るまで、口の帳面から予想した
  その音（帳面に無い指令なら「？」）。聞き分ける係は、聞いた音のうち予想に合う音だけを自分の声とし（親の語として覚えない）、
  この係へ「自分の声|親の語」の形で渡す。同じ回の親の語もふつうに聞く（段階9まで、写しの間は親の声も全部捨てていた）。
  戻ってきた音は口の帳面と自分の声の列になる。ECHO_MAX_TICKS 回たっても戻らない音（耳へ返さない環境）は待つのをやめる

受け取るもの：wordplan（言いたい語の荷札）、listen（TEXT：親の知っている語を空白区切り。自分の声が戻った回は「自分の声|親の語」）。
出すもの：voiceout（MOTOR 22。1音の口の指令、黙る回はゼロ）、listen（TEXT：自分の声の写し＝戻るはずの音）。
goal・habitmove・reach からの矢印と habitmove・cb への矢印は段階7では使わない（ゼロ）。

生まれつき（手で決める）：喃語の出し方・数え方・言う条件と、下の数字。**経験で変わる**：口の帳面・聞いた語の回数・M。
これらは get_extra_state で保存に入る（save.py）。save_voice / load_voice でこの係だけ保存・読み込みもできる。
"""

from __future__ import annotations

import math
import random
from collections import Counter

import torch

from .. import mouth
from ..tagging import word_tag
from .base import Box
from .listen import SELF_SEP, UNKNOWN_SOUND
from .semantic import WORD_TRACE_S

TICK_S = 0.1
SYLL_TICKS = 3            # 1音を出す間隔（3回＝0.3秒、1秒に約3音。喃語の音節の速さ・記憶ベース）［Tier2］
BABBLE_SYLLABLES = (1, 2)  # 喃語1回の音の数（仕様書 §2-2）［Tier3］
REDUP_BIAS = 2.0          # 2音目が直前と同じ形に戻る傾き（前の太郎 B と同じ）［Tier3・B の値］
BABBLE_INTERVAL_S = 45.0  # ふだんの喃語の平均の間隔（前の太郎 B の穏やかな時間の喃語）［Tier3・B の値］
RESP_LATENCY_S = 2.0      # 親が応えたあと、次の声までの平均の待ち（やりとりの間。記憶ベース・要確認）［Tier3］
M_WINDOW_S = 5 * WORD_TRACE_S   # 親の語のあと、自分の声を結び付ける秒数（聞いた語の名残りが消えるまで＝意味の辞書と同じ）
SIM_MIN = 0.5             # 「似た声」：口の形で測って半分より多く同じ［Tier3・仮置き］
M_MIN = 3.0               # 言える強さ：よく似た声を3回（意味の辞書の RECALL_MIN と同じ「3回」）［Tier3・仮置き］
WORD_COS = 0.9            # 届いた荷札と M の行の語が同じとみなす似かた（意味の辞書の MATCH_COS と同じ）
ECHO_TICKS = 4            # 口の指令を出してから、その音がこの係へ戻るまでの回数（配線の遅れ：声の出口1・空気から耳の入口1・
                          # 聞き分け1・この係1）。これより早く届いた音は自分の声ではない
ECHO_MAX_TICKS = 10       # 自分の声が耳から戻るのを待つ最長の回数（1秒）［Tier3］

M_WINDOW_TICKS = int(round(M_WINDOW_S / TICK_S))


def _draw_ticks(rng: random.Random, mean_s: float) -> int:
    return max(1, int(round(rng.expovariate(1.0 / mean_s) / TICK_S)))


class Voice(Box):
    BOX_ID = "voice"

    def __init__(self, learning: bool = False, babbling: bool = False, seed: int = 0) -> None:
        super().__init__()
        self.notebook: Counter = Counter()     # {(口の指令4つ, 出た音): 回数}
        self.lexicon: Counter = Counter()      # {親の語: 聞いた回数}
        self.M: dict[str, Counter] = {}        # {親の語: {自分の声: 強さ}}
        self.n_free = 0                        # これまでに出した自由な喃語の回数
        self.learning = bool(learning)
        self.babbling = bool(babbling)
        self.rng = random.Random(seed)
        self._inverse: dict[str, tuple] = {}
        self._by_cmd: dict[tuple, set] = {}
        self._row_tags: dict[str, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.queue: list[tuple] = []           # これから出す口の指令
        self.fifo: list[tuple] = []            # 出して、耳から戻るのを待っている (指令, 出した回)
        self.emit_wait = 0
        self.own = ""                          # いま出している自分の声（耳から戻った音）
        self.own_start = None
        self.last_own_end = None
        self.recent: list[tuple[str, int]] = []   # (親の語, 聞いた回)
        self.prev_plan = None
        self.countdown = _draw_ticks(self.rng, BABBLE_INTERVAL_S)
        self.log: list[dict] = []              # 記録用（試験場が読んで消す）
        self.last_decision = None              # 言う語が届いたときの判断（記録用）

    # ---- 保存 ----
    def get_extra_state(self):
        return {
            "notebook": [[*p, c, n] for (p, c), n in self.notebook.items()],
            "lexicon": dict(self.lexicon),
            "M": {w: dict(r) for w, r in self.M.items()},
            "n_free": self.n_free,
        }

    def set_extra_state(self, state) -> None:
        self.notebook = Counter({(tuple(x[:4]), x[4]): x[5] for x in state.get("notebook", [])})
        self.lexicon = Counter(state.get("lexicon", {}))
        self.M = {w: Counter(r) for w, r in state.get("M", {}).items()}
        self.n_free = int(state.get("n_free", 0))
        self._rebuild_inverse()

    def save_voice(self, path, meta: dict | None = None) -> None:
        torch.save({"state": self.get_extra_state(), "meta": meta or {}}, path)

    def load_voice(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        self.set_extra_state(blob["state"])
        return blob.get("meta", {})

    # ---- 口の帳面 ----
    def _rebuild_inverse(self) -> None:
        self._inverse = {}
        self._by_cmd = {}
        for (p, c) in self.notebook:
            self._by_cmd.setdefault(p, set()).add((p, c))
        for (p, c), n in self.notebook.items():
            if c not in self._inverse or n > self.notebook[(self._inverse[c], c)]:
                self._inverse[c] = p

    def note(self, params, char: str) -> None:
        """この指令でこの音が出た。同じ指令で前に出ていた別の音の数えは消す（体が育つと、同じ指令から別の音が
        出るようになる。前の太郎の帳面の順モデル forward_map も最新の音で上書きしていた）。"""
        stale = [c for (p, c) in self._by_cmd.get(params, ()) if c != char]
        for c in stale:
            del self.notebook[(params, c)]
            if self._inverse.get(c) == params:
                del self._inverse[c]
                best = max(((p, n) for (p, cc), n in self.notebook.items() if cc == c), key=lambda x: x[1], default=None)
                if best is not None:
                    self._inverse[c] = best[0]
        self._by_cmd[params] = {(params, char)}
        self.notebook[(params, char)] += 1
        best = self._inverse.get(char)
        if best is None or self.notebook[(params, char)] > self.notebook[(best, char)]:
            self._inverse[char] = params

    def known_sounds(self) -> set[str]:
        return set(self._inverse)

    def plan(self, text: str):
        """言う順番の箱：語を1音ずつ口の指令にする。帳面に無い音があれば None（言わない）。"""
        if not text or any(c not in self._inverse for c in text):
            return None
        return [self._inverse[c] for c in text]

    # ---- 喃語 ----
    def free_babble(self) -> list[tuple]:
        n = self.rng.choice(BABBLE_SYLLABLES)
        out = []
        for i in range(n):
            p = []
            for k, size in enumerate(mouth.SIZES):
                if i == 0:
                    p.append(self.rng.randrange(size))
                else:
                    w = [math.exp(REDUP_BIAS) if v == out[-1][k] else 1.0 for v in range(size)]
                    p.append(self.rng.choices(range(size), weights=w)[0])
            out.append(tuple(p))
        return out

    def pick_word(self):
        """前の太郎 B6-2：聞いた回数で語を選ぶか、自由な喃語（None）か。候補は全部の音が帳面にある語だけ。"""
        words = [(w, n) for w, n in self.lexicon.items() if self.plan(w) is not None]
        if not words:
            return None
        r = self.rng.random() * (sum(self.lexicon.values()) + self.n_free)
        for w, n in words:
            if r < n:
                return w
            r -= n
        return None                                  # 残り（言えない語の分と、自分の喃語の分）は自由な喃語

    def babble(self) -> None:
        w = self.pick_word()
        self.queue = self.plan(w) if w else self.free_babble()
        if not w and self.learning:
            self.n_free += 1
        self.log.append({"t": self.t, "event": "babble", "word": w or ""})

    # ---- M ----
    def count_m(self, own: str) -> None:
        for w, tw in self.recent:
            if self.own_start - tw > M_WINDOW_TICKS or tw > self.own_start:
                continue
            s = mouth.similarity(w, own)
            if s > SIM_MIN:
                self.M.setdefault(w, Counter())[own] += s

    def row_for(self, tag):
        best, best_c = None, WORD_COS
        for w in self.M:
            if w not in self._row_tags:
                self._row_tags[w] = word_tag(w)
            c = float(self._row_tags[w] @ tag)
            if c >= best_c:
                best, best_c = w, c
        return best

    def strongest(self, w):
        r = self.M.get(w)
        if not r:
            return None, 0.0
        v, s = max(r.items(), key=lambda kv: kv[1])
        return v, float(s)

    # ---- 1回 ----
    def busy(self) -> bool:
        return bool(self.queue or self.fifo or self.own)

    def ticks_until_active(self) -> float:
        """試験場が何も起きない時間を飛ばすため：次に自分から声を出すまでの回数（何かしている最中は0）。"""
        if self.busy():
            return 0
        return max(0, self.countdown) if self.babbling else math.inf

    def skip(self, n: int) -> None:
        """何も起きない n 回を飛ばす（喃語の待ちだけが進む）。"""
        self.t += n
        self.countdown -= n
        self.recent = [(w, tw) for w, tw in self.recent if self.t - tw <= M_WINDOW_TICKS]

    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        heard = inbox.get("listen") or ""
        # 段階9：聞き分ける係は「聞いた自分の声|親の語…」の形で渡す（自分の声が無い回は親の語だけ）
        own_heard, _sep, parent_words = heard.rpartition(SELF_SEP) if SELF_SEP in heard else ("", "", heard)
        self.fifo = [(p, t0) for p, t0 in self.fifo if self.t - t0 <= ECHO_MAX_TICKS]
        if own_heard and self.fifo and self.t - self.fifo[0][1] >= ECHO_TICKS:   # 自分の声が耳から戻った
            for c in own_heard:
                if not self.fifo or self.t - self.fifo[0][1] < ECHO_TICKS:
                    break
                p, _ = self.fifo.pop(0)
                if self.learning:
                    self.note(p, c)
                self.own += c
        if parent_words:                             # 親の語（自分が話している最中に聞いた親の語も）
            for w in parent_words.split():
                if self.learning:
                    self.lexicon[w] += 1
                self.recent.append((w, self.t))
                self.log.append({"t": self.t, "event": "parent", "word": w})
            if self.last_own_end is not None and self.t - self.last_own_end <= M_WINDOW_TICKS:
                self.countdown = min(self.countdown, _draw_ticks(self.rng, RESP_LATENCY_S))
                self.last_own_end = None             # 1回の声に応えてもらったのは1回
        self.recent = [(w, tw) for w, tw in self.recent if self.t - tw <= M_WINDOW_TICKS]

        if not self.own and not self.queue and not self.fifo:
            self.own_start = None                    # 耳へ戻らなかった声（写しの待ちが切れた）
        if self.own and not self.queue and not self.fifo:   # 自分の声を言い終えた
            if self.learning:
                self.count_m(self.own)
            self.log.append({"t": self.t, "event": "own", "voice": self.own})
            self.own, self.own_start, self.last_own_end = "", None, self.t

        tag = inbox.get("wordplan")
        has = tag is not None and float(torch.linalg.vector_norm(tag)) > 0.0
        onset = has and (self.prev_plan is None or float(self.prev_plan @ tag) < 0.99)
        self.prev_plan = tag.clone() if has else None
        if onset and not self.busy():
            w = self.row_for(tag)
            v, s = self.strongest(w) if w else (None, 0.0)
            seq = self.plan(v) if v is not None and s >= M_MIN else None
            self.last_decision = {"t": self.t, "word": w or "", "voice": v or "", "strength": s,
                                  "say": seq is not None}
            self.log.append({"event": "decide", **self.last_decision})
            if seq is not None:
                self.queue = list(seq)

        if self.babbling:
            self.countdown -= 1
            if self.countdown <= 0 and not self.busy():
                self.babble()
                self.countdown = _draw_ticks(self.rng, BABBLE_INTERVAL_S)

        self.emit_wait -= 1
        if self.queue and self.emit_wait <= 0:
            p = self.queue.pop(0)
            if self.own_start is None:
                self.own_start = self.t
            self.fifo.append((p, self.t))
            out["voiceout"] = mouth.to_motor(p)
            self.emit_wait = SYLL_TICKS
        # 写し：聞き分ける係に音が届く回（出してから ECHO_TICKS−1 回目）に届くように、1つ前の回に出す。
        # 段階9：中身は「いま戻るはずの自分の音」。口の帳面（この指令でどの音が出たか）から予想したかなを、出した順に空白区切りで。
        # 帳面にまだ無い指令は UNKNOWN_SOUND（自分の音が1つ戻るが、何の音かは分からない）
        out["listen"] = " ".join(self.predict_sound(p) for p, t0 in self.fifo
                                 if ECHO_TICKS - 2 <= self.t - t0 <= ECHO_MAX_TICKS)
        return out

    def predict_sound(self, params) -> str:
        """口の帳面の順向き：この指令で最後に出た音（帳面に無ければ UNKNOWN_SOUND）。"""
        got = self._by_cmd.get(params)
        return next(iter(got))[1] if got else UNKNOWN_SOUND
