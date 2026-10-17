"""音を語に切る・聞き分ける係（2段目 習慣）。

**段階5（2026-10-16）：手で書いた語の一覧（ぼおる・こっぷ）を、聞いて覚える形に替えた（仕様書
`2026-10-16_段階5_仕様書_聞き分ける係をNNに.md`）。**

* 中身：**次の音を当てるNN**（かな1つ＝1音ずつ聞く小さな再帰ネット、GRU。Elman 1990 の単純再帰ネットと
  同じ考え）。1つの発話を聞き終えるたびに、その発話で次の音の当て方を1回だけ直す（learning=True のとき）
* **語の切れ目**：NNが「ここで発話が終わりそう」と思う所（終わりの予想が、これまでの平均より高い所）で切る
  （Christiansen・Allen・Seidenberg 1998。前の太郎の Segmenter.segment_end_prob と同じ）。
  「ぼ→お→る」は続きやすく、「る」のあとは「は」「だ」「を」…とばらつき、発話の終わりにも来やすい
* **足がかり**：語1つだけで聞いた発話（親の話しかけの約1割、Brent・Siskind 2001。記憶ベース）を
  KNOWN_MIN 回以上聞き、しかもほかの発話の中にも KNOWN_MIN 種類以上で聞いたら「知っている語」にする
  （「りんごはどこ」のように、まるごと何度も聞くだけの文は語にしない）。
  段階6（2026-10-17）：1語だけで KNOWN_MIN 回以上聞き、中に知っている語を含まないかたまりも語にする
  （スミスとユーの実験のように、新しい語を単独でしか聞かない場合。「おおきいぼおる」「りんごはどこ」は
  知っている語を含むので語にしない）。発話まるごとが知っている語なら、切らずに1語として聞く。文の中にそれがあれば、まずそこで切る
  （長いもの優先・重ならない。前の太郎の分節第2案と同じ）
* **意味の辞書へ渡す語**：切ったかたまりのうち、最初の「知っている語」。その荷札は、かなの並びを
  決まった乱数で数にしたもの（tagging.word_tag＝生まれつきの耳の符号）。知っている語が無ければゼロ
  （「でんしゃはどこ？」では何も探さない）

* **段階7（2026-10-17）**：声の係（voice）から自分の声の写し（AMOUNT）が 1 で来ている間に聞いた音は、自分の声。
  次の音の当て方も回数も直さず（自分の喃語を親の語として覚えない）、意味の辞書へも送らず、そのかなを声の係へ渡す
  （口の帳面を作るため。TEXT）。写しが無い間は、聞いた発話の中の知っている語を空白区切りで声の係へ（TEXT）、
  最初の知っている語の荷札を言う語の係（wordplan）へ渡す（意味の辞書へ渡すものと同じ）

* **段階8（2026-10-18）**：発話の中の知っている語を**全部、順に**送る（今までは最初の1つだけ。「ぼおるないね」の「ない」が落ちていた）。
  聞いた回に1つ目、次の回に2つ目…と1回に1つずつ意味の辞書と言う語の係へ（語ごとに意味の辞書の名残りが1つできる）

NNの中の状態から語の荷札を作る案は、1試行目で違う語どうしが似すぎた（似かた 平均0.62・最大0.89、
わんわん／にゃんにゃん）ので、仕様書 §2 の決めどおり生まれつきの耳の符号に替えた。

生まれつき（手で決める）：NNの大きさ・直し方（Adam、LR）、切れ目の線（終わりの予想＞平均）、
KNOWN_MIN、平均の更新の速さ。**経験で変わる**：NNの重み、知っている語の一覧（聞いた回数）、終わりの予想の平均。
耳だけで覚えた中身は save_ear / load_ear で保存・読み込みする（脳全体の保存 save.py にも入る）。
"""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn

from ..tagging import word_tag
from .base import Box

KANA = [chr(c) for c in range(0x3041, 0x3097)] + ["ー"]
BOS, EOS = 0, 1
INDEX = {c: i + 2 for i, c in enumerate(KANA)}
VOCAB = len(KANA) + 2
EMB, HIDDEN = 32, 64      # NNの大きさ［仮置き・前の太郎と同じくらいの小ささ］
LR = 3e-3                 # 1発話ごとに1回直す強さ［仮置き］
KNOWN_MIN = 2             # 語1つだけで何回聞いたら「知っている語」か［仮置き］
MEAN_RATE = 0.01          # 終わりの予想の平均を、1音ごとにどれだけ新しい値へ寄せるか［仮置き］
INIT_SEED = 0             # NNの重みの最初の値（毎回同じにする）


def kana_only(text: str) -> str:
    return "".join(c for c in text if c in INDEX)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB)
        self.gru = nn.GRU(EMB, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, VOCAB)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h)


class Listen(Box):
    BOX_ID = "listen"

    def __init__(self, learning: bool = False) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(INIT_SEED)
            self.net = _Net()
        self.register_buffer("end_mean", torch.zeros(()))
        self.register_buffer("n_heard", torch.zeros((), dtype=torch.int64))
        self.alone: Counter = Counter()    # 発話まるごとを聞いた回数（語1つだけの発話が知っている語になる）
        self.chunks: Counter = Counter()   # 切り出したかたまりを聞いた回数（記録用）
        self.learning = bool(learning)
        self._opt = None
        self._known = None
        self.reset()

    def reset(self) -> None:
        self.last_heard: list[str] = []
        self.last_chunks: list[list[str]] = []
        self.last_word = ""
        self.last_self = ""
        self._queue: list[str] = []          # 段階8：まだ意味の辞書へ送っていない知っている語（順に1回1つ）

    # ---- 保存（数え上げは文字列なので extra_state に入れる） ----
    def get_extra_state(self):
        return {"alone": dict(self.alone), "chunks": dict(self.chunks)}

    def set_extra_state(self, state) -> None:
        self.alone = Counter(state.get("alone", {}))
        self.chunks = Counter(state.get("chunks", {}))
        self._known = None

    def save_ear(self, path, meta: dict | None = None) -> None:
        torch.save({"state_dict": self.state_dict(), "meta": meta or {}}, path)

    def load_ear(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(blob["state_dict"])
        self._opt = None
        return blob.get("meta", {})

    # ---- 耳の中身 ----
    def known(self) -> set[str]:
        if self._known is None:
            heard = [u for u in self.alone]
            rep = [u for u, n in self.alone.items() if n >= KNOWN_MIN]
            base = {u for u in rep if sum(1 for v in heard if len(v) > len(u) and u in v) >= KNOWN_MIN}
            self._known = base | {u for u in rep if not any(k in u for k in base)}
        return self._known

    def _predict(self, s: str):
        """音ごとに (その音の次で発話が終わる見込み) を返す。"""
        t = [INDEX[c] for c in s]
        with torch.no_grad():
            p = self.net(torch.tensor([[BOS] + t])).softmax(-1)[0]
        return [float(p[i + 1, EOS]) for i in range(len(t))]

    def segment(self, utterance: str) -> list[str]:
        s = kana_only(utterance)
        n = len(s)
        if n == 0:
            return []
        if s in self.known():
            return [s]
        endp = self._predict(s)
        cuts, used = {0, n}, [False] * n
        for k in sorted((k for k in self.known() if 2 <= len(k) < n), key=len, reverse=True):
            for st in range(n - len(k) + 1):
                if s[st:st + len(k)] == k and not any(used[st:st + len(k)]):
                    used[st:st + len(k)] = [True] * len(k)
                    cuts.update((st, st + len(k)))
        mean = float(self.end_mean)
        for j in range(1, n):
            if used[j - 1] and used[j] and j not in cuts:
                continue                     # 知っている語の中では切らない
            if endp[j - 1] > mean:
                cuts.add(j)
        cs = sorted(cuts)
        return [s[a:b] for a, b in zip(cs, cs[1:])]

    def hear(self, utterance: str) -> None:
        """1つの発話で覚える：次の音の当て方を1回直し、回数と平均を更新する。"""
        s = kana_only(utterance)
        if not s:
            return
        if self._opt is None:
            self._opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        t = [INDEX[c] for c in s]
        logits = self.net(torch.tensor([[BOS] + t]))[0]
        loss = nn.functional.cross_entropy(logits, torch.tensor(t + [EOS]))
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        with torch.no_grad():
            p = logits.softmax(-1)
            m = float(self.end_mean)
            for i in range(len(t) - 1):        # 文の途中の音だけ（最後の音のあとは必ず終わる）
                m += MEAN_RATE * (float(p[i + 1, EOS]) - m)
            self.end_mean.fill_(m)
        self.alone[s] += 1
        self.n_heard += 1
        if self.alone[s] in (1, KNOWN_MIN):   # 知っている語の一覧が変わりうるのは、新しい発話か、回数が線に届いたときだけ
            self._known = None

    def step(self, inbox):
        out = self.passthrough()
        self.last_heard, self.last_chunks, self.last_word = [], [], ""
        text = inbox.get("earin") or ""
        own = inbox.get("voice")
        if own is not None and float(own) > 0.5:
            # 段階7：自分の声の写しが来ている間に聞いた音は自分の声。覚えず、意味の辞書へも送らず、声の係へそのまま返す
            self.last_self = kana_only(text)
            out["voice"] = self.last_self
            return self._send_next(out)
        if not text:
            return self._send_next(out)
        known = self.known()
        found = []
        for u in text.split():               # 1回に2つ届いたら空白で区切られている
            ch = self.segment(u)
            self.last_heard.append(u)
            self.last_chunks.append(ch)
            found += [c for c in ch if c in known]
            if self.learning:
                self.hear(u)
                self.chunks.update(ch)
        self._queue += found
        out["voice"] = " ".join(found)
        return self._send_next(out)

    def _send_next(self, out):
        if self._queue:
            self.last_word = self._queue.pop(0)
            out["semantic"] = word_tag(self.last_word)
            out["wordplan"] = word_tag(self.last_word)
        return out
