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

* **段階7（2026-10-17）**（段階9で下のとおり変えた）：声の係（voice）から自分の声の写し（AMOUNT）が 1 で来ている間に聞いた音は、自分の声。
  次の音の当て方も回数も直さず（自分の喃語を親の語として覚えない）、意味の辞書へも送らず、そのかなを声の係へ渡す
  （口の帳面を作るため。TEXT）。写しが無い間は、聞いた発話の中の知っている語を空白区切りで声の係へ（TEXT）、
  最初の知っている語の荷札を言う語の係（wordplan）へ渡す（意味の辞書へ渡すものと同じ）

* **段階8（2026-10-18）**：発話の中の知っている語を**全部、順に**送る（今までは最初の1つだけ。「ぼおるないね」の「ない」が落ちていた）。
  聞いた回に1つ目、次の回に2つ目…と1回に1つずつ意味の辞書と言う語の係へ（語ごとに意味の辞書の名残りが1つできる）

* **段階9（2026-10-18）：自分が話している間も聞く**（人間も自分の声と一緒に他人の声を聞く）。前は自分の声の写しが来ている
  間に聞いた音を**全部**自分の声として捨てていたので、太郎が声を出している最中の親の名づけも捨てていた（look_diag：
  名づけ37回のうち7回）。今は、声の係から届く写しは「いま耳に戻ってくるはずの自分の音」（声の係の口の帳面が、出した口の
  指令から予想したかな。帳面にまだ無い指令なら UNKNOWN_SOUND）。耳に届いた発話（空白区切り）のうち、
  予想と同じ音を1つずつ自分の声とみる。予想と同じ音が無ければ（体が育って予想と違う音が出た、帳面に無い指令）、
  1音だけの発話を自分の声とみる（口は1回に1音しか出さない）。自分の声とみた音だけを覚えず・意味の辞書へ送らず、
  「聞いた自分の声」として声の係へ渡す（口の帳面のため。last_self にも残す）。**残りの発話（同じ回の親の声）は
  ふだんどおり聞く**。声の係へは「自分の声|親の知っている語…」の形で渡す（自分の声が無い回は今までどおり親の語だけ）

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
from .earin import normalize_kana
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
UNKNOWN_SOUND = "？"      # 自分の声の写しで「自分の音が1つ戻るが、何の音かはまだ分からない」
SELF_SEP = "|"            # 声の係へ渡す TEXT で、聞いた自分の声と親の語を分ける印


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

    @staticmethod
    def split_self(utterances: list[str], expected: str) -> tuple[str, list[str]]:
        """届いた発話の並びから、写し（いま戻るはずの自分の音、空白区切り）に合う音を抜き出す。(自分の声, 残り)。"""
        rest = list(utterances)
        own = []
        for exp in expected.split():
            k = next((i for i, u in enumerate(rest) if exp != UNKNOWN_SOUND and kana_only(u) == exp), None)
            if k is None:                    # 予想どおりの音が無い：1音だけの発話を自分の声とみる（口は1回に1音）
                k = next((i for i, u in enumerate(rest) if len(kana_only(u)) == 1), None)
            if k is not None:
                own.append(kana_only(rest.pop(k)))
        return "".join(own), rest

    def use_board_ear(self, table: dict[str, list[str]]) -> None:
        """盤面の世界の耳の下駄（3段目、2026-10-19）：{親の一言（かなにそろえた物）: 単語の列}。
        LLM で先に作った表（core_v2/tools/make_board_speech.py）。表にある一言が届いたら、切り分けずに
        その単語を順に送る（聞き分ける係の出口と同じ形）。あとで本物の聞き分け（この係の中身）に替える。"""
        self.__dict__["_board_ear"] = {normalize_kana(k): [normalize_kana(w) for w in v] for k, v in table.items()}

    def step(self, inbox):
        out = self.passthrough()
        self.last_heard, self.last_chunks, self.last_word = [], [], ""
        text = inbox.get("earin") or ""
        ear = self.__dict__.get("_board_ear")
        if ear is not None:
            if text:
                self.last_heard = [text]
                self._queue += ear.get(text, [])
            return self._send_next(out)
        expected = inbox.get("voice") or ""
        utterances = text.split()            # 1回に2つ届いたら空白で区切られている
        self.last_self = ""
        if expected:
            # 段階9：写しに合う音だけが自分の声。覚えず、意味の辞書へも送らない。残り（親の声）はふだんどおり聞く
            self.last_self, utterances = self.split_self(utterances, str(expected))
        if not utterances:
            if self.last_self:
                out["voice"] = self.last_self + SELF_SEP
            return self._send_next(out)
        known = self.known()
        found = []
        for u in utterances:
            ch = self.segment(u)
            self.last_heard.append(u)
            self.last_chunks.append(ch)
            found += [c for c in ch if c in known]
            if self.learning:
                self.hear(u)
                self.chunks.update(ch)
        self._queue += found
        out["voice"] = (self.last_self + SELF_SEP if self.last_self else "") + " ".join(found)
        return self._send_next(out)

    def _send_next(self, out):
        if self._queue:
            self.last_word = self._queue.pop(0)
            out["semantic"] = word_tag(self.last_word)
            out["wordplan"] = word_tag(self.last_word)
        return out
