"""段階9の試し：意味の辞書を「まばらな語の表し方」にした版（試験場だけ。本物の太郎の係はまだ替えない）。

仕様書 `2026-10-18_段階9_仕様書_意味の辞書まばら化.md`。文献 `文献調査/2026-10-18_まばらな語の表し方.md`。

**語の細胞の集まり**（N_POOL＝1000 個）：聞いた語の荷札 q（64）を、決まった乱数の行列で 1000 に広げ、いちばん強い K＝10 個だけが
光る（k-WTA。ハエの嗅覚の「広げて少数だけ残す」、海馬の歯状回）。どの細胞の組になるかは競い合いで決まり、語ごとに手で決めない。
20語の組どうしが重なるのは 190 組のうち 27 組、重なっても 10 個のうち 1 個（ぶうぶう・がおお だけ 2 個）（測った）。今の 64×64 の表の「混ざり」（|cos| 平均0.11）より
ずっと小さい。広げる行列・K・N_POOL は生まれつき（学ばない）。
**結び付き**（経験で育つ。ヘッブ則、ほうびは使わない）：名残りが消えて覚えるとき（今の覚え方Bと同じ時刻・同じ窓）、光った語の細胞 j
それぞれが
* 見た目の予想 A_j（64）を、そのとき見ていた見た目（＋驚きの荷札）の**平均**へ近づける：A_j += (a − A_j)/c_j（c_j＝その細胞が覚えた回数）
* 音の結び付き：そのとき聞こえていた物の音を、同じ作り方で音の細胞（1000 から 10）にし、光った音の細胞 i それぞれが
  T_i（語の細胞 1000 の上の重み）を、そのとき光っていた語の細胞の組の平均へ近づける。窓に音が無かった回は動かさない。
  （直し1：最初は語の細胞の側で音の組を平均したが、同じ部屋の別の語の窓にも音が入るので、すぷうんの音が こっぷ・ばなな にも
  「いつも一緒の音」として同じ強さで入り、すぷうん 0%、があがあ→さかな になった。結び付きは「送る側の細胞ごと」に平均する。
  語→見た目は語の細胞が送る側、音→語は音の細胞が送る側。1つの決まり）
* 語の形 Q_j（64）を、聞いた語の荷札の平均へ（逆に引いたとき、言う語の係へ出す荷札）
平均にする（足し続けない）ので、たくさん聞いた語がほかの語に勝つことが無い（Oja 1982・Grossberg の outstar と同じ「重みの大きさを保つ」形）。
**思い出す**（語→見た目、gaze へ）：光った細胞の A_j の平均（64）。光った細胞の覚えた回数の真ん中の値が RECALL_N（3）未満なら出さない
（平均にすると、ほかの語とたまたま重なった1個の細胞だけで聞いたことのない語が思い出せてしまう）
（今の「同じ組み合わせ3回で出る」と同じ）。gaze が受け取る形は今と同じ 64 の荷札なので、ほかの係は変わらない。
**逆に引く**（見た目・音→語）：語の細胞 j の強さ h_j ＝ A_j·a ＋ Σ_i T_ij z_s,i /K。強い K 個の Q_j の平均を出す。

**物の音の時間の形**（sound_tag_time。生まれつきの耳の前段、Chi・Ru・Shamma 2005 を小さくしたもの）：
対数メル32帯の音を、時間で4つに切った帯の平均（4×32、音の全体の大きさを引く）と、8つの帯の組ごとの「変わる速さ」
（10ms 先・40ms 先との差の大きさの平均、2×8、平均を引く）の2つの塊にし、塊ごとに長さ1にそろえて並べ、決まった乱数で64へ。
今の荷札（帯の時間平均だけ）では あひる・犬 0.98、猫・電車 0.96 だったのが、0.84・0.70 になる（7種類×4個体で測った）。
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..boxes import soundin as si
from ..boxes.base import ENV
from ..boxes.semantic import Semantic
from ..formats import DTYPE, TAG_DIM, make_tag

N_POOL = 1000          # 語の細胞・音の細胞の数（生まれつき）
K = 10                 # 一度に光る数（生まれつき。log2(1000)≈10：Willshaw の表で一番たくさん覚えられる数のめやす）
RECALL_N = 3           # 思い出すのに要る、覚えた回数の平均（今の RECALL_MIN と同じ「3回」）
WORD_SEED, SOUND_SEED = 31, 32
N_SLICES = 4           # 音を時間で切る数
N_GROUPS = 8           # 変わる速さを測る帯の組
LAGS = (1, 4)          # 10ms 先・40ms 先
TIME_PROJ_SEED = 20261019


def _proj(seed: int, n_in: int, n_out: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_out, n_in, generator=g, dtype=torch.float64).to(DTYPE)


_PW = _proj(WORD_SEED, TAG_DIM, N_POOL)
_PS = _proj(SOUND_SEED, TAG_DIM, N_POOL)


def kwta(x: torch.Tensor, k: int = K) -> torch.Tensor:
    """いちばん強い k 個を 1、ほかを 0（x がゼロなら全部 0）。"""
    z = torch.zeros_like(x)
    if float(torch.linalg.vector_norm(x)) == 0.0:
        return z
    z[torch.topk(x, k).indices] = 1.0
    return z


def word_code(q: torch.Tensor) -> torch.Tensor:
    return kwta(_PW @ make_tag(q))


def sound_code(n: torch.Tensor) -> torch.Tensor:
    return kwta(_PS @ make_tag(n))


# ---------------------------------------------------------------- 物の音の時間の形
_tproj = None


def _spec(wave) -> np.ndarray:
    y = np.asarray(wave, dtype=np.float64).reshape(-1)
    win, hop = int(si.WIN_S * si.SR), int(si.HOP_S * si.SR)
    if len(y) < win + 4 * hop * N_SLICES:
        y = np.pad(y, (0, win + 4 * hop * N_SLICES - len(y)))
    frames = np.stack([y[s:s + win] for s in range(0, len(y) - win + 1, hop)]) * np.hanning(win)
    p = np.abs(np.fft.rfft(frames, si.NFFT)) ** 2
    return np.log(p @ si.filterbank().T + 1e-10)        # 時間 × 32帯


def time_features(wave) -> np.ndarray:
    S = _spec(wave)
    sl = np.stack([c.mean(0) for c in np.array_split(S, N_SLICES)]) - S.mean()
    a = sl.reshape(-1)
    G = S.reshape(len(S), N_GROUPS, -1).mean(2)
    b = np.concatenate([np.abs(G[lag:] - G[:-lag]).mean(0) for lag in LAGS])
    b = b - b.mean()
    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)   # noqa: E731
    return np.concatenate([unit(a), unit(b)])


def sound_tag_time(wave) -> torch.Tensor:
    global _tproj
    f = time_features(wave)
    if _tproj is None:
        _tproj = np.random.default_rng(TIME_PROJ_SEED).standard_normal((len(f), TAG_DIM)) / math.sqrt(TAG_DIM)
    return make_tag((f @ _tproj).astype(np.float32))


class SoundinTime(si.Soundin):
    """物の音の入口の、時間の形を残す版（試験場の比べる用）。荷札の作り方だけが違う。"""

    def step(self, inbox):
        w = inbox.get(ENV)
        if w is not None and len(np.asarray(w).reshape(-1)) > 0:
            out = self.passthrough()
            self.tag = sound_tag_time(w)
            self.left = max(1, math.ceil(len(np.asarray(w).reshape(-1)) / si.SR / si.TICK_S - 1e-9))
            self.n_heard += 1
            for k in out:
                out[k] = self.tag
            self.left -= 1
            return out
        return super().step({})


# ---------------------------------------------------------------- まばらな意味の辞書
class SparseSemantic(Semantic):
    """覚える時刻・窓・名残りは今の Semantic（覚え方B）と同じ。表 W の代わりに語の細胞の結び付きを使う。"""

    def __init__(self, learning: bool = False) -> None:
        super().__init__(learning=learning, learn_mode="B")
        self.A = torch.zeros(N_POOL, TAG_DIM, dtype=DTYPE)     # 語の細胞 → 見た目の予想
        self.Q = torch.zeros(N_POOL, TAG_DIM, dtype=DTYPE)     # 語の細胞 → 語の形
        self.T = torch.zeros(N_POOL, N_POOL, dtype=DTYPE)      # 音の細胞 → 語の細胞（行＝音の細胞）
        self.c = torch.zeros(N_POOL, dtype=DTYPE)              # 語の細胞ごとに覚えた回数
        self.cs = torch.zeros(N_POOL, dtype=DTYPE)             # 音の細胞ごとに語と一緒に覚えた回数

    # 見ていた見た目（＋驚き）と音を分けて持つ：[見た目 64 | 音 64]
    def _seen_vec(self, lst, u: float, n=None):
        a = None
        if lst and len(lst) >= 2 and float(torch.linalg.vector_norm(lst[-2][0])) > 0.0:
            a = make_tag(lst[-2][0])
        if u > 0.0:
            a = (a if a is not None else 0.0) + u * self._surprise()
        if a is None and n is None:
            return None
        z = torch.zeros(TAG_DIM, dtype=DTYPE)
        return torch.cat([a if a is not None else z, n if n is not None else z])

    @staticmethod
    def _surprise():
        from ..boxes.semantic import SURPRISE_TAG
        return SURPRISE_TAG

    def _trace_step(self, lst, new_q) -> None:
        super()._trace_step(lst, new_q)
        if new_q is not None and not self._pending[-1]["v"].shape[0] == 2 * TAG_DIM:
            self._pending[-1]["v"] = torch.zeros(2 * TAG_DIM, dtype=DTYPE)

    def _learn_pending(self, p) -> None:
        if p["w"] == 0.0:
            self.last_event = "learn_skip"
            return
        v = p["v"] / p["w"]
        a, n = v[:TAG_DIM], v[TAG_DIM:]
        z = word_code(p["q"])
        idx = z.nonzero().flatten()
        has_a = float(torch.linalg.vector_norm(a)) > 0.0
        has_n = float(torch.linalg.vector_norm(n)) > 0.0
        if has_a:
            self.c[idx] += 1.0
            self.A[idx] += (make_tag(a) - self.A[idx]) / self.c[idx].unsqueeze(1)
            self.Q[idx] += (make_tag(p["q"]) - self.Q[idx]) / self.c[idx].unsqueeze(1)
        if has_n:      # 直し1：光った音の細胞が、そのとき光っていた語の細胞の組の平均を覚える（音の細胞ごとにそろえる）
            js = sound_code(n).nonzero().flatten()
            self.cs[js] += 1.0
            self.T[js] += (z - self.T[js]) / self.cs[js].unsqueeze(1)
        self.n_learn += 1
        self.last_event = "learn"

    def recall(self, q):
        z = word_code(q)
        idx = z.nonzero().flatten()
        cnt = float(self.c[idx].median())
        self.last_recall_norm = cnt
        if cnt < RECALL_N:
            return None
        return make_tag(self.A[idx].mean(0))

    def drive(self, a=None, n=None) -> torch.Tensor:
        """語の細胞の強さ h ＝ A·a ＋ Tᵀz_s/K（見た目・音から語へ）。"""
        h = torch.zeros(N_POOL, dtype=DTYPE)
        if a is not None and float(torch.linalg.vector_norm(a)) > 0.0:
            h = h + self.A @ make_tag(a)
        if n is not None and float(torch.linalg.vector_norm(n)) > 0.0:
            h = h + self.T.T @ sound_code(n) / K
        return h

    def word_scores(self, words_q: torch.Tensor, a=None, n=None) -> torch.Tensor:
        """試験場用：語ごとに、その語の細胞の組の強さの合計。"""
        h = self.drive(a, n)
        return torch.tensor([float(h @ word_code(q)) for q in words_q], dtype=DTYPE)

    def recall_word(self, lst, u: float = 0.0, n=None):
        self.last_word_norm = 0.0
        x = self._seen_vec(lst, u, n)
        if x is None:
            return None
        a, s = x[:TAG_DIM], x[TAG_DIM:]
        h = self.drive(a, s)
        z = kwta(h)
        idx = z.nonzero().flatten()
        cnt = float(self.c[idx].median()) if len(idx) else 0.0
        self.last_word_norm = cnt
        if cnt < RECALL_N or float(h[idx].sum()) <= 0.0:
            return None
        return make_tag(self.Q[idx].mean(0))
