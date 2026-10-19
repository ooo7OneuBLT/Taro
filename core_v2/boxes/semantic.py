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
* **驚きの荷札（段階8、2026-10-18、ユーザー承認の仕様書 `2026-10-18_段階8_仕様書_ない.md`）**：驚きの係から驚きの強さ u（0〜1）が
  届く。覚え方Bで毎回数える見た目に u·S を足す（S＝驚きの荷札 tagging.state_tag("予想外")、生まれつきの符号）。物を見ていない回も
  u＞0 なら数える。逆に引くときも Wᵀ(a＋u·S)。u＝0 なら段階3〜7と1ビットも同じ。
  「ない」と驚きのつながりは手で書かない。親が消える場面で言う「○○ないね」を、場面をまたいで数えるだけで生まれる
* **物の音の荷札（段階9、2026-10-18、ユーザー承認の仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md`）**：物の音の入口（soundin）から
  音の荷札 N が届いている回は、覚え方Bで毎回数える見た目に v·N を足す（v＝1。段階8の u·S と同じ足し方）。逆に引くときも
  Wᵀ(a＋u·S＋v·N)。物が見えていなくても音だけで語を引ける。音が届かなければ段階8と1ビットも同じ
* **聞いた語が続けて届く（段階8）**：聞き分ける係は1つの発話の知っている語を1回に1つずつ順に送る。前の回と違う語が届いたら新しい語
* **まばらな語の表し方（段階9、2026-10-18、ユーザー決定「推奨通りでいいよ」、仕様書 `2026-10-18_段階9_仕様書_意味の辞書まばら化.md`）**：
  既定（memory="sparse"）は、表 W の代わりに**語の細胞 N_POOL（1000）個**を使う。聞いた語の荷札を決まった乱数の行列で 1000 に広げ、
  いちばん強い K（10）個だけが光る（競い合い。語ごとに手で決めない。広げる行列・N_POOL・K は生まれつき）。
  覚える時刻・窓・名残り・驚きの荷札の足し方は表 W のときと同じ。覚えるとき、光った語の細胞 j それぞれが
  見た目の予想 A_j を「そのとき見ていた見た目（＋u·S）」の**平均**へ、語の形 Q_j を聞いた語の荷札の平均へ近づける
  （A_j は見た目があった回だけ、Q_j は見た目か音があった回。音だけ聞こえて物が見えない回でも、音→語は覚える）。
  物の音は見た目と足さずに別に持ち、音を同じ作り方で音の細胞（1000 から 10）にし、光った音の細胞 i が T_i を
  「そのとき光っていた語の細胞の組」の平均へ近づける（送る側の細胞ごとに平均。Oja 1982・outstar）。
  思い出す（→gaze）：光った語の細胞の A_j の平均（64、今と同じ形）。光った細胞の覚えた回数の真ん中の値が RECALL_N（3）未満なら出さない。
  逆に引く（→wordplan）：語の細胞の強さ h ＝ A·(a＋u·S) ＋ Tᵀz_s/K、強い K 個の Q_j の平均（その細胞の覚えた回数の真ん中の値が3以上のとき）。
  細胞（語の細胞・音の細胞）は、3回覚えるまで逆向きには働かない（思い出す線と同じ「3回」）
  memory="table" で前の表 W（1ビットも同じ）に戻せる（比べる用）
* **まちがえそうな物も一緒に思い出す（段階9、2026-10-18）**：目的の係へは、聞いた語の見た目の予想（先頭）に続けて、
  「物ふつうの見た目」G（今まで見た物の見た目の平均。まだ語を知らない太郎が思い浮かべる物）と、聞いた語の細胞以外の覚えた細胞の
  見た目の予想のうち、聞いた語の予想に近い順に N_RIVALS 個（同じ中身の細胞は1つに数える）を、(荷札, ゼロの地図) の列で渡す。
  目を向ける係は、見えている物が「聞いた語の予想に、まちがえそうな物のどれよりも似ている」ときだけ探し物とする（gaze.py）。
  本物の物（同じ種類どうしの似かた 0.79）では、決まった線 0.7 に正しい物の半分ほどが届かず、A が 73% に落ちていたため。
  N_RIVALS は書式の上限（MAX_OBJECTS 8）から、先頭と G の2つを引いた数。手で合わせた数ではない。
  まちがえそうな物の雑音は自分専用の乱数で（ほかの係の乱数の並びを変えない）。表 W（memory="table"）・段階2の表では先頭だけ
  （目を向ける係は前の線 0.7 で決める）
学ぶこと自体のやり方（ヘッブ則・ETA・RECALL_MIN・覚え方）は生まれつき＝手で決める。
W（まばら版は A・Q・T・c・cs）の中身だけが経験で変わる。どれも buffer なので save.py の保存に入り、reset() では消えない。

* **盤面の世界（4段目「あれなに？」、2026-10-19。use_board() で入る）**：目の入口から「目の前の物」(見た目の荷札, 地図) が
  届く。聞いた語（listen→semantic）が届いた回に目の前に物があれば、まばら版の覚え方で1回結び付ける（覚え方は上と同じ _bind。
  場面をまたいで、いつも一緒に出る組だけが積み上がる）。1分に1回（rhythm の SEM_PHASE）、目の前の物の名前を
  どれくらい知っているか k を出す。k は「物 → 語 → 物」の引き直しが元の物に戻るか：目の前の物から逆に語を引き（3回以上
  覚えた細胞だけ）、その語から見た目の予想を引いて、目の前の物との似かた c から k ＝ (c−KNOW_LO)/(KNOW_HI−KNOW_LO)（0〜1）。
  （1回でも覚えた細胞で引く案は、1回だけ一緒に聞いた語の細胞が見た目にぴったり合って勝ち、だめだった）
  どの物とも一緒に聞く語（まんま・だね）は、引き直すと物の平均になり元の物に戻らないので、知っていることにならない
  - semantic→want：わからなさ 1−k（目の前に物が無ければ0）。何か行動したくなるもと（知りたい）
  - semantic→worth：わかった ＝ 同じ物について k が前の分より上がった分（ほうびの3つめ「予想の外れが減る」）
  名前を何と呼ぶかは教えない（親が言った語が目の前の物と結び付くだけ）

段階2の手で登録した表（register / load_table、`core_v2/tools/register_appearance.py`）も残す。
表に1件でも登録してあれば、段階2と同じく表を引く（段階3の合格テスト②で使う）。
"""

import math

import torch

from ..formats import GRID, MAX_OBJECTS, TAG_DIM, DTYPE, make_tag
from ..tagging import NOISE_SIGMA, noisy, state_tag, word_tag
from .base import Box

MATCH_COS = 0.9          # 段階2の表引き
ETA = 1.0                # 1回に足す強さ［仮置き］
WORD_TRACE_S = 1.0       # 聞いた語の名残りが薄れる速さ（生まれつき・1歳の値。育つと伸びる。逸脱リスト登録）［Tier2］
LOOK_WINDOW_S = 3.0      # 語が届く前、見ていた物を数える秒数（物の名簿と同じ。Yu・Smith 2012）［Tier2］
TICK_S = 0.1
TRACE_TICKS = int(round(5 * WORD_TRACE_S / TICK_S))   # 名残りがほぼ消える（1%）まで
SURPRISE_TAG = state_tag("予想外")   # 段階8：驚きの荷札（生まれつき）
RECALL_MIN = 2.5         # 同じ組み合わせ3回で ‖r‖≈3、2回では出ない［仮置き・合格テスト③④で確かめる］

# ---- まばらな語の表し方（段階9。生まれつき） ----
MEMORY = "sparse"        # 既定。"table"＝前の 64×64 の表 W（比べる用）
N_POOL = 1000            # 語の細胞・音の細胞の数
K = 10                   # 一度に光る数（log2(1000)≈10：Willshaw の表で一番たくさん覚えられる数のめやす）
RECALL_N = 3             # 思い出すのに要る、光った細胞の覚えた回数の真ん中の値（表 W の「同じ組み合わせ3回」と同じ）
WORD_SEED, SOUND_SEED = 31, 32
N_RIVALS = MAX_OBJECTS - 2   # まちがえそうな物の数（書式の上限 8 − 聞いた語の予想 − 物ふつうの見た目）
_ZMAP = torch.zeros(GRID, GRID, dtype=DTYPE)
# 盤面4段目：「知っている」の引き直しの似かた。物の荷札の揺れ（1つの数あたり0.1）で、正しい名前なら約0.78、
# 6つの物どれとも聞く語なら約0.3［仮置き］
KNOW_LO, KNOW_HI = 0.35, 0.7
SAME_OBJECT = 0.3        # 盤面4段目：見た目の荷札の似かたがこれより大きければ同じ物（worth.SAME_PERSON と同じ）
NAME_KNOWN = 0.5         # 盤面5段目：知る度合いがこれ以上なら、目の前の物の名前の語を思い出の帳面へ渡す


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
    """語の荷札 → 光る語の細胞（1000 のうち K 個が 1）。"""
    return kwta(_PW @ make_tag(q))


def sound_code(n: torch.Tensor) -> torch.Tensor:
    """物の音の荷札 → 光る音の細胞（1000 のうち K 個が 1）。"""
    return kwta(_PS @ make_tag(n))


class Semantic(Box):
    BOX_ID = "semantic"

    def __init__(self, learning: bool = False, learn_mode: str = "B", memory: str | None = None) -> None:
        super().__init__()
        self.table: dict[str, torch.Tensor] = {}   # 段階2：{語: 見た目の荷札}
        self.memory = memory or MEMORY
        if self.memory == "table":
            self.register_buffer("W", torch.zeros(TAG_DIM, TAG_DIM, dtype=DTYPE))
        elif self.memory == "sparse":
            self.register_buffer("A", torch.zeros(N_POOL, TAG_DIM, dtype=DTYPE))    # 語の細胞 → 見た目の予想
            self.register_buffer("Q", torch.zeros(N_POOL, TAG_DIM, dtype=DTYPE))    # 語の細胞 → 語の形
            self.register_buffer("T", torch.zeros(N_POOL, N_POOL, dtype=DTYPE))     # 音の細胞（行）→ 語の細胞
            self.register_buffer("c", torch.zeros(N_POOL, dtype=DTYPE))             # 語の細胞ごとに見た目と覚えた回数
            self.register_buffer("cq", torch.zeros(N_POOL, dtype=DTYPE))            # 語の細胞ごとに覚えた回数（見た目か音）
            self.register_buffer("cs", torch.zeros(N_POOL, dtype=DTYPE))            # 音の細胞ごとに覚えた回数
            self.register_buffer("G", torch.zeros(TAG_DIM, dtype=DTYPE))            # 物ふつうの見た目（見た物の平均）
            self.register_buffer("g_n", torch.zeros((), dtype=DTYPE))               # その回数
        else:
            raise ValueError(f"memory は table か sparse：{memory!r}")
        self.register_buffer("n_learn", torch.zeros((), dtype=torch.int64))
        self.learning = bool(learning)
        self.learn_mode = learn_mode
        self._gen = torch.Generator().manual_seed(0)
        self._gen_r = torch.Generator().manual_seed(1)     # まちがえそうな物の雑音（言う語の雑音の並びを変えない）
        self.reset()

    def reset(self) -> None:
        for k in ("_board_n", "_board_k", "_board_aha", "_board_unknown"):
            self.__dict__.pop(k, None)
        self._prev_heard = False
        self._prev_q = None
        self._u = 0.0                    # 段階8：前の回に届いた驚きの強さ
        self._n = None                   # 段階9：届いている物の音の荷札（v＝1。無ければ None）
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
    @property
    def sparse(self) -> bool:
        # 段階9より前に pickle した係（試験場の続きの写し）には memory が無い＝表 W
        return self.__dict__.get("memory", "table") == "sparse"

    @property
    def recall_line(self) -> float:
        """思い出す線（表 W：‖W q‖、まばら版：覚えた回数の真ん中の値）。どちらも「同じ組み合わせ3回」。"""
        return RECALL_N if self.sparse else RECALL_MIN

    def learn(self, appearance_tag, word_q, strength: float = 1.0) -> None:
        a, q = make_tag(appearance_tag), make_tag(word_q)
        if self.sparse:        # 覚え方A（比べる用）：強さは使わず、見た目を1回分の平均に入れる
            self._bind(a, None, q)
            return
        self.W += ETA * strength * torch.outer(a, q)

    def _bind(self, a, n, q) -> None:
        """まばら版で1回覚える。a：見た目（＋驚き）、n：物の音（無ければ None）、q：語の荷札。"""
        z = word_code(q)
        idx = z.nonzero().flatten()
        has_a = a is not None and float(torch.linalg.vector_norm(a)) > 0.0
        has_n = n is not None and float(torch.linalg.vector_norm(n)) > 0.0
        if has_a or has_n:     # 語の形は、見た目でも音でも、覚えるたびに平均へ（逆に引くときの語）
            self.cq[idx] += 1.0
            self.Q[idx] += (make_tag(q) - self.Q[idx]) / self.cq[idx].unsqueeze(1)
        if has_a:
            self.c[idx] += 1.0
            self.A[idx] += (make_tag(a) - self.A[idx]) / self.c[idx].unsqueeze(1)
        if has_n:
            js = sound_code(n).nonzero().flatten()     # 音の細胞が、そのとき光っていた語の細胞の組の平均を覚える
            self.cs[js] += 1.0
            self.T[js] += (z - self.T[js]) / self.cs[js].unsqueeze(1)

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
        """語 → 見た目の予想（64）。まだ3回覚えていなければ None。"""
        if self.sparse:
            idx = word_code(q).nonzero().flatten()
            if len(idx) == 0:
                self.last_recall_norm = 0.0
                return None
            cnt = float(self.c[idx].median())
            self.last_recall_norm = cnt
            return make_tag(self.A[idx].mean(0)) if cnt >= RECALL_N else None
        r = self.W @ q
        n = float(torch.linalg.vector_norm(r))
        self.last_recall_norm = n
        return make_tag(r) if n >= RECALL_MIN else None

    def rivals(self, q) -> list:
        """聞いた語のまちがえそうな物の見た目（雑音なし）：[物ふつうの見た目 G] ＋ ほかの語の見た目の予想を、聞いた語の予想に近い順に
        N_RIVALS 語（まばら版だけ。ほかは空）。ほかの語は、聞いた語の細胞以外の覚えた細胞 j から、その細胞が覚えた語の形 Q_j で
        思い出す（語の細胞の組 word_code(Q_j) の見た目の予想の平均）。聞いた語や先に選んだ語と細胞の半分より多くが重なる組は同じ語とみなして使わない
        （ちがう語どうしの重なりは多くて2個。2つの語の細胞に光る細胞の語の形は2つの語の混ざりで、組がそれぞれと半分ほど重なる）。
        細胞1つずつだと同じ語の細胞が何度も入り、まちがえそうな語が3つほどしか入らなかった（2026-10-18）。"""
        if not self.sparse:
            return []
        idx = word_code(q).nonzero().flatten()
        mine = set(idx.tolist())
        r = make_tag(self.A[idx].mean(0))
        g = getattr(self, "G", None)
        out = [make_tag(g)] if g is not None and float(torch.linalg.vector_norm(g)) > 0.0 else []
        ok = self.c >= RECALL_N
        ok[idx] = False
        cand = ok.nonzero().flatten()
        if len(cand) == 0:
            return out
        An = self.A[cand] / torch.linalg.vector_norm(self.A[cand], dim=1, keepdim=True).clamp_min(1e-12)
        used = [mine]
        for j in torch.argsort(An @ r, descending=True).tolist():
            if len(used) > N_RIVALS:
                break
            z = word_code(self.Q[cand[j]]).nonzero().flatten()
            key = set(z.tolist())
            if any(len(key & u) > K // 2 for u in used) or float(self.c[z].median()) < RECALL_N:
                continue
            used.append(key)
            out.append(make_tag(self.A[z].mean(0)))
        return out

    def recall_list(self, q):
        """目的の係へ渡す列：[(聞いた語の見た目の予想, ゼロ), (まちがえそうな物, ゼロ)…]。思い出せなければ None。
        予想には今までと同じ大域の乱数で雑音、まちがえそうな物には自分専用の乱数で雑音。"""
        r = self.recall(q)
        if r is None:
            return None
        riv = [make_tag(v + NOISE_SIGMA * torch.randn(TAG_DIM, generator=self.__dict__.setdefault(
            "_gen_r", torch.Generator().manual_seed(1)))) for v in self.rivals(q)]
        return [(noisy(r), _ZMAP)] + [(v, _ZMAP) for v in riv]

    def recall_strength(self, q) -> float:
        """語を思い出す強さ（線 recall_line と比べる）。"""
        self.recall(q)
        return self.last_recall_norm

    def drive(self, a=None, n=None) -> torch.Tensor:
        """まばら版：見た目（＋驚き）a・物の音 n から、語の細胞の強さ h ＝ A·a ＋ Tᵀz_s/K。
        細胞は、3回（RECALL_N、思い出す線と同じ）覚えるまで逆向きには働かない。平均で覚えるので、1回だけ覚えた細胞の
        見た目の予想もよく覚えた細胞と同じ長さになり、たまたま驚きの窓で1回覚えた語の細胞が「ない」に勝ってしまう
        （段階8の試験場で見つけた：1日目、驚きだけで引くと、1回だけ覚えた細胞10個が「ない」の細胞より強かった）。"""
        h = torch.zeros(N_POOL, dtype=DTYPE)
        if a is not None and float(torch.linalg.vector_norm(a)) > 0.0:
            h = h + (self.A @ make_tag(a)) * (self.c >= RECALL_N)
        if n is not None and float(torch.linalg.vector_norm(n)) > 0.0:
            h = h + self.T.T @ (sound_code(n) * (self.cs >= RECALL_N)) / K
        return h

    def reverse(self, a=None, n=None):
        """見た目（＋驚き）a・物の音 n → (語の形 64, 強さ)。線では切らない（試験場・記録用）。何も無ければ (None, 0)。"""
        z = torch.zeros(TAG_DIM, dtype=DTYPE)
        a = a if a is not None else z
        n = n if n is not None else z
        if float(torch.linalg.vector_norm(a)) == 0.0 and float(torch.linalg.vector_norm(n)) == 0.0:
            return None, 0.0
        if not self.sparse:
            r = self.W.T @ (a + n)
            return r, float(torch.linalg.vector_norm(r))
        h = self.drive(a, n)
        idx = kwta(h).nonzero().flatten()
        if len(idx) == 0 or float(h[idx].sum()) <= 0.0:
            return None, 0.0
        return make_tag(self.Q[idx].mean(0)), float(self.cq[idx].median())

    def word_scores(self, words_q: torch.Tensor, a=None, n=None) -> torch.Tensor:
        """語ごとの強さ（試験場用）。表 W：q·Wᵀ(a＋n)、まばら版：その語の細胞の組の強さの合計。"""
        z = torch.zeros(TAG_DIM, dtype=DTYPE)
        a = a if a is not None else z
        n = n if n is not None else z
        if not self.sparse:
            return words_q @ (self.W.T @ (a + n))
        h = self.drive(a, n)
        return torch.stack([h @ word_code(q) for q in words_q])

    def _seen_vec(self, lst, u: float, n=None):
        """毎回数える見た目：今見ている行の荷札 a（無ければゼロ）＋ u·S（段階8）＋ v·N（段階9、物の音、v＝1）。何も無ければ None。"""
        a = None
        if lst and len(lst) >= 2 and float(torch.linalg.vector_norm(lst[-2][0])) > 0.0:
            a = make_tag(lst[-2][0])
        if u > 0.0:
            a = (a if a is not None else 0.0) + u * SURPRISE_TAG
        if self.sparse:        # まばら版：見た目（＋驚き）と音を分けて持つ [見た目 64 | 音 64]
            if a is None and n is None:
                return None
            z = torch.zeros(TAG_DIM, dtype=DTYPE)
            return torch.cat([a if a is not None else z, n if n is not None else z])
        if n is not None:
            a = (a if a is not None else 0.0) + n
        return a

    def recall_word(self, lst, u: float = 0.0, n=None):
        """段階7：逆向きに引く。名簿の「今見ている行」の荷札 a から語 Wᵀa を思い出す（同じ表を逆に引くだけ）。
        段階8：驚きの強さ u があれば Wᵀ(a＋u·S)。"""
        self.last_word_norm = 0.0
        x = self._seen_vec(lst, u, n)
        if x is None:
            return None
        if self.sparse:
            r, cnt = self.reverse(x[:TAG_DIM], x[TAG_DIM:])
            self.last_word_norm = cnt
            return r if r is not None and cnt >= RECALL_N else None
        r = self.W.T @ x
        n = float(torch.linalg.vector_norm(r))
        self.last_word_norm = n
        return make_tag(r) if n >= RECALL_MIN else None

    def _learned_names(self) -> tuple:
        return ("A", "Q", "T", "c", "cq", "cs", "G", "g_n") if self.sparse else ("W",)

    def save_learned(self, path, meta: dict | None = None) -> None:
        blob = {k: getattr(self, k).clone() for k in self._learned_names()}
        torch.save({**blob, "memory": self.memory, "n_learn": int(self.n_learn), "meta": meta or {}}, path)

    def load_learned(self, path) -> dict:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("memory", "table") != self.memory:
            raise ValueError(f"覚えた中身の形が違う（保存 {blob.get('memory', 'table')}・今 {self.memory}）")
        for k in self._learned_names():
            if k in blob:          # 物ふつうの見た目 G は 2026-10-18 から（前の保存には無い＝ゼロのまま）
                getattr(self, k).copy_(blob[k])
        self.n_learn.fill_(int(blob.get("n_learn", 0)))
        return blob.get("meta", {})

    def use_board(self) -> None:
        self.__dict__["_board"] = True

    def knowing(self, a) -> float:
        """盤面：見た目 a の名前をどれくらい知っているか（0〜1）。物 → 語 → 物 の引き直しが元に戻る度合い。"""
        a = make_tag(a)
        word, strength = self.reverse(a)
        if word is None or strength < RECALL_N:
            return 0.0
        r = self.recall(word)
        if r is None:
            return 0.0
        c = float(r @ a)
        return min(1.0, max(0.0, (c - KNOW_LO) / (KNOW_HI - KNOW_LO)))

    def board_step(self, inbox, out):
        from ..rhythm import OUT_PHASE, SEM_PHASE, phase
        n = self.__dict__.get("_board_n", 0)
        seen = inbox.get("eyein") or []
        obj = seen[0][0] if seen else None
        q = inbox.get("listen")
        # 語と物を結ぶのは、大人もその物を見て話している時だけ（地図の [1, 0]。一緒に見ている時の語を
        # その物の名前と取る、Baldwin 1991。忙しい時の「まっててね」などを物と結んで、わかったが出ていた）
        attend = bool(seen) and float(seen[0][1][1, 0]) > 0.5
        if obj is not None and attend and q is not None and float(torch.linalg.vector_norm(q)) > 0.0:
            self._bind(make_tag(obj), None, make_tag(q))
            self.n_learn += 1
            self.__dict__["_board_heard"] = True       # この分に物と語を結んだ（わかったが変わりうる）
        if phase(n) == SEM_PHASE:      # わからなさ（この分に選ぶ係へ）・目の前の物の名前（思い出の帳面へ）
            k = self.knowing(obj) if obj is not None else 0.0
            self.__dict__["_board_unknown"] = (1.0 - k) if obj is not None else 0.0
            word = self.reverse(make_tag(obj))[0] if obj is not None and k >= NAME_KNOWN else None
            self.__dict__["_board_name"] = make_tag(word) if word is not None else torch.zeros(TAG_DIM, dtype=DTYPE)
        if phase(n) == OUT_PHASE:      # わかった（この分に聞いた語を覚え終えてから。次の分の初めに worth が読む）
            # わかった＝その物について、今までで一番わかった度合いを超えた分（覚えたものどうしの混ざりで
            # 度合いが上下するので、下がって戻っただけの分は数えない）
            # 物と語を結ばなかった分は数えない（見るたびの揺れで度合いが変わるだけ）。物は荷札の似かたで見分ける
            # （見るたびに揺れるので、同じ物どうしの似かたは0.6ほど。前は0.9で比べていて、毎分「初めての物」になり
            # わかったが毎分出ていた）
            heard = self.__dict__.pop("_board_heard", False)
            k = self.knowing(obj) if obj is not None else 0.0
            best = self.__dict__.setdefault("_board_best", [])       # [(物の荷札, 一番わかった度合い)]
            sims = [float(o @ obj) for o, _ in best] if obj is not None else []
            i = max(range(len(sims)), key=sims.__getitem__) if sims and max(sims) > SAME_OBJECT else None
            aha = 0.0
            if obj is not None:
                if i is None:
                    best.append((make_tag(obj), k))                  # 初めて見た物：今の度合いを覚えるだけ
                else:
                    aha = max(0.0, k - best[i][1]) if heard else 0.0
                    best[i] = (best[i][0], max(k, best[i][1]))
            self.__dict__["_board_aha"] = aha
        elif phase(n) == 0:
            self.__dict__["_board_aha"] = 0.0
        self.__dict__["_board_n"] = n + 1
        out["want"] = torch.tensor(self.__dict__.get("_board_unknown", 0.0), dtype=DTYPE)
        out["worth"] = torch.tensor(self.__dict__.get("_board_aha", 0.0), dtype=DTYPE)
        out["episodic"] = self.__dict__.get("_board_name", torch.zeros(TAG_DIM, dtype=DTYPE))
        return out

    def step(self, inbox):
        out = self.passthrough()
        if self.__dict__.get("_board"):
            return self.board_step(inbox, out)
        self.last_event = ""
        q = inbox.get("listen")
        heard = q is not None and float(torch.linalg.vector_norm(q)) > 0.0
        first = heard and (not self._prev_heard or float(self._prev_q @ q) < 0.99)
        self._prev_heard = heard
        self._prev_q = q if heard else None
        u = inbox.get("surprise")
        self._u = float(u) if u is not None else 0.0
        n = inbox.get("soundin")
        self._n = n if n is not None and float(torch.linalg.vector_norm(n)) > 0.0 else None
        self._trace_step(inbox.get("objfile"), q if (self.learning and first and self.learn_mode == "B"
                                                     and not self.table) else None)
        if not self.table:
            w = self.recall_word(inbox.get("objfile"), self._u, self._n)
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
                out["goal"] = [(noisy(self.table[best]), _ZMAP)]
            return out
        lst = self.recall_list(q)
        if lst is not None:
            out["goal"] = lst
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
        held = self._seen_vec(lst, 0.0)    # 飛ばすのは驚きの無い間だけ（試験場が決める）
        self._count_seen(lst, n)
        keep_n = int(round(LOOK_WINDOW_S / TICK_S))
        self._held_hist = (self._held_hist + [held] * min(n, keep_n))[-keep_n:]
        self._prev_heard = False
        self._prev_q = None
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

    def _count_seen(self, lst, n: int) -> None:
        """物ふつうの見た目 G：見ている物の見た目の平均（数えるだけ。語は要らない）。同じ物を n 回見たとして足す。"""
        if self.learning and self.sparse and lst and len(lst) >= 2 and float(torch.linalg.vector_norm(lst[-2][0])) > 0.0:
            self.g_n += float(n)
            self.G += n * (make_tag(lst[-2][0]) - self.G) / self.g_n

    def _trace_step(self, lst, new_q) -> None:
        """覚え方B：今見ている行を覚えておき、名残りの残っている語ごとに足す。名残りが消えたら覚える。"""
        held = self._seen_vec(lst, self._u, self._n)
        self._count_seen(lst, 1)
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
            v = torch.stack(seen).sum(0) if seen else torch.zeros((2 if self.sparse else 1) * TAG_DIM, dtype=DTYPE)
            self._pending.append({"q": new_q.clone(), "k": 0, "v": v, "w": float(len(seen))})

    def _learn_pending(self, p) -> None:
        if p["w"] == 0.0:
            self.last_event = "learn_skip"
            return
        if self.sparse:
            v = p["v"] / p["w"]
            self._bind(v[:TAG_DIM], v[TAG_DIM:], p["q"])
            self.n_learn += 1
            self.last_event = "learn"
            return
        self.W += ETA * torch.outer(p["v"] / p["w"], make_tag(p["q"]))
        self.n_learn += 1
        self.last_event = "learn"
