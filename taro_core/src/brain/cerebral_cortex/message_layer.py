"""言いたいことの層（MessageLayer） — 物の塊と述語の塊を別々に選んで並べる。

【出典】仕様_M6_言いたいことの層_2026-09-07.md 後半§1。土台＝塊レベル層
（chunk_vocab.py・仕様_M5_塊レベル層_2026-09-07.md）。

3つの数え上げ表で成り立つ（すべて数え上げ＝睡眠の再生を通さない一発記憶。
逸脱その49として登録）：
  state_count  塊ごとに「あるとき／消えたときに聞かれた回数」
  pred_count   状態ごとに「親が言った塊（役割=述語のものだけ）」の回数
  role_bigram  役割の並び（<s>→noun→pred→</s> 等）の回数

役割判定（role()）：仕様書の本文は「P(gone|chunk)とbaseのP(gone)の差の絶対値が
role_dev_thresh以上ならpred」としつつ、注意書きで「baseのP(gone)が小さいと
差が閾値未満になりうる」問題を指摘し、代替として「比が2倍以上または半分以下」
を実装担当が採用してよいとしている。本実装はその比の方式を採用した
（役割判定の例：「ないね」P(gone|c)=1・base P(gone)≈0.2→比5倍→pred、
「だね」P(gone|c)=0・base P(gone)≈0.2→比≈0倍→pred、「かばん」は比≈1倍→noun）。
role_dev_threshは仕様書のコンストラクタ引数と互換のため引数として残すが、
比の方式では使わない（未使用。作業記録「仕様に無かった判断」に記載）。
"""


class MessageLayer:
    STATES = ("here", "gone")   # 状態の箱の値。None（注意中の物なし）は数えない

    # 【Tier3・2026-09-07】比の判定のしきい値（2倍/0.5倍）。仕様書本文の
    #   注意書きが例示した値をそのまま採用（文献根拠なし・恣意的）。
    ROLE_RATIO_HIGH = 2.0
    ROLE_RATIO_LOW = 0.5
    # 【Tier3・2026-09-07】役割の並び（compose）を最大何手たどるか。塊は
    #   noun/predの2種類しか無いので3手あれば十分（無限ループ防止の歯止め）。
    _MAX_ORDER_STEPS = 3
    # role_bigramがまだ空のとき（学習の最初期）に使う既定の並び。日本語は
    #   述語が文末に来ることが多いため noun→pred を仮の初期値にする
    #   （Tier3・恣意的。data が育てば role_bigram側の実測に置き換わる）。
    _DEFAULT_ORDER = ("noun", "pred")

    def __init__(self, role_dev_thresh=0.3, min_count=3):
        self.role_dev_thresh = role_dev_thresh   # 【未使用】docstring参照
        self.min_count = min_count
        self.state_count = {}   # chunk_id -> {"here": n, "gone": n}
        self.pred_count = {}    # state -> {chunk_id: n}
        self.role_bigram = {}   # role_prev("<s>"|"noun"|"pred") -> {role_next: n}
        self.base = {"here": 0, "gone": 0}   # 状態の基準率（発話1文ごとに1加算）

    def observe(self, chunk_ids, state):
        """親の発話1文（塊id列、特殊トークン除く）と、そのときの状態を数える。

        state は "here"|"gone" 以外なら何もしない（呼び出し側が None を渡す
        ケース＝注意中の物が無いとき）。
        """
        if state not in self.STATES:
            return
        # 1) base と state_count を更新
        self.base[state] = self.base.get(state, 0) + 1
        for cid in chunk_ids:
            cid = int(cid)
            sc = self.state_count.setdefault(cid, {"here": 0, "gone": 0})
            sc[state] += 1
        # 2) 各塊の役割を role() で判定し、3) 役割列に <s>/</s> を付けて
        #    role_bigram を更新する（未知(None)は役割列から除く＝順番表は
        #    noun/predの並びだけを数える）。
        roles = []
        for cid in chunk_ids:
            r = self.role(int(cid))
            if r is not None:
                roles.append((int(cid), r))
        seq = ["<s>"] + [r for _cid, r in roles] + ["</s>"]
        for i in range(len(seq) - 1):
            prev, nxt = seq[i], seq[i + 1]
            bucket = self.role_bigram.setdefault(prev, {})
            bucket[nxt] = bucket.get(nxt, 0) + 1
        # 4) 役割が pred の塊は pred_count[state] に加算
        for cid, r in roles:
            if r == "pred":
                bucket = self.pred_count.setdefault(state, {})
                bucket[cid] = bucket.get(cid, 0) + 1

    def role(self, chunk_id):
        """"noun" | "pred" | None(未知＝観測min_count未満、またはbase未形成)。

        判定は比の方式（docstring参照）：
        max(P(gone|c), P(here|c)) 側と base の対応する比が2倍以上または
        0.5倍以下なら「状態と一緒に変わる」＝pred。そうでなければ noun。
        """
        sc = self.state_count.get(int(chunk_id))
        if sc is None:
            return None
        n = sc["here"] + sc["gone"]
        if n < self.min_count:
            return None
        total = self.base["here"] + self.base["gone"]
        # 【注意・2026-09-07】base側がhere/goneの片方しかまだ経験していない
        #   （例：goneをまだ一度も見ていない）と、比較先の基準率が0か1に
        #   潰れて交差乗算の式が常に片側へ倒れ、min_count以上聞いた塊が
        #   軒並みpred判定される（実測で確認）。両方の状態を最低1回ずつ
        #   基準側が経験するまではNone（未知）にする。
        if total == 0 or self.base["gone"] == 0 or self.base["here"] == 0:
            return None
        # 【なぜ、2026-09-07】P(gone|c)/P(gone) を素直に浮動小数の割り算2回で
        #   出すと、ちょうど境界値（2.0/0.5）のときにepsの丸めで境界の内側へ
        #   寄ってしまう（実測：sc={"gone":59,"here":0}, base 100/100の
        #   ケースでratioが1.99999...となり2.0以上の判定を1つ取りこぼした）。
        #   割り算を1回にまとめる交差乗算（cross-multiplication）にすれば
        #   epsが要らず境界値も正しく判定できる：
        #   P(gone|c) >= H*P(gone) <=> sc["gone"]*total >= H*base["gone"]*n
        gone_c, total_c = sc["gone"], n
        gone_b, total_b = self.base["gone"], total
        high_lhs = gone_c * total_b
        high_rhs = self.ROLE_RATIO_HIGH * gone_b * total_c
        low_lhs = gone_c * total_b
        low_rhs = self.ROLE_RATIO_LOW * gone_b * total_c
        if high_lhs >= high_rhs or low_lhs <= low_rhs:
            return "pred"
        return "noun"

    def _best_role_order(self):
        """role_bigramの<s>から</s>までを、各歩でいちばん多く数えられた次の役割へ
        たどった並び（例 ["noun", "pred"]）。role_bigramが空ならDEFAULT_ORDER。
        """
        if not self.role_bigram:
            return list(self._DEFAULT_ORDER)
        order = []
        cur = "<s>"
        seen = set()
        for _ in range(self._MAX_ORDER_STEPS):
            nxts = self.role_bigram.get(cur)
            if not nxts:
                break
            nxt = max(nxts.items(), key=lambda kv: kv[1])[0]
            if nxt == "</s>" or nxt in seen:
                break
            order.append(nxt)
            seen.add(nxt)
            cur = nxt
        return order if order else list(self._DEFAULT_ORDER)

    def compose(self, state, noun_candidates):
        """名詞の塊＋述語の塊を役割の順に並べる。

        noun_candidates: [(chunk_id, prob)]（塊GRUの先頭塊の分布のうち、
        役割 noun のものだけを呼び出し側が渡す）。
        戻り値: (chunk_ids, conf, roles)。conf = noun_prob * pred_prob
        （pred_count[state]が空なら pred は選ばれず conf は noun_prob のまま）。
        """
        order = self._best_role_order()
        best_noun = max(noun_candidates, key=lambda kv: kv[1]) if noun_candidates else None
        noun_id, noun_prob = best_noun if best_noun is not None else (None, 0.0)

        pred_table = self.pred_count.get(state, {}) if state is not None else {}
        pred_id, pred_prob = None, 1.0
        if pred_table:
            pred_id, pred_n = max(pred_table.items(), key=lambda kv: kv[1])
            total_pred = sum(pred_table.values())
            pred_prob = (pred_n / total_pred) if total_pred else 0.0

        chunk_ids, roles = [], []
        for r in order:
            if r == "noun" and noun_id is not None:
                chunk_ids.append(noun_id)
                roles.append("noun")
            elif r == "pred" and pred_id is not None:
                chunk_ids.append(pred_id)
                roles.append("pred")
        conf = noun_prob * pred_prob
        return chunk_ids, conf, roles

    def state_dict(self):
        return {
            "state_count": {int(k): dict(v) for k, v in self.state_count.items()},
            "pred_count": {k: {int(kk): int(vv) for kk, vv in v.items()}
                           for k, v in self.pred_count.items()},
            "role_bigram": {k: dict(v) for k, v in self.role_bigram.items()},
            "base": dict(self.base),
            "role_dev_thresh": self.role_dev_thresh,
            "min_count": self.min_count,
        }

    def load_state_dict(self, d):
        self.state_count = {int(k): dict(v)
                             for k, v in d.get("state_count", {}).items()}
        self.pred_count = {k: {int(kk): int(vv) for kk, vv in v.items()}
                           for k, v in d.get("pred_count", {}).items()}
        self.role_bigram = {k: dict(v) for k, v in d.get("role_bigram", {}).items()}
        self.base = dict(d.get("base", {"here": 0, "gone": 0}))
        self.role_dev_thresh = d.get("role_dev_thresh", self.role_dev_thresh)
        self.min_count = d.get("min_count", self.min_count)
