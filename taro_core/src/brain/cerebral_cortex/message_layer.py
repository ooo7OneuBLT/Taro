"""言いたいことの層（MessageLayer） — 物の塊と述語の塊を別々に選んで並べる。

【出典】仕様_M6_言いたいことの層_2026-09-07.md 後半§1。
【M6b改訂・2026-09-07】仕様_M6b_役割は見た目との結び付きで_2026-09-07.md。
【M6b追記・2026-09-07（同ファイル末尾「追記」）】key_visから全塊共通の平均鍵
（global_mean）を引いた中心化ベクトルで vis_mean/vis_cons を計算するよう変更。
土台＝塊レベル層（chunk_vocab.py・仕様_M5_塊レベル層_2026-09-07.md）。

5つの数え上げ・累積平均で成り立つ（すべて数え上げ／移動平均＝睡眠の再生を
通さない一発記憶。逸脱その49として登録）：
  state_count  塊ごとに「あるとき／消えたときに聞かれた回数」（述語表にのみ使う）
  global_mean  全塊共通の「見えていた鍵（見た目、64次元）」の移動平均（α=0.05）。
               机・壁など全塊に共通する成分（視野全体CLSの共有成分）を推定する
  vis_mean     塊ごとに「一緒に見えていた鍵からglobal_meanを引いた中心化ベクトル」
               の移動平均
  vis_cons     塊ごとに「入ってきた中心化ベクトルと、そのとき現在だった vis_mean
               とのコサイン類似度」の移動平均（α=0.1）。値が高い＝毎回だいたい
               同じ物と一緒に出た（名詞らしい）。値が低い＝出るたびに違う物と
               一緒だった（述語らしい）
  pred_count   状態ごとに「親が言った塊（役割=述語のものだけ）」の回数
  role_bigram  役割の並び（<s>→noun→pred→</s> 等）の回数

【M6→M6bで変えた理由】M6は役割を「状態と一緒に変わるか」（state_countの比較）
だけで決めていた。すると、黙って隠した教えていない3語は「消えたときに聞かれない」
点で「だね／だよ」と同じに見え、述語側に誤分類された（研究日誌2026-09-07
「F2-87/87b M6」節）。人間の子が名詞と述語を分ける手がかりは、本来「その語が
特定の物と一緒に出るか（名詞）、どの物とも出るか（述語）」である。M6bはそこに
戻し、role()の判定を状態との関係(state_count)からvis_consの1次元k-means
（k=2）に差し替えた。state_countは述語表（pred_count）を作るためだけに残る。

【M6b追記で変えた理由】机上確認（F2-87モデルの海馬16件）で、述語（だね・
ないね）のvis_consも0.95以上と高く出た。脳の見た目ベクトル（視野全体のCLS）は
机・壁などの共有成分が大きく、別の物どうしでもコサインが高いため。全塊共通の
平均鍵を引いて共有成分を消してから比べるよう直した。
"""

import numpy as np


class MessageLayer:
    STATES = ("here", "gone")   # 状態の箱の値。None（注意中の物なし）は数えない

    # 【Tier3・2026-09-07・M6b】vis_consの移動平均の速さ。lexicon.py contrastの
    #   eta_pull(0.05)より少し速いが桁は同じ（文献根拠なし・恣意的、仕様書指定値）。
    VIS_ALPHA = 0.1
    # 【Tier3・2026-09-07・M6b追記】全塊共通の平均鍵(global_mean)の移動平均の
    #   速さ。VIS_ALPHA(0.1)より遅くした＝個々の塊のvis_consより緩やかに動く
    #   「背景（共有成分）」を推定する狙い（文献根拠なし・恣意的、仕様書指定値）。
    GLOBAL_ALPHA = 0.05
    # 【Tier3・2026-09-07】1次元k-means(k=2)の反復回数の歯止め（無限ループ防止）。
    #   塊の名簿規模に対して十分（通常数回で収束）。
    _KMEANS_MAX_ITER = 50
    # 【Tier3・2026-09-07】役割の並び（compose）を最大何手たどるか。塊は
    #   noun/predの2種類しか無いので3手あれば十分（無限ループ防止の歯止め）。
    _MAX_ORDER_STEPS = 3
    # role_bigramがまだ空のとき（学習の最初期）に使う既定の並び。日本語は
    #   述語が文末に来ることが多いため noun→pred を仮の初期値にする
    #   （Tier3・恣意的。data が育てば role_bigram側の実測に置き換わる）。
    _DEFAULT_ORDER = ("noun", "pred")

    def __init__(self, role_dev_thresh=0.3, min_count=3):
        # 【未使用・M6の名残】M6のratio方式のコンストラクタ引数と互換のため
        #   引数として残すが、M6bのk-means方式では使わない
        #   （作業記録「仕様に無かった判断」に記載）。
        self.role_dev_thresh = role_dev_thresh
        self.min_count = min_count
        self.state_count = {}   # chunk_id -> {"here": n, "gone": n}
        self.global_mean = None  # 全塊共通の平均鍵（64次元、numpy配列）。M6b追記
        self.vis_mean = {}      # chunk_id -> list[float](中心化ベクトルの移動平均)
        self.vis_cons = {}      # chunk_id -> float（コサイン類似度の移動平均）
        self.pred_count = {}    # state -> {chunk_id: n}
        self.role_bigram = {}   # role_prev("<s>"|"noun"|"pred") -> {role_next: n}
        self.base = {"here": 0, "gone": 0}   # 状態の基準率（発話1文ごとに1加算）

    def observe(self, chunk_ids, state, key_vis=None):
        """親の発話1文（塊id列、特殊トークン除く）と、そのときの状態・見た目を数える。

        state は "here"|"gone" 以外なら何もしない（呼び出し側が None を渡す
        ケース＝注意中の物が無いとき）。
        key_vis: このtickで見えていた物の鍵（64次元、numpy配列かリスト）。
        None なら vis_mean/vis_cons の更新をスキップする（呼び出し側が視覚投射を
        まだ持たない等の場合の後方互換。役割判定には観測が要るので、その場合
        role()はNoneのまま＝この塊はまだ役割不明として扱われる）。
        """
        if state not in self.STATES:
            return
        # 1) base と state_count を更新
        self.base[state] = self.base.get(state, 0) + 1
        _centered = None
        if key_vis is not None:
            _key = np.asarray(list(key_vis), dtype=np.float64)
            _centered = self._center_and_update_global(_key)
        for cid in chunk_ids:
            cid = int(cid)
            sc = self.state_count.setdefault(cid, {"here": 0, "gone": 0})
            sc[state] += 1
            if _centered is not None:
                self._update_vis(cid, _centered)
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

    def _center_and_update_global(self, key):
        """全塊共通の平均鍵(global_mean)を引いた中心化ベクトルを返す（M6b追記）。

        コサイン(_update_vis)と同じ流儀で、中心化には「更新前の」global_mean
        を使い、その後にglobal_meanをEMA(GLOBAL_ALPHA)で更新する。初回観測は
        基準となる平均が無いのでkeyそのもので初期化し、中心化ベクトルは
        ゼロベクトル（自分自身との差）になる。
        """
        if self.global_mean is None:
            self.global_mean = key.copy()
            return key - self.global_mean
        centered = key - self.global_mean
        self.global_mean = self.global_mean + self.GLOBAL_ALPHA * (key - self.global_mean)
        return centered

    def _update_vis(self, cid, key):
        """塊cidのvis_mean/vis_consを、今回の中心化ベクトルkeyで1回分更新する
        （M6b。keyはすでにglobal_meanを引いた値＝_center_and_update_globalの
        戻り値）。

        初回観測は「現在のvis_mean」がまだ無い（コサインを取る相手が無い）ので
        vis_meanをkeyそのもので初期化し、vis_consは1.0（自分自身と完全一致）
        にする（lexicon.py contrastモードのproto初期化と同じ考え方）。
        """
        vm = self.vis_mean.get(cid)
        if vm is None:
            self.vis_mean[cid] = key.copy()
            self.vis_cons[cid] = 1.0
            return
        cos = _cosine(key, vm)
        prev_cons = self.vis_cons.get(cid, cos)
        self.vis_cons[cid] = prev_cons + self.VIS_ALPHA * (cos - prev_cons)
        self.vis_mean[cid] = vm + self.VIS_ALPHA * (key - vm)

    def role(self, chunk_id):
        """"noun" | "pred" | None(未知＝観測min_count未満、または群分けできない)。

        判定（M6b）：観測min_count以上かつvis_consを持つ塊**全体**を対象に、
        vis_consの値を1次元k-means（k=2、初期値＝最小と最大）で2群に分け、
        値が高い群＝"noun"、低い群＝"pred"とする（doc/仕様_M6b参照）。
        """
        sc = self.state_count.get(int(chunk_id))
        if sc is None:
            return None
        n = sc["here"] + sc["gone"]
        if n < self.min_count:
            return None
        if self.vis_cons.get(int(chunk_id)) is None:
            return None
        labels = self._role_labels()
        return labels.get(int(chunk_id))

    def _eligible_chunks(self):
        """観測min_count以上、かつvis_consを持つ塊idの一覧（role()の対象集合）。"""
        out = []
        for cid, sc in self.state_count.items():
            if (sc["here"] + sc["gone"]) >= self.min_count and cid in self.vis_cons:
                out.append(cid)
        return out

    def _role_labels(self):
        """全塊のvis_consを1次元k-means(k=2)で2群に分け、
        {chunk_id: "noun"|"pred"} を返す（対象が2群に分けられないときは空dict）。
        """
        ids = self._eligible_chunks()
        if len(ids) < 2:
            # 【M6b】1点以下では「ばらつきの大小」を比べる相手がいない＝
            #   まだ判定できない（Noneのまま）。
            return {}
        vals = {cid: self.vis_cons[cid] for cid in ids}
        lo = min(vals.values())
        hi = max(vals.values())
        if lo == hi:
            # 全塊のvis_consが同値＝分けようがない（学習極初期を想定）。
            return {}
        c0, c1 = lo, hi
        assign = {}
        for _ in range(self._KMEANS_MAX_ITER):
            new_assign = {}
            for cid, v in vals.items():
                new_assign[cid] = 0 if abs(v - c0) <= abs(v - c1) else 1
            if new_assign == assign:
                assign = new_assign
                break
            assign = new_assign
            g0 = [vals[cid] for cid in ids if assign[cid] == 0]
            g1 = [vals[cid] for cid in ids if assign[cid] == 1]
            new_c0 = (sum(g0) / len(g0)) if g0 else c0
            new_c1 = (sum(g1) / len(g1)) if g1 else c1
            if new_c0 == c0 and new_c1 == c1:
                break
            c0, c1 = new_c0, new_c1
        high_cluster = 0 if c0 >= c1 else 1
        return {cid: ("noun" if assign[cid] == high_cluster else "pred")
                for cid in ids}

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
            "global_mean": (np.asarray(self.global_mean, dtype=np.float64).tolist()
                             if self.global_mean is not None else None),
            "vis_mean": {int(k): list(np.asarray(v, dtype=np.float64).tolist())
                         for k, v in self.vis_mean.items()},
            "vis_cons": {int(k): float(v) for k, v in self.vis_cons.items()},
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
        _gm = d.get("global_mean")
        self.global_mean = np.asarray(_gm, dtype=np.float64) if _gm is not None else None
        self.vis_mean = {int(k): np.asarray(v, dtype=np.float64)
                         for k, v in d.get("vis_mean", {}).items()}
        self.vis_cons = {int(k): float(v) for k, v in d.get("vis_cons", {}).items()}
        self.pred_count = {k: {int(kk): int(vv) for kk, vv in v.items()}
                           for k, v in d.get("pred_count", {}).items()}
        self.role_bigram = {k: dict(v) for k, v in d.get("role_bigram", {}).items()}
        self.base = dict(d.get("base", {"here": 0, "gone": 0}))
        self.role_dev_thresh = d.get("role_dev_thresh", self.role_dev_thresh)
        self.min_count = d.get("min_count", self.min_count)


def _cosine(a, b):
    """2つのnumpy配列のコサイン類似度。どちらかがゼロベクトルなら0.0を返す
    （ゼロ割りを避ける。実務上、視覚投射の出力がぴったりゼロになることは
    ほぼ無いが、初期状態や異常値の防御として置く）。
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
