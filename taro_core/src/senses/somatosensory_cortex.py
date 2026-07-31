"""体性感覚系(触覚)の脳内経路：視床VPL核（部位別集約） → 一次体性感覚野S1（部位間統合）。

【なぜこのファイルか、2026-07-24】従来の TouchEncoder は 12,000次元近い生の触覚を1枚の
巨大変換層に流し込んでいた。これは解剖学的にズレている：人間の触覚は
「皮膚→末梢神経→脊髄→視床VPL核→S1」と伝わる過程で、**部位ごとに整理されたマップ**を
経由する。VPL核は部位別受容野を持ち(顔・手・体幹・脚・足が内側から外側へ配置)、S1は
それを統合したホムンクルス表現を持つ(手や顔の領域は大きい=皮質拡大)。

【2026-07-31 の作り直し：センサ点数に依存しない要約へ】
前の版は部位ごとに `nn.Linear(センサ点数×3, dim)` を置いていた。これは
**センサ点数が入力次元に直結する**作りで、体を育てると壊れる：

    月齢      触覚flat次元   head の Linear
    0ヶ月      4,824         入力924 → 出力17
    4ヶ月      9,804         入力1347 → 出力21

しかも壊れ方が**静かだった**。`index_select` は作った時点のインデックス表を持ち続け、
配列が長くなる方向の変化では**範囲内に収まって例外にならない**。
0ヶ月で作った脳に4ヶ月の観測を入れると、落ちずに**別の部位を読んだ値**が返る
（落とし穴チェックリスト 項86）。

⇒ 部位ごとの出力を**点数に依存しない3つの要約**に置き換えた：

    有無   その部位のどこかに触れたか        max‖f‖ を tanh
    強さ   どれくらいの圧か                  mean‖f‖ を tanh
    重心   その部位の**どこに**触れたか      ‖f‖で重み付けした位置の平均（3次元）

部位あたり5次元。点が何個あっても5次元。**体が育っても層の形が変わらない。**
成長時は `rebuild()` でインデックス表と位置表だけ差し替え、**学習した重みは保つ**。

【人間模倣からの逸脱・2026-07-31 登録】
先行研究（國吉研）は触覚を**まとめない**。生の点を全結合で脊髄・皮質へ渡し、
「どの点がどの筋と相関するか」の発見を学習に委ねる：

  - 森・國吉 2010 日本ロボット学会誌 28(8), 1014-1024
    触覚細胞→αモータニューロン／介在ニューロンへ**全結合**、共分散則で学習
    https://www.jstage.jst.go.jp/article/jrsj/28/8/28_8_1014/_pdf
  - Yamada, Mori, Kuniyoshi 2016 *Sci Rep* 6:27893
    3,000点→皮質912ニューロンへ**多対多のランダム投射**。
    人手の区分は「脚・体幹・腕・頭」の4カテゴリだけで、しかも数値要約ではなく
    **トポグラフィック（近傍性を保つ）配線ルール**

  それができるのは**國吉研のモデルは体のサイズが固定だから**（胎児1体・早産児）。
  太郎は 0→4ヶ月で体を育てるので、生のままでは層が保てない。
  ⇒ **理由のある逸脱**。将来、体を固定した実験をするなら生の点に戻す選択肢がある。

【文献根拠】
- 視床VPL核の体部位マップ [Tier1]：Padberg et al. 2009 *Cereb Cortex*；
  Kaas, Nelson, Sur, Dykes, Merzenich 1984 *J Comp Neurol* 226；
  Leplus et al. 2024 *Ann Clin Transl Neurol* (脳深部刺激の臨床)
- S1のホムンクルス表現・皮質拡大 [Tier1]：教科書レベル
- 部位ごとの皮質の割り当て [Tier1]：Saadon-Grosman, Loewenstein, Arzy 2020
  *Hum Brain Mapp* 41(13):3620-3646「体性感覚野の割り当ては体表面積に比例しない」
  上肢46.3% / 唇28.9% / 体幹15.8% / 下肢9.1%
  → 部位の重み `part_weight` の**初期値**に使う（学習で動く）
- 末梢の触覚受容器密度 [Tier1]：Corniani & Saal 2020 *J Neurophysiol*
  指先241 / 手掌58 / 体幹9 units/cm²。注意**末梢密度と皮質割り当ては相関しない**
  (r=0.40, P=0.42) ので、重みの根拠には皮質側（Saadon-Grosman）を採った

【何がやっていないか(将来課題)】
- S1→S2の高次階層。今は最上流(視床VPL+S1)まで。
- 相反抑制的な結合や、体部位間のトポロジカル結合(隣接部位が互いに影響)。今は独立集約。
- 「手のひらか甲か」の区別は重心の3次元で**粗く**しか出ない（部位内の連続位置なので
  向きの情報は残るが、明示的な面の区別はしていない）。
- 温度・痛覚・かゆみ（触覚以外の体性感覚）。MIMo に無い。
"""
import numpy as np
import torch
import torch.nn as nn


# 体部位グループの定義。MIMoのbody名から意味のあるグループへ集約する。
# 意図：手指のような多数の細分geomが「1つの手」として脳に届く前に、まず指ごとにまとめる。
# 内側→外側の順は問わない(統合層で学習で決まる)。
# [Tier3・簡略化] グループ分けの粒度は解剖学的に厳密でなく、体部位の意味的な塊で切っている。
#
# 3つめの要素は**皮質の割り当てカテゴリ**。part_weight の初期値に使う。
#   [Tier1] Saadon-Grosman et al. 2020：上肢46.3 / 唇28.9 / 体幹15.8 / 下肢9.1（％）
#   注意：太郎の身体（MIMo）に「唇」という独立した部位は無いので head に含めた。
#     ＝head の重みは「顔＋唇」の合計として扱われる。人間より頭が重くなる方向の近似。
_BODY_GROUPS = [
    # 頭（唇の割り当てもここに含める。MIMoに唇のbodyが無いため）
    ("head", ["head"], "head"),
    # 眼(触覚上は目のセンサはほぼ無いか特殊なので個別扱い)
    ("eyes", ["left_eye", "right_eye"], "head"),
    # 体幹
    ("chest", ["chest", "upper_body"], "trunk"),
    ("hip", ["hip", "lower_body"], "trunk"),
    # 腕
    ("right_upper_arm", ["right_upper_arm"], "upper_limb"),
    ("left_upper_arm", ["left_upper_arm"], "upper_limb"),
    ("right_lower_arm", ["right_lower_arm"], "upper_limb"),
    ("left_lower_arm", ["left_lower_arm"], "upper_limb"),
    # 手のひら(手全体を1つのグループにまとめて指の各節と対等に扱う)
    ("right_palm", ["right_hand"], "upper_limb"),
    ("left_palm", ["left_hand"], "upper_limb"),
    # 右指5本(親指含む)を各指1グループに集約
    # 注意：【2026-07-31 修正】薬指(rf)と小指(lf)の指節が**入れ違っていた**。
    #   誤：right_rf = [lfmetacarpal, lfknuckle, lfmiddle, rfdistal]  ← 小指3節＋薬指の先
    #       right_lf = [rfknuckle, rfmiddle, lfdistal]              ← 薬指2節＋小指の先
    #   ⇒ 各グループが「1本の指」になっておらず、2本の指の断片が混ざっていた。
    #     実測でも left_rf 376点 / left_ff 53点 と7倍の偏りが出ていた。
    #   MIMo の実際の親子関係（mujoco の body_parentid で確認）：
    #     hand → rfknuckle → rfmiddle → rfdistal                    （薬指＝3節）
    #     hand → lfmetacarpal → lfknuckle → lfmiddle → lfdistal      （小指＝4節）
    ("right_thumb", ["right_thbase", "right_thhub", "right_thdistal"], "upper_limb"),
    ("right_ff", ["right_ffknuckle", "right_ffmiddle", "right_ffdistal"], "upper_limb"),
    ("right_mf", ["right_mfknuckle", "right_mfmiddle", "right_mfdistal"], "upper_limb"),
    ("right_rf", ["right_rfknuckle", "right_rfmiddle", "right_rfdistal"], "upper_limb"),
    ("right_lf", ["right_lfmetacarpal", "right_lfknuckle", "right_lfmiddle",
                  "right_lfdistal"], "upper_limb"),
    # 左指5本
    ("left_thumb", ["left_thbase", "left_thhub", "left_thdistal"], "upper_limb"),
    ("left_ff", ["left_ffknuckle", "left_ffmiddle", "left_ffdistal"], "upper_limb"),
    ("left_mf", ["left_mfknuckle", "left_mfmiddle", "left_mfdistal"], "upper_limb"),
    ("left_rf", ["left_rfknuckle", "left_rfmiddle", "left_rfdistal"], "upper_limb"),
    ("left_lf", ["left_lfmetacarpal", "left_lfknuckle", "left_lfmiddle",
                 "left_lfdistal"], "upper_limb"),
    # 脚
    ("right_upper_leg", ["right_upper_leg"], "lower_limb"),
    ("left_upper_leg", ["left_upper_leg"], "lower_limb"),
    ("right_lower_leg", ["right_lower_leg"], "lower_limb"),
    ("left_lower_leg", ["left_lower_leg"], "lower_limb"),
    # 足
    ("right_foot", ["right_foot", "right_toes", "right_big_toe"], "lower_limb"),
    ("left_foot", ["left_foot", "left_toes", "left_big_toe"], "lower_limb"),
]

# 一次体性感覚野の割り当て（％）。[Tier1] Saadon-Grosman, Loewenstein, Arzy 2020
#   *Hum Brain Mapp* 41(13):3620-3646
# 注意：唇28.9％は head に合算している（MIMoに唇のbodyが無い）。
_CORTEX_SHARE = {
    "head": 28.9 + 0.0,     # 唇の割り当て。顔そのものの値は原論文に別掲が無いのでここに含める
    "trunk": 15.8,
    "upper_limb": 46.3,
    "lower_limb": 9.1,
    "other": 1.0,           # グループに載らなかった部位の受け皿。小さくしておく
}

# 部位ごとの出力は「有無・強さ・重心xyz」の5つ。センサ点数によらず固定。
FEATURES_PER_PART = 5


class TouchMap:
    """触覚flat配列を「点→部位」に対応づける地図。体を作り直すたびに作り変える。

    Attributes:
        group_names: list[str]  部位グループ名（G個）
        part_of_point: (N,) int64  各センサ点がどのグループか
        positions: (N, 3) float32  各センサ点の位置。**グループごとに中心を引いて
            最大半径で割った正規化座標**（おおむね [-1,1]）。重心の計算に使う。
        counts: (G,) float32  グループごとの点数
        n_points: int  センサ点の総数 N
        total_dim: int  触覚flat配列の長さ（= N * 3）
    """

    def __init__(self, group_names, part_of_point, positions, counts):
        self.group_names = group_names
        self.part_of_point = part_of_point
        self.positions = positions
        self.counts = counts
        self.n_points = int(part_of_point.shape[0])
        self.total_dim = self.n_points * 3


def build_touch_map(model, data, touch):
    """MIMoの環境から TouchMap を作る。

    Args:
        model: mujoco model
        data: mujoco data（**初期姿勢で mj_forward 済みのもの**）。
            グループが複数のbodyにまたがるとき（指の節・足と足指）、
            body ローカル座標のままでは原点が違って重心が意味を持たない。
            初期姿勢のグローバル座標に直してから正規化する。
        touch: env.unwrapped.touch (mimoTouch.Touch)

    Returns:
        TouchMap

    注意：【2026-07-31】`touch.sensor_positions` のキーは**触覚クラスによって意味が違う**。
        DiscreteTouch  … キーは geom_id
        TrimeshTouch   … キーは body_id   ← MIMo v2 の既定はこちら
      キーを geom_id と決め打ちして `model.geom_bodyid[key]` で引くと
      **まったく別の部位に点数が割り当てられる**（落とし穴 項85）。
    注意：flat配列の並びは `flatten_sensor_dict` が決めており、**sorted(touch.meshes)** 順。
      `sensor_positions` のキーを sorted しても同じになるはずだが、
      並びの決定権は meshes 側にあるので**そちらを優先して読む**。
    """
    keys_are_body = type(touch).__name__ == "TrimeshTouch"
    order = sorted(getattr(touch, "meshes", None) or touch.sensor_positions.keys())

    # ---- 1. flat配列の順に、各点の (body_id, ローカル位置) を並べる ----------
    body_of_point = []
    local_pos = []
    for key in order:
        pts = np.asarray(touch.sensor_positions[key], dtype=np.float64)
        bid = int(key) if keys_are_body else int(model.geom_bodyid[key])
        body_of_point.extend([bid] * pts.shape[0])
        local_pos.append(pts)
    body_of_point = np.asarray(body_of_point, dtype=np.int64)
    local_pos = np.concatenate(local_pos, axis=0) if local_pos else np.zeros((0, 3))
    n_points = local_pos.shape[0]

    # ---- 2. 初期姿勢のグローバル座標に直す -----------------------------------
    #   world = xpos[body] + xmat[body] @ local
    #   注意：関節が動けば実際の相対位置は変わる。ここで固定するのは
    #     「グループ内での点の並び」だけなので、同一部位の細分（指の節など）では
    #     ずれは小さい。[Tier3・工学的近似]
    world_pos = np.zeros((n_points, 3), dtype=np.float64)
    for bid in np.unique(body_of_point):
        sel = body_of_point == bid
        R = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
        world_pos[sel] = local_pos[sel] @ R.T + np.asarray(data.xpos[bid], dtype=np.float64)

    # ---- 3. body名 → グループ番号 の対応を作る --------------------------------
    name_to_group = {}
    group_names = []
    group_category = []
    for gi, (gname, body_names, category) in enumerate(_BODY_GROUPS):
        group_names.append(gname)
        group_category.append(category)
        for bn in body_names:
            name_to_group[bn] = gi
    other_index = None

    part_of_point = np.full(n_points, -1, dtype=np.int64)
    for bid in np.unique(body_of_point):
        bname = model.body(int(bid)).name
        gi = name_to_group.get(bname)
        if gi is None:
            if other_index is None:
                other_index = len(group_names)
                group_names.append("other")
                group_category.append("other")
            gi = other_index
        part_of_point[body_of_point == bid] = gi

    # 実際に点を持たないグループは落とす（番号を詰め直す）
    used = [gi for gi in range(len(group_names)) if np.any(part_of_point == gi)]
    remap = {old: new for new, old in enumerate(used)}
    part_of_point = np.asarray([remap[g] for g in part_of_point], dtype=np.int64)
    kept_names = [group_names[g] for g in used]
    kept_category = [group_category[g] for g in used]
    n_groups = len(kept_names)

    # ---- 4. グループごとに位置を正規化（中心を引いて最大半径で割る）-----------
    positions = np.zeros((n_points, 3), dtype=np.float32)
    counts = np.zeros(n_groups, dtype=np.float32)
    for gi in range(n_groups):
        sel = part_of_point == gi
        p = world_pos[sel]
        counts[gi] = float(sel.sum())
        c = p.mean(axis=0)
        d = p - c
        r = float(np.abs(d).max())
        positions[sel] = (d / r if r > 1e-9 else d).astype(np.float32)

    tm = TouchMap(kept_names, part_of_point, positions, counts)
    tm.category = kept_category
    return tm


def build_touch_map_from_env(env):
    """gym の環境から TouchMap を作る。呼び出し側の定型処理をまとめただけ。

    注意：`data.xpos` は前向き計算をしないと更新されない。ここで `mj_forward` を
      1回だけ打つ。qpos/qvel は変えない（派生量を計算するだけ）ので、
      **学習の乱数列にも状態にも影響しない**。
    """
    import mujoco
    u = env.unwrapped
    mujoco.mj_forward(u.model, u.data)
    return build_touch_map(u.model, u.data, u.touch)


def _initial_part_weights(categories):
    """皮質の割り当て（％）を部位グループへ配る。平均1になるよう正規化して返す。

    同じカテゴリの中は**部位数で等分**する。[Tier3・簡略化]
      人間では上肢の中でも指先の割り当てが特に大きいが、部位ごとの内訳の定量値が
      手に入らなかった。等分にしておき、`part_weight` を学習可能にして脳に決めさせる。
    """
    from collections import Counter
    n_in_cat = Counter(categories)
    w = np.asarray([_CORTEX_SHARE.get(c, 1.0) / n_in_cat[c] for c in categories],
                   dtype=np.float32)
    return w / max(float(w.mean()), 1e-9)


class SomatosensoryCortex(nn.Module):
    """視床VPL核（部位別集約） + 一次体性感覚野S1（部位間統合）。

    入力：MIMoの触覚flat配列(1次元、長さ = センサ点数×3)。
    出力：触覚 embedding(embedding_dim 次元、既定64)。

    **センサ点数に依存する重みを1つも持たない。** 体が育ったら `rebuild()` で
    地図だけ差し替える（学習した重みはそのまま使える）。
    """

    def __init__(self, touch_map, embedding_dim=64):
        """
        Args:
            touch_map: build_touch_map の戻り値
            embedding_dim: S1の統合出力次元。fusion側の他感覚に揃える(既定64)。
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self._install_map(touch_map)
        self.n_groups = len(touch_map.group_names)

        # 部位の重み（視床→S1の投射の太さ＝皮質の割り当てに相当）。
        # 初期値は Saadon-Grosman 2020 の皮質割り当て。学習で動く。
        w = _initial_part_weights(getattr(touch_map, "category",
                                          ["other"] * self.n_groups))
        self.part_weight = nn.Parameter(torch.as_tensor(w, dtype=torch.float32))

        # 触覚の値の大きさは環境しだいなので、tanh に入る前の倍率を学習可能にしておく。
        self.presence_gain = nn.Parameter(torch.tensor(1.0))
        self.strength_gain = nn.Parameter(torch.tensor(1.0))

        # 部位間統合層(S1相当)。部位×5 → embedding_dim。
        self.integrate = nn.Linear(self.n_groups * FEATURES_PER_PART, embedding_dim)

    # ------------------------------------------------------------ 地図の差し替え
    def _install_map(self, touch_map):
        """地図（点→部位の対応と位置）をバッファとして持つ。学習しない値。"""
        self.group_names = list(touch_map.group_names)
        self.n_points = touch_map.n_points
        self.total_dim = touch_map.total_dim
        self.register_buffer("_part_of_point",
                             torch.as_tensor(touch_map.part_of_point, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_positions",
                             torch.as_tensor(touch_map.positions, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_counts",
                             torch.as_tensor(touch_map.counts, dtype=torch.float32),
                             persistent=False)

    def rebuild(self, touch_map):
        """体を作り直したときに地図だけ差し替える。**学習した重みは保つ。**

        部位グループの並びが変わると、学習した `part_weight` と `integrate` の
        対応が崩れる。そのときは黙って続けず例外で止める
        （落とし穴 項86「エラーが出ずに動いた、を動いたと読まない」）。
        """
        old = list(self.group_names)
        new = list(touch_map.group_names)
        if old != new:
            raise AssertionError(
                "体を作り直したら触覚の部位グループが変わった。\n"
                f"  前: {old}\n  後: {new}\n"
                "  学習した部位の重みが別の部位に対応してしまうので止めた。")
        dev = self.part_weight.device
        self._install_map(touch_map)
        self._part_of_point = self._part_of_point.to(dev)
        self._positions = self._positions.to(dev)
        self._counts = self._counts.to(dev)
        return self

    # ------------------------------------------------------------ 前向き計算
    def part_features(self, touch_flat):
        """部位ごとの要約を返す。(..., 部位数, 5) ＝ 有無・強さ・重心x・重心y・重心z。

        統合層に入る**手前**の値。中身を見たいとき（検査・可視化・デバッグ）は
        forward ではなくこれを呼ぶ。**部位の重みはまだ掛けていない。**
        """
        if touch_flat.shape[-1] != self.total_dim:
            raise AssertionError(
                f"触覚の次元が地図と合わない：入力{touch_flat.shape[-1]} / "
                f"地図{self.total_dim}。体を作り直したなら rebuild() を呼ぶ必要がある")
        lead = touch_flat.shape[:-1]
        G = len(self.group_names)

        # 各センサ点の力ベクトル(3成分)の大きさ
        f = touch_flat.reshape(*lead, self.n_points, 3)
        mag = torch.linalg.vector_norm(f, dim=-1)                 # (..., N)

        idx = self._part_of_point.expand(*lead, self.n_points)    # (..., N)
        zeros = torch.zeros(*lead, G, dtype=mag.dtype, device=mag.device)

        # ① 有無：部位内で最も強い点（どこかに触れたか）
        peak = zeros.scatter_reduce(-1, idx, mag, reduce="amax", include_self=True)
        presence = torch.tanh(peak * self.presence_gain)

        # ② 強さ：部位内の平均圧
        sum_mag = zeros.scatter_add(-1, idx, mag)                 # (..., G)
        strength = torch.tanh(sum_mag / self._counts * self.strength_gain)

        # ③ 重心：圧で重み付けした位置の平均（部位内のどこに触れたか）
        wp = (mag.unsqueeze(-1) * self._positions).transpose(-1, -2)   # (..., 3, N)
        idx3 = self._part_of_point.expand(*lead, 3, self.n_points)
        csum = torch.zeros(*lead, 3, G, dtype=mag.dtype,
                           device=mag.device).scatter_add(-1, idx3, wp)
        centroid = (csum / (sum_mag.unsqueeze(-2) + 1e-6)).transpose(-1, -2)  # (..., G, 3)

        return torch.cat([presence.unsqueeze(-1), strength.unsqueeze(-1), centroid],
                         dim=-1)                                   # (..., G, 5)

    def forward(self, touch_flat):
        """touch_flat: (..., n_points*3) の触覚flat配列。戻り値: (..., embedding_dim)。"""
        feat = self.part_features(touch_flat)
        feat = feat * self.part_weight.unsqueeze(-1)
        lead = feat.shape[:-2]
        return self.integrate(feat.reshape(*lead, feat.shape[-2] * FEATURES_PER_PART))

    # ------------------------------------------------------------ 表示
    def summary(self):
        """デバッグ用サマリー(初回起動時にprintすると設計が一目で確認できる)。"""
        total_params = sum(p.numel() for p in self.parameters())
        G = len(self.group_names)
        lines = [
            f"SomatosensoryCortex: {G}部位, 合計{total_params:,}パラメータ",
            f"  センサ点{self.n_points:,}点(flat {self.total_dim:,}次元)"
            f" → 部位ごとに有無/強さ/重心xyz の{FEATURES_PER_PART}つ"
            f" → {G * FEATURES_PER_PART}次元 → 統合{self.embedding_dim}次元",
            "  注意パラメータ数はセンサ点数によらない（体が育っても層の形は変わらない）",
        ]
        w = self.part_weight.detach().cpu().numpy()
        cnt = self._counts.detach().cpu().numpy()
        for name, n, wi in zip(self.group_names, cnt, w):
            lines.append(f"    {name:20s}: {int(n):5d}点  重み{wi:6.3f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 旧API（centroid化する前）との橋渡し。
#   注意：`build_sensor_layout` は 2026-07-31 に廃止した。
#     部位ごとに `nn.Linear(点数×3, dim)` を作る作りで、体を育てると
#     **静かに別の部位を読む**（落とし穴 項86）。呼び出し側は build_touch_map へ。
def build_sensor_layout(model, touch):
    raise NotImplementedError(
        "build_sensor_layout は 2026-07-31 に廃止。build_touch_map(model, data, touch) を使う。\n"
        "  理由：部位ごとの層がセンサ点数に依存していて、体を育てると\n"
        "        古いインデックス表のまま別の部位を読んでいた（落とし穴 項86）")
