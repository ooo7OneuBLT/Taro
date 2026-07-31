"""体性感覚系(触覚)の脳内経路：視床VPL核（部位別集約） → 一次体性感覚野S1（部位間統合）。

【なぜこのファイルか、2026-07-24】従来の TouchEncoder は 12,000次元近い生の触覚を1枚の
巨大変換層に流し込んでいた。これは解剖学的にズレている：人間の触覚は
「皮膚→末梢神経→脊髄→視床VPL核→S1」と伝わる過程で、**部位ごとに整理されたマップ**を
経由する。VPL核は部位別受容野を持ち(顔・手・体幹・脚・足が内側から外側へ配置)、S1は
それを統合したホムンクルス表現を持つ(手や顔の領域は大きい=皮質拡大)。

【文献根拠】
- 視床VPL核の体部位マップ [Tier1]：Padberg et al. 2009 *Cereb Cortex*；
  Kaas, Nelson, Sur, Dykes, Merzenich 1984 *J Comp Neurol* 226；
  Leplus et al. 2024 *Ann Clin Transl Neurol* (脳深部刺激の臨床)
- S1のホムンクルス表現・皮質拡大 [Tier1]：教科書レベル
- S1・S2の階層構造 [Tier1]：Wikipedia "Secondary somatosensory cortex"

【設計】
- 入力：MIMoの触覚センサをsorted(geom_id)順に連結した1次元配列(既存のflatten_sensor_dictの
  出力そのもの)。長さは触覚パラメータで変わる(乳児acuity 2倍粗く化なら約2000次元)。
- ①部位別集約層(VPL核相当)：意味のある体部位でグループ化し、部位ごとに小さい変換層で
  集約。各部位の出力次元は max(4, min(24, int(√センサ点数))) で、大きい部位ほど大きい表現
  (皮質拡大の粗い近似)。
- ②部位間統合層(S1相当)：全部位の集約結果を連結し、1つの変換層で最終的な触覚embed(64次元)へ。

【何がやっていないか(将来課題)】
- S1→S2の高次階層(S2は"予測が的中する場所"、Distributed functions of detection and
  discrimination 2015)。今は最上流(視床VPL+S1)まで。予測は e_growth_train.py の ln_prop で
  この embed を予測対象にする形で対応(=BのRND式)。
- 相反抑制的な結合や、体部位間のトポロジカル結合(隣接部位が互いに影響)。今は独立集約。
- 発達スケジュール(月齢で触覚acuityが変わる)。今は固定。
"""
import math
import numpy as np
import torch
import torch.nn as nn


# 体部位グループの定義。MIMoのbody名から意味のあるグループへ集約する。
# 意図：手指のような多数の細分geomが「1つの手」として脳に届く前に、まず指ごとにまとめる。
# 内側→外側の順は問わない(統合層で学習で決まる)。
# [Tier3・簡略化] グループ分けの粒度は解剖学的に厳密でなく、体部位の意味的な塊で切っている。
_BODY_GROUPS = [
    # 頭
    ("head", ["head"]),
    # 眼(触覚上は目のセンサはほぼ無いか特殊なので個別扱い)
    ("eyes", ["left_eye", "right_eye"]),
    # 体幹
    ("chest", ["chest", "upper_body"]),
    ("hip", ["hip", "lower_body"]),
    # 腕
    ("right_upper_arm", ["right_upper_arm"]),
    ("left_upper_arm", ["left_upper_arm"]),
    ("right_lower_arm", ["right_lower_arm"]),
    ("left_lower_arm", ["left_lower_arm"]),
    # 手のひら(手全体を1つのグループにまとめて指の各節と対等に扱う)
    ("right_palm", ["right_hand"]),
    ("left_palm", ["left_hand"]),
    # 右指5本(親指含む)を各指1グループに集約
    # 注意：【2026-07-31 修正】薬指(rf)と小指(lf)の指節が**入れ違っていた**。
    #   誤：right_rf = [lfmetacarpal, lfknuckle, lfmiddle, rfdistal]  ← 小指3節＋薬指の先
    #       right_lf = [rfknuckle, rfmiddle, lfdistal]              ← 薬指2節＋小指の先
    #   ⇒ 各グループが「1本の指」になっておらず、2本の指の断片が混ざっていた。
    #     実測でも left_rf 376点 / left_ff 53点 と7倍の偏りが出ていた。
    #   MIMo の実際の親子関係（mujoco の body_parentid で確認）：
    #     hand → rfknuckle → rfmiddle → rfdistal                    （薬指＝3節）
    #     hand → lfmetacarpal → lfknuckle → lfmiddle → lfdistal      （小指＝4節）
    ("right_thumb", ["right_thbase", "right_thhub", "right_thdistal"]),
    ("right_ff", ["right_ffknuckle", "right_ffmiddle", "right_ffdistal"]),
    ("right_mf", ["right_mfknuckle", "right_mfmiddle", "right_mfdistal"]),
    ("right_rf", ["right_rfknuckle", "right_rfmiddle", "right_rfdistal"]),
    ("right_lf", ["right_lfmetacarpal", "right_lfknuckle", "right_lfmiddle", "right_lfdistal"]),
    # 左指5本
    ("left_thumb", ["left_thbase", "left_thhub", "left_thdistal"]),
    ("left_ff", ["left_ffknuckle", "left_ffmiddle", "left_ffdistal"]),
    ("left_mf", ["left_mfknuckle", "left_mfmiddle", "left_mfdistal"]),
    ("left_rf", ["left_rfknuckle", "left_rfmiddle", "left_rfdistal"]),
    ("left_lf", ["left_lfmetacarpal", "left_lfknuckle", "left_lfmiddle", "left_lfdistal"]),
    # 脚
    ("right_upper_leg", ["right_upper_leg"]),
    ("left_upper_leg", ["left_upper_leg"]),
    ("right_lower_leg", ["right_lower_leg"]),
    ("left_lower_leg", ["left_lower_leg"]),
    # 足
    ("right_foot", ["right_foot", "right_toes", "right_big_toe"]),
    ("left_foot", ["left_foot", "left_toes", "left_big_toe"]),
]


def build_sensor_layout(model, touch):
    """MIMoの環境から、部位グループごとの (グループ名, flat配列上のインデックス列) を作る。

    Args:
        model: mujoco model
        touch: env.unwrapped.touch (mimoTouch.Touch)

    Returns:
        layout: list of (group_name, indices_ndarray, n_points)
          indices_ndarray は sorted(sensor_positions) 順に並んだ flat 配列上のインデックス
          (各センサ点は3成分なので、indicesは3個ずつ連続)
        total_dim: flat配列の全長(= sum(n_points) * 3)、検証用
    """
    # sorted(geom_id) 順に flat配列が並ぶ(mimoTouch.flatten_sensor_dictと同じ規則)
    # 注意：【2026-07-31 修正】`touch.sensor_positions` のキーは
    #   **触覚クラスによって意味が違う**。
    #     DiscreteTouch  … キーは geom_id
    #     TrimeshTouch   … キーは body_id   ← MIMo v2 の既定はこちら
    #   従来はキーを geom_id と決め打ちして `model.geom_bodyid[key]` で body を引いていた。
    #   ⇒ TrimeshTouch では**まったく別の部位に点数が割り当てられる**。
    #     実測では「左右で点数が桁違い（rfdistal 右96 vs 左412）」「脚にセンサーが無い」
    #     という**存在しない現象**が観測されていた（実際は左右対称・脚にも346点ある）。
    #   ⇒ キーが body_id かどうかを実際に確かめてから引く。
    keys_sorted = sorted(touch.sensor_positions.keys())
    # flat配列上の各キーの開始位置と点数を計算
    geom_offset = {}
    offset = 0
    for k in keys_sorted:
        n_pts = touch.sensor_positions[k].shape[0]
        geom_offset[k] = (offset, offset + n_pts * 3, n_pts)
        offset += n_pts * 3
    total_dim = offset
    # キーが body_id を指しているか（TrimeshTouch）を判定する。
    #   body_id なら `model.body(k)` が引けて、かつ geom_bodyid 経由と食い違う。
    keys_are_body = type(touch).__name__ == "TrimeshTouch"
    geom_ids_sorted = keys_sorted

    # body_name → キー のマップ
    #   TrimeshTouch はキーが body_id なので**そのまま**引く。
    #     DiscreteTouch はキーが geom_id なので geom_bodyid 経由で引く。
    name_to_geoms = {}
    for gid in geom_ids_sorted:
        body_id = int(gid) if keys_are_body else int(model.geom_bodyid[gid])
        name = model.body(body_id).name
        name_to_geoms.setdefault(name, []).append(gid)

    layout = []
    matched_names = set()
    for group_name, body_names in _BODY_GROUPS:
        indices = []
        n_points_group = 0
        for bn in body_names:
            for gid in name_to_geoms.get(bn, []):
                s, e, n_pts = geom_offset[gid]
                indices.extend(range(s, e))
                n_points_group += n_pts
                matched_names.add(bn)
        if n_points_group > 0:
            layout.append((group_name, np.array(indices, dtype=np.int64), n_points_group))

    # マッチしなかったbody名(想定外の部位)は "other" グループにまとめて拾う
    unmatched_indices = []
    unmatched_points = 0
    for name, gids in name_to_geoms.items():
        if name in matched_names:
            continue
        for gid in gids:
            s, e, n_pts = geom_offset[gid]
            unmatched_indices.extend(range(s, e))
            unmatched_points += n_pts
    if unmatched_points > 0:
        layout.append(("other", np.array(unmatched_indices, dtype=np.int64), unmatched_points))

    return layout, total_dim


class SomatosensoryCortex(nn.Module):
    """視床VPL核（部位別集約） + 一次体性感覚野S1（部位間統合）。

    入力：MIMoの触覚flat配列(1次元)。
    出力：触覚 embedding(embedding_dim 次元、既定64)。
    """

    def __init__(self, layout, embedding_dim=64):
        """
        Args:
            layout: build_sensor_layout の戻り値の1つめ。
                    list of (group_name, indices_ndarray, n_points)
            embedding_dim: S1の統合出力次元。fusion側の他感覚に揃える(既定64)。
        """
        super().__init__()
        self.layout = layout
        self.group_names = [g[0] for g in layout]
        # 各部位グループの出力次元 = max(4, min(24, int(√センサ点数)))
        # 大きい部位ほど大きい表現(皮質拡大の粗い近似)。[Tier3・簡略化・単調近似]
        self.part_dims = [max(4, min(24, int(math.sqrt(g[2])))) for g in layout]

        # 部位別集約層(視床VPL核相当)。入力=そのグループの センサ点数×3成分、出力=part_dim。
        self.part_encoders = nn.ModuleList([
            nn.Linear(g[2] * 3, dim) for g, dim in zip(layout, self.part_dims)
        ])
        # 部位別インデックスをテンソル化してバッファ登録(GPU移動時も追従)
        for i, (name, idx, _n) in enumerate(layout):
            self.register_buffer(f"_idx_{i}", torch.as_tensor(idx, dtype=torch.long),
                                 persistent=False)

        # 部位間統合層(S1相当)。全部位のembedを連結→embedding_dim。
        self.total_part_dim = sum(self.part_dims)
        self.integrate = nn.Linear(self.total_part_dim, embedding_dim)

    def forward(self, touch_flat):
        """touch_flat: (..., total_dim) の触覚flat配列。戻り値: (..., embedding_dim)。"""
        parts = []
        for i, enc in enumerate(self.part_encoders):
            idx = getattr(self, f"_idx_{i}")
            part_in = touch_flat.index_select(-1, idx)
            parts.append(torch.relu(enc(part_in)))
        cat = torch.cat(parts, dim=-1)
        return self.integrate(cat)

    def summary(self):
        """デバッグ用サマリー(初回起動時にprintすると設計が一目で確認できる)。"""
        total_params = sum(p.numel() for p in self.parameters())
        lines = [f"SomatosensoryCortex: {len(self.layout)}部位, 合計{total_params:,}パラメータ",
                 f"  部位別集約→統合({self.total_part_dim}→{self.integrate.out_features})"]
        for (name, _idx, n_pts), dim in zip(self.layout, self.part_dims):
            lines.append(f"    {name:20s}: {n_pts:4d}点×3成分 → {dim}次元")
        return "\n".join(lines)
