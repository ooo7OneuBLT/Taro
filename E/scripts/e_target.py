"""【E1】予測対象（太郎が「次にこう感じるはず」と予測する中身）を作る。

【従来（目標C〜D）】予測対象は**固有感覚621次元のみ**だった：
    target = layer_norm(obs["observation"])
視覚は「自分が動くと視界全体が動く(egomotion)ので、他人の動きと区別できない」という理由で
予測対象から外していた（目標D）。

【E1で視覚を足す理由】
E1の問いは「**既存の本能(progress報酬)だけで、自分の手を見る行動が創発するか**」。
progress報酬は「**予測が上達したぶん**」なので、**予測対象に入っていないものは報酬を生まない**。
視覚が予測対象になければ、太郎が手を見ても一切得をしない＝創発しようがない。
そして**自分の手**が相手なら、Dで問題だったegomotionは邪魔者ではなく**学習の材料そのもの**になる
（自分の運動と完全に対応して動く唯一の視覚対象）。

【★2つの設計判断】
(1) **視覚エンコーダは凍結する（RND式）**
    予測する側と正解側が同じ学習中のエンコーダだと、「エンコーダが出力を平坦にすれば予測が当たる」
    という抜け道ができて学習が崩壊する。目標Cで実際に踏んだ罠（`fusion.freeze()`）。
    ここでは**別インスタンスの凍結エンコーダ**を正解側に使う。
(2) **固有感覚と視覚を別々に layer_norm してから連結する**
    全体を一度に layer_norm すると、621次元の固有感覚が平均・分散を支配し、64次元の視覚が
    埋もれる。D0で踏んだ「触覚が次元数に薄められる」罠・E1で接触が消えた罠と同じ構造
    （621次元では効果量 d=-0.005 だが腕の次元だけ見ると d=-0.22 だった）。
    → ブロックごとに正規化して、少なくとも**スケールでは薄まらない**ようにする。

【⚠️それでも残る「薄まり」】
MSEを取ると寄与は次元数比のまま（視覚 64/685 = 9.3%）。これを重み付けで補正するのは
**恣意的**なので今はしない。まず等重みで実装し、**視覚が予測誤差に信号として現れるか**を
測ってから判断する（接触のときと同じ「関門」の測り方）。
"""
import numpy as np
import torch
import torch.nn.functional as F

VISION_EMB_DIM = 64        # VisionEncoder の出力次元（fusion と同じ）


class PredictionTarget:
    """予測対象を作る係。固有感覚（＋任意で視覚）を、ブロックごとに正規化して連結する。

    Args:
        frozen_fusion: **凍結済み**の MinimalFusion（正解側専用。学習側と別インスタンス）。
        use_vision: 視覚を予測対象に入れるか。False なら従来と完全に同一の出力になる。
    """

    def __init__(self, frozen_fusion=None, use_vision=True):
        self.fusion = frozen_fusion
        self.use_vision = bool(use_vision and frozen_fusion is not None
                               and getattr(frozen_fusion, "vision", None) is not None)
        self._warned = False

    def dim(self, obs):
        return int(self(obs).shape[0])

    def describe(self):
        return ("固有感覚621 + 視覚64（凍結・別正規化）" if self.use_vision
                else "固有感覚621のみ（従来と同一）")

    def __call__(self, obs):
        v = torch.as_tensor(np.asarray(obs["observation"]), dtype=torch.float32)
        prop = F.layer_norm(v, v.shape)              # ★従来の ln_prop と完全に同じ処理
        if not self.use_vision:
            return prop.detach()
        if "eye_left" not in obs or "eye_right" not in obs:
            if not self._warned:
                print("⚠️ obsに視覚が無いので予測対象は固有感覚のみになります")
                self._warned = True
            return prop.detach()
        with torch.no_grad():                        # 正解側は勾配を流さない（RND式）
            e = self.fusion.vision(obs["eye_left"], obs["eye_right"])
        vis = F.layer_norm(e, e.shape)               # ★視覚は別に正規化＝スケールで埋もれない
        return torch.cat([prop, vis], dim=-1).detach()


def make_frozen_fusion(like_fusion, touch_dim=0, vision_res=0):
    """正解側専用の**凍結した**融合層を作る（学習側とは別インスタンス・別の初期重み）。

    ⚠️同じ重みをコピーしてはいけない。RNDの要点は「正解側が学習に引きずられないこと」で、
    別インスタンスかつ requires_grad=False であればよい。初期重みが違っても問題ない
    （むしろ予測側が正解側を"当てにいく"対象として機能する）。
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, "taro_core"))
    from fusion import MinimalFusion
    f = MinimalFusion(touch_dim=touch_dim, vision_res=vision_res)
    return f.freeze()
