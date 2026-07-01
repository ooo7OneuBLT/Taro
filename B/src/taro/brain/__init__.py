"""
太郎の脳 — 人間の脳の部品に対応した構造

外部コードは `from taro.brain import X` だけで全部品にアクセスできる。
ファイルの移動・リネーム時はこの__init__.pyだけ変更すればよい。
"""
from taro.brain.cortex import TaroBrain, Vocabulary
from taro.brain.cerebellum import Cerebellum
from taro.brain.left_frontal_lobe import BrocasArea
from taro.brain.basal_ganglia import TaroLearner
from taro.brain.hippocampus import Hippocampus
from taro.brain.insula import Insula
from taro.brain.instincts import (compute_imitation_reward, compute_prediction_reward,
                                   Dopamine, Habituation, LocusCoeruleus,
                                   compute_total_reward, Homeostasis, Critic)
