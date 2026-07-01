"""太郎の本能 — 生まれつき備わっている行動のもと"""
from taro.brain.instincts.imitation import compute_imitation_reward, compute_alignment_credit
from taro.brain.instincts.prediction import compute_prediction_reward
from taro.brain.instincts.dopamine import Dopamine
from taro.brain.instincts.habituation import Habituation
from taro.brain.instincts.locus_coeruleus import LocusCoeruleus
from taro.brain.instincts.reward import compute_total_reward
from taro.brain.instincts.homeostasis import Homeostasis
from taro.brain.instincts.critic import Critic
