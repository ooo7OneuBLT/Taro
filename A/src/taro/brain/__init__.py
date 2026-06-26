"""
太郎の脳 — 人間の脳の部品に対応した構造

cortex.py        大脳皮質（知覚・判断・運動指令）
cerebellum.py    小脳（運動スキル・順モデル・逆モデル）
basal_ganglia.py 大脳基底核（行動選択・強化学習）
hippocampus.py   海馬（エピソード記憶・将来用）
instincts/       本能（模倣・予測・ドーパミン・馴化・青斑核）
"""
from taro.brain.cortex import TaroBrain, Vocabulary
from taro.brain.cerebellum import Cerebellum
from taro.brain.speech_planner import SpeechPlanner
from taro.brain.basal_ganglia import TaroLearner
from taro.brain.hippocampus import Hippocampus
