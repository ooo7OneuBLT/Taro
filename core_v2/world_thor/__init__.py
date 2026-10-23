"""家の場面（AI2-THOR、2026-10-19、計画①「太郎の脳を家につなぐ」）。

目だけの場面（world_eye）の MuJoCo の机の代わりに、AI2-THOR 5.0.0 の家（台所 FloorPlan1〜・居間 FloorPlan201〜）の中に
赤ちゃんの目（床から 0.6m）を1つ置く。太郎の脳・目の入口・目を向ける反射は、目だけの場面とまったく同じ形でつなぐ
（env.step・obs["eye_left"]・info["parent_utterance"]・env._orienting.set_map_target）。

PC での下見（依頼13〜15）で分かったこと：
* Windows 用のビルドは無い。WSL2 の Ubuntu 22.04 の中で、窓あり（platform 指定なし＝Linux64、WSLg・GPU で描く）なら動く。
  画面なしの CloudRendering は WSL では Vulkan が無く動かない
* Controller には server_timeout=900（最初の起動が遅い）と makeAgentsVisible=False（体を映さない）が要る
* 0.6m の目は AddThirdPartyCamera（エージェントの 0.3m 前、下へ約10度）で作れる。1歩 約55ミリ秒（RotateRight 毎秒約18回）
* TeleportObject（forceKinematic）で物を目の前へ動かせる。PickupObject も通る

置き場所：
* env.py    … 環境本体（ThorEnv）。reset で座る場所を決め（place=(x, z, yaw) で手で、auto は控え）、親がおもちゃ（toys＝4）を太郎の前の床に並べる。ai2thor は使うときにだけ読む（クラウドのテストと、ほかの道具は ai2thor 無しで回る）
* fake.py   … ai2thor の代わりの小さな偽物（FakeController）。テストと、クラウドで配線を確かめるため
* parent.py … 見えない親の手（follow-in：太郎が1秒見た物を目の前へ持ってきて、振って、名前を言い、5秒で戻す。
              bring-in：5秒名前の物を見なければ親が物を持ってきて、太郎が見てから名前を言う）と、
              AI2-THOR の物の種類 → 子どもの語の表
* eye.py    … 目の入口の作り方（dinov2＝本物の目。seg＝クラウドの確かめ用の代わりの目、DINOv2 ではない）
* taro.py   … 家の場面の太郎（grow_names.EyeTaro と同じ1回の回し方）と、名づけで育つ太郎（ThorGrower）、見る時間の記録

道具：python -m core_v2.tools.thor_watch（動画）。--first-frame で最初の目の絵と上から見た絵だけ（座る場所を選ぶため）。watch_scene に --world thor を付けても同じ。
"""

from .env import ThorEnv  # noqa: F401
