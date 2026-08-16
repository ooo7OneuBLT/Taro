# taro_setup.py 取扱説明書（AI用）

本体: C:\claude\AI\Taro\run\taro_setup.py（959行）
最終同期: 2026-08-14 / 本体の行数: 959

## このファイルは何をするか
`run/config.py` の Config と、構築済みの env（環境）を受け取り、太郎の中身
（脳・融合層・学習器・神経調節・小脳・海馬・触覚順応・目標指向探索の周辺装置）一式を
組み立てて `Taro` インスタンスとして返す。測り方も環境の作り方もここには入れない
（測定は`run/plugins/`、環境構築は`run/plugins/common/scene.py`の仕事）。

## 主要なクラス・関数
| 名前 | 行 | 何をするか |
|---|---|---|
| `_reflex_common_joint_indices` | 118-161 | 四肢すべての関節を、行動配列のindexで部位別（右腕/左腕/右脚/左脚）に返す |
| `_reflex_common_groups` | 164-198 | `common_drive_grouping`設定から{group名:[関節index,...]}を作る |
| `_setup_reflex_common` | 201-236 | reflex_common駆動モードの配線本体。`Taro.__init__`と`on_body_change`の両方から呼ばれる |
| `_DoubleTouchBonusContributor`（クラス） | 239-256 | ダブルタッチがhitしたら固定ボーナスを返す、reward_contributorsの1要素 |
| `_MouthTouchBonusContributor`（クラス） | 259-307 | 口元への自己接触の立ち上がり検出で1回だけ報酬を返す、reward_contributorsの1要素 |
| `_MouthTouchBonusContributor.reset` | 289-291 | エピソード境界で立ち上がり検出の前tick状態を初期化する |
| `_MouthTouchBonusContributor.compute` | 293-307 | 口元presence・接触presenceのしきい値判定と立ち上がり検出をして報酬を返す |
| `Taro`（クラス） | 310-959 | 太郎そのもの（脳・学習器・神経調節・小脳・海馬）。envは保持しない |
| `Taro.__init__` | 325-578 | 融合層→行動次元→reset→予測対象次元→脳→CPG/reflex_common→学習器・神経調節→小脳→続きから読み込み→努力コスト重み→reach_goal/double_touchの一式を、乱数消費順を厳守して組み立てる |
| `Taro._load` | 581-632 | 保存済みモデルを、形が合う層だけ読み込む（続きから学習） |
| `Taro.on_body_change` | 635-697 | 体を作り直したとき、触覚地図・reflex_common・触覚順応・腕indexを新しい体のものに差し替える |
| `Taro.apply_touch_adaptation` | 700-727 | 触覚の順応を1回進め、obs["touch_percept"]を追加した新しいdictを返す |
| `Taro.encode_target` | 730-789 | 予測誤差の「正解」を作る（固有感覚+任意で触覚/視覚/前庭をブロックごとlayer_normして連結） |
| `Taro.encode_reach_goal` | 792-827 | 手先位置の目標表現g（腕7次元＋自己接触15次元）を作る（案C・Goal Babbling段階1） |
| `Taro.dummy_reach_goal` | 829-851 | reach_goalの陰性対照（もっともらしい値域だが意味的相関のないランダム目標）を作る |
| `Taro.block_pe` | 853-878 | 予測誤差をブロックごとに平均してから足す（次元数の希釈を防ぐ） |
| `Taro.infer_latent` | 881-883 | [感覚,前回行動]→内部表現zを推論する（brainへの委譲） |
| `Taro.act_mean` | 885-888 | 決定的な行動平均を返す（評価・agency測定用） |
| `Taro.motor_drive` | 890-894 | 行動の平均と揺らぎを返す（運動野＋小脳のブレンド） |
| `Taro.infer_goal_action` | 896-910 | 凍結した順モデル(nat_head)を反転し、目標感覚gに届く行動を推論する（legacy、固有感覚全体が対象） |
| `Taro.infer_reach_goal_action` | 912-931 | 凍結したreach_headを反転し、目標reach_goal gに届く行動を推論する（案C版） |
| `Taro.init_state` | 933-936 | 学習ループが持つ「いまの状態」の初期値を返す |
| `Taro.save` | 938-959 | 自己モデル一式と条件（config）をまとめて保存する |

## 触るときの注意
- `Taro.__init__`は乱数消費順を厳守する必要がある（ファイル冒頭コメント：env作成→fusion→
  target_fusion→env.reset(seed)→予測対象次元を測る→brain→脊髄CPG→learner→神経調節→
  小脳→モデル読み込み）。順序を1つ変えると同じシードでも別の初期値になり、過去の実験と
  比較できなくなる（落とし穴チェックリスト項3）。
- `reach_space`（goal_babbling and goal_space=="reach_self"）・`double_touch_bonus`・
  `mouth_touch_bonus`関連のブロック（489-578行）はいずれも「既定OFFなら一度も実行されず
  1ビットも既存挙動を変えない」設計。ここを触るときは既定条件のガードを崩さないこと。
- `on_body_change`（635-697行）は、体が成長で変わったときに触覚地図などを差し替える。
  差し替えを忘れても例外は出ず、配列が長くなる方向の変化では古いインデックスが範囲内に
  収まり静かに別の部位を読む（落とし穴チェックリスト項86）。新しい「体が変わったら
  引き直すべき状態」を追加するときは、このメソッドに登録を忘れないこと。
- `encode_target`（730-789行）は「凍結した別インスタンス（target_fusion、RND式）」と
  「固有感覚と視覚を別々にlayer_normしてから連結」という2つの設計判断を守っている。
  ここを崩すと過去に3回踏んだ罠（次元数の多い層が支配して少ない層が埋もれる）を再現する。
- `block_pe`（853-878行）は`self.blocks`が1個以下なら`mse(pred, target)`に完全一致する
  特例を持つ。`touch_embed`のような単一ブロックを狙ってスライスしたいときにこの特例に
  落ちて「全身まるごとの誤差」を返してしまう罠が実際にあった（`run/trainer.py`の
  self_touch_interest_bonus実装コメント862-868行参照）。

## よくある変更の場所
- 脳の構成要素（融合層・学習器・神経調節・小脳など）を追加/変更したいなら
  `Taro.__init__`の①〜⑨の節番号コメント（393-578行）を目印に、対応する節へ追記する。
- 続きから学習するときの読み込み対象を増やしたいなら `Taro._load`（581-632行）に
  同じ「保存されていなければ警告して既定値のまま」というパターンで追加する。
- 体が成長する実験で新しく引き直すべき状態があるなら `Taro.on_body_change`（635-697行）。
- 報酬に新しい寄与（接触ボーナス等）を足したいなら、`_DoubleTouchBonusContributor`
  （239-256行）や`_MouthTouchBonusContributor`（259-307行）と同じ「compute(ctx)->float」
  規約のクラスを作り、`Taro.__init__`末尾（543-578行）で`self.reward_contributors`に
  追加する（`run/trainer.py`側のforループが自動で呼ぶ）。
- 予測誤差の正解ベクトルの中身（何を予測対象に含めるか）を変えたいなら
  `Taro.encode_target`（730-789行）。目標指向探索の目標表現を変えたいなら
  `Taro.encode_reach_goal`（792-827行、既存のencode_targetとは独立に保つ設計）。
