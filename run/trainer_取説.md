# trainer.py 取扱説明書（AI用）

本体: run\trainer.py（1,005行）
最終同期: 2026-08-14 / 本体の行数: 1005

## このファイルは何をするか
`run/config.py` の Config と `run/taro_setup.py` の Taro を受け取り、環境と太郎を噛み合わせて
学習ループを回す（`Trainer.run`）。測定はしない（測るのは `run/plugins/` 側で、ここは呼ぶだけ）。
チェックポイントごとにプラグインへ「測って良い区間」を与え、測定前後で体の状態を完全に控えて
戻す（`_snapshot`/`_restore`）ことで、学習の軌道に測定が影響しないようにする。

## 主要なクラス・関数
| 名前 | 行 | 何をするか |
|---|---|---|
| `_mj_arrays` | 41-62 | MuJoCoの`data`から書き換え可能な配列を機械的に全部集めてコピーする |
| `_release_renderers` | 65-93 | 眼球カメラのOpenGL描画用メモリを明示的に解放する（体を作り直す前に呼ぶ） |
| `Trainer`（クラス） | 96-979 | 太郎を「生きて学ぶ」させる本体クラス。測らない |
| `Trainer.__init__` | 103-113 | cfg・plugins・env/taroの入れ物を初期化するだけ |
| `Trainer.build` | 115-169 | 4系統の乱数を撒く→シーン構築→Taro構築→ctx/probe_ctx構築。乱数消費順が命 |
| `Trainer.step_k` | 172-212 | 1判断でK物理ステップ進める。reflex_common有効時はtick毎に反射+共通駆動を再計算 |
| `Trainer.reset_state` | 214-227 | env.reset()して状態初期化。触覚順応の適用・reward_contributorsのreset呼び出し |
| `Trainer._build_probe_ctx` | 230-255 | `e_probes.ProbeContext`（自己モデル測定用の入れ物）を組み立てる |
| `Trainer._build_ctx` | 257-273 | プラグインに渡す`Ctx`（読み取り専用の入れ物）を組み立てる |
| `Trainer._regrow` | 276-370 | 保存を挟まず体（env）だけ作り直す。脳・経験・オプティマイザは保持したまま成長させる |
| `Trainer.consolidate` | 373-395 | 睡眠中の記憶定着。海馬の経験をバッチ再生し予測経路を復習で固める |
| `Trainer._snapshot` | 417-480 | 体・内臓・乱数（4系統）・筋・視覚キャッシュ・反射の状態をまるごと控える |
| `Trainer._numeric_attrs` | 483-501 | オブジェクトが持つ数値・配列属性を機械的に控える（model/dataは除外） |
| `Trainer._restore` | 503-558 | `_snapshot`で控えた状態に戻す。測定が体を進めた分を巻き戻す |
| `Trainer._record` | 561-592 | 全プラグインの値＋太郎の状態を1行のログ辞書にまとめてlog_rowへ渡す |
| `Trainer._checkpoint` | 595-630 | 測定前後でsnapshot/restoreを挟みつつプラグインのon_checkpointを呼び、記録・進捗表示する |
| `Trainer.run` | 633-979 | 学習ループ本体（e_growth_train.py 1175-1304行の写し）。1ステップの感覚→行動→報酬→学習を回す |
| `close_env` | 982-991 | 環境を閉じる（レンダラ解放込み）。閉じる処理自体の失敗で元の例外を隠さない |
| `train` | 994-1005 | 設定から学習を1本回す入口関数。Trainerを作りbuild→run、finallyでclose_env |

## 触るときの注意
- 乱数を消費する順序を変えると同じシードでも別の学習になる（`build`冒頭のコメント、
  落とし穴チェックリスト項3）。太郎の乱数は4系統（torch/np/random/env.reset(seed=)）で、
  1つでも撒き忘れると再現しない（2026-07-30に発覚した実例あり）。
- `_snapshot`/`_restore`は「測定を無かったことにする」仕組み。ここを崩すと再現性が壊れる
  （落とし穴チェックリスト項79・80）。qpos/qvelだけでは足りず、qacc・qacc_warmstart・
  センサ値・筋の活性化ダイナミクス（`actuation_model`側、Python側の配列）・視覚キャッシュ・
  内臓（HybridEnv側）・反射（VOR等）まで機械的に控える必要がある。
- `_regrow`（体を育てる）は視覚レンダラのメモリリークに注意（OpenGL error 0x505の実例、
  2026-07-30）。触覚の次元は`observation`とは別キーなので次元チェックを素通りする
  （落とし穴チェックリスト項86）。
- `run`内の`reflex_common`分岐（2026-08-11導入）・`goal_babbling`のreach_self分岐
  （2026-08-02〜）・`double_touch`/`mouth_touch_bonus`分岐（2026-08-03〜2026-08-12）は
  いずれも「既定OFFなら1バイトも動作を変えない」設計。既定値を変えずにこれらを触ると
  既存実験全部の再現性が壊れるので、既定の分岐条件（`if reach_space:` 等）を先に確認する。
- タグ文字列（`_checkpoint`内の`print`）はASCIIのみに保つ。Windows cp932で出せない文字
  （上付き数字・絵文字）で学習が落ちた実例が2回ある（落とし穴チェックリスト項35）。

## よくある変更の場所
- 報酬の計算式を変えたいなら `run` 内 792-925行あたり（`rew_task`の決定、努力コスト、
  CAPSペナルティ、`reward_contributors`ループ、`self_touch_interest_bonus`）。
- 何か新しい測定値をログに残したいなら `_record`（561-592行）か、`ctx.last`/`ctx.last_reward`
  （`run`内774-780行・919-925行）に「置いておくだけ」で追加し、プラグイン側で読ませる。
- チェックポイント間隔・保存周りを変えたいなら `_checkpoint`（595-630行）と`run`末尾の
  `if (i + 1) % ckpt == 0:`（962-965行）。
- 体を育てる実験の挙動を変えたいなら `_regrow`（276-370行）。触覚地図・reach_goalの
  再計算もここに集約されている。
- 学習ループに新しい駆動モード（CPG以外）を足すなら `step_k`（172-212行）の
  `reflex_common_active`分岐と、`taro_setup.py`側の`_setup_reflex_common`をセットで見る。
