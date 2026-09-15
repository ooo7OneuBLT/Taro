"""ダブルタッチ（頭への自己接触）検出器 ── 報酬に繋ぐための本能版。

【重要：これはDopamine・LocusCoeruleusとは性質が違う】
Dopamine（Schultz 1997, [Tier1]）・LocusCoeruleus（実在する脳幹核）は、検証された
人間の生理機構を模したもの。**このファイルはそれとは違う。** 「ダブルタッチという
感覚イベントが特殊である」こと自体は一次資料で確認されているが（Rochat 1998・2003）、
「それが報酬を生む」「報酬を検出して足す専用回路がある」という部分は**検証されていない、
うまくいくか分からない工学的な当てずっぽう**である（Hoffmann 2017は"might constitute"
＝仮説止まりと確認済み。`doc/文献調査/リーチング/自己接触の報酬性_2026-08-02.md`）。
このファイルをDopamine等と同じ`taro_core/src/brain/`に置くのは「太郎の学習に実際に
関与する仕組みは実験段階でもcoreに置く（二重実装を避ける、例外なし）」という配置
ルールに従うためであり、「これが人間の本物のメカニズムだから」ではない。

【なぜtaro_coreに作り直すか、2026-08-03】
`run/plugins/common/double_touch.py`（測定専用）で6000ステップの実機実験
（`E/experiments/double_touch_検出確認_6000_seed0.json`）を回し、以下が分かった：

    ・「頭」でのダブルタッチは実際に検出できた（初検出ステップ2148、以降継続的に検出）。
      dtouch_count_head は reach_touch_head（reach_success.pyの頭タッチ回数）と
      全チェックポイントで完全一致 → 頭の一致は信頼できる自己接触の指標。
    ・「胸」は既知の座面confound（body_supportが常に胸に接触）で信用できない。
    ・「反対の手（left_palm）」は一度も検出されなかった → 未解明の想定外として
      別途扱う（このタスクの対象外）。

仕様：作業記録（非公開）

**方針（2026-08-03時点）**：報酬に繋ぐ対象は「頭」のみにする（胸・反対の手は対象外）。

Taroのプラグイン（`run/plugins/`）は「太郎の外から測る道具」であり、学習・報酬には
一切関与しないという既存の設計原則がある（reach_success.pyの作業記録に明記済み）。
報酬に繋ぐ＝太郎の本能の一部にする、ということなので、判定ロジックを
taro_coreへ本能として作り直す（`run/plugins/common/double_touch.py` は測定専用として
そのまま残す。二重実装ではなく「測定用」と「本能用」で役割が違う）。

【文献的背景、Tier3・工学的判断】
`doc/文献調査/リーチング/自己接触の報酬性_2026-08-02.md`。
「自己接触は他者接触より報酬が大きい」という一般化は一次資料では支持されない
（Hoffmann 2017は仮説止まり）。支持されるのは「ダブルタッチ（同時に2チャンネルの
触覚情報が来る）は質的に特殊な入力パターンである」という点のみ。これを報酬信号
として使うこと自体は**Tier3・工学的な設計判断**であり、実証された人間のメカニズム
そのものではない。この逸脱は `doc/人間模倣からの逸脱リスト.md` に追記済み。

【presenceの取り方】
`run/plugins/common/double_touch.py`・`reach_success.py`・`encode_reach_goal` と
全く同じ考え方（`part_features()`、presenceしきい値判定、`group_names` で毎回
名前検索）を踏襲する。

【2026-08-05追記：全身の複数部位への一般化・taro自身の脳への依存を断つ】
設計：作業記録（非公開）
仕様：作業記録（非公開）

変更点は2つ：

  (1) `touched_name`（単数、既定"head"）を`touched_names`（複数、既定`("head",)`、
      OR判定＝いずれか1部位でもしきい値を超えればヒット）へ一般化した。
      【対象部位をどこまで広げてよいか、文献的な結論（設計1-2節・5節）】
      「頭」という部位の選定自体に文献的な特別さは無い。現状「頭のみ」を対象に
      している理由は測定上の信頼性（reach_success.pyの頭タッチ回数との完全一致）
      であって、Rochat(1998)の定義が「頭」という体の部位を名指ししているわけでは
      ない。だが、だからといって「全身どこでもよい」への一般化は文献的に支持
      されない。Rochat(1998)の定義（"the double touch of the cutaneous surface of
      the hand contacting the cutaneous surface of the facial region...Contact by
      the baby with any other physical object, surface, or person in the
      environment will never correspond to a double-touch intermodal event."）は、
      「顔（の代替である頭）」という組み合わせを他の全ての接触と対比させて
      定義したものであり、「顔以外の自分の体の部位」への一般化を許すものでは
      ない。よって、この一般化は[Tier3・工学的判断]：「機構として複数部位を
      扱えるようにしたこと」自体に文献根拠は無い。既定値は`("head",)`のまま
      変えていない＝実験ファイルで明示指定しない限り、既存実験の挙動は
      1ビットも変わらない。胸・腰・手足・目などを対象に加える場合は、実験ごとに
      個別の混同源の確認が要る（胸は座面confoundが既知）。目は防御的な瞬目反射が
      知られ、人間の赤ちゃんが積極的に自分の目を触れて報われる行動として確立
      していないため、対象に加えないこと（`前提.md`「人間の赤ちゃんがしないことは
      入れない」）。

  (2) `touch_module`引数（taro自身の`target_fusion.touch`、凍結インスタンス・
      RND式・勾配なしを渡す経路）を廃止し、`__init__`で`touch_map`を受け取り、
      自前で`SomatosensoryCortex`を構築する（`run/plugins/common/contact_reward.py`
      の`setup()`が`build_touch_map_from_env`+`SomatosensoryCortex(touch_map,
      embedding_dim=64)`を作るのと同じパターン）。
      理由：`SomatosensoryCortex.part_features()`の出力は`presence_gain`・
      `strength_gain`（`nn.Parameter`、初期値1.0）に依存するが、`target_fusion.touch`
      は"RND式・凍結インスタンス・勾配なし"であり一度も最適化されない＝ゲインは
      永遠に初期値1.0のまま。よって同じ`touch_map`から新しく作った自前インスタンス
      も同じ初期値1.0を持ち、出力は理論上一致する（実装時に実測で裏取り済み。
      作業記録参照）。これにより、`cfg.touch=False`のシーン（taro自身の脳が触覚を
      使っていない場合）でも、環境から取れる生の触覚観測だけでこのクラスが
      単独で動くようになった＝reach_self・touch=true専用という制約が外れた。

  (3) 部位名の存在確認（`_check_groups`、`contact_reward.py`と同じ流儀）と
      `rebuild(touch_map)`（`taro_setup.py`の`on_body_change`から呼ばれる。成長で
      触覚の点数が変わっても、古い地図を参照し続けないようにするため）を追加した。

  toucher側（触れる手）は今回変更していない（`reach_arm_side`固定のまま）。
  Rochatの定義はどちらの手かを区別していないため、両手を対象にする拡張は
  定義への忠実化にあたるが、対象部位を広げることとは別の軸であり、今回の
  スコープには含めない（`doc/人間模倣からの逸脱リスト.md`⑦・やることリストへ
  申し送り済み）。

【2026-08-12追記：口元への自己接触報酬】
仕様：作業記録（非公開）
設計（人間模倣・値そのもの）：作業記録（非公開）5-1節

「口元」は"head"グループの**一部（48点/308点）**であり、既存の
touch_map（グループ単位）では表現できない。既存の`DoubleTouchDetector`を拡張し、
口元presenceを扱えるようにする（新しい検出器クラスは作らない、仕様1〜2節）。

  - `mouth_point_mask()`（下記モジュール関数）：頭の触覚308点のうち、頭の
    ローカル座標で前後方向(x)・上下方向(z)の割合しきい値を満たす点を
    「口元」として選び、touch_map.n_points 長のbool配列を返す。
    しきい値は絶対座標でなく割合で持つ（体が成長して頭のサイズが変わっても
    追従するため）。
  - `DoubleTouchDetector.__init__`に`mouth_mask=None`（任意引数）を追加。
    渡さなければ既存の挙動は1ビットも変わらない（口元機能は使われない）。
  - `mouth_presence()`：口元マスクで選んだ点の力の最大値をtanhで正規化して返す。
    `gain`は`SomatosensoryCortex.presence_gain`と同じ流儀で固定値1.0
    （学習しない。一貫性のため）。
  - `rebuild()`：成長で触覚の点数・並びが変わるたびに、口元マスクも
    `mouth_point_mask()`で再計算する（`model`・`touch`を新たに渡せるよう拡張）。

[Tier3・工学的判断]口元の切り出しは、人間の唇の感覚受容器の実際の分布を
再現したものではなく、頭の一部を座標で人為的に区切っただけの近似である。
`doc/人間模倣からの逸脱リスト.md`に追記済み。
"""
import numpy as np
import torch

from somatosensory_cortex import SomatosensoryCortex


def mouth_point_mask(touch_map, model, touch, x_frac=0.60, z_frac=0.40,
                      head_group="head"):
    """touch_map の "head" グループの点のうち、頭のローカル座標で
    前後方向(x)が x_min+x_frac*(x_max-x_min) より前、
    上下方向(z)が z_min+z_frac*(z_max-z_min) より下、
    の両方を満たす点を「口元」として選び、touch_map.n_points と同じ長さの
    bool配列（口元ならTrue）を返す。

    しきい値は絶対座標でなく割合で決める（体が成長して頭のサイズが変わっても
    追従するため。`E/docs/figures/口元領域の候補_2026-08-12.png`、
    作業記録（非公開） 5-1節）。

    注意：`touch.sensor_positions[head_body_id]` は「頭のローカル座標」
    （`build_touch_map_from_env`の元になっている`touch.sensor_positions`そのもの、
    TrimeshTouchではキーがbody_id）。`touch_map.positions` はグループごとに
    中心を引いて正規化した座標なので、ここでは使わない（絶対的な前後・上下の
    意味を失っているため）。

    Args:
        touch_map: `build_touch_map`（または`build_touch_map_from_env`）の戻り値
        model: mujoco model
        touch: env.unwrapped.touch（mimoTouch.Touch。生のtouchオブジェクト）
        x_frac: 前後方向のしきい値の割合（既定0.60）
        z_frac: 上下方向のしきい値の割合（既定0.40）
        head_group: 頭のグループ名（既定"head"）

    Returns:
        numpy.ndarray[bool]  長さ touch_map.n_points
    """
    if head_group not in touch_map.group_names:
        raise AssertionError(
            f"mouth_point_mask: touch_map に '{head_group}' グループが無い。\n"
            f"  いまの部位: {touch_map.group_names}")
    head_bid = int(model.body(head_group).id)
    sp = np.asarray(touch.sensor_positions[head_bid], dtype=np.float64)
    x_min, x_max = float(sp[:, 0].min()), float(sp[:, 0].max())
    z_min, z_max = float(sp[:, 2].min()), float(sp[:, 2].max())
    local_mouth = (sp[:, 0] > x_min + x_frac * (x_max - x_min)) & \
                  (sp[:, 2] < z_min + z_frac * (z_max - z_min))

    head_gid = touch_map.group_names.index(head_group)
    head_idx = np.where(touch_map.part_of_point == head_gid)[0]
    # 【なぜ、仕様2節】head_idx の各要素は sp（touch.sensor_positions[head_bid]）の
    #   行と**同じ順序で1対1対応する**（build_touch_mapが各bodyキーの点を
    #   sorted(meshes)順のまま連結してpart_of_pointを作っているため）。
    #   対応が崩れていないかを点数の一致で検算してから使う
    #   （落とし穴チェックリスト項86「エラーが出ずに動いた、を動いたと読まない」）。
    if head_idx.shape[0] != sp.shape[0]:
        raise AssertionError(
            f"mouth_point_mask: head_idx({head_idx.shape[0]}点) と "
            f"touch.sensor_positions[head]({sp.shape[0]}点) の点数が合わない。\n"
            "  build_touch_map の並びの前提（bodyごとの点をsorted(meshes)順のまま"
            "連結）が崩れている可能性がある。")

    mask = np.zeros(touch_map.n_points, dtype=bool)
    mask[head_idx[local_mouth]] = True
    return mask


class DoubleTouchDetector:
    """toucher（reach_arm_sideの手のひら）と touched_names（既定 頭のみ）の presence が
    同じtickで両方しきい値を超えたら True を返す（touched_namesはOR判定＝いずれか
    1部位でもヒットすればヒット）。

    Dopamine・LocusCoeruleusと同じ粒度：小さく・状態を持たない
    （このクラス自体はステップをまたいだ「時間方向の」状態を持たない。しきい値・
    自前のSomatosensoryCortex・対象部位のリストだけを保持する）。

    【なぜ自前でSomatosensoryCortexを持つか、2026-08-05】taro自身の脳
    （`target_fusion.touch`）に依存すると、`cfg.touch=False`のシーンでは
    `target_fusion.touch`自体が存在せず動かない。このクラスは「太郎がこの感覚を
    使っているかに関わらず、物理的な接触という事実そのものは計算できる」という
    工学的な割り切り（`contact_reward.py`と同じ考え方）で、環境から取れる生の
    触覚観測を自前で処理する。
    """

    def __init__(self, touch_map, threshold=0.5, touched_names=("head",),
                 mouth_mask=None, mouth_x_frac=0.60, mouth_z_frac=0.40,
                 mouth_head_group="head"):
        # しきい値。[Tier3・工学的判断]既存プラグイン(double_touch.py)と同じ値・
        #   同じ考え方（presence = tanh(peak) なので 0.5 は peak≈0.55 に相当する
        #   適当な中間値。文献的な根拠は無い）。
        self.threshold = float(threshold)
        # touched_names＝報酬に繋ぐ「触れられる側」の部位一覧。既定は頭のみ
        #   （既存実験の挙動を変えない後方互換。2026-08-05・全身一般化の設計
        #   1-2節：対象を広げること自体は文献根拠の無いTier3の工学的判断）。
        self.touched_names = tuple(touched_names)
        # 【2026-08-05】自前でSomatosensoryCortexを構築する（taro自身の脳への
        #   依存を断つ。上記docstring(2)参照）。
        self._touch_cortex = SomatosensoryCortex(touch_map, embedding_dim=64)
        self._check_groups()
        # 【2026-08-12・口元自己接触報酬】mouth_mask=None（既定）のままなら、
        #   以下の属性は一切参照されない＝既存の呼び出し元（reach_self・
        #   double_touch_bonus単体）の挙動は1ビットも変わらない。
        #   mouth_x_frac/mouth_z_frac/mouth_head_groupは値そのものではなく
        #   「rebuild()で口元マスクを再計算するためのレシピ」として保持する
        #   （実装担当の判断。仕様3節「実装時に確認すること」）。
        self._mouth_mask = None
        if mouth_mask is not None:
            self._mouth_mask = torch.as_tensor(mouth_mask, dtype=torch.bool)
        self._mouth_x_frac = float(mouth_x_frac)
        self._mouth_z_frac = float(mouth_z_frac)
        self._mouth_head_group = str(mouth_head_group)

    def _check_groups(self):
        """touched_names が触覚の地図に存在するかを確認する
        （`run/plugins/common/contact_reward.py`の`_check_groups`と同じ流儀。
        存在しない部位名を黙って無視せず、学習開始前にAssertionErrorで止める
        ＝落とし穴チェックリスト項86「エラーが出ずに動いた、を動いたと読まない」）。
        """
        names = self._touch_cortex.group_names
        missing = [nm for nm in self.touched_names if nm not in names]
        if missing:
            raise AssertionError(
                f"DoubleTouchDetector の touched_names に触覚の地図に無い部位がある: "
                f"{missing}\n  いまの部位: {names}")

    def rebuild(self, touch_map, model=None, touch=None):
        """体を作り直したときに地図を差し替える（成長対応、2026-08-05追加）。

        `taro_setup.py`の`on_body_change`から、既存の`fusion.touch`専用の早期return
        より**前**に呼ばれる必要がある（そうしないと`cfg.touch=False`のシーンでは
        この呼び出しが一度も実行されない。設計1-7節・実装ノウハウ2026-08-05項）。

        【2026-08-12追記】口元マスクを使っている場合（`self._mouth_mask is not None`）、
        成長で触覚の点数・並びが変わるため`mouth_point_mask()`を再計算する。
        再計算には`model`・`touch`（生のtouchオブジェクト）が要るので、
        引数で受け取れるよう拡張した。口元マスクを使っていなければ
        （既存の呼び出し元）、`model`・`touch`が渡されなくても今まで通り動く
        （挙動不変）。
        """
        self._touch_cortex.rebuild(touch_map)
        self._check_groups()
        if self._mouth_mask is not None:
            if model is None or touch is None:
                raise AssertionError(
                    "DoubleTouchDetector.rebuild: 口元マスクを使っているのに"
                    " model/touch が渡されなかった。\n"
                    "  成長後も古い口元マスク（古い点数・古い並び）を参照し続けてしまう"
                    "（落とし穴チェックリスト項86「エラーが出ずに動いた、を動いたと"
                    "読まない」）。呼び出し元（taro_setup.pyのon_body_change）で"
                    "model=env.unwrapped.model, touch=env.unwrapped.touch を渡すこと。")
            new_mask = mouth_point_mask(
                touch_map, model, touch,
                x_frac=self._mouth_x_frac, z_frac=self._mouth_z_frac,
                head_group=self._mouth_head_group)
            self._mouth_mask = torch.as_tensor(new_mask, dtype=torch.bool)

    def mouth_presence(self, touch_flat, gain=1.0):
        """口元の presence（0〜1）を返す。仕様3節。

        touch_flat を (n_points,3) に変形 → 各点の力の大きさ(ノルム) →
        self._mouth_mask で選んだ点の最大値(peak) → tanh(peak*gain)。

        gain は学習しない固定値（既定1.0）。`SomatosensoryCortex.presence_gain`の
        初期値と同じ流儀（そちらも一度もoptimizerに渡らず永遠に初期値のままである
        ことがdocstringに明記されているので、口元側も同じ扱いにして一貫性を保つ）。

        口元にセンサ点が1つも無い場合（マスク未設定、mouth_mask=Noneのまま構築した
        場合）は0.0を返す。
        """
        if self._mouth_mask is None:
            return 0.0
        with torch.no_grad():
            f = touch_flat.reshape(-1, 3)
            mag = torch.linalg.vector_norm(f, dim=-1)
            mask = self._mouth_mask.to(mag.device)
            if not bool(mask.any()):
                return 0.0
            peak = mag[mask].max()
            return float(torch.tanh(peak * gain))

    def detect(self, touch_flat, toucher_name, touched_names=None):
        """toucher_name: 触れている側の部位名（例 "right_palm"）
        touch_flat: 触覚の生の観測（to_tensor済み）
        touched_names: 触れられる側の部位名のイテラブル（省略時は self.touched_names。
          いずれか1部位でもしきい値を超えればヒット＝OR判定）

        戻り値: (bool 一致したか, toucher_presence, touched_presence, hit_names)
          touched_presence: {部位名: presence値} の辞書（ログ・チューニング用）
          hit_names: しきい値を超えて実際にヒットした部位名のリスト（0件ならヒットなし）
        """
        names = self._touch_cortex.group_names
        use_names = tuple(touched_names) if touched_names is not None else self.touched_names
        # 毎回名前で検索する（reach_success.py・既存プラグインと同じ流儀。
        #   整数indexをキャッシュすると、体の作り直しでグループの並びが
        #   変わったとき静かに別の部位を読む＝落とし穴チェックリスト項86）。
        missing = [nm for nm in (toucher_name, *use_names) if nm not in names]
        if missing:
            raise AssertionError(
                f"DoubleTouchDetector.detect の対象部位が触覚の地図に無い: {missing}\n"
                f"  いまの部位: {names}")
        with torch.no_grad():
            feat = self._touch_cortex.part_features(touch_flat)     # (G, 5)
        ti = names.index(toucher_name)
        toucher_presence = float(feat[ti, 0])
        toucher_touching = toucher_presence > self.threshold
        touched_presence = {}
        hit_names = []
        for nm in use_names:
            gi = names.index(nm)
            p = float(feat[gi, 0])
            touched_presence[nm] = p
            if toucher_touching and p > self.threshold:
                hit_names.append(nm)
        hit = bool(hit_names)
        return hit, toucher_presence, touched_presence, hit_names
