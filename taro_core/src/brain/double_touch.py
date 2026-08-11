"""ダブルタッチ（頭への自己接触）検出器 ── 報酬に繋ぐための本能版。

【重要：これはDopamine・LocusCoeruleusとは性質が違う】
Dopamine（Schultz 1997, [Tier1]）・LocusCoeruleus（実在する脳幹核）は、検証された
人間の生理機構を模したもの。**このファイルはそれとは違う。** 「ダブルタッチという
感覚イベントが特殊である」こと自体は一次資料で確認されているが（Rochat 1998・2003）、
「それが報酬を生む」「報酬を検出して足す専用回路がある」という部分は**検証されていない、
うまくいくか分からない工学的な当てずっぽう**である（Hoffmann 2017は"might constitute"
＝仮説止まりと確認済み。`E/docs/リーチング/文献調査/自己接触の報酬性_2026-08-02.md`）。
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
`E/docs/リーチング/文献調査/自己接触の報酬性_2026-08-02.md`。
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
"""
import torch

from somatosensory_cortex import SomatosensoryCortex


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

    def __init__(self, touch_map, threshold=0.5, touched_names=("head",)):
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

    def rebuild(self, touch_map):
        """体を作り直したときに地図を差し替える（成長対応、2026-08-05追加）。

        `taro_setup.py`の`on_body_change`から、既存の`fusion.touch`専用の早期return
        より**前**に呼ばれる必要がある（そうしないと`cfg.touch=False`のシーンでは
        この呼び出しが一度も実行されない。設計1-7節・実装ノウハウ2026-08-05項）。
        """
        self._touch_cortex.rebuild(touch_map)
        self._check_groups()

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
