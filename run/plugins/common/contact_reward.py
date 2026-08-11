"""指定した部位どうしの自己接触の瞬間とそれ以外の瞬間で、太郎の脳がすでに持っている
報酬(rew)・ドーパミンのRPE(rpe)の値そのものに差があるかを直接測る。

【なぜ要るか、2026-08-03】ダブルタッチ報酬（頭への自己接触に+0.2の人工ボーナスを
足す）を試したが、n=5シードで効果が消えた。何かを足す前に、「自己接触が起きた
瞬間」と「それ以外の瞬間」で、太郎が元から持っている報酬・RPEに自然な差がある
かどうかをまず直接測る。差が無ければ、自己接触に注目した一連の実験（姿勢バイアス
版含む）の前提自体が崩れる、という重要な分岐点になる測定。
仕様：作業記録（非公開）
設計：作業記録（非公開）
（7節「2026-08-03追記」が最終版。おもちゃ接触の分類は使わない＝設計が既存ログ
14本を実測したところ、同じシーン・goal_space=reach_selfではおもちゃへの接触が
14本すべてで0回だった。reach_self（自己接触目標）にはおもちゃへ向かう動機が
組み込まれておらず、おもちゃ接触を比較対象にすること自体が無理筋、という整理）。

【2026-08-04追記：selfの定義を「頭または胸への接触」に拡張した】
設計：作業記録（非公開）
（2節「選択肢2」採用）。

【2026-08-05追記：reach_self・touch=true への依存を除去し、一般化した】
仕様：作業記録（非公開）
背景：このプラグインは今まで `taro.reach_touch_groups`・`taro.cfg.reach_arm_side`
（goal_babbling=true・goal_space="reach_self" のときだけ存在）と、
`taro.target_fusion.touch.part_features()`（`taro.touch=true` で脳が触覚を
使っているときだけ存在）に依存しており、reach_self・touch=true の実験でしか
使えなかった。「柵なし自発運動シーン」（goal_babbling無し・touch=false=既定）では
このプラグインが動かなかった。

変更点は2つ：
  (1) `toucher`・`touched_groups` を実験ファイルの `plugins.contact_reward` 欄から
      読む形にした。reach_self設定がある実験（`plugins.contact_reward: true`だけ書く
      既存の書き方）は、いままで通り `taro.cfg.reach_arm_side`・
      `taro.reach_touch_groups` から自動導出する（後方互換。既存12〜13本の実験
      ファイルは1文字も変更不要）。reach_self設定が無い実験では、明示的に
      `{"toucher": "right_palm", "touched_groups": ["head", "chest"]}` のように
      指定する。
  (2) `taro.target_fusion.touch`（taro自身の脳の一部。touch=trueでしか存在しない）
      への依存を断ち、このプラグインが**自前で**`SomatosensoryCortex`インスタンスを
      構築して判定する。根拠：`SomatosensoryCortex`の`presence_gain`・
      `strength_gain`はリテラル定数`nn.Parameter(torch.tensor(1.0))`で初期化され、
      `part_features()`のコード自体（`taro_core/src/senses/somatosensory_cortex.py`
      362〜398行）はそれと`touch_map`（`group_names`・`part_of_point`・`positions`・
      `counts`）以外の学習可能パラメータ（`part_weight`・`integrate`）を一切
      参照しない。よって`part_features()`の出力は、学習が何tick進んでも、
      どのインスタンスで計算しても、`touch_map`が同じであれば常に同じ値になる
      （taroの脳が触覚を使っていなくても、環境からは触覚の生データ`obs["touch"]`が
      常に取れる。`run/taro_setup.py:35`の`build_touch_map_from_env`、290行付近の
      `on_body_change`の使用例を参照。環境側は`cfg.touch`の設定に関わらず常に
      触覚センサーを計算している）。
      これにより、`touch=false`で学習したモデル（今回使いたい「良い自発運動」の
      selfmodel_v3含む）でも、このプラグインだけで自己接触を測定できる。
      頭の判定も、以前は`ctx.last_double_touch`（reach_selfのときしか計算されない）
      経由だったが、胸と同じ自前判定に統一した（このファイルのon_step参照）。

【胸を含めることの限界（Tier3・工学的近似。double_touch.pyと同じ限界）】
presence（`SomatosensoryCortex.part_features()`の①有無）は接触の**発生源**
（相手が何か）を区別しない。つまり「toucher（手のひら）とtouched_groupsの部位が
互いに接触した」保証ではなく、「同じtickでtoucherと当該部位が、たまたま別々に
何かへ触れていた」可能性を排除できない近似である（double_touch.pyのdocstring
参照）。特に胸は、double_touch.py・run/config.pyのコメントに明記されている通り
**座面との接触という別の混同源**を持つ（リクライニングした姿勢で胸が座面に
触れ続ける可能性がある）。この限界は`doc/人間模倣からの逸脱リスト.md`の
double_touch.py分（本文③）に既に登録済みの限界を、胸を含めることで再度踏襲する形。

【分類は2区分のみ】
    "self" ＝ そのtickで toucher と touched_groups のいずれか1部位以上との
              presenceしきい値判定が成立していた（このプラグインが自前で判定、
              下記on_step参照）
    "none" ＝ それ以外の全ての瞬間
おもちゃ判定は行わない（`E/scripts/e_toy_touch.py`・`run/plugins/common/toy_touch.py`
は一切importしない）。

【rew・rpeの読み方】
`ctx.last`（on_step 用）ではなく、新設の`ctx.last_reward`（{"rew":..., "rpe":...}）
を`on_step_late`で読む。理由は`run/plugins/base.py`のon_step_lateのdocstring・
`run/trainer.py`の該当箇所を参照（rew・rpeはon_stepが呼ばれる時点ではまだ
計算されていない）。接触presenceの判定自体はrew・rpeに依存しないため、on_step
（rew・rpe確定"前"）で計算して`self._last_hit_parts`に置いておき、on_step_late
でrew・rpeと合成する。

実験ファイルでの書き方：

    reach_self設定がある実験（従来通り、自動導出）:
        "plugins": {"contact_reward": true}

    reach_self設定が無い実験（明示的に指定する。2026-08-05新設）:
        "plugins": {"contact_reward": {
            "toucher": "right_palm",
            "touched_groups": ["head", "chest"]
        }}

注意：run.type=train でのみ使える（rew・rpeは太郎の脳が行動を選び学習した結果
として毎回計算される値であり、run.type=measure（脳を通さない）ではrew・rpe自体が
計算されないため）。touch=false・goal_babbling無しの実験でも使える
（2026-08-05以降。触覚の判定自体はこのプラグインが自前で行うため）。

【2026-08-05追記：自己接触/外界接触の区別（世界目標＝world_targets）】
上記の"self"/"none"の2区分（presenceベース）は一切変えず、新しい設定キー
`world_targets`を指定したときだけ、MuJoCoの実際の接触ペア(`data.contact`)を
使った正確な外界接触判定を追加する。既存の"self"（presence近似）とは判定方式が
異なる別のカテゴリとして扱う（設計3節・実装担当の申し送り参照。presence近似
自体の精度向上は今回の対象外）。

    world_targets 未指定           機能OFF（既存実験そのまま。新規コードに
                                    一切到達しない。設定キーの有無で判定）
    world_targets: []              絞り込みなし（太郎自身以外の全geomが対象。
                                    内訳も記録）
    world_targets: ["fence_post"]  前方一致するgeom/bodyだけに絞る

実験ファイルでの書き方（2026-08-05新設）:

    "plugins": {"contact_reward": {
        "toucher": "right_palm",
        "touched_groups": ["head", "chest"],
        "world_targets": ["fence_post"]
    }}

`world_toucher_bodies`（省略可）を指定すると、外界接触の「自分側」のbody名を
明示指定できる。省略時は`toucher`（例"right_palm"）から
`somatosensory_cortex.body_names_of_group()`で実body名を自動導出する。

分類は4区分（self_and_ext / self_only / ext_only / neither）。既存の
self.CATS=("self","none")とは別の`self.WORLD_CATS`に集計する（設計4-3節）。
設計・実装レベルの詳細は`run/plugins/common/body_geoms.py`のdocstring参照。
"""

import os

import torch

from run.plugins.base import Plugin
from run.taro_setup import to_tensor
# 【なぜ、2026-08-05】taro自身の脳（target_fusion.touch、touch=trueでしか
#   存在しない）に依存せず、このプラグインが自前で触覚を判定するために使う。
#   run.taro_setup を先にimportした時点でsys.pathにsensesフォルダが追加されている
#   （run/taro_setup.py 20-29行）ため、この単純なimportで解決できる。
from somatosensory_cortex import build_touch_map_from_env, SomatosensoryCortex
# 【2026-08-05追記：外界接触（柵等）の区別】既存のimport行(上)は変えず、新規の
#   import行を追加するだけにする。body_names_of_groupはtaro_core側の読み取り専用
#   関数（_BODY_GROUPSを検索するだけ）。geoms_of_body等は新設の共有モジュール。
#   設計：作業記録（非公開）
#   仕様：作業記録（非公開）
from somatosensory_cortex import body_names_of_group
from run.plugins.common.body_geoms import geoms_of_body, geoms_by_name_prefix, contact_pairs_between

# presence判定のしきい値。double_touch.pyのDEFAULT_THRESHOLDと同じ値・
#   同じ考え方（[Tier3・工学的判断]。presence=tanh(peak)なので0.5はpeak≈0.55に
#   相当する適当な中間値。文献的根拠は無い）。実験ファイルで
#   plugins.contact_reward.threshold を指定すれば上書きできる（double_touch.pyと
#   同じ流儀）。
_DEFAULT_THRESHOLD = 0.5


def _label_for_geom(model, g):
    """geomの表示名を1つ決める（world_targets=[]の内訳報告用、2026-08-05追加）。

    geom自身に名前があればそれを使い、無ければ属すbodyの名前を使う
    （床・座面・背もたれ等、geomに個別の名前が無いケースの受け皿）。
    """
    gname = model.geom(g).name or ""
    if gname:
        return gname
    bid = int(model.geom_bodyid[g])
    return model.body(bid).name or f"geom{g}"


class ContactReward(Plugin):
    name = "contact_reward"

    def setup(self, ctx):
        if ctx.brain is None:
            raise ValueError(
                "contact_reward は太郎の脳が要る（run.type=train で使う）。\n"
                "  measure（脳を通さず環境だけ進める）では rew・rpe 自体が計算されない")
        taro = getattr(ctx, "taro", None)
        # 【2026-08-05：reach_self依存の除去・設定駆動化】
        #   明示指定を優先し、無ければreach_self設定から後方互換で自動導出する。
        self.toucher = self.config.get("toucher")
        self.touched_groups = self.config.get("touched_groups")
        if self.toucher is None and taro is not None and getattr(taro.cfg, "reach_arm_side", None):
            self.toucher = f"{taro.cfg.reach_arm_side}_palm"
        if self.touched_groups is None and taro is not None and getattr(taro, "reach_touch_groups", None):
            self.touched_groups = ["head", "chest"]     # 既存の挙動をそのまま踏襲
        if self.toucher is None or not self.touched_groups:
            raise ValueError(
                "contact_reward には toucher・touched_groups が要る。\n"
                "  reach_self設定が無い実験では、plugins.contact_reward.toucher と "
                "touched_groups を明示的に指定すること（例：{\"toucher\":\"right_palm\","
                "\"touched_groups\":[\"head\",\"chest\"]}）")
        self.threshold = float(self.config.get("threshold", _DEFAULT_THRESHOLD))
        # 【2026-08-05：taroの脳への依存を断つ】自前のSomatosensoryCortexを構築する。
        #   taro.target_fusion.touch は cfg.touch=true でしか存在しないため、
        #   touch=false の実験（今回使いたい selfmodel_v3 等）ではこちらを使う。
        #   part_features() は学習可能パラメータを一切参照しない純粋な計算式なので
        #   （このファイル冒頭のdocstring参照）、taro側のインスタンスと数値は
        #   一致するはず（実装後に実測で裏取りする。単体確認は作業記録参照）。
        touch_map = build_touch_map_from_env(ctx.env)
        self._touch_cortex = SomatosensoryCortex(touch_map, embedding_dim=64)
        self._check_groups()
        self._last_hit_parts = []
        # 自己接触tickの生ログ（2026-08-04）。selfカテゴリのtickだけ、
        #   集計せず1行=1tickでそのまま保持する（自己接触は稀なので平均にすると
        #   内訳の意味が薄れる、という研究者の要望）。既定OFF＝実験ファイルで
        #   plugins.contact_reward.self_trace_out を指定したときだけ動く。
        #   noneカテゴリは対象外（今まで通り集計のみ）。
        self.self_trace_out = self.config.get("self_trace_out")
        self.self_trace_rows = []
        # 分類は2区分のみ（2026-08-03、設計7節でコーディネーターが
        #   おもちゃ判定を除いて簡略化した最終版。2026-08-04、selfの定義を
        #   「頭または胸への接触」に拡張したが区分そのものは2区分のまま。
        #   2026-08-05、touched_groupsを一般化したが区分は変えていない）。
        self.CATS = ("self", "none")
        self.steps = 0
        # 通し（学習全体）の集計
        self.total_count = {c: 0 for c in self.CATS}
        self.total_rew_sum = {c: 0.0 for c in self.CATS}
        self.total_rpe_sum = {c: 0.0 for c in self.CATS}
        # ============================================================
        # 【2026-08-05追記：外界接触（柵等）の区別】ここから下は
        #   if self._world_enabled: のガード1つの中だけに新規コードを追記する。
        #   上のCATS・total_count等、既存の行は1行も変えていない。
        #   world_targetsキー自体が実験ファイルに無ければ_world_enabled=Falseに
        #   なり、以後の新規コードには物理的に到達しない（後方互換の機械的保証）。
        # ============================================================
        self._world_enabled = "world_targets" in self.config
        if self._world_enabled:
            self._setup_world(ctx)
        # 区間（記録の区切り）の集計
        self._seg_reset()

    def _setup_world(self, ctx):
        """world_targets が指定されたときだけ呼ばれる（setup から）。

        toucher側（太郎自身の、外界と接触したかを見たい部位）と、外界側
        （柵・おもちゃ等、絞り込み対象）のgeom集合を作る。
        """
        # ---- toucher側のgeom集合 ----------------------------------------
        #   world_toucher_bodies が明示指定されていればそれを使う。無ければ
        #   self.toucher（触覚グループ名。例"right_palm"）から実body名を
        #   自動導出する（仕様3節：グループ名と実body名は一致しないため
        #   taro_core側のbody_names_of_group()を経由する）。
        toucher_bodies = self.config.get("world_toucher_bodies")
        if not toucher_bodies:
            toucher_bodies = body_names_of_group(self.toucher)
        self._world_toucher_bodies = list(toucher_bodies)
        self._world_toucher_geoms = set()
        for bn in self._world_toucher_bodies:
            self._world_toucher_geoms |= geoms_of_body(ctx.model, bn, include_children=True)

        # ---- 外界側のgeom集合 ----------------------------------------------
        #   world_targets: [] （空リスト）＝絞り込みなし＝太郎自身以外の
        #   全geomを対象にする。太郎自身のgeom集合はroot body "mimo_location"
        #   から求め、全geomから引いた残りを使う。
        #   world_targets: ["fence_post"] のような非空リスト＝前方一致で絞る。
        self._world_targets = list(self.config.get("world_targets") or [])
        if self._world_targets:
            self._world_target_geoms = geoms_by_name_prefix(
                ctx.model, self._world_targets, include_children=True)
            self._check_world_targets()
        else:
            taro_geoms = geoms_of_body(ctx.model, "mimo_location", include_children=True)
            self._world_target_geoms = {
                g: _label_for_geom(ctx.model, g)
                for g in range(ctx.model.ngeom) if g not in taro_geoms}

        # 4区分の集計（既存のself.CATS・total_countとは別の入れ物）。
        #   【なぜhasattrで守るか】_setup_worldはon_body_change（体を作り直す
        #   たび）からも呼ばれる。base.Pluginのon_body_changeの規約
        #   「累積した測定値は消さない」に合わせ、geom集合は毎回作り直すが、
        #   集計カウンタは最初の1回（setupから呼ばれたとき）だけ初期化する。
        if not hasattr(self, "total_world_count"):
            self.WORLD_CATS = ("self_and_ext", "self_only", "ext_only", "neither")
            self.total_world_count = {c: 0 for c in self.WORLD_CATS}
            # 外界側「どの物体に何回触れたか」の内訳（geoms_by_name_prefixが返した
            #   「一致した名前」ごとの回数。report()の要求）
            self.total_world_breakdown = {}
        self._last_world_hit_pairs = []

    def _check_world_targets(self):
        """world_targets が非空リストなのに、1件もgeomが一致しなかったら止める。

        【Q2（承認済み）】設計8節Q2・実装担当の仕様どおり、既存の
        _check_groups と同じ流儀でAssertionErrorを出す（黙って0のまま
        動かし続けない。シーンの取り違えを早期に発見するため）。
        world_targets=[]（絞り込みなし）のときはこのチェックは当てはまらない。
        """
        if self._world_targets and not self._world_target_geoms:
            raise AssertionError(
                f"contact_reward の world_targets が1件もgeom/bodyに一致しない: "
                f"{self._world_targets}\n"
                f"  シーンの取り違えの可能性がある（項86・既存の_check_groupsと同じ流儀）")

    def _check_groups(self):
        """toucher・touched_groups が触覚の地図に存在するかを確認する
        （double_touch.py._check_groups と同じ流儀）。
        """
        names = self._touch_cortex.group_names
        missing = [nm for nm in (self.toucher, *self.touched_groups) if nm not in names]
        if missing:
            raise AssertionError(
                f"contact_reward の対象部位が触覚の地図に無い: {missing}\n"
                f"  いまの部位: {names}")

    def on_body_change(self, ctx):
        # 【なぜ、2026-08-04（2026-08-05に自前のSomatosensoryCortexへ差し替え）】
        #   体を作り直すとTouchMap（点→部位の対応）がrebuildされる。
        #   taro.on_body_change・double_touch.on_body_change と同じ理由で
        #   ここでも地図を差し替えて確認し直す（落とし穴チェックリスト 項86）。
        tm = build_touch_map_from_env(ctx.env)
        self._touch_cortex.rebuild(tm)
        self._check_groups()
        # 【2026-08-05追記】体を作り直すとgeom idが変わりうるため、world側の
        #   geom集合も作り直す。既存の上の3行は変えていない。
        if self._world_enabled:
            self._setup_world(ctx)

    def on_step(self, ctx):
        # on_step_late（rew・rpe確定後）に渡すため、接触presenceだけ先に判定して
        #   おく。double_touch.py の on_step 実装と全く同じやり方
        #   （ctx.last["obs_out"]["touch"]をto_tensorで読み、part_features()で
        #   presenceを得て、group_namesで毎回名前で検索する）。
        #   【2026-08-05】taro.target_fusion.touch ではなく self._touch_cortex
        #   （自前のインスタンス）を使う。
        last = getattr(ctx, "last", None)
        if last is None:
            # 異常系（on_stepが呼ばれなかった等）の安全弁。
            self._last_hit_parts = []
            return
        obs = last["obs_out"]
        touch = to_tensor(obs["touch"])
        with torch.no_grad():
            feat = self._touch_cortex.part_features(touch)     # (G, 5)
        names = self._touch_cortex.group_names
        ti = names.index(self.toucher)     # 毎回名前で検索する（double_touch.pyと同じ流儀）
        toucher_presence = float(feat[ti, 0])
        toucher_touching = toucher_presence > self.threshold
        hit_parts = []
        if toucher_touching:
            for nm in self.touched_groups:
                gi = names.index(nm)
                presence = float(feat[gi, 0])
                # ダブルタッチ判定＝toucher側とtouched側の両方が閾値を超える
                #   （double_touch.pyの判定をそのまま踏襲）。
                if presence > self.threshold:
                    hit_parts.append(nm)
        self._last_hit_parts = hit_parts
        # 【2026-08-05追記】外界接触の判定はrew・rpeを使わないのでon_stepで
        #   計算してよい（仕様のとおり）。既存の上のself判定コードは変えていない。
        if self._world_enabled:
            self._last_world_hit_pairs = contact_pairs_between(
                ctx.data, self._world_toucher_geoms, set(self._world_target_geoms.keys()))

    def _seg_reset(self):
        self._seg_steps = 0
        self._seg_count = {c: 0 for c in self.CATS}
        self._seg_rew_sum = {c: 0.0 for c in self.CATS}
        self._seg_rpe_sum = {c: 0.0 for c in self.CATS}
        # 【2026-08-05追記】既存の上3行は変えていない。
        if getattr(self, "_world_enabled", False):
            self._seg_world_count = {c: 0 for c in self.WORLD_CATS}

    def on_step_late(self, ctx):
        # on_step_late は rew・rpe が確定した"後"に呼ばれる（base.Plugin docstring参照）。
        #   ctx.last_reward が無い＝古い trainer（この変更が入る前）で呼ばれた場合の
        #   安全弁。通常は毎tick必ず存在する。
        lr = getattr(ctx, "last_reward", None)
        if lr is None:
            return
        rew, rpe = lr["rew"], lr["rpe"]
        # 【2026-08-05】頭・胸（touched_groupsの各部位）とも on_step で自前判定した
        #   self._last_hit_parts を使う（以前は頭だけ ctx.last_double_touch 経由
        #   だったが、reach_selfのときしか計算されないため統一した）。
        hit_parts = self._last_hit_parts
        cat = "self" if hit_parts else "none"
        self.steps += 1
        self._seg_steps += 1
        for d in (self._seg_count, self.total_count):
            d[cat] += 1
        for d in (self._seg_rew_sum, self.total_rew_sum):
            d[cat] += float(rew)
        for d in (self._seg_rpe_sum, self.total_rpe_sum):
            d[cat] += float(rpe)
        if cat == "self" and self.self_trace_out:
            # part列：発火した部位名を"+"区切りで連結する。touched_groupsが
            #   3部位以上でも動く汎用の書き方（"head"固定にしない、仕様3節）。
            self.self_trace_rows.append({
                "step": ctx.step,
                "pe_fast": round(float(lr["pe_fast"]), 6),
                "pe_slow": round(float(lr["pe_slow"]), 6),
                "progress_core": round(float(lr["progress_core"]), 6),
                "surprise_trace": round(float(lr["surprise_trace"]), 6),
                "progress": round(float(lr["progress"]), 6),
                "rew": round(float(rew), 6),
                "rpe": round(float(rpe), 6),
                "part": "+".join(hit_parts),
            })
        # 【2026-08-05追記：4区分（self_and_ext/self_only/ext_only/neither）】
        #   既存の上のself/none判定・集計は一切変えていない。ここは別カウンタ
        #   （self.WORLD_CATS・self.total_world_count）への追記のみ。
        if self._world_enabled:
            ext = bool(self._last_world_hit_pairs)
            is_self = bool(hit_parts)
            if is_self and ext:
                wcat = "self_and_ext"
            elif is_self:
                wcat = "self_only"
            elif ext:
                wcat = "ext_only"
            else:
                wcat = "neither"
            for d in (self._seg_world_count, self.total_world_count):
                d[wcat] += 1
            if ext:
                # 「どの外界物体に何回触れたか」の内訳。同じtickに複数点で
                #   同じ物体へ触れても、その物体は1回として数える
                #   （物体単位の"触れたtick数"として扱う。指と接触点を分けて
                #   数えると常時接触に近い部位で数値が意味を失うため）。
                ext_names_this_tick = {
                    self._world_target_geoms.get(extg, f"geom{extg}")
                    for _tg, extg in self._last_world_hit_pairs}
                for nm in ext_names_this_tick:
                    self.total_world_breakdown[nm] = self.total_world_breakdown.get(nm, 0) + 1

    def metrics(self, ctx):
        if not self._seg_steps:
            return None
        out = {}
        for c in self.CATS:
            n = self._seg_count[c]
            out[f"creward_{c}_n"] = n
            out[f"creward_{c}_rew_mean"] = round(self._seg_rew_sum[c] / n, 6) if n else ""
            out[f"creward_{c}_rpe_mean"] = round(self._seg_rpe_sum[c] / n, 6) if n else ""
        # 【2026-08-05追記】既存の上のcreward_*キーは1つも変えていない。新しい
        #   接頭辞（creward_world_*）にして既存キーと被らないようにする。
        if self._world_enabled:
            for c in self.WORLD_CATS:
                out[f"creward_world_{c}_n"] = self._seg_world_count[c]
        self._seg_reset()
        return out

    def line(self, ctx):
        if not self.steps:
            return None
        n_self = self.total_count["self"]
        return f"creward_self={n_self}/{self.steps}"

    def report(self, ctx):
        if not self.steps:
            return None
        out = {
            "測ったステップ数": self.steps,
            "分類": list(self.CATS),
            "回数_通し": dict(self.total_count),
            "toucher": self.toucher,
            "touched_groups": list(self.touched_groups),
        }
        for c in self.CATS:
            n = self.total_count[c]
            out[f"rew平均_通し_{c}"] = round(self.total_rew_sum[c] / n, 6) if n else "未達成"
            out[f"rpe平均_通し_{c}"] = round(self.total_rpe_sum[c] / n, 6) if n else "未達成"
        out["注意"] = (
            "分類は self（toucher と touched_groups のいずれかとの接触が起きたtick）／"
            "none（それ以外の全て）の2区分のみ。おもちゃ接触は比較対象にしていない"
            "（reach_selfにはおもちゃへ向かう動機が組み込まれておらず、"
            "同じシーンでの既存ログ14本すべてでおもちゃ接触が0回だったため、"
            "設計の整理で比較対象から外した。これはreach_self以外の実験には"
            "当てはまらない可能性があるので、新しい用途では別途確認すること）。")
        out["注意_接触頻度について"] = (
            "touched_groupsのいずれかへの接触が一部のランで6000tick中35%程度"
            "（胸の場合で実測2106/6000）に達するほど頻発することがあり、"
            "progress報酬が前提とする「レアな驚き」という問題設定から外れる"
            "可能性がある。目安として接触tickの比率が10%を超えるシードは"
            "比較対象から除外する候補とし、実際の比率は実行後のCSVから計算してから"
            "最終的な線引きを決めること。この判定自体は実装段階の仕事ではなく、"
            "実行後の分析段階で行う。")
        if self.self_trace_out and self.self_trace_rows:
            path = self.self_trace_out if os.path.isabs(self.self_trace_out) else os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            os.pardir, os.pardir, os.pardir)), self.self_trace_out)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            cols = ["step", "pe_fast", "pe_slow", "progress_core",
                    "surprise_trace", "progress", "rew", "rpe", "part"]
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(cols)
                for r in self.self_trace_rows:
                    w.writerow([r.get(c, "") for c in cols])
            out["self_trace_出力"] = {"件数": len(self.self_trace_rows), "パス": self.self_trace_out}
        # 【2026-08-05追記：外界接触（world_targets）の内訳】既存の出力キーは
        #   1行も変えていない。世界接触分は別のキー名でぶら下げる。
        if self._world_enabled:
            out["world_targets"] = list(self._world_targets)
            out["world_toucher_bodies"] = list(self._world_toucher_bodies)
            out["回数_通し_world4区分"] = dict(self.total_world_count)
            out["外界物体ごとの接触tick数"] = dict(self.total_world_breakdown)
            out["注意_world区分について"] = (
                "self_and_ext/self_only/ext_only/neitherの4区分は、既存のself/none"
                "（presenceしきい値による近似判定）と、外界接触（MuJoCoの実際の"
                "接触ペアdata.contactによる正確な判定）を組み合わせたもの。"
                "自己接触側(self)は引き続きTier3近似のままで、今回の実装では"
                "精度を上げていない（設計3節の申し送り）。")
        return out
