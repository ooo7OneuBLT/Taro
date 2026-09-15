"""ダブルタッチ（自己接触の一致）を検出する ── 触れた側・触れられた側の
両方から**同時に**触覚情報が来る、という入力パターンを直接測る。

【なぜ要るか、2026-08-02】手先位置の目標表現（案C・reach_self）はB（本物の自己接触
目標）とD（陰性対照）で実際の接触回数に差が出なかった（`E/logs/goal_stage1_v2/`
の6実験）。原因を調べたところ、Taroの現在の報酬（`taro_core/src/brain/dopamine.py`
へ渡す `progress`、`run/trainer.py:708-719`）は視覚・固有感覚・触覚を**すべて1つの
スカラーに混ぜた**ものであり、「自分に触れたこと」に選択的に反応する経路が
存在しないと判明した。仕様：
作業記録（非公開）

【文献調査の結論】（`doc/文献調査/リーチング/自己接触の報酬性_2026-08-02.md`）
「自己接触は他者接触より感覚的な報酬・衝撃が大きい」という主張は一次資料では
支持されない。支持されるのはもっと弱い主張——「自己接触（ダブルタッチ：触れた側・
触れられた側の両方から同時に触覚情報が来る）は、外部の物への接触（片方からしか
情報が来ない）と**質的に異なる入力パターンである**」という点のみ。

方針：「自己接触に高い報酬を与える」設計ではなく、「ダブルタッチという入力パターン
を検出できる仕組みを、まず作る」。**この道具は検出だけを行う。報酬には一切繋がない**
（`run/trainer.py` の reward計算・`dopamine.py`・`learning_progress.py` は変更しない）。

【何を測るか・二重実装しない】
`reach_success.py` と全く同じ作り・同じ経路を踏襲する。

    触れている側（toucher） = reach_arm_side の手のひらグループ
        例：cfg.reach_arm_side == "right" のとき "right_palm"
    触れられる側（touched） = taro.reach_touch_groups（既存。例 ["head","chest","left_palm"]）

    ダブルタッチ = toucher の presence > しきい値  かつ
                  touched のいずれか1つ以上の presence > しきい値
                  （同じtickで両方成立したとき）

presence の取り方・しきい値の考え方は reach_success.py と同一
（`taro.target_fusion.touch.part_features()` の①有無、`group_names` で毎回名前検索、
既定しきい値0.5・`config`で上書き可）。`target_fusion.touch` は RND式の凍結インスタンス
（勾配を受け取らず学習中ずっと初期値のまま）＝ここで測る presence は**学習の進み具合に
左右されない**安定した物差しになる（落とし穴チェックリスト 項19）。

【重要な限界・2026-08-02、Tier3・工学的近似（根拠となる文献値は無い）】
`SomatosensoryCortex.part_features()` は接触の**発生源**（相手が何か）を区別しない
（`doc/文献調査/リーチング/自己接触の報酬性_2026-08-02.md` 6節、reach_success.py の
chest confound と同じ限界）。つまりこの検出器の「一致」は、

    「toucherとtouchedが互いに接触した」という保証ではなく、
    「toucherとtouchedが、たまたま同じtickで別々に何かに触れていた」可能性を
    排除できない（例：右手がおもちゃに触れている間に、頭が別に座面や環境に
    触れている、等）

という**近似**である。しきい値に依存しない生の presence 値（区間内の平均・最大）を
毎回CSVへ一緒に記録し、この近似の妥当性をあとから確認できるようにしてある。
この逸脱は `doc/人間模倣からの逸脱リスト.md` にも追記済み（本文③参照）。

実験ファイルでの書き方（taro.goal_babbling=true, taro.goal_space="reach_self" が前提）:
    "plugins": {"double_touch": true}
    "plugins": {"double_touch": {"threshold": 0.5}}

注意：run.type=train で goal_space=reach_self のときだけ使える
（reach_success.py と同じ「太郎の脳が要る」「reach_touch_groups が要る」の考え方）。
"""
import torch

from run.plugins.base import Plugin
from run.taro_setup import to_tensor

# 「触れた」と判定する presence のしきい値。[Tier3・工学的判断]
#   reach_success.py と同じ値・同じ考え方（presence = tanh(peak) なので 0.5 は
#   peak≈0.55 に相当する適当な中間値。文献的な根拠は無い）。
#   しきい値に依存しない生の presence 値も必ず一緒に記録するので、あとから振り直せる。
DEFAULT_THRESHOLD = 0.5


class DoubleTouch(Plugin):
    name = "double_touch"

    def setup(self, ctx):
        if ctx.brain is None:
            raise ValueError(
                "double_touch は太郎の脳が要る（run.type=train で使う）。\n"
                "  measure（脳を通さず環境だけ進める）では測れない")
        taro = getattr(ctx, "taro", None)
        if taro is None or getattr(taro, "reach_touch_groups", None) is None:
            raise ValueError(
                "double_touch は自己接触を目標にする設定が要る"
                "（taro.goal_babbling=true, taro.goal_space=\"reach_self\"）。\n"
                "  いまの実験ファイルはこの設定になっていない（reach_touch_groups が無い）")
        self.taro = taro
        self.threshold = float(self.config.get("threshold", DEFAULT_THRESHOLD))
        # toucher（触れている側）＝腕側の手のひら。taro.cfg は Taro.__init__ で
        #   保持されている実験設定そのもの（run/taro_setup.py 83行）。
        self.toucher = f"{taro.cfg.reach_arm_side}_palm"
        self.touched = list(taro.reach_touch_groups)     # 例: ["head","chest","left_palm"]
        self._check_groups()
        self.steps = 0
        # 通し（学習全体）の集計
        self.total_double = {nm: 0 for nm in self.touched}
        self._double_ever = {nm: False for nm in self.touched}
        self.first_double_step = None      # 初めてどれか1つとダブルタッチしたステップ
        # 区間（記録の区切り）の集計
        self._seg_reset()

    def _check_groups(self):
        """toucher・touched の対象部位が触覚の地図に存在するかを確認する
        （taro.on_body_change と独立に。reach_success._check_groups と同じ流儀）。
        """
        names = self.taro.target_fusion.touch.group_names
        missing = [nm for nm in [self.toucher, *self.touched] if nm not in names]
        if missing:
            raise AssertionError(
                f"double_touch の対象部位が触覚の地図に無い: {missing}\n"
                f"  いまの部位: {names}")

    def on_body_change(self, ctx):
        # 【なぜ、2026-08-02】体を作り直すと TouchMap（点→部位の対応）が rebuild
        #   される。reach_success.on_body_change と同じ理由でここでも確認し直す
        #   （落とし穴チェックリスト 項86「配列が長くなる方向の変化は例外にならない」）。
        self._check_groups()

    def _seg_reset(self):
        self._seg_steps = 0
        self.seg_double = {nm: 0 for nm in self.touched}
        self._toucher_presence_sum = 0.0
        self._toucher_presence_max = 0.0
        self._presence_sum = {nm: 0.0 for nm in self.touched}
        self._presence_max = {nm: 0.0 for nm in self.touched}

    def on_step(self, ctx):
        last = getattr(ctx, "last", None)
        if last is None:
            return
        obs = last["obs_out"]
        touch = to_tensor(obs["touch"])
        with torch.no_grad():
            feat = self.taro.target_fusion.touch.part_features(touch)     # (G, 5)
        names = self.taro.target_fusion.touch.group_names
        self.steps += 1
        self._seg_steps += 1

        ti = names.index(self.toucher)     # 毎回名前で検索する（reach_success と同じ流儀）
        toucher_presence = float(feat[ti, 0])
        self._toucher_presence_sum += toucher_presence
        if toucher_presence > self._toucher_presence_max:
            self._toucher_presence_max = toucher_presence
        toucher_touching = toucher_presence > self.threshold

        for nm in self.touched:
            gi = names.index(nm)
            presence = float(feat[gi, 0])
            self._presence_sum[nm] += presence
            if presence > self._presence_max[nm]:
                self._presence_max[nm] = presence
            # ダブルタッチ＝同じtickで toucher・touched の両方が閾値を超えたとき
            if toucher_touching and presence > self.threshold:
                self.total_double[nm] += 1
                self.seg_double[nm] += 1
                self._double_ever[nm] = True
        if self.first_double_step is None and any(self._double_ever[nm] for nm in self.touched):
            self.first_double_step = self.steps

    def metrics(self, ctx):
        if not self._seg_steps:
            return None
        out = {
            f"dtouch_toucher_{self.toucher}_presence_mean":
                round(self._toucher_presence_sum / self._seg_steps, 4),
            f"dtouch_toucher_{self.toucher}_presence_max":
                round(self._toucher_presence_max, 4),
        }
        for nm in self.touched:
            out[f"dtouch_count_{nm}"] = self.seg_double[nm]
            # touched側 presence の生の値。reach_success.py の reach_presence_* 列と
            #   重複してよい（この道具単体でも中身が追えるようにするため、仕様2節）。
            out[f"dtouch_presence_mean_{nm}"] = round(self._presence_sum[nm] / self._seg_steps, 4)
            out[f"dtouch_presence_max_{nm}"] = round(self._presence_max[nm], 4)
        out["dtouch_parts_hit"] = sum(1 for nm in self.touched if self.seg_double[nm] > 0)
        # 「まだ達成していない」ときはキー自体を出さない → CSVでは空欄になる
        #   （run/main.py の _csv_logger が未出現のキーを "" で埋める仕組みを使う）
        if self.first_double_step is not None:
            out["dtouch_first_step"] = self.first_double_step
        self._seg_reset()
        return out

    def line(self, ctx):
        if not self.steps:
            return None
        hit = sum(1 for nm in self.touched if self.total_double[nm] > 0)
        return f"dtouch={hit}/{len(self.touched)}部位"

    def report(self, ctx):
        if not self.steps:
            return None
        sec = self.steps * ctx.dt
        per_min = {nm: round(60.0 * self.total_double[nm] / sec, 4) if sec > 0 else 0.0
                   for nm in self.touched}
        return {
            "toucher（触れている側）": self.toucher,
            "touched（触れられる側候補）": self.touched,
            "ダブルタッチ回数_通し_部位別": dict(self.total_double),
            "1分あたり回数_部位別": per_min,
            "何部位とダブルタッチしたか_通し": sum(1 for nm in self.touched
                                          if self.total_double[nm] > 0),
            "初めてダブルタッチしたステップ": (self.first_double_step
                                       if self.first_double_step is not None else "未達成"),
            "測ったステップ数": self.steps,
            "しきい値": self.threshold,
            "限界（Tier3・工学的近似）": (
                "SomatosensoryCortexのpart_featuresは接触の発生源を区別しない。"
                "この「一致」は「toucherとtouchedが互いに接触した」保証ではなく、"
                "「同じtickで別々に何かへ触れていた」可能性を排除できない近似"
                "（reach_success.pyのchest confoundと同じ限界）。"),
            "注意": ("presence(生の値)の平均・最大はCSVの dtouch_presence_mean_* / "
                    "dtouch_presence_max_* 列と dtouch_toucher_*_presence_* 列に"
                    "区間ごと記録済み。しきい値を振り直すのに使える"),
        }
