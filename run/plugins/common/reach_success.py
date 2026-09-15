"""自己接触が実際に起きているか（頭・胸・反対の手に触れたか）を直接測る。

【なぜ要るか、2026-08-02】手先位置の目標表現（案C・reach_self、
`E/docs/リーチング/` 設計）は4条件×2シードで完走したが、比較に使った指標
（classify・margin・corr・persist）は**自己モデルの学習具合を測る間接指標**で、
「実際に頭・胸・反対の手に触れているか」を一度も直接測っていなかった。
B（新実装）とD（陰性対照）にこの間接指標では差が出なかったが、
それが「自己接触が創発していない」ことを意味するとは限らない
（間接指標の感度が足りないだけの可能性がある）。この道具は物理的な接触の
有無を直接数える。仕様：
作業記録（非公開）

【何を測るか・二重実装しない】
`encode_reach_goal`（`run/taro_setup.py`）が目標の触覚成分を作るのに使っているのと
**同じ**部位グループ（`taro.reach_touch_groups` ＝ head・chest・opposite_palm）・
**同じ**触覚経路（`taro.target_fusion.touch`）を、ここでも再利用する。
部位のインデックス構築（TouchMap・SomatosensoryCortex）は一切ここに書かない。

`target_fusion.touch` は RND式の「凍結した別インスタンス」（encode_target・
encode_reach_goal と共通の設計判断）で、**勾配を一切受け取らず、学習中ずっと
初期値のまま**。そのため `presence_gain` は常に1.0固定＝この道具が返す
「触れたかどうか」は**学習の進み具合に左右されない**安定した物差しになる
（落とし穴チェックリスト 項19「関門は学習に依存する量で測らない」を満たす）。

【接触の判定】
`SomatosensoryCortex.part_features()` の①有無
（presence = tanh(部位内で最も強い点の圧 × presence_gain)）を使う。
presence は [0,1) の連続値。「触れた」と判定するしきい値は既定 0.5
（[Tier3・工学的判断、根拠となる文献値は無い]）。
しきい値に張り付いていないか後から確認できるよう、**しきい値で決めた
0/1の回数だけでなく、presence の生の値（区間内の平均・最大）も一緒に
CSVへ記録する**（movement_units.py と同じ考え方。落とし穴チェックリスト
項41「閾値は文献で確認する」・項17「配線チェック」を参照）。

実験ファイルでの書き方（taro.goal_babbling=true, taro.goal_space="reach_self" が前提）:
    "plugins": {"reach_success": true}
    "plugins": {"reach_success": {"threshold": 0.5}}

注意：run.type=train で goal_space=reach_self のときだけ使える
（自己接触を目標にしていない実験・measure/viewでは使えない。setup で明確に止める。
 self_model.py と同じ「太郎の脳が要る」の考え方を踏襲）。
"""
import torch

from run.plugins.base import Plugin
from run.taro_setup import to_tensor

# 「触れた」と判定する presence のしきい値。[Tier3・工学的判断]
#   presence = tanh(peak) なので 0.5 は peak≈0.55 に相当する適当な中間値。
#   文献的な根拠は無い。しきい値に依存しない生の presence 値も必ず一緒に
#   記録するので、あとから振り直せる（movement_units.py の min_gap_ms と同じ流儀）。
DEFAULT_THRESHOLD = 0.5


class ReachSuccess(Plugin):
    name = "reach_success"

    def setup(self, ctx):
        """ctx.brain または taro.reach_touch_groups が無ければ例外を出す。しきい値と対象部位（taro.reach_touch_groups）を決めて部位の存在を確認し、通し・区間の集計カウンタを初期化する。戻り値は無い。
        """
        if ctx.brain is None:
            raise ValueError(
                "reach_success は太郎の脳が要る（run.type=train で使う）。\n"
                "  measure（脳を通さず環境だけ進める）では測れない")
        taro = getattr(ctx, "taro", None)
        if taro is None or getattr(taro, "reach_touch_groups", None) is None:
            raise ValueError(
                "reach_success は自己接触を目標にする設定が要る"
                "（taro.goal_babbling=true, taro.goal_space=\"reach_self\"）。\n"
                "  いまの実験ファイルはこの設定になっていない（reach_touch_groups が無い）")
        self.taro = taro
        self.threshold = float(self.config.get("threshold", DEFAULT_THRESHOLD))
        self.groups = list(taro.reach_touch_groups)     # 例: ["head","chest","left_palm"]
        self._check_groups()
        self.steps = 0
        # 通し（学習全体）の集計
        self.total_touches = {nm: 0 for nm in self.groups}
        self._touched_ever = {nm: False for nm in self.groups}
        self.all3_step = None      # 初めて全部位に触れたステップ。まだなら None
        # 区間（記録の区切り）の集計
        self._seg_steps = 0
        self._seg_reset()

    def _check_groups(self):
        """対象部位が触覚の地図に存在するかを確認する（taro.on_body_change と独立に）。"""
        names = self.taro.target_fusion.touch.group_names
        missing = [nm for nm in self.groups if nm not in names]
        if missing:
            raise AssertionError(
                f"reach_success の対象部位が触覚の地図に無い: {missing}\n"
                f"  いまの部位: {names}")

    def on_body_change(self, ctx):
        # 【なぜ、2026-08-02】体を作り直すと TouchMap（点→部位の対応）が rebuild
        #   される。encode_reach_goal と同じく部位は**毎回名前で検索**するので
        #   索引はキャッシュしていないが、対象部位そのものが消えていないかは
        #   taro 側（`Taro.on_body_change`）とは独立にここでも確認する
        #   （落とし穴チェックリスト 項86「配列が長くなる方向の変化は例外にならない」）。
        """体を作り直した直後に呼ばれる。対象部位が触覚の地図にまだ存在するかを再確認する。戻り値は無い。"""
        self._check_groups()

    def _seg_reset(self):
        self.seg_touches = {nm: 0 for nm in self.groups}
        self._presence_sum = {nm: 0.0 for nm in self.groups}
        self._presence_max = {nm: 0.0 for nm in self.groups}

    def on_step(self, ctx):
        """毎ステップ呼ばれる。触覚観測から対象各部位のpresenceを計算し、しきい値を超えた部位を「触れた」として通し・区間のカウンタに加算し、全部位に初めて触れたステップを記録する。戻り値は無い。
        """
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
        for nm in self.groups:
            gi = names.index(nm)     # 毎回名前で検索する（encode_reach_goal と同じ流儀）
            presence = float(feat[gi, 0])
            self._presence_sum[nm] += presence
            if presence > self._presence_max[nm]:
                self._presence_max[nm] = presence
            if presence > self.threshold:
                self.total_touches[nm] += 1
                self.seg_touches[nm] += 1
                self._touched_ever[nm] = True
        if self.all3_step is None and all(self._touched_ever[nm] for nm in self.groups):
            self.all3_step = self.steps

    def metrics(self, ctx):
        """区間内にステップが無ければ None を返す。部位ごとの接触回数・presence平均・最大、触れた部位数をまとめた辞書を返し、区間の集計をリセットする。"""
        if not self._seg_steps:
            return None
        out = {}
        for nm in self.groups:
            out[f"reach_touch_{nm}"] = self.seg_touches[nm]
            out[f"reach_presence_mean_{nm}"] = round(self._presence_sum[nm] / self._seg_steps, 4)
            out[f"reach_presence_max_{nm}"] = round(self._presence_max[nm], 4)
        out["reach_parts_covered"] = sum(1 for nm in self.groups if self.seg_touches[nm] > 0)
        # 「まだ達成していない」ときはキー自体を出さない → CSVでは空欄になる
        #   （run/main.py の _csv_logger が未出現のキーを "" で埋める仕組みを使う）
        if self.all3_step is not None:
            out["reach_all3_step"] = self.all3_step
        self._seg_reset()
        return out

    def line(self, ctx):
        """まだ測定していなければ None を返す。触れた部位数と対象部位総数を「reach=N/M部位」という短い文字列で返す。"""
        if not self.steps:
            return None
        covered = sum(1 for nm in self.groups if self.total_touches[nm] > 0)
        return f"reach={covered}/{len(self.groups)}部位"

    def report(self, ctx):
        """まだ測定していなければ None を返す。対象部位、部位別の接触回数と1分あたり回数、全部位に初めて触れたステップ、しきい値をまとめた辞書を返す。"""
        if not self.steps:
            return None
        sec = self.steps * ctx.dt
        per_min = {nm: round(60.0 * self.total_touches[nm] / sec, 4) if sec > 0 else 0.0
                   for nm in self.groups}
        return {
            "対象部位": self.groups,
            "触れた回数_通し": dict(self.total_touches),
            "1分あたり回数": per_min,
            "何部位に触れたか_通し": sum(1 for nm in self.groups if self.total_touches[nm] > 0),
            "初めて全部位に触れたステップ": (self.all3_step if self.all3_step is not None
                                       else "未達成"),
            "測ったステップ数": self.steps,
            "しきい値": self.threshold,
            "注意": ("presence(生の値)の平均・最大はCSVの reach_presence_mean_* / "
                    "reach_presence_max_* 列に区間ごと記録済み。しきい値を振り直すのに使える"),
        }
