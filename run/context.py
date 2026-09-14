"""実行中の「いま」をまとめた入れ物。プラグインはこれだけを受け取る。

【なぜ入れ物にするか、2026-07-30】プラグインが env や太郎の中身へ勝手に手を伸ばすと、
「測る道具が太郎を変えてしまう」事故が起きる（＝観測装置が対象を変える）。
渡すものを1つに絞り、**読むだけ**という約束を明示する。

注意：プラグインは ctx の中身を**書き換えない**。測る・見るだけ。
  太郎を変える設定は実験ファイルの `taro` 欄が担う。
"""


class Ctx:
    """実行中の状態。プラグインへ渡す唯一の入れ物。

    属性:
        env      : 環境（HybridEnv 等）。model/data は env.unwrapped から
        model    : MuJoCo の model（よく使うので取り出してある）
        data     : MuJoCo の data
        brain    : 太郎の脳（無い実験＝measure のみ なら None）
        spec     : 実験ファイル（辞書）そのもの
        scene    : シーンの辞書
        step     : いまのステップ数
        n_steps  : 全体のステップ数
        dt       : 1ステップの秒数（K × DT）
        log      : 追記用の関数 log(行の辞書)
    """

    def __init__(self, *, env, spec, scene, n_steps, dt, brain=None, log=None):
        self.env = env
        u = env.unwrapped
        self.model = u.model
        self.data = u.data
        self.brain = brain
        self.spec = spec
        self.scene = scene
        self.step = 0
        self.n_steps = n_steps
        self.dt = dt
        self._log = log or (lambda row: None)
        # 【なぜ、2026-08-15】プラグインが「学習を打ち切りたい」と伝える手段が
        #   無かった（学習ループには早期終了の仕組みが無かった）。
        #   停止シグナルの土台として2つの属性を追加する（run/trainer.py が
        #   on_checkpointの直後にstop_requestedを見てbreakする）。
        #   注意：既存のどのプラグインもこの属性を読み書きしていないので、
        #   既存挙動は変わらない（仕様「作るもの」1節）。
        self.stop_requested = False
        self.stop_reason = None


    # 【段B-2a・2026-09-13】太郎が自分の作業台（taro.tick）へ書いた値を、
    #   道具からは今までどおり `ctx.<名前>` で読めるようにする。
    #   **写さない＝参照する。** 写すと、写した瞬間と道具が読む瞬間がずれて
    #   1周期古い値を読むことになる（道具は tick の途中で読むため）。
    #   自分（Ctx）に直接置かれた値があればそちらが勝つ＝
    #   `taro.visual_attention` を使わない古い実験（道具が自分で作って
    #   ctx へ置く経路）もそのまま動く。
    _BRAIN_KEYS = frozenset(['attended_object', 'attention_point', 'attention_switch_t', 'efference', 'goal_point', 'last_babble', 'last_produce', 'last_vision_vec', 'last_world_pred', 'object_files', 'priority_map_result', 'priority_map_result_t', 'salience_map', 'surprise_trace', 'vanish_misses', 'world_pred_by_file', 'world_pred_inputs', 'world_predictor_grad_report'])

    def __getattr__(self, name):
        # 通常の属性探索が失敗したときだけ呼ばれる
        if name in Ctx._BRAIN_KEYS:
            taro = self.__dict__.get("taro")          # 再帰を避けるため辞書から直接
            tick = getattr(taro, "tick", None) if taro is not None else None
            if tick is not None:
                try:
                    return getattr(tick, name)
                except AttributeError:
                    pass
        raise AttributeError(name)

    def log(self, row):
        self._log(row)

    @property
    def sim_sec(self):
        """シミュレーション上の経過秒数。"""
        return self.step * self.dt
