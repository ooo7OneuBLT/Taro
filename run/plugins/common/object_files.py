# -*- coding: utf-8 -*-
"""測る道具（兼・配線係）：MobileSAMの物体切り出し（[既製部品] 逸脱その42・登録済み）
とDINOv2（[既製部品] 視覚の既製部品）で見た物を、物体ファイル
（`taro_core/src/brain/cerebral_cortex/parietal_lobe/intraparietal_sulcus.py` の
ObjectFileSystem）に渡して追跡を更新する、**本番の走行で初めて動かす**道具。

【なぜ、2026-09-04】太郎はMobileSAMによる物の切り出しと、物体ファイルによる
「同じ物が動いている」追跡の両方をすでに持っているが、どちらも書き捨ての
検証スクリプト（`F/scripts/f65_mobilesam_detect_probe.py` 等）でしか動かした
ことが無く、太郎が実際に学習している最中には一度も働いていない。
仕様：`F/docs/物体ファイルと注意/仕様_物体ファイルプラグイン接続_2026-09-04.md`。

【役割】観測はするが、太郎の体・脳・環境は変えない（`run/plugins/base.py` の規約）。
  唯一の例外は `ctx.object_files` を新しく置くこと（他のプラグイン・今後の
  脳側の配線がここから読めるようにするための「置き場」であり、太郎を書き換える
  行為ではない）。

【読むもの】
  ctx.data.time          … sim時刻（`interval_s` ごとに検出するかの判定に使う）
  ctx.last["obs_out"]    … 直近の観測。`eye_left`（全視野・視野60度）を使う。
                           `eye_left_fovea`（視野15度）は複数物体が入らないため使わない。

【今回やらないこと】物体ファイルの中身を太郎の脳の入力に繋ぐことは範囲外
  （2語文の設計で決める）。ここは「動いていて、観測できる」までが範囲。

【config（実験ファイルの `plugins.object_files` に書く。既定値付き）】
    interval_s      : 1.0   # sim時間で何秒ごとに検出するか（制御周期1秒に合わせた既定）
    events_out      : None  # CSVの出力先（相対パスはリポジトリルート基準。Noneなら書かない）
    device          : None  # None→環境変数 TARO_VISION_DEVICE→cuda→cpu の順（load_mobilesamの既定に合わせる）
    points_per_side : None  # MobileSAMの自動マスク生成の点の密度。Noneなら load_mobilesam の既定。速度が足りなければ下げる

【events_out CSVの列】
    step, sim_time, n_dets, n_files, file_id, x, y, area, event,
    misses, since_seen, app_cos_created
    物体ファイル1つにつき1行。event は matched/created/lost/unmatched のいずれか。
    物体ゼロ（このステップで検出も既存の追跡イベントも無い）のときは file_id 空で1行だけ書く。

【frames_out（config、任意。既定None＝描画しない）】
    設定すると、検出を行った各コマ（interval_sごと）で `img224`（検出器に渡した
    224x224画像）の上に検出（黄丸）と物体ファイル（色つき四角＋文字）を重ねて
    PNGへ保存する（`frame_<step:05d>.png`）。目視用（動画は
    `F/scripts/f73b_stitch_detection_frames.py` で作る）。既存のCSV・検出・
    対応づけロジックには一切触れない（描画は読むだけ）。
    仕様：`F/docs/二語文/仕様_M1.5b_視界動画_検出と物体ファイルの重ね描き_2026-09-05.md`。
    色：matched=緑、unmatched=灰、created=青、lost=赤。

【なぜ、2026-09-05・M1.5持続確認】以前は matched/created/lost の3種しか行が出ず、
    「対応がつかず、まだ削除されていない」物体ファイル（隠されている間の状態そのもの）が
    ログに1行も残らなかった。仕様：F/docs/二語文/仕様_M1.5_物体ファイルの持続確認_
    2026-09-05.md。追加した3列と `unmatched` イベントはロジックを一切変えず、
    ObjectFileSystemが既に持っている値（misses・since_seen）と、プラグイン側で
    新たに保持する「作成時のappearance」との比較（app_cos_created）を読むだけ。
    既存の列順は変えない（末尾に追加）ので、F2-67系の解析道具はそのまま読める。

【なぜ、2026-09-06・M3 注意中の物体ファイルと消失信号】太郎の側に「○○が消えた」
    という状態を作る区切り。仕様：F/docs/二語文/仕様_M3_注意中の物体ファイルと
    消失信号_2026-09-06.md。追加した config キー（`attend` 既定False）は検出・
    対応づけロジック（ofs.step等）には一切触れず、その結果（f.pos/f.area/f.misses）を
    読むだけ。既定Falseでは新設メソッドが1つも呼ばれず、既存の挙動・出力は不変。

【attend用config（既定は全てOFF相当で既存挙動は不変）】
    attend           : false   # true で「注意中のファイル」の選択と消失信号を有効化
    attend_radius    : 56      # 画像中央(112,112)からこの距離(px)以内を「中央付近」とみなす
    vanish_misses    : 1       # misses がこの値以上で「消えた」
    attend_out       : None    # 注意.csv の出力先（相対パスはリポジトリルート基準）

【注意.csvの列】
    step, sim_time, attended_id, visible, misses, vanished, dist_center, area,
    nearest_word, nearest_cos, target_word, target_cos
    検出コマ（interval_sごと）に1行。nearest_word/target_wordは
    ctx.taro.lexicon.proto（DINOv2の見た目ベクトル→語のプロトタイプ）との
    コサイン類似度で決める（word_similarity_map.pyの手本と同じ読み方）。

【ctx.attended_object（辞書 or None）】
    setupで None を置く（ctx.object_files と同じ流儀）。attend=Trueのとき検出コマ
    ごとに更新：
    {"file_id", "visible", "misses", "since_seen", "pos", "area", "vanished",
     "last_seen_vec", "last_seen_time", "t"}
    注意中のファイルが無ければ（一度も注意していない／lostで削除された）None。
"""
import csv
import os
import sys
import time

from run.plugins.base import Plugin

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir))


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(_REPO_ROOT, p)


class ObjectFiles(Plugin):
    name = "object_files"

    def setup(self, ctx):
        self.interval_s = float(self.config.get("interval_s", 1.0))
        events_out = self.config.get("events_out")
        self.events_out = _abs_path(events_out) if events_out else None
        self.device = self.config.get("device")
        self.points_per_side = self.config.get("points_per_side")
        # 【2026-09-09・仕様_物体ファイルの人間寄せ_凝集性・連続性・上限】
        #   既定は全てNone（凝集性・連続性・上限とも従来どおりOFF）＝既定不変。
        self.cohesion_gap_px = self.config.get("cohesion_gap_px")
        revive_window_s = self.config.get("revive_window_s")
        self.revive_window_s = None if revive_window_s is None else float(revive_window_s)
        self.revive_cos = float(self.config.get("revive_cos", 0.8))
        self.revive_dist_px = float(self.config.get("revive_dist_px", 40.0))
        max_files = self.config.get("max_files")
        self.max_files = None if max_files is None else int(max_files)
        max_missed = self.config.get("max_missed")
        self.max_missed = None if max_missed is None else int(max_missed)
        # 【2026-09-09・仕様_物体ファイルの重複をなくす「後半」4節】既定は全て
        #   ObjectFileSystemの既定値と同じ（coast_max_s=None・uncertainty_penalty=0.0・
        #   exclusive_dist_px=None・exclusive_cos=0.7）＝既定不変。
        coast_max_s = self.config.get("coast_max_s")
        self.coast_max_s = None if coast_max_s is None else float(coast_max_s)
        self.uncertainty_penalty = float(self.config.get("uncertainty_penalty", 0.0))
        exclusive_dist_px = self.config.get("exclusive_dist_px")
        self.exclusive_dist_px = None if exclusive_dist_px is None else float(exclusive_dist_px)
        # 【2026-09-09・追記_物体ファイルの重複をなくす「追記」1節】JSONの
        #   null（＝Python側のNone）がそのまま届く。既定は0.7（キー省略時のみ）。
        #   明示的にnullを渡せば見た目の条件を外す（既定不変：キー省略なら従来と
        #   1ビットも変わらない）。
        exclusive_cos = self.config.get("exclusive_cos", 0.7)
        self.exclusive_cos = None if exclusive_cos is None else float(exclusive_cos)
        # 【同「追記」1節】既定False＝従来どおりexclusive_dist_pxそのまま（既定不変）。
        self.exclusive_scale_by_size = bool(self.config.get("exclusive_scale_by_size", False))
        # 【同「追記」2節】既定None＝従来どおり延長条件は速度を見ない（既定不変）。
        coast_min_speed_px_s = self.config.get("coast_min_speed_px_s")
        self.coast_min_speed_px_s = (None if coast_min_speed_px_s is None
                                      else float(coast_min_speed_px_s))
        # 【2026-09-09・仕様_見る側_道を1本にする】確認と切り出しの2本の道を
        #   `_detect_frame` 1本にまとめた（旧「予測して確かめる検出」の見回り
        #   タイマー・確認専用の設定は全て廃止。後方互換の分岐は残さない）。
        self._point_predictor = None
        # 【2026-09-07・メモリ削減候補(a)】既定None＝SamAutomaticMaskGeneratorの
        #   既定値(64)のまま＝1ビットも変わらない。渡したときだけ上書きする。
        self.points_per_batch = self.config.get("points_per_batch")
        # 【2026-09-07・メモリ削減候補(c)】既定False＝従来どおり呼ばない。
        self.empty_cache_after_detect = bool(self.config.get("empty_cache_after_detect", False))
        frames_out = self.config.get("frames_out")
        self.frames_out = _abs_path(frames_out) if frames_out else None
        self._frame_font = None
        if self.frames_out:
            os.makedirs(self.frames_out, exist_ok=True)

        # 【2026-09-09・仕様_見る側の段構成_実装】段0・段1・段6。既定は全てNone
        #   （無効）＝既定不変（仕様書10節「既定不変」）。段4の
        #   appearance_gate_cosもここでまとめて読む。
        # 【2026-09-10・見る側3段目】場所の地図で注意を決め、注意が向いた場所に
        #   だけ記録を作る。attention=None（既定）なら1行も通らない＝既定不変。
        self.attention_cfg = self.config.get("attention")
        # 【2026-09-10・4段目】注意が移った先へ目も向けるか。既定False＝既定不変。
        self.gaze_from_attention = bool(self.config.get("gaze_from_attention", False))
        self._gaze_cmd_sent = 0
        self._sal = None
        self._spri = None
        self.efference_copy_cfg = self.config.get("efference_copy")
        self.preattentive_cfg = self.config.get("preattentive")
        self.priority_cfg = self.config.get("priority")
        appearance_gate_cos = self.config.get("appearance_gate_cos")
        self.appearance_gate_cos = (None if appearance_gate_cos is None
                                     else float(appearance_gate_cos))
        # 【2026-09-09・仕様_見る側_道を1本にする 後半2節】旧・確認専用の位置
        #   ゲート設定を統合した（確認の道自体を削除。ハンガリアン法の門に
        #   一本化）。既定"mahal"＝ObjectFileSystem側の既定と一致するので
        #   無条件で渡してよい（既定不変）。
        self.pos_gate = self.config.get("pos_gate", "mahal")
        self._ec = None
        self._pre = None
        self._priority = None
        self._first_scan_done = False
        self._warned_no_orienting = False
        # 【2026-09-09・仕様_見る側_道を1本にする 後半1節】前コマの探索点
        #   （段3c）。まだ無ければNone。
        self._prev_explore_point = None

        self._lazy_import()

        if self.preattentive_cfg is not None:
            self._pre = self._PreattentiveMap(
                diff_thresh=float(self.preattentive_cfg.get("diff_thresh", 20.0)),
                min_blob_area_px=int(self.preattentive_cfg.get("min_blob_area_px", 30)),
                max_blobs=int(self.preattentive_cfg.get("max_blobs", 8)))
        if self.attention_cfg is not None:
            from brain.midbrain.salience_map import SalienceMap
            from brain.cerebral_cortex.parietal_lobe.spatial_priority_map import (
                SpatialPriorityMap)
            _c = int(self.attention_cfg.get("cell", 28))
            self._sal = SalienceMap(
                cell=_c,
                weights=tuple(self.attention_cfg.get("weights", (1.0, 1.0, 1.0, 1.0))),
                dog_iters=int(self.attention_cfg.get("dog_iters", 10)))
            self._spri = SpatialPriorityMap(
                cell=_c, img_size=224.0,
                ior_tau_s=float(self.attention_cfg.get("ior_tau_s", 0.7)),
                ior_gain=float(self.attention_cfg.get("ior_gain", 1.0)),
                acc_tau_s=float(self.attention_cfg.get("acc_tau_s", 0.11)))
        if self.priority_cfg is not None:
            explore_interval_s = self.priority_cfg.get("explore_interval_s")
            self._priority = self._PriorityMap(
                w_size=float(self.priority_cfg.get("w_size", 1.0)),
                w_center=float(self.priority_cfg.get("w_center", 0.0)),
                w_motion=float(self.priority_cfg.get("w_motion", 0.0)),
                w_z=float(self.priority_cfg.get("w_z", 0.0)),
                w_nov=float(self.priority_cfg.get("w_nov", 0.0)),
                w_ior=float(self.priority_cfg.get("w_ior", 0.0)),
                ior_tau_s=float(self.priority_cfg.get("ior_tau_s", 2.0)),
                w_hab=float(self.priority_cfg.get("w_hab", 0.0)),
                hysteresis=float(self.priority_cfg.get("hysteresis", 0.0)),
                switch_delay_s=float(self.priority_cfg.get("switch_delay_s", 0.0)),
                explore_interval_s=(None if explore_interval_s is None
                                     else float(explore_interval_s)),
                explore_ior_s=float(self.priority_cfg.get("explore_ior_s", 3.0)))

        import torch
        if self.device is None:
            # load_mobilesam() の既定解決と同じ順にする（config省略時に食い違わないため）
            want = os.environ.get("TARO_VISION_DEVICE")
            self.device = want if want else ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        ).eval().to(self.device)

        self._mgen = self._load_mobilesam(device=self.device,
                                           points_per_batch=self.points_per_batch)
        if self.points_per_side is not None:
            # load_mobilesam（object_detector.py）にpoints_per_sideを渡す口が無いので、
            # 生成器の point_grids を直接作り直す（object_detector.py は変更しない）。
            from mobile_sam.utils.amg import build_all_layer_point_grids
            self._mgen.point_grids = build_all_layer_point_grids(
                int(self.points_per_side),
                self._mgen.crop_n_layers,
                self._mgen.crop_n_points_downscale_factor)

        ofs_kwargs = dict(
            revive_window_s=self.revive_window_s,
            revive_cos=self.revive_cos,
            revive_dist_px=self.revive_dist_px,
            max_files=self.max_files,
            # 【2026-09-09・重複をなくす「後半」4節】既定値がObjectFileSystem側の
            #   既定と一致するので、max_missedと違い無条件で渡してよい（既定不変）。
            coast_max_s=self.coast_max_s,
            uncertainty_penalty=self.uncertainty_penalty,
            exclusive_dist_px=self.exclusive_dist_px,
            exclusive_cos=self.exclusive_cos,
            pos_gate=self.pos_gate,
            # 【2026-09-09・追記「直し」1〜2節】既定値がObjectFileSystem側の既定と
            #   一致する（exclusive_scale_by_size=False・coast_min_speed_px_s=None）
            #   ので無条件で渡してよい（既定不変）。frame_dt_s は「1コマ＝検出周期」
            #   の換算に使うのでinterval_sをそのまま渡す
            #   （coast_min_speed_px_sを使うときだけ参照される＝既定不変）。
            exclusive_scale_by_size=self.exclusive_scale_by_size,
            coast_min_speed_px_s=self.coast_min_speed_px_s,
            frame_dt_s=self.interval_s,
            # 【2026-09-09・仕様_見る側の段構成_実装】既定値がObjectFileSystem側の
            #   既定(None)と一致するので無条件で渡してよい（既定不変）。
            appearance_gate_cos=self.appearance_gate_cos,
        )
        if self.max_missed is not None:
            # 既定Noneのときは渡さない＝ObjectFileSystemの既定値(20)のまま（既定不変）。
            ofs_kwargs["max_missed"] = self.max_missed
        # 【2026-09-10・仕様_消え方で持ち時間を決める】vanish_budget（辞書 or None）。
        #   Noneなら一切渡さない＝従来どおり max_missed で消す（既定不変）。
        #   例：{"abrupt_frames": 5, "occluded_frames": 50, "look_back": 4}
        self.vanish_budget = self.config.get("vanish_budget")
        if self.vanish_budget is not None:
            ofs_kwargs["vanish_budget"] = self.vanish_budget
        self.ofs = self._ObjectFileSystem(**ofs_kwargs)
        # 他のプラグイン・今後の脳側の配線から読めるように置く（ctx は属性を自由に足せる）
        ctx.object_files = self.ofs

        # 【M3・2026-09-06】既定Falseでは以下のattend関連の状態・処理は一切使われない。
        self.attend = bool(self.config.get("attend", False))
        self.attend_radius = float(self.config.get("attend_radius", 56))
        self.vanish_misses = int(self.config.get("vanish_misses", 1))
        # 【M7b-1・2026-09-09】trainer.py._world_predictor_stepが「消えたか」の
        #   判定にvanish_missesと同じ基準を使うための置き場（読むだけ・ここでは
        #   書き込む以外何もしない）。仕様「後半」4節。attend=Falseでは
        #   _process_attention自体が呼ばれないため使われない。
        ctx.vanish_misses = self.vanish_misses
        # 【M7b-1・2026-09-09】注意の加点用の係数（既定0＝既定不変）。仕様「後半」4節。
        self.attend_surprise_gain = float(self.config.get("attend_surprise_gain", 0.0))
        attend_out = self.config.get("attend_out")
        self.attend_out = _abs_path(attend_out) if attend_out else None
        self._attended_id = None
        self._attended_last_seen_vec = None
        self._attended_last_seen_time = None
        # 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う(0)】
        #   直近の検出コマで注意中のファイルが可視だったか。on_step冒頭の
        #   毎tick更新（検出コマでなくても）で使う。既定Falseでは一切参照されない
        #   （attend=Falseならon_step冒頭のifを通らない）。
        self._attended_visible = False
        self._last_nearest_word = ""
        self._last_vanished = False
        # 親の発話履歴から「目標(target id)→語」を引くための表（決めたこと3の
        # target_word用。word_learning.pyのword_forと同じ考え方：最初に見た
        # 命名文をその物の語とする。cause="vanish"（「○○ないね」の全文）と
        # cause="voice"（相槌等）は除く）。
        self._target_word = {}
        self._last_target = None
        self.attend_rows = []
        ctx.attended_object = None

        self._last_t = float("-inf")
        self.rows = []
        # 【M1.5・2026-09-05】ファイルごとの「作成時のappearance」。created の瞬間に
        #   保存し、lost で捨てる（仕様書「1.」）。app_cos_created の分母に使う。
        self._created_appearance = {}
        self._detect_ms_sum = 0.0
        self._detect_n = 0
        self._seg_n_files = []
        self._seg_n_dets = []
        self._seg_detect_ms = []
        # 【2026-09-09・仕様_見る側_道を1本にする】検出コマの処理を1本化した
        #   ので、mode別（旧confirm_ms/scan_ms）の内訳は無くなった
        #   （mode_countsで「frame」「first」の件数だけ数える）。
        self._mode_counts = {}

    def _lazy_import(self):
        for sub in ("senses", ""):
            p = os.path.join(_REPO_ROOT, "taro_core", "src", sub) if sub else \
                os.path.join(_REPO_ROOT, "taro_core", "src")
            if p not in sys.path:
                sys.path.insert(0, p)
        # 旧パス brain.object_files は転送のみなので使わない。
        from object_detector import (patch_features, detect, load_mobilesam,
                                      make_point_predictor, segment_at_points)
        from brain.cerebral_cortex.parietal_lobe.intraparietal_sulcus import ObjectFileSystem
        from brain.midbrain.efference_copy import EfferenceCopy
        from brain.midbrain.preattentive_map import PreattentiveMap
        from brain.cerebral_cortex.parietal_lobe.priority_map import PriorityMap
        self._patch_features = patch_features
        self._detect = detect
        self._load_mobilesam = load_mobilesam
        self._make_point_predictor = make_point_predictor
        self._segment_at_points = segment_at_points
        self._ObjectFileSystem = ObjectFileSystem
        self._EfferenceCopy = EfferenceCopy
        self._PreattentiveMap = PreattentiveMap
        self._PriorityMap = PriorityMap

    def _get_point_predictor(self):
        """段2（`segment_at_points`）で使う点プロンプト用の`SamPredictor`。
        1回だけ作って使い回す。"""
        if self._point_predictor is None:
            self._point_predictor = self._make_point_predictor(self._mgen)
        return self._point_predictor

    def on_step(self, ctx):
        # 【2026-09-09・仕様_見る側の段構成_実装 1節】段0 遠心性コピー。毎tick
        #   呼ぶ（検出コマでなくても）。efference_copy=None（既定）では一切
        #   呼ばれず、ctx.efferenceも置かれない＝既定不変。
        if self.efference_copy_cfg is not None:
            self._ensure_efference_copy(ctx)
            _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
            orienting = getattr(_u, "_orienting", None) if _u is not None else None
            if orienting is None and not self._warned_no_orienting:
                print("[efference] WARNING _orienting が読めない。全て0/Falseとして続行")
                self._warned_no_orienting = True
            ctx.efference = self._ec.update(orienting, float(ctx.data.time))

        if self.attend:
            # 【M3】親の発話は検出コマ(interval_s)より細かい頻度で来るので、
            #   「目標→語」の表だけは毎tick更新する（読むだけ・attend=Falseでは
            #   一切呼ばれない）。
            self._track_parent_target(ctx)
            # 【M4b・2026-09-06】M4(0)の毎tick上書き（旧195-212行）はここにあった。
            #   引っ込め中（検出コマとコマの間）に空の机の視覚で上書きしてしまい、
            #   GRUが「空の机＝消えた」を学習した（f81実測）。検出コマの処理
            #   （_process_attention、self.ofs.step直後）でだけ控えるように変更し、
            #   このtick単位の分岐は削除した。
        t = float(ctx.data.time)
        if t - self._last_t < self.interval_s:
            return
        self._last_t = t

        last = getattr(ctx, "last", None) or {}
        obs = last.get("obs_out")
        if obs is None or "eye_left" not in obs:
            return

        import numpy as np
        from PIL import Image
        img = np.asarray(obs["eye_left"])
        img224 = np.array(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                           .resize((224, 224)))

        # 【2026-09-09・仕様_見る側の段構成_実装 1節】段0が有効なとき、段3の
        #   predict()に渡すshift_pred（無効時はNone＝既定不変）。
        shift_pred = None
        if self.efference_copy_cfg is not None:
            shift_pred = (getattr(ctx, "efference", None) or {}).get("shift_pred")

        # prev_by_id は _detect_frame()（内部でofs.stepを呼ぶ）が呼ばれる**前**（＝削除される
        # 前）に確定させる（predict_only()等が位置を書き換えても同じ参照な
        # ので値は変わらない＝既存の呼び出し順と1ビットも変わらない）。
        prev_by_id = {f.id: f for f in self.ofs.files}
        # 【2026-09-09・枚数の上限】押し出し禁止の対象＝現在注意中のファイル
        # （_process_attentionはこのofs.stepより後に呼ぶので、ここではまだ
        # 前コマの注意状態を使う。attend=Falseなら常にNone＝従来どおり）。
        protect_id = self._attended_id if self.attend else None

        # 【2026-09-09・仕様_見る側_道を1本にする】検出コマの処理は_detect_frame
        #   1本（旧「確認」道・旧「切り出し」道の2本を統合）。
        mode, res, dt_ms, onset_extra = self._detect_frame(ctx, img224, t, protect_id)
        dets = onset_extra.pop("new_dets", [])

        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

        if self.attend:
            # 【M3】検出・対応づけ結果（res・self.ofs.files）を読むだけ。
            #   frames_out描画（次のif）が this tick の注意状態を使えるよう、
            #   描画より先に済ませる。
            self._process_attention(ctx, t, res)

        if self.frames_out:
            self._save_detection_frame(ctx, t, img224, dets, res, prev_by_id, mode=mode)

        self._detect_ms_sum += dt_ms
        self._detect_n += 1
        n_files = len(self.ofs.files)
        n_dets = len(dets)
        self._seg_n_files.append(n_files)
        self._seg_n_dets.append(n_dets)
        self._seg_detect_ms.append(dt_ms)

        if self.events_out is None:
            return

        cur_by_id = {f.id: f for f in self.ofs.files}
        matched_ids = set(file_id for file_id, _det_idx, _residual in res["matched"])
        created_ids = set(res["created"])

        def _app_cos(file_id, appearance):
            """現在appearanceと作成時appearanceのコサイン類似度。作成時appearance
            が無ければ空文字（本来起きない：createdの瞬間に必ず保存するため）。"""
            created = self._created_appearance.get(file_id)
            if created is None or appearance is None:
                return ""
            a = np.asarray(appearance, dtype=np.float64)
            b = np.asarray(created, dtype=np.float64)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            return round(float(a @ b / denom), 6)

        def _cohesion_for(f):
            """【2026-09-09・仕様_見る側の段構成_実装 3節】createdイベントの
            cohesion列。dets（このコマの検出）の中から同じ位置(誤差1px以内)の
            ものを探し、その"cohesion_reason"を返す（無ければ""）。"""
            for d in dets:
                dp = d.get("pos")
                if dp is None:
                    continue
                if abs(dp[0] - f.pos[0]) <= 1.0 and abs(dp[1] - f.pos[1]) <= 1.0:
                    return d.get("cohesion_reason", "") or ""
            return ""

        events = []
        for file_id, det_idx, residual in res["matched"]:
            f = cur_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "matched",
                               f.misses, f.since_seen, _app_cos(file_id, f.appearance)))
        for file_id in res["created"]:
            f = cur_by_id.get(file_id)
            if f is not None:
                # 【M1.5】作成の瞬間のappearanceを保存する（このファイルのapp_cos_createdの基準点）。
                self._created_appearance[file_id] = np.array(f.appearance, dtype=np.float64, copy=True)
                events.append((file_id, f.pos[0], f.pos[1], f.area, "created",
                               f.misses, f.since_seen, _app_cos(file_id, f.appearance)))
        # 【2026-09-09・仕様_見る側の段構成_実装「後半」段4・新イベント
        #   individuated】位置は既存カードの近くにあるのに見た目で「別の物」と
        #   判定して新カードを作った件。作成の行とは別行で記録する（作成行は
        #   上のcreatedループで既に出ている）。appearance_gate_cos/
        #   exclusive_dist_pxがNone（既定）なら常に空リスト＝既定不変。
        for file_id in res.get("individuated", []):
            f = cur_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "individuated",
                               f.misses, f.since_seen, _app_cos(file_id, f.appearance)))
        # 【2026-09-09・連続性・新イベント revived】墓場から復活したファイル。
        #   created と同じ扱いで「復活した瞬間のappearance」を新しい基準点にする
        #   （lost時にcreated_appearanceは既にpopされているため、ここが新規登録になる）。
        for file_id in res.get("revived", []):
            f = cur_by_id.get(file_id)
            if f is not None:
                self._created_appearance[file_id] = np.array(f.appearance, dtype=np.float64, copy=True)
                events.append((file_id, f.pos[0], f.pos[1], f.area, "revived",
                               f.misses, f.since_seen, _app_cos(file_id, f.appearance)))
        revived_ids = set(res.get("revived", []))
        # 【2026-09-09・重複をなくす「後半」4節・新イベント absorbed】既存カードに
        #   吸収された検出。そのカードの行として記録する（misses=0に更新済み）。
        absorbed_pairs = res.get("absorbed", [])
        absorbed_ids = set(file_id for _det_idx, file_id in absorbed_pairs)
        for _det_idx, file_id in absorbed_pairs:
            f = cur_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "absorbed",
                               f.misses, f.since_seen, _app_cos(file_id, f.appearance)))
        # 【M1.5・新イベント unmatched】このコマで対応がつかず、まだ削除されていない
        #   ファイル全部（＝matched でも created でもない、いま self.ofs.files に
        #   残っているファイル）。x/y/area は predict() 後の予測値
        #   （ObjectFile.pos が返すのはその値。step()冒頭で毎ファイルpredict()済み）。
        for f in self.ofs.files:
            if (f.id in matched_ids or f.id in created_ids or f.id in revived_ids
                    or f.id in absorbed_ids):
                continue
            events.append((f.id, f.pos[0], f.pos[1], f.area, "unmatched",
                           f.misses, f.since_seen, _app_cos(f.id, f.appearance)))
        for file_id in res["lost"]:
            # prev_by_id は同じObjectFileインスタンスへの参照なので、この一連の
            # step()内でのmisses/since_seenの増加（削除直前の値）がそのまま読める
            # （オブジェクトは self.ofs.files から外れただけで、参照自体は生きている）。
            f = prev_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "lost",
                               f.misses, f.since_seen, _app_cos(file_id, f.appearance)))
            else:
                events.append((file_id, "", "", "", "lost", "", "", ""))
            self._created_appearance.pop(file_id, None)
        if not events:
            events.append(("", "", "", "", "", "", "", ""))

        for file_id, x, y, area, event, misses, since_seen, app_cos_created in events:
            # 【2026-09-09・仕様_見る側_道を1本にする 後半1節】検出コマの処理を
            #   1本化したので、conf_iou/conf_area_ratio/conf_emb_cos/conf_reject
            #   （確認専用の診断列）は無い。app_conf（appearance_confのEMA）も
            #   フィールドごと削除したので出さない（中5）。
            # 【2026-09-09・仕様_見る側の段構成_実装 3節】cohesion列は
            #   event=="created"の行にだけ書く。
            f_for_row = (prev_by_id.get(file_id) if event == "lost"
                         else cur_by_id.get(file_id))
            cohesion = _cohesion_for(f_for_row) if (event == "created" and f_for_row is not None) else ""
            self.rows.append({
                "step": ctx.step, "sim_time": round(t, 3),
                "n_dets": n_dets, "n_files": n_files,
                "file_id": file_id, "x": x, "y": y, "area": area, "event": event,
                "misses": misses, "since_seen": since_seen,
                "app_cos_created": app_cos_created,
                # 【2026-09-09・仕様_見る側_道を1本にする】mode列は"frame"
                #   （最初のコマだけ"first"）に一本化。
                "mode": mode,
                # 【2026-09-09・道を1本にする 後半1節】段1・段2・段4の列。
                #   preattentive/efference_copy無効時は空文字のまま。
                "n_points": onset_extra.get("n_points", ""),
                "n_blobs": onset_extra.get("n_blobs", ""),
                "n_unexplained": onset_extra.get("n_unexplained", ""),
                "ec_res_shift": onset_extra.get("ec_res_shift", ""),
                "ec_res_noshift": onset_extra.get("ec_res_noshift", ""),
                "ec_moving": onset_extra.get("ec_moving", ""),
                "cohesion": cohesion,
                # 【2026-09-10・消え方で持ち時間を決める】explained＝消えたことの
                #   説明のつき具合（0〜1、見失った瞬間に決まる）、budget＝残量。
                #   vanish_budget 無効時は 0.0/1.0 のまま動かないので空文字にする。
                "explained": (round(float(f_for_row.explained), 3)
                              if (self.vanish_budget is not None and f_for_row is not None
                                  and getattr(f_for_row, "misses", 0) > 0) else ""),
                "budget": (round(float(f_for_row.budget), 3)
                           if (self.vanish_budget is not None and f_for_row is not None) else ""),
                # 【2026-09-09・追記1「直し」3】当て付きの点の数／当てに合う
                #   大きさが無くて捨てた点の数。
                "n_points_expect": onset_extra.get("n_points_expect", ""),
                "n_reject_scale": onset_extra.get("n_reject_scale", ""),
            })

    def _detect_frame(self, ctx, img224, t, protect_id):
        """【2026-09-09・仕様_見る側_道を1本にする 後半1節】検出コマの処理
        1本（旧「確認」道・旧「切り出し」道を統合）。

        0. eff = ctx.efference（無効なら shift=(0,0)・moving=False）
        1. pre = self._pre.update(...)（段1。無効なら blobs=[]）
        2. self.ofs.predict_only(t, shift=eff["shift_pred"])（shiftを必ず渡す＝重大1の直し）
        3. points を集める：3a カードの予測位置、3b onset（misses==0のカードの
           予測bboxに入らない塊の重心）、3c 前コマの探索点、3d 最初のコマだけは
           全体切り出しに差し替え
        4. dets = segment_at_points(...)（set_imageは1回、背景の足切り・
           重複除けはsegment_at_points側で行う）
        5. res = self.ofs.step(dets, t=t, protect_id=protect_id, skip_predict=True)
        6. priority = self._priority.update(candidates, ...)
           （candidatesは「見えていて中央attend_radius以内」のカードだけ＝重大2の直し）

        Returns:
            (mode, res, dt_ms, onset_extra)
            mode: "first"（最初のコマ）または"frame"（それ以外・一本化）
            onset_extra: CSV追加列用の辞書（n_points, n_blobs, n_unexplained,
                ec_res_shift, ec_res_noshift, ec_moving, new_dets）
        """
        import numpy as np
        from PIL import Image

        t0 = time.perf_counter()
        onset_extra = {"n_points": "", "n_blobs": "", "n_unexplained": "",
                        "ec_res_shift": "", "ec_res_noshift": "", "ec_moving": "",
                        # 【2026-09-09・追記1「直し」3】最初のコマ("first")は
                        #   segment_at_pointsを通らないので空文字のまま（既定不変）。
                        "n_points_expect": "", "n_reject_scale": ""}

        # ---- 0. 遠心性コピー（無効ならshift=(0,0)・moving=False） --------------
        eff = getattr(ctx, "efference", None) or {}
        shift_actual = eff.get("shift_actual", (0.0, 0.0))
        shift_pred = eff.get("shift_pred")
        moving = bool(eff.get("moving", False))

        # ---- 3d. 最初のコマだけ全体切り出し -------------------------------------
        if not self._first_scan_done:
            self._first_scan_done = True
            p, n = self._patch_features(self._model, img224, self.device)
            dets = self._detect(p, n, img=img224, mask_generator=self._mgen,
                                 cohesion_gap_px=self.cohesion_gap_px)
            if self.empty_cache_after_detect:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # 段1・段6の状態（前コマ画像・探索地図）だけは最初のコマでも更新して
            # おく（無効なら中身は空のまま）。
            if self._pre is not None:
                gray = np.array(Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8))
                                 .convert("L"), dtype=np.float32)
                self._pre.update(gray, shift_actual, moving)
            self.ofs.predict_only(t, shift=shift_pred)
            res = self.ofs.step(dets, t=t, protect_id=protect_id, skip_predict=True)
            onset_extra["new_dets"] = dets
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return "first", res, dt_ms, onset_extra

        # ---- 1. 段1：前注意の地図（無効ならblobs=[]） ---------------------------
        if self._pre is not None:
            gray = np.array(Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8))
                             .convert("L"), dtype=np.float32)
            pre_res = self._pre.update(gray, shift_actual, moving)
        else:
            pre_res = {"blobs": [], "motion_mean": 0.0, "motion_mean_noshift": 0.0,
                       "static_sal": None, "valid": False}
        onset_extra["n_blobs"] = len(pre_res["blobs"])
        onset_extra["ec_res_shift"] = round(pre_res["motion_mean"], 4)
        onset_extra["ec_res_noshift"] = round(pre_res["motion_mean_noshift"], 4)
        onset_extra["ec_moving"] = moving

        # ---- 1b. 【2026-09-10・見る側1〜2段目】目立ちの地図 → 場所の優先度地図。
        #      attention=None（既定）なら1行も通らない。
        attn_res = None
        if self._spri is not None:
            sal_res = self._sal.update(img224, shift_actual, moving)
            attn_res = self._spri.update(sal_res["salience"], shift_px=shift_pred,
                                          dt=self.interval_s)
            ctx.salience_map = sal_res["salience"]
            ctx.attention_point = attn_res["winner_px"]
            onset_extra["attn_x"] = round(attn_res["winner_px"][0], 1)
            onset_extra["attn_y"] = round(attn_res["winner_px"][1], 1)
            onset_extra["attn_switched"] = attn_res["switched"]
            # 【2026-09-10・見る側4段目】意思の視線。注意が移った瞬間だけ、
            #   その場所を反射の「意思の目標」として渡す。反射は残したまま
            #   （人間も上丘の反射は無くならない）、2系統が同じ筋へつながる形。
            #   gaze=False（既定）なら1行も通らない。
            if self.gaze_from_attention and attn_res["switched"]:
                _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
                _or = getattr(_u, "_orienting", None) if _u is not None else None
                if _or is not None and hasattr(_or, "set_voluntary_target"):
                    wx, wy = attn_res["winner_px"]
                    # 視野の方向（[-1,1]、右・上が正）へ。画像の y は下向きなので反転。
                    _or.set_voluntary_target((wx - 112.0) / 112.0,
                                              (112.0 - wy) / 112.0)
                    self._gaze_cmd_sent += 1
                    onset_extra["gaze_cmd"] = 1

        # ---- 2. カードを予測位置へ進める（★shiftを必ず渡す・重大1の直し） -------
        self.ofs.predict_only(t, shift=shift_pred)

        # ---- 3. 点を集める -------------------------------------------------------
        # 【2026-09-09・追記1「直し」1】各点に「このくらいの大きさのはず」の
        #   当て（expect_area、面積比0-1・無ければNone）を付ける
        #   （segment_at_pointsのマスク選びの手がかり。カードの予測位置は
        #   そのカードの面積、動きの塊はその塊の面積比、探索点は当てが無い）。
        points = []
        # 3a. カードの予測位置（misses に関わらず全カード）：当て＝f.area
        #     【2026-09-10・3段目】これは「追い続ける」ための点。ここからは
        #     新しい記録を作らない（can_create=False）。
        _atten = self._spri is not None
        for f in self.ofs.files:
            points.append({"pos": f.pos, "expect_area": max(float(f.area), 0.0),
                           "can_create": not _atten})
        # 3b. onset：misses==0のカードの予測bboxに重心が入らない塊：当て＝塊の面積比
        unexplained = []
        for b in pre_res["blobs"]:
            cx, cy = b["cx"], b["cy"]
            in_file = False
            for f in self.ofs.files:
                if f.misses != 0:
                    continue
                fx, fy = f.pos
                r = (max(f.area, 0.0) ** 0.5) * 224.0 * 0.75 + 8.0
                if ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5 <= r:
                    in_file = True
                    break
            if not in_file:
                unexplained.append(b)
        onset_extra["n_unexplained"] = len(unexplained)
        for b in unexplained:
            bx, by, bw, bh = b["bbox"]
            blob_area = (float(bw) * float(bh)) / (224.0 * 224.0)
            points.append({"pos": (b["cx"], b["cy"]), "expect_area": blob_area})
        # 3c. 前コマの探索点（あれば）：当て無し（None）
        if self._prev_explore_point is not None and not _atten:
            points.append({"pos": self._prev_explore_point, "expect_area": None})
        # 3d. 【2026-09-10・3段目】注意が向いた1点。ここからだけ新しい記録を作る。
        #     すでにその場所に見えているカードがあるなら足さない（既に記録済み）。
        if _atten and attn_res is not None:
            wx, wy = attn_res["winner_px"]
            covered = False
            for f in self.ofs.files:
                if f.misses != 0:
                    continue
                r = (max(f.area, 0.0) ** 0.5) * 224.0 * 0.75 + 8.0
                if ((wx - f.pos[0]) ** 2 + (wy - f.pos[1]) ** 2) ** 0.5 <= r:
                    covered = True
                    break
            if not covered:
                points.append({"pos": (wx, wy), "expect_area": None,
                               "can_create": True})
            onset_extra["attn_covered"] = covered
        onset_extra["n_points"] = len(points)

        # ---- 4. 集めた点をSAMに1回だけ渡して切り出す -----------------------------
        dets = []
        n_points_expect = 0
        n_reject_scale = 0
        if points:
            p, n = self._patch_features(self._model, img224, self.device)
            pp = self._get_point_predictor()
            dets, seg_stats = self._segment_at_points(pp, img224, points, p, n,
                                                        blobs=pre_res["blobs"])
            n_points_expect = seg_stats.get("n_points_expect", 0)
            n_reject_scale = seg_stats.get("n_reject_scale", 0)
            if self.empty_cache_after_detect:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        onset_extra["new_dets"] = dets
        onset_extra["n_points_expect"] = n_points_expect
        onset_extra["n_reject_scale"] = n_reject_scale

        # ---- 5. 全部まとめて1つの照合にかける -----------------------------------
        res = self.ofs.step(dets, t=t, protect_id=protect_id, skip_predict=True)

        # ---- 6b. 【2026-09-10・3段目】注意するカード＝注意が向いた点にいちばん
        #      近い、見えているカード（注意の半径の中）。人間は場所を選び、
        #      そこにある物が記録になる（順序は場所が先）。
        if self._spri is not None and attn_res is not None:
            wx, wy = attn_res["winner_px"]
            foa_r = self._spri.foa_radius_px
            best, best_d = None, None
            for f in self.ofs.files:
                if f.misses != 0:
                    continue
                d = ((wx - f.pos[0]) ** 2 + (wy - f.pos[1]) ** 2) ** 0.5
                if d <= foa_r and (best_d is None or d < best_d):
                    best, best_d = f.id, d
            ctx.priority_map_result = {
                "attended_id": best, "switch_signal": attn_res["switched"],
                "priority": {}, "explore_point": None, "ior": {},
                "switch_decided": attn_res["switched"]}
            ctx.priority_map_result_t = t
            if attn_res["switched"]:
                ctx.attention_switch_t = t
            onset_extra["attn_id"] = "" if best is None else best

        # ---- 6. 優先度地図（見えていて中央attend_radius以内のカードだけ・重大2の直し） --
        explore_point = None
        if self._priority is not None:
            CENTER_X, CENTER_Y = 112.0, 112.0
            candidates = [f for f in self.ofs.files
                          if f.misses == 0
                          and ((f.pos[0] - CENTER_X) ** 2 + (f.pos[1] - CENTER_Y) ** 2) ** 0.5
                          <= self.attend_radius]
            motion_by_id = {}
            for f in self.ofs.files:
                fx, fy = f.pos
                for b in pre_res["blobs"]:
                    bx, by, bw, bh = b["bbox"]
                    if bx <= fx <= bx + bw and by <= fy <= by + bh:
                        motion_by_id[f.id] = motion_by_id.get(f.id, 0.0) + b["mean_abs_diff"]
            surprise_trace = getattr(ctx, "surprise_trace", {}) or {}
            priority_result = self._priority.update(
                candidates, t, motion_by_id, pre_res["static_sal"], surprise_trace,
                {}, {}, moving, self.interval_s)
            explore_point = priority_result.get("explore_point")
            ctx.priority_map_result = priority_result
            ctx.priority_map_result_t = t
        self._prev_explore_point = explore_point

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return "frame", res, dt_ms, onset_extra

    def _ensure_efference_copy(self, ctx):
        """【2026-09-09・仕様_見る側の段構成_実装 1節】fovyはenv側から読む
        （setup時点ではctx.envがまだ無いことがあるため遅延構築。1回だけ
        作って使い回す）。cam_fovyが読めなければ60度で代用し、報告用に
        警告を1回だけ出す（仕様10節「止まる」条件：60度で代用して続け、
        報告に書く、を選んだ）。"""
        if self._ec is not None or self.efference_copy_cfg is None:
            return
        import math
        cfg = self.efference_copy_cfg
        fovy = 60.0
        _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
        if _u is not None:
            try:
                cid = int(_u.model.camera("eye_left").id)
                fovy = float(_u.model.cam_fovy[cid])
            except Exception as e:
                print("[efference] WARNING cam_fovy が読めない。60度で代用: %r" % (e,))
        else:
            print("[efference] WARNING ctx.env が無く cam_fovy を読めない。60度で代用")
        f_px = 112.0 / math.tan(math.radians(fovy / 2.0))
        self._ec = self._EfferenceCopy(
            f_px=f_px,
            sign_h=float(cfg.get("sign_h", 1.0)),
            sign_v=float(cfg.get("sign_v", 1.0)),
            moving_extra_ticks=int(cfg.get("moving_extra_ticks", 1)))

    def _save_detection_frame(self, ctx, sim_time, img224, dets, res, prev_by_id,
                               mode="frame"):
        """目視用：検出（黄丸）と物体ファイル（色つき四角）をimg224へ重ねてPNG保存する。
        検出・対応づけ・既存CSVには一切触れない（読むだけ）。"""
        import math
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        if self._frame_font is None:
            from run.plugins.common.view_video import _FONT_PATH
            self._frame_font = ImageFont.truetype(_FONT_PATH, 12)
        img = Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8)).convert("RGB")
        dr = ImageDraw.Draw(img)
        for d in dets:
            cx, cy = d["pos"]
            r = math.sqrt(d["area"] * 224 * 224 / math.pi)
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0))
        matched_ids = set(file_id for file_id, _det_idx, _residual in res["matched"])
        created_ids = set(res["created"])
        revived_ids = set(res.get("revived", []))
        for f in self.ofs.files:
            if f.id in matched_ids:
                color = (0, 200, 0)       # matched=緑
            elif f.id in revived_ids:
                color = (170, 60, 220)    # 【2026-09-09・連続性】revived=紫
            elif f.id in created_ids:
                color = (60, 60, 255)     # created=青
            else:
                color = (160, 160, 160)   # unmatched=灰
            self._draw_file_box(dr, f, color)
        for file_id in res["lost"]:
            f = prev_by_id.get(file_id)
            if f is not None:
                self._draw_file_box(dr, f, (255, 0, 0))   # lost=赤
        if self.attend and self._attended_id is not None:
            # 【M3・仕様書「描画」節】注意中のファイルを太枠(width=3)で強調し、
            #   消えていれば左上に赤字、見た目に一番近い語を四角の下に出す。
            att_f = next((f for f in self.ofs.files if f.id == self._attended_id), None)
            if att_f is not None:
                x, y = att_f.pos
                dr.rectangle([x - 8, y - 8, x + 8, y + 8], outline=(255, 140, 0), width=3)
                if self._last_nearest_word:
                    dr.text((x - 8, y + 9), self._last_nearest_word,
                             fill=(255, 140, 0), font=self._frame_font)
            if self._last_vanished:
                dr.text((4, 28), "VANISHED #%s" % self._attended_id,
                         fill=(255, 0, 0), font=self._frame_font)
        # 【2026-09-09・仕様_見る側_道を1本にする】左上にmodeのラベル
        #   （"frame"/"first"の2種に一本化）。
        _mode_label = {"first": "FIRST"}.get(mode, "FRAME")
        dr.text((4, 2), _mode_label, fill=(255, 255, 0), font=self._frame_font)
        # 【2026-09-09・重複をなくす「後半」4節】吸収された検出＝紫の小さい丸
        #   （そのカードの現在位置に描く。revivedの箱と違う形なので区別できる）。
        for _det_idx, file_id in res.get("absorbed", []):
            f = next((ff for ff in self.ofs.files if ff.id == file_id), None)
            if f is not None:
                x, y = f.pos
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(180, 60, 220))
        dr.text((4, 14), "t=%.1fs dets=%d files=%d" % (sim_time, len(dets), len(self.ofs.files)),
                 fill=(255, 255, 255), font=self._frame_font)
        img.save(os.path.join(self.frames_out, "frame_%05d.png" % ctx.step))

    def _draw_file_box(self, dr, f, color):
        x, y = f.pos
        dr.rectangle([x - 8, y - 8, x + 8, y + 8], outline=color)
        dr.text((x + 9, y - 9), "#%s m%s" % (f.id, f.misses), fill=color, font=self._frame_font)

    # ---- M3・2026-09-06 注意中の物体ファイルと消失信号 --------------------------

    def _track_parent_target(self, ctx):
        """親が今どの物を指しているか（target id）と、その語（初回の命名文の
        text）の対応を毎tick更新する（決めたこと3のtarget_word用）。
        cause="vanish"（「○○ないね」の全文）とcause="voice"（相槌等、必ずしも
        単語ではない）は語の登録元から除く（targetの更新自体はする）。"""
        for ev in (getattr(ctx, "last_parent_utterance", None) or []):
            tgt = ev.get("target")
            if tgt is None:
                continue
            self._last_target = tgt
            cause = ev.get("cause") or ""
            text = ev.get("text")
            if cause in ("", "repeat") and text and tgt not in self._target_word:
                self._target_word[tgt] = text

    @staticmethod
    def _cos(a, b):
        """word_similarity_map.py._cos と同じ実装（1個も変えない）。"""
        import numpy as np
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na <= 0 or nb <= 0:
            return 0.0
        return float((a / na) @ (b / nb))

    def _lexicon_proto(self, ctx):
        """(proto, vocab) を返す。学習の最初期でlexiconが空ならNoneのペア。"""
        taro = getattr(ctx, "taro", None)
        lexicon = getattr(taro, "lexicon", None) if taro is not None else None
        proto = getattr(lexicon, "proto", None) if lexicon is not None else None
        hearing = getattr(taro, "hearing", None) if taro is not None else None
        vocab = getattr(hearing, "vocab", None) if hearing is not None else None
        if not proto or vocab is None:
            return None, None
        return proto, vocab

    def _nearest_lexicon_word(self, ctx, vec):
        """vecに一番近い（コサイン最大）語彙プロトタイプの語と類似度を返す。
        見つからなければ("", "")。"""
        proto, vocab = self._lexicon_proto(ctx)
        if proto is None:
            return "", ""
        best_word, best_cos = "", None
        for chunk, v in proto.items():
            s = vocab.decode(chunk)
            if not s:
                continue
            c = self._cos(vec, v)
            if best_cos is None or c > best_cos:
                best_word, best_cos = s, c
        if best_cos is None:
            return "", ""
        return best_word, round(best_cos, 6)

    def _word_cos(self, ctx, vec, word):
        """vecと指定した語(word)のプロトタイプとのコサイン類似度。無ければ""。"""
        proto, vocab = self._lexicon_proto(ctx)
        if proto is None:
            return ""
        for chunk, v in proto.items():
            if vocab.decode(chunk) == word:
                return round(self._cos(vec, v), 6)
        return ""

    def _process_attention(self, ctx, t, res):
        """検出コマ(interval_sごと)に呼ばれる。仕様書「決めたこと」1〜3を実装する。
        検出・対応づけロジック（self.ofs.step等）には一切触れない（結果を読むだけ）。"""
        import math
        import numpy as np

        CENTER_X, CENTER_Y = 112.0, 112.0
        cur_by_id = {f.id: f for f in self.ofs.files}

        # ---- 決めたこと1：どの物体ファイルに注意しているか --------------------
        visible = [f for f in self.ofs.files if f.misses == 0]
        candidates = [f for f in visible
                      if math.hypot(f.pos[0] - CENTER_X, f.pos[1] - CENTER_Y)
                      <= self.attend_radius]
        priority_result = None
        if (self._spri is not None
                and getattr(ctx, "priority_map_result_t", None) == t
                and getattr(ctx, "priority_map_result", None) is not None):
            # 【2026-09-10・見る側3段目】場所の地図が決めた注意をそのまま使う。
            #   物の一覧から選び直さない（順序は「場所が先、物が後」）。
            priority_result = ctx.priority_map_result
            if priority_result["attended_id"] is not None:
                self._attended_id = priority_result["attended_id"]
        elif self._priority is not None:
            # 【2026-09-09・仕様_見る側の段構成_実装 5節】段6。priority設定が
            #   有効なときだけ置き換える（無効時は下のelseで従来どおり1行も
            #   変えない）。中央56pxの絞りはPriorityMapの外（上のcandidates）で
            #   従来どおり適用し、PriorityMapには候補だけ渡す。
            if (getattr(ctx, "priority_map_result_t", None) == t
                    and getattr(ctx, "priority_map_result", None) is not None):
                # `_detect_predictive_preattentive` が同じtickで既に1回だけ
                # update()を呼んでいる（探索点のため）。ここで2回目を呼ぶと
                # 内部状態（切り替えの遅れ・IOR）が壊れるので使い回す。
                priority_result = ctx.priority_map_result
            else:
                moving = bool((getattr(ctx, "efference", None) or {}).get("moving", False))
                _trace = getattr(ctx, "surprise_trace", {}) or {}
                priority_result = self._priority.update(
                    candidates, t, {}, None, _trace, {}, {}, moving, self.interval_s)
            if priority_result["attended_id"] is not None:
                self._attended_id = priority_result["attended_id"]
        elif candidates:
            # 中央付近に複数あれば一番大きいものを取る（1個の物が部位ごとに
            # 複数ファイルへ分裂する既知の性質＝M1.5bで判明への対処）。
            # 【M7b-1・2026-09-09】仕様「後半」4節：「今の面積」に「その物の驚きの
            #   余韻」を足す。gain=0（既定）なら surprise_trace を足しても面積の
            #   大小関係は変わらない＝既定不変。surprise_traceはtrainer.pyが
            #   ctx.surprise_traceに置く（無ければ空辞書扱い）。
            gain = self.attend_surprise_gain
            _trace = getattr(ctx, "surprise_trace", {}) or {}
            self._attended_id = max(
                candidates,
                key=lambda f: f.area + gain * _trace.get(f.id, 0.0)).id
        # 無ければ前回のattendedを保持（消えた物を追い続ける）。
        lost_ids = set(res["lost"])
        if self._attended_id is not None and self._attended_id in lost_ids:
            # attendedがlostで削除された＝これ以上追い続ける物が無い。
            self._attended_id = None
            self._attended_last_seen_vec = None
            self._attended_last_seen_time = None
            self._attended_visible = False

        attended_f = (cur_by_id.get(self._attended_id)
                      if self._attended_id is not None else None)
        # 【M4・2026-09-06(0)】次の検出コマまでの毎tick更新（on_step冒頭）が
        #   使う「直近の検出コマで可視だったか」をここで確定させる。
        self._attended_visible = attended_f is not None and attended_f.misses == 0

        # ---- 決めたこと2：「消えた物の見た目」の控え ---------------------------
        # 【M4b：検出コマで一致した瞬間だけ控える（毎tickだと引っ込め中に
        #   空の机で上書きされる。f81で実測）】確認して実際にその物が見つかった
        #   瞬間（self._attended_visible、検出コマ内）だけ控えを更新する。
        #   確認と確認の間は控えを触らない。
        if self._attended_visible and ctx.last_vision_vec is not None:
            self._attended_last_seen_vec = np.array(
                ctx.last_vision_vec, dtype=np.float64, copy=True)
            self._attended_last_seen_time = t
            if getattr(ctx, "attended_object", None) is not None:
                ctx.attended_object["last_seen_vec"] = self._attended_last_seen_vec
                ctx.attended_object["last_seen_time"] = self._attended_last_seen_time

        # ---- 決めたこと3の前段：消失信号 -------------------------------------
        _prev_vanished = self._last_vanished
        vanished = attended_f is not None and attended_f.misses >= self.vanish_misses
        self._last_vanished = bool(vanished)
        # 【M4c・2026-09-06】vanished が偽→真に変わった瞬間だけ、世界（親）へ
        #   合図する。声の合図（run/trainer.py:934 _taro_voice_signal）と同じ
        #   流儀。仕様：F/docs/二語文/仕様_M4c_親は太郎が気づいてから
        #   「ないね」と言う_2026-09-06.md。人間の親は子の頭の中を直接読めない
        #   （視線や探す動きから推し量る）ので、これは人間模倣からの逸脱
        #   （世界が太郎の内部状態を読む）。doc/人間模倣からの逸脱リスト.md
        #   「その45」に登録済み。
        if vanished and not _prev_vanished:
            _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
            if _u is not None:
                _u._taro_noticed_gone_time = float(ctx.data.time)

        # ---- nearest_word（描画・CSV両方で使うのでここで一度だけ計算） --------
        if self._attended_last_seen_vec is not None:
            nearest_word, _nearest_cos = self._nearest_lexicon_word(
                ctx, self._attended_last_seen_vec)
        else:
            nearest_word, _nearest_cos = "", ""
        self._last_nearest_word = nearest_word

        # 【2026-09-09・仕様_見る側の段構成_実装 5節】切り替えた合図。priority
        #   無効なら常にFalse・ctx.attention_switch_tは触らない＝既定不変。
        switch_signal = bool(priority_result["switch_signal"]) if priority_result else False
        if switch_signal:
            ctx.attention_switch_t = t

        # ---- ctx.attended_object ----------------------------------------------
        if attended_f is None:
            ctx.attended_object = None
        else:
            ctx.attended_object = {
                "file_id": attended_f.id,
                "visible": attended_f.misses == 0,
                "misses": attended_f.misses,
                "since_seen": attended_f.since_seen,
                "pos": attended_f.pos,
                "area": attended_f.area,
                "vanished": vanished,
                "last_seen_vec": self._attended_last_seen_vec,
                "last_seen_time": self._attended_last_seen_time,
                "t": t,
            }
            if priority_result is not None:
                # 【2026-09-09・仕様_見る側の段構成_実装 5節】既存キーは変えず追加のみ。
                ctx.attended_object["switch_signal"] = switch_signal
                ctx.attended_object["priority"] = priority_result.get("priority", {})

        if self.attend_out is None:
            return

        target_word = (self._target_word.get(self._last_target, "")
                        if self._last_target is not None else "")
        target_cos = ""
        if target_word and self._attended_last_seen_vec is not None:
            target_cos = self._word_cos(ctx, self._attended_last_seen_vec, target_word)

        if attended_f is not None:
            dist_center = round(math.hypot(attended_f.pos[0] - CENTER_X,
                                            attended_f.pos[1] - CENTER_Y), 3)
            area = attended_f.area
            misses_v = attended_f.misses
            visible_v = (attended_f.misses == 0)
            vanished_v = vanished
        else:
            dist_center = area = misses_v = visible_v = vanished_v = ""

        # 【2026-09-09・仕様_見る側の段構成_実装 5節】priority/switch/explore_x/
        #   explore_y/ior列。priority無効時は空文字のまま＝既定不変。
        priority_v = switch_v = explore_x_v = explore_y_v = ior_v = ""
        # 【2026-09-09・追記2「直し」3】switch_decided＝このコマで新しく
        #   切り替えを「決めた」か（実際に移るのは switch_delay_s 秒後・
        #   switch列のまま）。priority無効時は空文字のまま＝既定不変。
        switch_decided_v = ""
        if priority_result is not None:
            priority_v = round(priority_result["priority"].get(self._attended_id, 0.0), 6) \
                if self._attended_id is not None else ""
            switch_v = switch_signal
            switch_decided_v = bool(priority_result.get("switch_decided", False))
            ep = priority_result.get("explore_point")
            if ep is not None:
                explore_x_v, explore_y_v = round(ep[0], 2), round(ep[1], 2)
            ior_v = round(priority_result.get("ior", {}).get(self._attended_id, 0.0), 6) \
                if self._attended_id is not None else ""

        self.attend_rows.append({
            "step": ctx.step, "sim_time": round(t, 3),
            "attended_id": self._attended_id if self._attended_id is not None else "",
            "visible": visible_v, "misses": misses_v, "vanished": vanished_v,
            "dist_center": dist_center, "area": area,
            "nearest_word": nearest_word, "nearest_cos": _nearest_cos,
            "target_word": target_word, "target_cos": target_cos,
            "priority": priority_v, "switch": switch_v,
            "explore_x": explore_x_v, "explore_y": explore_y_v, "ior": ior_v,
            "switch_decided": switch_decided_v,
        })

    def metrics(self, ctx):
        if not self._seg_n_files:
            return None
        out = {
            "object_files_n_files": round(sum(self._seg_n_files) / len(self._seg_n_files), 3),
            "object_files_n_dets": round(sum(self._seg_n_dets) / len(self._seg_n_dets), 3),
            "object_files_detect_ms": round(
                sum(self._seg_detect_ms) / len(self._seg_detect_ms), 2),
        }
        # 【2026-09-10・4段目】意思で撃った回数と反射で撃った回数（記録用）。
        if self.gaze_from_attention:
            _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
            _or = getattr(_u, "_orienting", None) if _u is not None else None
            if _or is not None:
                out["gaze_vol_fired"] = int(getattr(_or, "vol_fired", 0))
                out["gaze_reflex_fired"] = int(getattr(_or, "reflex_fired", 0))
                out["gaze_cmd_sent"] = int(self._gaze_cmd_sent)
        self._seg_n_files, self._seg_n_dets, self._seg_detect_ms = [], [], []
        return out

    def line(self, ctx):
        return f"物体:{len(self.ofs.files)}"

    def report(self, ctx):
        if self.rows and self.events_out:
            os.makedirs(os.path.dirname(self.events_out) or ".", exist_ok=True)
            with open(self.events_out, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_time", "n_dets", "n_files",
                           "file_id", "x", "y", "area", "event",
                           "misses", "since_seen", "app_cos_created", "mode",
                           # 【2026-09-09・仕様_見る側_道を1本にする 後半1節】
                           #   確認専用の診断列(conf_iou等)・app_confは廃止（中5）。
                           #   代わりにn_points（段3で集めた点の数）を足す。
                           "n_points", "n_blobs", "n_unexplained", "ec_res_shift",
                           "ec_res_noshift", "ec_moving", "cohesion",
                           # 【2026-09-09・追記1「直し」3】当て付きの点の数／
                           #   当てに合う大きさが無くて捨てた点の数。
                           "n_points_expect", "n_reject_scale",
                           # 【2026-09-10・消え方で持ち時間を決める】
                           "explained", "budget"])
                for r in self.rows:
                    w.writerow([r["step"], r["sim_time"], r["n_dets"], r["n_files"],
                               r["file_id"], r["x"], r["y"], r["area"], r["event"],
                               r["misses"], r["since_seen"], r["app_cos_created"],
                               r["mode"], r.get("n_points", ""), r.get("n_blobs", ""),
                               r.get("n_unexplained", ""),
                               r.get("ec_res_shift", ""), r.get("ec_res_noshift", ""),
                               r.get("ec_moving", ""), r.get("cohesion", ""),
                               r.get("n_points_expect", ""), r.get("n_reject_scale", ""), r.get("explained", ""), r.get("budget", "")])
        if self.attend_rows and self.attend_out:
            os.makedirs(os.path.dirname(self.attend_out) or ".", exist_ok=True)
            with open(self.attend_out, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_time", "attended_id", "visible", "misses",
                           "vanished", "dist_center", "area", "nearest_word",
                           "nearest_cos", "target_word", "target_cos",
                           # 【2026-09-09・仕様_見る側の段構成_実装 5節】末尾に追加した列
                           #（priority無効時は空文字。既存列は変えない＝追加のみ）。
                           "priority", "switch", "explore_x", "explore_y", "ior",
                           # 【2026-09-09・追記2「直し」3】決めたコマ（実際に移った
                           #   コマは従来どおりswitch列）。
                           "switch_decided"])
                for r in self.attend_rows:
                    w.writerow([r["step"], r["sim_time"], r["attended_id"], r["visible"],
                               r["misses"], r["vanished"], r["dist_center"], r["area"],
                               r["nearest_word"], r["nearest_cos"], r["target_word"],
                               r["target_cos"], r.get("priority", ""), r.get("switch", ""),
                               r.get("explore_x", ""), r.get("explore_y", ""), r.get("ior", ""),
                               r.get("switch_decided", "")])
        return {
            "検出回数": self._detect_n,
            "処理ms_平均": (round(self._detect_ms_sum / self._detect_n, 2)
                          if self._detect_n else None),
            "最終n_files": len(self.ofs.files),
        }
