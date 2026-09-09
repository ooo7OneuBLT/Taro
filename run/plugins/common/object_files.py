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
        self.exclusive_cos = float(self.config.get("exclusive_cos", 0.7))
        # 【2026-09-09・仕様_予測して確かめる検出】確認・見回り・横取り。既定は
        #   全てNone（scan_interval_s=None）＝従来どおり毎回全体切り出し（既定不変）。
        scan_interval_s = self.config.get("scan_interval_s")
        self.scan_interval_s = None if scan_interval_s is None else float(scan_interval_s)
        self.confirm_iou_thresh = float(self.config.get("confirm_iou_thresh", 0.7))
        self.confirm_area_tol = float(self.config.get("confirm_area_tol", 0.5))
        scan_change_thresh = self.config.get("scan_change_thresh")
        self.scan_change_thresh = None if scan_change_thresh is None else float(scan_change_thresh)
        scan_on_misses = self.config.get("scan_on_misses")
        self.scan_on_misses = None if scan_on_misses is None else int(scan_on_misses)
        self._last_scan_t = float("-inf")
        self._prev_small = None
        self._point_predictor = None
        # 【2026-09-09・追記「直し」1】見失い横取りは「立ち上がり」で1回だけ。
        #   misses>=scan_on_missesに“なった瞬間”のfile.idだけをここに入れ、matched/
        #   revived/lost（=misses==0に戻った）で外す。空のままなら従来と挙動不変。
        self._miss_fired = set()
        # 【2026-09-09・追記「直し」4】scan_changeの連続発火を防ぐ不応期（秒）。
        scan_change_refractory_s = self.config.get("scan_change_refractory_s", 1.0)
        self.scan_change_refractory_s = float(scan_change_refractory_s)
        self._last_scan_change_t = float("-inf")
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

        self._lazy_import()

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
        )
        if self.max_missed is not None:
            # 既定Noneのときは渡さない＝ObjectFileSystemの既定値(20)のまま（既定不変）。
            ofs_kwargs["max_missed"] = self.max_missed
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
        # 【2026-09-09・仕様_予測して確かめる検出】mode別の直近値（confirm_ms・scan_ms）。
        #   scan_interval_s=Noneのときは常にmode="scan"のままなのでobject_files_scan_ms
        #   ＝従来のobject_files_detect_msと同じ値になる（既定不変・新しい列が増えるだけ）。
        self._seg_confirm_ms = []
        self._seg_scan_ms = []
        self._mode_counts = {}

    def _lazy_import(self):
        for sub in ("senses", ""):
            p = os.path.join(_REPO_ROOT, "taro_core", "src", sub) if sub else \
                os.path.join(_REPO_ROOT, "taro_core", "src")
            if p not in sys.path:
                sys.path.insert(0, p)
        # 旧パス brain.object_files は転送のみなので使わない。
        from object_detector import (patch_features, detect, load_mobilesam,
                                      make_point_predictor, confirm_points)
        from brain.cerebral_cortex.parietal_lobe.intraparietal_sulcus import ObjectFileSystem
        self._patch_features = patch_features
        self._detect = detect
        self._load_mobilesam = load_mobilesam
        self._make_point_predictor = make_point_predictor
        self._confirm_points = confirm_points
        self._ObjectFileSystem = ObjectFileSystem

    def on_step(self, ctx):
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

        confirm_debug = None
        if self.scan_interval_s is None:
            # 従来どおり：常に全体切り出し（既定不変）。mode は新規の"scan"固定
            # （CSV・metrics に列が増えるだけで、判定・タイミングは一切変えない）。
            t0 = time.perf_counter()
            p, n = self._patch_features(self._model, img224, self.device)
            dets = self._detect(p, n, img=img224, mask_generator=self._mgen,
                                 cohesion_gap_px=self.cohesion_gap_px)
            if self.empty_cache_after_detect:
                # 【2026-09-07・メモリ削減候補(c)】既定Falseでは1行も実行されない。
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            mode = "scan"
            payload = dets
        else:
            # 【2026-09-09・仕様_予測して確かめる検出】確認・見回り・横取りの分岐。
            # confirmのときpayload＝confirm_update()用のカードごとの判定リスト
            #（追記2「直し」1節）、それ以外＝ofs.step()用のdetsリスト（従来どおり）。
            mode, payload, dt_ms, confirm_debug = self._detect_predictive(img224, t)

        prev_by_id = {f.id: f for f in self.ofs.files}
        # 【2026-09-09・枚数の上限】押し出し禁止の対象＝現在注意中のファイル
        # （_process_attentionはこのofs.stepより後に呼ぶので、ここではまだ
        # 前コマの注意状態を使う。attend=Falseなら常にNone＝従来どおり）。
        protect_id = self._attended_id if self.attend else None
        if self.scan_interval_s is not None and mode == "confirm":
            # 【2026-09-09・追記2「直し」1節】確認は頼んだカードにだけ結ぶ
            # （ofs.step()のハンガリアン法には渡さない＝別カードへの誤結合や
            #  新規カード作成を起こさない）。predict()はpredict_only()で
            #  済んでいるのでconfirm_updateは呼ばない（内部でも呼ばない）。
            res = self.ofs.confirm_update(payload, t=t)
            # n_dets（下のn_dets = len(dets)で使う）＝「一致した結果の数」
            #（追記2「直し」3節）。matched=Trueのものだけを数える。
            dets = [p for p in payload if p.get("matched")]
        else:
            dets = payload
            res = self.ofs.step(dets, t=t, protect_id=protect_id)
        # 【2026-09-09・追記「直し」1】このコマで一致(matched)・復活(revived)・
        #   削除(lost)されたfile_idは、見失いの立ち上がり判定から外す
        #   （misses==0に戻った/カード自体が無くなった＝再発火の資格を得る）。
        self._miss_fired -= set(fid for fid, _det_idx, _residual in res["matched"])
        self._miss_fired -= set(res.get("revived", []))
        self._miss_fired -= set(res.get("lost", []))

        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1
        if mode == "confirm":
            self._seg_confirm_ms.append(dt_ms)
        else:
            self._seg_scan_ms.append(dt_ms)

        if self.attend:
            # 【M3】検出・対応づけ結果（res・self.ofs.files）を読むだけ。
            #   frames_out描画（次のif）が this tick の注意状態を使えるよう、
            #   描画より先に済ませる。
            self._process_attention(ctx, t, res)

        if self.frames_out:
            self._save_detection_frame(ctx, t, img224, dets, res, prev_by_id,
                                        mode=mode, confirm_debug=confirm_debug)

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
            self.rows.append({
                "step": ctx.step, "sim_time": round(t, 3),
                "n_dets": n_dets, "n_files": n_files,
                "file_id": file_id, "x": x, "y": y, "area": area, "event": event,
                "misses": misses, "since_seen": since_seen,
                "app_cos_created": app_cos_created,
                # 【2026-09-09・仕様_予測して確かめる検出】末尾に追加した列。
                #   scan_interval_s=Noneのときは常に"scan"（既定不変・列が増えるだけ）。
                "mode": mode,
            })

    def _detect_predictive(self, img224, t):
        """【2026-09-09・仕様_予測して確かめる検出_2026-09-09「後半」2節】
        確認（confirm）・見回り（scan）・横取り（scan_change/scan_miss）の分岐。
        `scan_interval_s` が指定されているときだけ呼ばれる（既定不変）。

        Returns:
            (mode, payload, dt_ms, confirm_debug)
            mode: "scan"/"scan_change"/"scan_miss"/"scan_empty"/"confirm"
            payload: mode=="confirm"のときは `ObjectFileSystem.confirm_update()`
                に渡すカードごとの判定リスト（追記2「直し」1節）。それ以外は
                `ObjectFileSystem.step()` に渡す検出のリスト（従来どおり）。
            dt_ms: この呼び出しにかかった時間（ミリ秒）
            confirm_debug: mode=="confirm"のときだけ、点ごとの結果のリスト
                （`_save_detection_frame` の描画用）。それ以外はNone。
        """
        import numpy as np
        from PIL import Image

        # ---- 横取り条件（安い順）：①見失いの連続 ②粗い画像差分 ------------------
        # 【安い順、仕様「後半」2節】まずmisses（既に持っている値、計算コスト0）を
        # 見て、それで決まらないときだけ32x32グレースケールの差分を計算する。
        force_scan_reason = None
        if len(self.ofs.files) == 0:
            force_scan_reason = "empty"
        elif self.scan_on_misses is not None:
            # 【2026-09-09・追記「直し」1】立ち上がりで1回だけ：misses>=kに
            #   “なった瞬間”のカードだけを数える（self._miss_firedに無いもの）。
            #   同じカードは次に一致する(matched/revived)かlostになるまで
            #   再発火しない（on_step側で_miss_firedから外す）。
            newly_fired = [f.id for f in self.ofs.files
                           if f.misses >= self.scan_on_misses and f.id not in self._miss_fired]
            if newly_fired:
                force_scan_reason = "miss"
                self._miss_fired.update(newly_fired)

        small = np.asarray(
            Image.fromarray(np.clip(img224, 0, 255).astype(np.uint8))
            .convert("L").resize((32, 32)), dtype=np.float64)
        if (force_scan_reason is None and self.scan_change_thresh is not None
                and self._prev_small is not None
                # 【2026-09-09・追記「直し」4】不応期：発火後scan_change_refractory_s
                #   秒は再発火しない（既定1.0秒）。
                and (t - self._last_scan_change_t) >= self.scan_change_refractory_s):
            diff = np.abs(small - self._prev_small)
            # 既存カードの周り（半径 = sqrt(area)*224*1.5 px、32x32尺度に換算）を除く。
            mask = np.ones((32, 32), dtype=bool)
            yy, xx = np.ogrid[:32, :32]
            for f in self.ofs.files:
                fx, fy = f.pos
                r_px = (max(f.area, 0.0) ** 0.5) * 224.0 * 1.5
                r32 = r_px / 224.0 * 32.0
                gx, gy = fx / 224.0 * 32.0, fy / 224.0 * 32.0
                mask &= ((xx + 0.5 - gx) ** 2 + (yy + 0.5 - gy) ** 2) > (r32 ** 2)
            vals = diff[mask]
            if vals.size and float(vals.mean()) > self.scan_change_thresh:
                force_scan_reason = "change"
                self._last_scan_change_t = t
        self._prev_small = small

        do_scan = force_scan_reason is not None or (t - self._last_scan_t >= self.scan_interval_s)

        if do_scan:
            t0 = time.perf_counter()
            p, n = self._patch_features(self._model, img224, self.device)
            dets = self._detect(p, n, img=img224, mask_generator=self._mgen,
                                 cohesion_gap_px=self.cohesion_gap_px)
            if self.empty_cache_after_detect:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._last_scan_t = t
            mode = {"change": "scan_change", "miss": "scan_miss",
                    "empty": "scan_empty"}.get(force_scan_reason, "scan")
            return mode, dets, dt_ms, None

        # ---- 確認：既存カードの予測位置を1点ずつ聞く（画像埋め込みは1回だけ）------
        t0 = time.perf_counter()
        self.ofs.predict_only(t)
        points = [f.pos for f in self.ofs.files]
        expected_areas = [f.area for f in self.ofs.files]
        file_ids = [f.id for f in self.ofs.files]
        pos_by_id = {f.id: f.pos for f in self.ofs.files}
        if self._point_predictor is None:
            self._point_predictor = self._make_point_predictor(self._mgen)
        # 【2026-09-09・追記2「直し」1節】結果にfile_idを付けて返してもらう
        #（confirm_updateで頼んだカードにだけ結ぶため）。
        results = self._confirm_points(self._point_predictor, img224, points, expected_areas,
                                        file_ids=file_ids,
                                        iou_thresh=self.confirm_iou_thresh,
                                        area_tol=self.confirm_area_tol)
        # 【2026-09-09・追記2「直し」2節】同じ物を複数のカードが確認したら1枚だけ
        #（固体性を確認にも適用）。IoU>=0.5 または重心距離<=10px を重複とみなし、
        # 注意中のカード＞古いidの順で1枚だけ残す。残りはmisses+=1（confirm_update
        # 側で"matched": False扱いになる）。
        kept_ids = self._dedup_confirm(results)
        payload = []
        confirm_debug = []
        for r in results:
            fid = r.get("file_id")
            if r.get("ok") and fid in kept_ids:
                payload.append({"file_id": fid, "matched": True,
                                 "pos": r["pos"], "area": r["area"]})
                confirm_debug.append({"pos": r["pos"], "ok": True})
            else:
                # 【仕様「見失い」】okでなかった、または重複で外れたカードは
                # matched=False（confirm_update側でmisses+=1が起きる）。
                payload.append({"file_id": fid, "matched": False})
                confirm_debug.append({"pos": r.get("pos", pos_by_id.get(fid)), "ok": False})
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return "confirm", payload, dt_ms, confirm_debug

    def _dedup_confirm(self, results):
        """【2026-09-09・仕様_予測して確かめる検出_2026-09-09「追記2」2節】
        `results`（`confirm_points` の生の戻り値、"ok"でないものも含む）の中で
        "ok" なもの同士を比べ、マスクのIoUが0.5以上、または重心の距離が10px
        以内なら「同じ物を見ている」とみなし、注意中のカード＞古いid(小さい方)
        の順で1枚だけ残す。残りのfile_idは戻り値の集合に含めない
        （＝呼び出し元でmatched=False扱いになり、自然に消える）。

        Returns: 残す（一致として採用する）file_idの集合。
        """
        import math
        import numpy as np
        ok = [r for r in results if r.get("ok")]
        if len(ok) <= 1:
            return set(r["file_id"] for r in ok)

        def _is_dup(a, b):
            ma, mb = a.get("mask"), b.get("mask")
            if ma is not None and mb is not None:
                inter = float(np.logical_and(ma, mb).sum())
                union = float(np.logical_or(ma, mb).sum())
                if union > 0 and (inter / union) >= 0.5:
                    return True
            ax, ay = a["pos"]
            bx, by = b["pos"]
            return math.hypot(ax - bx, ay - by) <= 10.0

        attended_id = self._attended_id if self.attend else None
        order = sorted(range(len(ok)),
                        key=lambda i: (ok[i]["file_id"] != attended_id, ok[i]["file_id"]))
        excluded = set()
        for pos_a, i in enumerate(order):
            if i in excluded:
                continue
            for j in order[pos_a + 1:]:
                if j in excluded:
                    continue
                if _is_dup(ok[i], ok[j]):
                    excluded.add(j)
        return set(ok[i]["file_id"] for i in range(len(ok)) if i not in excluded)

    def _save_detection_frame(self, ctx, sim_time, img224, dets, res, prev_by_id,
                               mode="scan", confirm_debug=None):
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
        # 【2026-09-09・仕様_予測して確かめる検出「後半」2節】左上にmodeのラベル。
        #   scan_empty はSCANの見た目のまま（ラベル一覧に無い＝そのまま"SCAN"扱い）。
        _mode_label = {"confirm": "CONFIRM", "scan": "SCAN",
                       "scan_change": "SCAN(change)", "scan_miss": "SCAN(miss)"
                       }.get(mode, "SCAN")
        dr.text((4, 2), _mode_label, fill=(255, 255, 0), font=self._frame_font)
        # 確認(confirm)で当たった点は緑の小さい点、外れた予測位置は赤の×。
        if mode == "confirm" and confirm_debug:
            for d in confirm_debug:
                px, py = d["pos"]
                if d["ok"]:
                    dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 255, 0))
                else:
                    dr.line([px - 4, py - 4, px + 4, py + 4], fill=(255, 0, 0), width=2)
                    dr.line([px - 4, py + 4, px + 4, py - 4], fill=(255, 0, 0), width=2)
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
        if candidates:
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

        self.attend_rows.append({
            "step": ctx.step, "sim_time": round(t, 3),
            "attended_id": self._attended_id if self._attended_id is not None else "",
            "visible": visible_v, "misses": misses_v, "vanished": vanished_v,
            "dist_center": dist_center, "area": area,
            "nearest_word": nearest_word, "nearest_cos": _nearest_cos,
            "target_word": target_word, "target_cos": target_cos,
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
        self._seg_n_files, self._seg_n_dets, self._seg_detect_ms = [], [], []
        # 【2026-09-09・仕様_予測して確かめる検出】mode別の直近値の平均。
        #   scan_interval_s=Noneのときはmode="scan"固定なので、この2つの列は
        #   常に「confirm_ms列は出ない・scan_ms==detect_ms」になる（既定不変）。
        if self._seg_confirm_ms:
            out["object_files_confirm_ms"] = round(
                sum(self._seg_confirm_ms) / len(self._seg_confirm_ms), 2)
            self._seg_confirm_ms = []
        if self._seg_scan_ms:
            out["object_files_scan_ms"] = round(
                sum(self._seg_scan_ms) / len(self._seg_scan_ms), 2)
            self._seg_scan_ms = []
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
                           "misses", "since_seen", "app_cos_created", "mode"])
                for r in self.rows:
                    w.writerow([r["step"], r["sim_time"], r["n_dets"], r["n_files"],
                               r["file_id"], r["x"], r["y"], r["area"], r["event"],
                               r["misses"], r["since_seen"], r["app_cos_created"],
                               r["mode"]])
        if self.attend_rows and self.attend_out:
            os.makedirs(os.path.dirname(self.attend_out) or ".", exist_ok=True)
            with open(self.attend_out, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_time", "attended_id", "visible", "misses",
                           "vanished", "dist_center", "area", "nearest_word",
                           "nearest_cos", "target_word", "target_cos"])
                for r in self.attend_rows:
                    w.writerow([r["step"], r["sim_time"], r["attended_id"], r["visible"],
                               r["misses"], r["vanished"], r["dist_center"], r["area"],
                               r["nearest_word"], r["nearest_cos"], r["target_word"],
                               r["target_cos"]])
        return {
            "検出回数": self._detect_n,
            "処理ms_平均": (round(self._detect_ms_sum / self._detect_n, 2)
                          if self._detect_n else None),
            "最終n_files": len(self.ofs.files),
        }
