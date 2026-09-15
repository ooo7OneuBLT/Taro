# -*- coding: utf-8 -*-
"""視覚と注意の手順そのもの（脳の部品）。

【なぜ、2026-09-13・設計_視覚を脳へ戻す】以前はこの手順一式が
`run/plugins/common/object_files.py`（測る道具の置き場）の中にあった。
道具は「太郎を変えない」約束（`run/plugins/base.py`）のはずなのに、実際には
物を見つけ・どこに注意するかを決め・眼球へ命令を出すところまでやっていた。
仕様：`doc/設計/設計_視覚を脳へ戻す_2026-09-13.md`。

このクラスは**手順の移設だけ**を行う。計算式・呼ぶ順・乱数の使い方は
`run/plugins/common/object_files.py` に元々あったコードから1つも変えていない
（ctx を受け取れないぶんの引数の受け渡し方だけを変えた）。

【段構成】`step()` が毎tick呼ばれる想定：
  段0 遠心性コピー（毎tick、無効なら何もしない）
  （注意の対象を毎tick更新する追跡だけ、毎tick行う）
  ここで sim時間の間隔（interval_s）に満たなければ打ち切り（`detected=False`）
  段1 前注意の地図（動いた塊）
  段2 目立ちの地図 → 段3 目的の地図（goal） → 段4 場所の優先度地図
  段5 点プロンプトで切り出す（MobileSAM）
  段6 物体ファイルの更新（カルマン＋ハンガリアン、ObjectFileSystem）
  段7 注意する物を決める・消失を判定する（`attend`設定時のみ）
  段8 注意が移った瞬間だけ眼球へ命令（`orienting.set_map_target`）

`step()` は ctx を受け取らない。呼び出し側（`run/plugins/common/object_files.py`）が
ctx から必要な値を取り出して引数で渡し、戻り値をまた ctx へ書き戻す。
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# taro_core/src/brain/cerebral_cortex/parietal_lobe から5つ上がリポジトリルート。
_REPO_ROOT = os.path.abspath(os.path.join(
    _HERE, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(_REPO_ROOT, p)


class VisualAttention:
    """1コマぶんの「見る」を最初から最後まで回す。ctxを受け取らない。"""

    def __init__(self, *, config):
        self.config = config or {}
        c = self.config
        # ---- 元 ObjectFiles.setup() のうち、知覚に関する部分をそのまま ----
        self.interval_s = float(c.get("interval_s", 1.0))
        self.device = c.get("device")
        self.points_per_side = c.get("points_per_side")
        # 【2026-09-09・仕様_物体ファイルの人間寄せ_凝集性・連続性・上限】
        #   既定は全てNone（凝集性・連続性・上限とも従来どおりOFF）＝既定不変。
        self.cohesion_gap_px = c.get("cohesion_gap_px")
        revive_window_s = c.get("revive_window_s")
        self.revive_window_s = None if revive_window_s is None else float(revive_window_s)
        self.revive_cos = float(c.get("revive_cos", 0.8))
        self.revive_dist_px = float(c.get("revive_dist_px", 40.0))
        max_files = c.get("max_files")
        self.max_files = None if max_files is None else int(max_files)
        max_missed = c.get("max_missed")
        self.max_missed = None if max_missed is None else int(max_missed)
        # 【2026-09-09・仕様_物体ファイルの重複をなくす「後半」4節】既定は全て
        #   ObjectFileSystemの既定値と同じ（coast_max_s=None・uncertainty_penalty=0.0・
        #   exclusive_dist_px=None・exclusive_cos=0.7）＝既定不変。
        coast_max_s = c.get("coast_max_s")
        self.coast_max_s = None if coast_max_s is None else float(coast_max_s)
        self.uncertainty_penalty = float(c.get("uncertainty_penalty", 0.0))
        exclusive_dist_px = c.get("exclusive_dist_px")
        self.exclusive_dist_px = None if exclusive_dist_px is None else float(exclusive_dist_px)
        # 【2026-09-09・追記_物体ファイルの重複をなくす「追記」1節】JSONの
        #   null（＝Python側のNone）がそのまま届く。既定は0.7（キー省略時のみ）。
        exclusive_cos = c.get("exclusive_cos", 0.7)
        self.exclusive_cos = None if exclusive_cos is None else float(exclusive_cos)
        self.exclusive_scale_by_size = bool(c.get("exclusive_scale_by_size", False))
        coast_min_speed_px_s = c.get("coast_min_speed_px_s")
        self.coast_min_speed_px_s = (None if coast_min_speed_px_s is None
                                      else float(coast_min_speed_px_s))
        # 【2026-09-09・仕様_見る側_道を1本にする】確認と切り出しの2本の道を
        #   `_detect_frame` 1本にまとめた。
        self._point_predictor = None
        # 【2026-09-07・メモリ削減候補(a)】既定None＝SamAutomaticMaskGeneratorの
        #   既定値(64)のまま＝1ビットも変わらない。渡したときだけ上書きする。
        self.points_per_batch = c.get("points_per_batch")
        # 【2026-09-07・メモリ削減候補(c)】既定False＝従来どおり呼ばない。
        self.empty_cache_after_detect = bool(c.get("empty_cache_after_detect", False))

        # 【2026-09-09・仕様_見る側の段構成_実装】段0・段1・段6。既定は全てNone
        #   （無効）＝既定不変（仕様書10節「既定不変」）。段4の
        #   appearance_gate_cosもここでまとめて読む。
        self.attention_cfg = c.get("attention")
        # 【2026-09-10・4段目】注意が移った先へ目も向けるか。既定False＝既定不変。
        self.gaze_from_attention = bool(c.get("gaze_from_attention", False))
        # 【2026-09-11・K1「記憶からの山」】上からの目的（5段目）。
        self.goal_cfg = c.get("goal")
        self._goal_id = None          # いま的にしているカードの id
        self._goal_until = -1.0       # この時刻まで同じ的を保つ［sim秒］
        self._goal_repick = 0         # このコマで選び直したか（2＝的が死んだので）
        self._gaze_cmd_sent = 0
        self._fire_probe = []
        self._sal = None
        self._spri = None
        self.efference_copy_cfg = c.get("efference_copy")
        self.preattentive_cfg = c.get("preattentive")
        self.priority_cfg = c.get("priority")
        appearance_gate_cos = c.get("appearance_gate_cos")
        self.appearance_gate_cos = (None if appearance_gate_cos is None
                                     else float(appearance_gate_cos))
        # 【2026-09-09・仕様_見る側_道を1本にする 後半2節】既定"mahal"＝
        #   ObjectFileSystem側の既定と一致するので無条件で渡してよい（既定不変）。
        self.pos_gate = c.get("pos_gate", "mahal")
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
            coast_max_s=self.coast_max_s,
            uncertainty_penalty=self.uncertainty_penalty,
            exclusive_dist_px=self.exclusive_dist_px,
            exclusive_cos=self.exclusive_cos,
            pos_gate=self.pos_gate,
            exclusive_scale_by_size=self.exclusive_scale_by_size,
            coast_min_speed_px_s=self.coast_min_speed_px_s,
            frame_dt_s=self.interval_s,
            appearance_gate_cos=self.appearance_gate_cos,
        )
        if self.max_missed is not None:
            # 既定Noneのときは渡さない＝ObjectFileSystemの既定値(20)のまま（既定不変）。
            ofs_kwargs["max_missed"] = self.max_missed
        # 【2026-09-10・仕様_消え方で持ち時間を決める】vanish_budget（辞書 or None）。
        self.vanish_budget = c.get("vanish_budget")
        if self.vanish_budget is not None:
            ofs_kwargs["vanish_budget"] = self.vanish_budget
        self.ofs = self._ObjectFileSystem(**ofs_kwargs)

        # 【M3・2026-09-06】既定Falseでは以下のattend関連の状態・処理は一切使われない。
        self.attend = bool(c.get("attend", False))
        self.attend_radius = float(c.get("attend_radius", 56))
        self.vanish_misses = int(c.get("vanish_misses", 1))
        # 【M7b-1・2026-09-09】注意の加点用の係数（既定0＝既定不変）。仕様「後半」4節。
        self.attend_surprise_gain = float(c.get("attend_surprise_gain", 0.0))
        attend_out = c.get("attend_out")
        self.attend_out = _abs_path(attend_out) if attend_out else None
        self._attended_id = None
        self._attended_last_seen_vec = None
        self._attended_last_seen_time = None
        # 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う(0)】
        self._attended_visible = False
        self._last_nearest_word = ""
        self._last_vanished = False
        self._target_word = {}
        self._last_target = None
        self.attend_rows = []
        self._attended_object = None

        self._last_t = float("-inf")

        # ---- 呼び出し側が毎回 step() に渡す「今tickの入力」の控え ----------------
        #   （元は ctx.taro / ctx.last_vision_vec / ctx.surprise_trace /
        #   ctx.last_parent_utterance を、呼ばれるたびに直接読んでいた。ctxを
        #   受け取れないので、step()の頭で一度だけ受け取って控える）。
        self._cur_vision_vec = None
        self._cur_surprise_trace = {}
        self._cur_vocab = None
        self._cur_step = None
        self.noticed_gone = False

        # ---- 元は ctx.* に置いていた「他のプラグイン・脳配線から読める」出力 ----
        #   （ctxを受け取れないので、ここに持ち、step()の戻り値で都度渡す。
        #   呼び出し側がそのままctxへ置き直す）。
        self.efference = None
        self._priority_map_result = None
        self._priority_map_result_t = None
        self.goal_point = None
        self.salience_map = None
        self.attention_point = None
        self.attention_switch_t = None

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

    # ------------------------------------------------------------------
    # 1コマぶんの「見る」を最初から最後まで回す。
    # ------------------------------------------------------------------
    def step(self, *, obs, orienting, t, step=None, fovy=None, surprise_trace=None,
             vocab=None, vision_vec=None, parent_target=None):
        """呼び出し側（プラグイン）が毎tick呼ぶ。ctxは受け取らない。

        obs           : ctx.last["obs_out"] 相当（Noneも可）
        orienting     : ctx.env.unwrapped._orienting 相当
        t             : ctx.data.time（sim時刻）
        step          : ctx.step 相当（注意.csvのstep列にそのまま使うだけ）
        fovy          : 遠心性コピーの初回構築だけに使うカメラ視野角
                        （呼び出し側が ctx.env から読んで渡す。cam_fovyが
                        読めない場合の既定・警告は呼び出し側の役目に変わった
                        ＝ctxを持たない側の必然的な変更。値・タイミングは
                        元の_ensure_efference_copyと同じ）
        surprise_trace: ctx.surprise_trace 相当
        vocab         : ctx.taro.hearing.vocab 相当
        vision_vec    : ctx.last_vision_vec 相当
        parent_target : ctx.last_parent_utterance 相当

        戻り値は辞書。CSV・PNGを書くのに必要な値を全部入れる。
        """
        self._cur_vision_vec = vision_vec
        self._cur_surprise_trace = surprise_trace or {}
        self._cur_vocab = vocab
        self._cur_step = step

        # 【2026-09-09・仕様_見る側の段構成_実装 1節】段0 遠心性コピー。毎tick
        #   呼ぶ（検出コマでなくても）。efference_copy=None（既定）では一切
        #   呼ばれず、self.efferenceも更新されない＝既定不変。
        if self.efference_copy_cfg is not None:
            self._ensure_efference_copy(fovy)
            if orienting is None and not self._warned_no_orienting:
                print("[efference] WARNING _orienting が読めない。全て0/Falseとして続行")
                self._warned_no_orienting = True
            self.efference = self._ec.update(orienting, float(t))
            # 【2026-09-10】配線が届いているかを走行の頭で1回だけ実測して出す
            #   （設定が届かず静かに既定値で走る事故が通算5件あったため）。
            if orienting is not None and not getattr(self, "_ec_reported", False):
                self._ec_reported = True
                print("[efference] 配線の確認: data=%s eye_qadr=%s neck=%s "
                      "version_h=%.4f eye_v=%.4f nu=%d"
                      % (orienting.data is not None, dict(orienting.eye_qadr),
                         list(orienting.neck_idx.keys()),
                         orienting._version_h_deg(),
                         orienting._angle_deg(orienting.eye_qadr["v"]),
                         orienting.n_actuator), flush=True)

        if self.attend:
            # 【M3】親の発話は検出コマ(interval_s)より細かい頻度で来るので、
            #   「目標→語」の表だけは毎tick更新する（読むだけ・attend=Falseでは
            #   一切呼ばれない）。
            self._track_parent_target(parent_target)
            # 【M4b・2026-09-06】M4(0)の毎tick上書きは検出コマの処理
            #   （_process_attention、self.ofs.step直後）でだけ行う。

        t = float(t)
        if t - self._last_t < self.interval_s:
            return self._result(detected=False)
        self._last_t = t

        if obs is None or "eye_left" not in obs:
            return self._result(detected=False)

        import numpy as np
        from PIL import Image
        img = np.asarray(obs["eye_left"])
        img224 = np.array(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                           .resize((224, 224)))

        # 【2026-09-09・仕様_見る側の段構成_実装 1節】段0が有効なとき、段3の
        #   predict()に渡すshift_pred（無効時はNone＝既定不変）。
        shift_pred = None
        if self.efference_copy_cfg is not None:
            shift_pred = (self.efference or {}).get("shift_pred")

        # prev_by_id は _detect_frame()（内部でofs.stepを呼ぶ）が呼ばれる**前**（＝削除される
        # 前）に確定させる。
        prev_by_id = {f.id: f for f in self.ofs.files}
        # 【2026-09-09・枚数の上限】押し出し禁止の対象＝現在注意中のファイル
        # （_process_attentionはこのofs.stepより後に呼ぶので、ここではまだ
        # 前コマの注意状態を使う。attend=Falseなら常にNone＝従来どおり）。
        protect_id = self._attended_id if self.attend else None

        # 【2026-09-09・仕様_見る側_道を1本にする】検出コマの処理は_detect_frame
        #   1本（旧「確認」道・旧「切り出し」道の2本を統合）。
        mode, res, dt_ms, onset_extra = self._detect_frame(orienting, img224, t, protect_id)
        dets = onset_extra.pop("new_dets", [])

        if self.attend:
            # 【M3】検出・対応づけ結果（res・self.ofs.files）を読むだけ。
            self._process_attention(t, res)

        return self._result(detected=True, mode=mode, res=res, dt_ms=dt_ms,
                             onset_extra=onset_extra, dets=dets, img224=img224,
                             prev_by_id=prev_by_id)

    def _result(self, *, detected, mode=None, res=None, dt_ms=None,
                onset_extra=None, dets=None, img224=None, prev_by_id=None):
        return {
            "detected": detected,
            "mode": mode, "res": res, "dt_ms": dt_ms,
            "onset_extra": onset_extra, "dets": dets, "img224": img224,
            "prev_by_id": prev_by_id,
            "files": self.ofs.files,
            "efference": self.efference,
            "attended_id": self._attended_id,
            "attended_object": self._attended_object,
            "priority_map_result": self._priority_map_result,
            "priority_map_result_t": self._priority_map_result_t,
            "goal_point": self.goal_point,
            "salience_map": self.salience_map,
            "attention_point": self.attention_point,
            "attention_switch_t": self.attention_switch_t,
            "last_nearest_word": self._last_nearest_word,
            "last_vanished": self._last_vanished,
            "noticed_gone": self.noticed_gone,
        }

    def _log_goal_terms(self, onset_extra, goal_info, goal_map, salience, attn_res):
        """【2026-09-11・K1】的の升と勝った升で、各項がいくらだったかを残す。

        これが無いと「効かなかった」の原因を分けられない（仕様 後半4節）。
        山が小さいのか／復帰抑制に消されたのか／溜めが目立ちに埋もれたのか。
        """
        import numpy as np
        cell = goal_map.shape[0]
        ior = attn_res["ior"]
        acc = getattr(self._spri, "acc", None)

        def at(cx, cy, tag):
            c = int(min(max(cx, 0), cell - 1))
            r = int(min(max(cy, 0), cell - 1))
            onset_extra["g_at_" + tag] = round(float(goal_map[r, c]), 4)
            onset_extra["sal_at_" + tag] = round(float(np.asarray(salience)[r, c]), 4)
            onset_extra["ior_at_" + tag] = round(float(np.asarray(ior)[r, c]), 4)
            onset_extra["acc_at_" + tag] = ("" if acc is None
                                             else round(float(np.asarray(acc)[r, c]), 4))

        if goal_info.get("goal_cx") != "":
            at(goal_info["goal_cx"], goal_info["goal_cy"], "goal")
        wc, wr = attn_res["winner_cell"]
        at(wc, wr, "win")

    def _goal_map(self, t, cell):
        """【2026-09-11・K1】上からの目的を cell×cell の地図にする。

        的にするのは**見失い中のカード**＝生きているが直近の照合で対応が
        つかなかったもの（`misses > 0`）。記憶だけが位置を知っている状態で、
        画像をいくら照らしても見つからない＝記憶の項にしかできないこと。

        位置はカードが自分で持っているので、**見た目の照合はしない**。
        選んだカードの位置に山（ガウス）を1つ置くだけ。

        注意：この関数は照合より**前**に走るので、`misses` は前コマまでの値。
        それでよい（「直近の検出時点で見失っていた」が的の定義）。

        Returns: (goal地図 or None, 記録用の辞書)
        """
        import numpy as np
        info = {"goal_id": "", "goal_cx": "", "goal_cy": "", "goal_repick": 0}
        if not self.goal_cfg:
            return None, info
        cfg = self.goal_cfg if isinstance(self.goal_cfg, dict) else {}
        g = float(cfg.get("g", 1.0))              # 山の高さ（目立ちは0〜1）
        sigma = float(cfg.get("sigma", 1.5))      # 山の広がり［升］
        hold_s = float(cfg.get("hold_s", 3.0))    # 同じ的を保つ時間［秒］

        # 見失い中のカード。**よく見えていたカードを優先**する（hits の多い順、
        #   同数なら id の若い順＝乱数を使わない・再現する）。
        cands = sorted([f for f in self.ofs.files if getattr(f, "misses", 0) > 0],
                       key=lambda f: (-int(getattr(f, "hits", 0)), f.id))
        if not cands:
            self._goal_id = None
            return None, info

        alive = {f.id: f for f in self.ofs.files}
        cur = alive.get(self._goal_id)
        if self._goal_id is not None and cur is None:
            self._goal_repick = 2          # 的が死んだ（lost）ので選び直す
            self._goal_id = None
        if (self._goal_id is None or cur is None
                or getattr(cur, "misses", 0) <= 0 or t >= self._goal_until):
            if self._goal_repick != 2:
                self._goal_repick = 1
            self._goal_id = cands[0].id
            self._goal_until = t + hold_s
        target = alive.get(self._goal_id)
        if target is None:
            return None, info

        step = 224.0 / float(cell)
        px, py = float(target.pos[0]), float(target.pos[1])
        cx, cy = px / step, py / step
        ys, xs = np.mgrid[0:cell, 0:cell]
        gmap = g * np.exp(-(((xs + 0.5) - cx) ** 2 + ((ys + 0.5) - cy) ** 2)
                          / (2.0 * sigma * sigma))
        info = {"goal_id": target.id, "goal_cx": round(cx, 2),
                "goal_cy": round(cy, 2), "goal_repick": self._goal_repick}
        self._goal_repick = 0
        return gmap.astype(np.float32), info

    def _detect_frame(self, orienting, img224, t, protect_id):
        """【2026-09-09・仕様_見る側_道を1本にする 後半1節】検出コマの処理
        1本（旧「確認」道・旧「切り出し」道を統合）。

        0. eff = self.efference（無効なら shift=(0,0)・moving=False）
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

        `orienting` は ctx.env.unwrapped._orienting 相当（呼び出し側から渡される）。

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
                        "n_points_expect": "", "n_reject_scale": "",
                        "n_reject_edge": "", "n_reject_area": "",
                        "n_reject_dedup": "",
                        # 【2026-09-11・K1】最初のコマは優先度地図を通らない
                        "goal_id": "", "goal_cx": "", "goal_cy": "", "goal_repick": "",
                        "g_at_goal": "", "sal_at_goal": "", "ior_at_goal": "",
                        "acc_at_goal": "", "g_at_win": "", "sal_at_win": "",
                        "ior_at_win": "", "acc_at_win": ""}

        # ---- 0. 遠心性コピー（無効ならshift=(0,0)・moving=False） --------------
        eff = self.efference or {}
        # 【2026-09-10】検出コマは2〜3tickに1回しか来ないので、1tickぶんの
        #   shift_actual ではなく、**前の検出コマからの合計**を受け取る
        #   （efference_copy.consume_shift のコメント参照。受け取ると 0 に戻る）。
        if self._ec is not None:
            shift_actual = self._ec.consume_shift()
        else:
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
                       "shift_meas": (0.0, 0.0), "static_sal": None, "valid": False}
        onset_extra["n_blobs"] = len(pre_res["blobs"])
        onset_extra["ec_res_shift"] = round(pre_res["motion_mean"], 4)
        onset_extra["ec_res_noshift"] = round(pre_res["motion_mean_noshift"], 4)
        onset_extra["ec_moving"] = moving
        # 【2026-09-10・測定器の直し】遠心性コピーの予告と、画像から測った実際の
        #   ずれを、そのまま並べて残す。
        _sm = pre_res.get("shift_meas", (0.0, 0.0))
        onset_extra["ec_dx_used"] = round(float(shift_actual[0]), 2)
        onset_extra["ec_dy_used"] = round(float(shift_actual[1]), 2)
        onset_extra["ec_dx_meas"] = round(float(_sm[0]), 2)
        onset_extra["ec_dy_meas"] = round(float(_sm[1]), 2)
        onset_extra["ec_eye_h"] = round(float(eff.get("eye_h", 0.0)), 3)
        onset_extra["ec_eye_v"] = round(float(eff.get("eye_v", 0.0)), 3)
        _sp = eff.get("shift_pred") or (0.0, 0.0)
        onset_extra["ec_dx_pred"] = round(float(_sp[0]), 2)
        onset_extra["ec_dy_pred"] = round(float(_sp[1]), 2)
        onset_extra["img_mean"] = round(float(np.mean(img224)), 2)
        onset_extra["img_std"] = round(float(np.std(img224)), 2)
        onset_extra["sacc_n"] = int(getattr(orienting, "n_saccades", 0)) if orienting else ""

        # ---- 1b. 【2026-09-10・見る側1〜2段目】目立ちの地図 → 場所の優先度地図。
        #      attention=None（既定）なら1行も通らない。
        attn_res = None
        if self._spri is not None:
            sal_res = self._sal.update(img224, shift_actual, moving)
            # 【2026-09-11・K1】5段目。goal 未設定なら None のまま＝既定不変。
            goal_map, goal_info = self._goal_map(t, self._spri.cell)
            attn_res = self._spri.update(sal_res["salience"], shift_px=shift_pred,
                                          dt=self.interval_s, goal=goal_map)
            onset_extra.update(goal_info)
            # 【2026-09-11】目的の場所を画素で外へ出す（view_video が重ねて描く）。
            #   目的が無いコマは None にして、前のコマの印が残らないようにする。
            _step_px = 224.0 / float(self._spri.cell)
            self.goal_point = (None if goal_info.get("goal_cx") == "" else
                               (goal_info["goal_cx"] * _step_px,
                                goal_info["goal_cy"] * _step_px))
            if goal_map is not None:
                self._log_goal_terms(onset_extra, goal_info, goal_map,
                                      sal_res["salience"], attn_res)
            self.salience_map = sal_res["salience"]
            self.attention_point = attn_res["winner_px"]
            onset_extra["attn_x"] = round(attn_res["winner_px"][0], 1)
            onset_extra["attn_y"] = round(attn_res["winner_px"][1], 1)
            onset_extra["attn_switched"] = attn_res["switched"]
            _wc, _wr = attn_res["winner_cell"]
            for _k, _v in sal_res["channels"].items():
                onset_extra["ch_" + _k] = round(float(_v[_wr, _wc]), 4)
            # 【2026-09-10・測定】撃った方向を、地図はどう評価していたか。
            if orienting is not None and getattr(orienting, "fire_log", None):
                _sal = sal_res["salience"]; _ior = attn_res["ior"]
                _cell = _sal.shape[0]; _step = 224.0 / _cell
                for (_ft, _h, _v, _kind) in orienting.fire_log:
                    _c = int(min(max((112.0 + _h * 112.0) / _step, 0), _cell - 1))
                    _r = int(min(max((112.0 - _v * 112.0) / _step, 0), _cell - 1))
                    self._fire_probe.append((_kind, float(_sal[_r, _c]),
                                              float(_ior[_r, _c])))
                orienting.fire_log = []
            # 【2026-09-10・見る側4段目】地図を読んで目を向ける道。注意が移った
            #   瞬間だけ、その場所を目標として渡す。自前で見つける道は残したまま
            #   （人間も上丘が自分で撃つ道は無くならない）、2つが同じ筋へつながる形。
            #   gaze=False（既定）なら1行も通らない。
            if self.gaze_from_attention and attn_res["switched"]:
                if orienting is not None and hasattr(orienting, "set_map_target"):
                    wx, wy = attn_res["winner_px"]
                    # 視野の方向（[-1,1]、右・上が正）へ。画像の y は下向きなので反転。
                    orienting.set_map_target((wx - 112.0) / 112.0,
                                              (112.0 - wy) / 112.0)
                    self._gaze_cmd_sent += 1
                    onset_extra["gaze_cmd"] = 1

        # ---- 2. カードを予測位置へ進める（★shiftを必ず渡す・重大1の直し） -------
        self.ofs.predict_only(t, shift=shift_pred)

        # ---- 3. 点を集める -------------------------------------------------------
        # 【2026-09-09・追記1「直し」1】各点に「このくらいの大きさのはず」の
        #   当て（expect_area、面積比0-1・無ければNone）を付ける
        points = []
        # 3a. カードの予測位置（misses に関わらず全カード）：当て＝f.area
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
            for _k in ("n_reject_edge", "n_reject_area", "n_reject_dedup"):
                onset_extra[_k] = seg_stats.get(_k, 0)
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
            self._priority_map_result = {
                "attended_id": best, "switch_signal": attn_res["switched"],
                "priority": {}, "explore_point": None, "ior": {},
                "switch_decided": attn_res["switched"]}
            self._priority_map_result_t = t
            if attn_res["switched"]:
                self.attention_switch_t = t
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
            surprise_trace = self._cur_surprise_trace
            priority_result = self._priority.update(
                candidates, t, motion_by_id, pre_res["static_sal"], surprise_trace,
                {}, {}, moving, self.interval_s)
            explore_point = priority_result.get("explore_point")
            self._priority_map_result = priority_result
            self._priority_map_result_t = t
        self._prev_explore_point = explore_point

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return "frame", res, dt_ms, onset_extra

    def _ensure_efference_copy(self, fovy):
        """1回だけ作って使い回す。

        【2026-09-13・ctxを持たない側への移設】fovyの解決（cam_fovyが読めない
        ときに60度で代用し警告を1回出す部分）は、ctx.envへのアクセスが要る
        ため呼び出し側（プラグイン）に移した。f_pxの計算式自体は変えていない。
        """
        if self._ec is not None or self.efference_copy_cfg is None:
            return
        import math
        cfg = self.efference_copy_cfg
        _fovy = 60.0 if fovy is None else float(fovy)
        f_px = 112.0 / math.tan(math.radians(_fovy / 2.0))
        self._ec = self._EfferenceCopy(
            f_px=f_px,
            sign_h=float(cfg.get("sign_h", 1.0)),
            sign_v=float(cfg.get("sign_v", 1.0)),
            moving_extra_ticks=int(cfg.get("moving_extra_ticks", 1)))

    # ---- M3・2026-09-06 注意中の物体ファイルと消失信号 --------------------------

    def _track_parent_target(self, parent_target):
        """親が今どの物を指しているか（target id）と、その語（初回の命名文の
        text）の対応を毎tick更新する（決めたこと3のtarget_word用）。
        cause="vanish"（「○○ないね」の全文）とcause="voice"（相槌等、必ずしも
        単語ではない）は語の登録元から除く（targetの更新自体はする）。"""
        for ev in (parent_target or []):
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

    # 【2026-09-15】_lexicon_proto / _nearest_lexicon_word / _word_cos は削除した。
    #   意味の表（lexicon.proto）を読んで 注意.csv の nearest_word / nearest_cos /
    #   target_cos を書くためだけの関数で、注意の動き自体には使っていなかった。
    #   表そのものを削除したため、これらの列は空になる。

    def _process_attention(self, t, res):
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
                and self._priority_map_result_t == t
                and self._priority_map_result is not None):
            # 【2026-09-10・見る側3段目】場所の地図が決めた注意をそのまま使う。
            #   物の一覧から選び直さない（順序は「場所が先、物が後」）。
            priority_result = self._priority_map_result
            if priority_result["attended_id"] is not None:
                self._attended_id = priority_result["attended_id"]
        elif self._priority is not None:
            # 【2026-09-09・仕様_見る側の段構成_実装 5節】段6。priority設定が
            #   有効なときだけ置き換える（無効時は下のelseで従来どおり1行も
            #   変えない）。中央56pxの絞りはPriorityMapの外（上のcandidates）で
            #   従来どおり適用し、PriorityMapには候補だけ渡す。
            if (self._priority_map_result_t == t
                    and self._priority_map_result is not None):
                # `_detect_frame` が同じtickで既に1回だけupdate()を呼んでいる
                # （探索点のため）。ここで2回目を呼ぶと内部状態（切り替えの
                # 遅れ・IOR）が壊れるので使い回す。
                priority_result = self._priority_map_result
            else:
                moving = bool((self.efference or {}).get("moving", False))
                _trace = self._cur_surprise_trace
                priority_result = self._priority.update(
                    candidates, t, {}, None, _trace, {}, {}, moving, self.interval_s)
            if priority_result["attended_id"] is not None:
                self._attended_id = priority_result["attended_id"]
        elif candidates:
            # 中央付近に複数あれば一番大きいものを取る（1個の物が部位ごとに
            # 複数ファイルへ分裂する既知の性質＝M1.5bで判明への対処）。
            # 【M7b-1・2026-09-09】仕様「後半」4節：「今の面積」に「その物の驚きの
            #   余韻」を足す。gain=0（既定）なら surprise_trace を足しても面積の
            #   大小関係は変わらない＝既定不変。
            gain = self.attend_surprise_gain
            _trace = self._cur_surprise_trace
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
        # 【M4・2026-09-06(0)】次の検出コマまでの毎tick更新（呼び出し側の
        #   毎tick呼び出し）が使う「直近の検出コマで可視だったか」をここで
        #   確定させる。
        self._attended_visible = attended_f is not None and attended_f.misses == 0

        # ---- 決めたこと2：「消えた物の見た目」の控え ---------------------------
        # 【M4b：検出コマで一致した瞬間だけ控える（毎tickだと引っ込め中に
        #   空の机で上書きされる。f81で実測）】確認して実際にその物が見つかった
        #   瞬間（self._attended_visible、検出コマ内）だけ控えを更新する。
        #   確認と確認の間は控えを触らない。
        if self._attended_visible and self._cur_vision_vec is not None:
            self._attended_last_seen_vec = np.array(
                self._cur_vision_vec, dtype=np.float64, copy=True)
            self._attended_last_seen_time = t
            if self._attended_object is not None:
                self._attended_object["last_seen_vec"] = self._attended_last_seen_vec
                self._attended_object["last_seen_time"] = self._attended_last_seen_time

        # ---- 決めたこと3の前段：消失信号 -------------------------------------
        _prev_vanished = self._last_vanished
        vanished = attended_f is not None and attended_f.misses >= self.vanish_misses
        self._last_vanished = bool(vanished)
        # 【M4c・2026-09-06】vanished が偽→真に変わった瞬間だけ、世界（親）へ
        #   合図する。声の合図（run/trainer.py:934 _taro_voice_signal）と同じ
        #   流儀。人間の親は子の頭の中を直接読めない（視線や探す動きから
        #   推し量る）ので、これは人間模倣からの逸脱（世界が太郎の内部状態を
        #   読む）。doc/人間模倣からの逸脱リスト.md「その45」に登録済み。
        #   【2026-09-13・設計_視覚を脳へ戻す】この合図（環境への書き込み）は
        #   今回の移設範囲外（別件・触らない）。呼び出し側が`noticed_gone`を
        #   見て、元と同じ場所（env.unwrapped._taro_noticed_gone_time）へ書く。
        self.noticed_gone = bool(vanished and not _prev_vanished)

        # ---- nearest_word（描画・CSV両方で使うのでここで一度だけ計算） --------
        # 【2026-09-15】意味の表の削除に伴い常に空（上のコメント参照）。
        nearest_word, _nearest_cos = "", ""
        self._last_nearest_word = nearest_word

        # 【2026-09-09・仕様_見る側の段構成_実装 5節】切り替えた合図。priority
        #   無効なら常にFalse・attention_switch_tは触らない＝既定不変。
        switch_signal = bool(priority_result["switch_signal"]) if priority_result else False
        if switch_signal:
            self.attention_switch_t = t

        # ---- attended_object ----------------------------------------------
        if attended_f is None:
            self._attended_object = None
        else:
            self._attended_object = {
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
                self._attended_object["switch_signal"] = switch_signal
                self._attended_object["priority"] = priority_result.get("priority", {})

        if self.attend_out is None:
            return

        target_word = (self._target_word.get(self._last_target, "")
                        if self._last_target is not None else "")
        target_cos = ""
        if target_word and self._attended_last_seen_vec is not None:
            # 【2026-09-15】意味の表の削除に伴い常に空。
            target_cos = ""

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
            "step": self._cur_step, "sim_time": round(t, 3),
            "attended_id": self._attended_id if self._attended_id is not None else "",
            "visible": visible_v, "misses": misses_v, "vanished": vanished_v,
            "dist_center": dist_center, "area": area,
            "nearest_word": nearest_word, "nearest_cos": _nearest_cos,
            "target_word": target_word, "target_cos": target_cos,
            "priority": priority_v, "switch": switch_v,
            "explore_x": explore_x_v, "explore_y": explore_y_v, "ior": ior_v,
            "switch_decided": switch_decided_v,
        })
