# -*- coding: utf-8 -*-
"""測る道具：物体ファイル・注意の見た目をCSV・PNGへ書き出す。

【いまの姿（段A・2026-09-13）】視覚と注意の**持ち主は太郎**
（実験ファイルの `taro.visual_attention`。実体は
`taro_core/.../parietal_lobe/visual_attention.py` の `VisualAttention`、
組み立ては `run/taro_setup.py: build_visual_attention`、
毎tick呼ぶのは `run/trainer.py: _visual_attention_step`）。
**このファイルは、その最後の結果を読んでCSV・PNGに残すだけ**。

  ⇒ **この道具を実験ファイルから消しても、太郎の能力は落ちない。**
     （確かめ方：`plugins.object_files` を消した走行で `太郎の発話.csv`・
       `世界の予測器.csv` が残した走行と完全一致することを確認済み）

  1. 太郎の `VisualAttention` の最後の結果を読む（setup / on_step）
  2. CSV・PNGへ書く（_save_detection_frame ほか）
  3. `metrics` / `report` を返す

【後方互換（いずれ消す）】`taro.visual_attention` を書かない古い実験では、
このファイルが**今までどおり自分で `VisualAttention` を作って毎tick呼び、
掲示板(ctx)へも置く**。これは規約違反なので `intervenes` で名乗ってある。
すべての実験が `taro.visual_attention` へ移ったら、後方互換ごと消す。

【なぜこうなったか、2026-09-13】以前はこのファイルが、物を見つけ・どこに注意するかを
決め・眼球へ命令を出すところまで全部やっていた（測る道具は「太郎を変えない」約束の
はずなのに、太郎そのものを動かしていた）。
仕様：`doc/設計_視覚を脳へ戻す_2026-09-13.md`（第1段）と
`doc/設計_太郎をCoreで完結させる_2026-09-13.md`（段A）。

【一度やって間違えたこと】その途中で「環境が持つ」形も作ったが誤りだった。
(1) 環境の step の中で呼ぶと位置が変わり、発話が読む「注意している物」が
前tick→今tickへずれる (2) 環境は太郎の語彙を覗けないので語と物が結びつかない。
環境側の経路は撤去済み（`E/scripts/e_toy_env.py` の同日コメント）。

計算式・呼ぶ順・乱数の使い方は移設前と1つも変えていない（移すだけ）。

【なぜ、2026-09-04（元の経緯）】太郎はMobileSAMによる物の切り出しと、物体ファイルに
よる「同じ物が動いている」追跡の両方をすでに持っているが、どちらも書き捨ての
検証スクリプト（`F/scripts/f65_mobilesam_detect_probe.py` 等）でしか動かした
ことが無く、太郎が実際に学習している最中には一度も働いていない。
仕様：`F/docs/物体ファイルと注意/仕様_物体ファイルプラグイン接続_2026-09-04.md`。

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

    # 【段A・2026-09-13】この道具は**2つの顔**を持つ。
    #   ・`taro.visual_attention` がある実験 … 読むだけ（正しい姿）
    #   ・無い実験（後方互換）           … 自分で視覚注意を作って動かす＝介入
    #   後方互換の側が残っている間は名乗る。すべての実験が
    #   `taro.visual_attention` に移ったら後方互換ごと消し、この行も消す。
    #   台帳その45（親へ「気づいた」を伝える・既定OFF）もこの経路に含まれる。
    intervenes = ("後方互換の経路のみ：taro.visual_attention が無い実験では"
                  "自分で視覚注意を作って動かす／台帳その45（親への合図・既定OFF）")

    # 物体ファイル.csv の列。**ここだけに書く**（ヘッダーと行の両方をここから作る）。
    #   列を足すときは、この表に足して `self.rows` の辞書に同じ名前で入れるだけ。
    #   2026-09-10：以前はヘッダーと行を別々に並べていたため、列を足したときに
    #   ヘッダーだけ増えて以降の値が1列ずつずれる事故が起きた。
    _EVENT_COLUMNS = [
        "step", "sim_time", "n_dets", "n_files",
        "file_id", "x", "y", "area", "event",
        "misses", "since_seen", "app_cos_created", "mode",
        # 【2026-09-09・仕様_見る側_道を1本にする 後半1節】確認専用の診断列
        #   (conf_iou等)・app_conf は廃止（中5）。代わりに n_points を足した。
        "n_points", "n_blobs", "n_unexplained",
        # 遠心性コピー：残差（ずらした／ずらさない）と、目が動いていたか。
        "ec_res_shift", "ec_res_noshift", "ec_moving",
        # 【2026-09-10】遠心性コピーの予告(used)と、画像から測った実測(meas)。
        "ec_dx_used", "ec_dy_used", "ec_dx_meas", "ec_dy_meas",
        # 【2026-09-10】眼球の角度そのもの。予告が0のとき「目が動いていない」のか
        #   「動いているのに予告が出ていない」のかを切り分けるため。
        "ec_eye_h", "ec_eye_v",
        # 【2026-09-10】注意が選んだ場所（画素）。目がどこを狙わされているかを見る。
        "attn_x", "attn_y",
        # 【2026-09-10・入力の総点検】見る側が使う入力を1つずつ実測するための列。
        #   ec_dx_pred/ec_dy_pred＝遠心性コピーの「これから動く分」（地図をずらす量）
        #   img_mean/img_std＝目の画像そのもの（止まったら壊れている）
        #   ch_*＝目立ちの4つの面が、勝った升でそれぞれいくつだったか
        #   sacc_n＝この検出コマまでに撃たれたサッケードの累計
        "ec_dx_pred", "ec_dy_pred", "img_mean", "img_std",
        "ch_明るさ", "ch_色", "ch_向き", "ch_動き", "sacc_n",
        "cohesion",
        # 【2026-09-09・追記1「直し」3】当て付きの点の数／当てに合う大きさが
        #   無くて捨てた点の数。
        "n_points_expect", "n_reject_scale",
        "n_reject_edge", "n_reject_area", "n_reject_dedup",
        # 【2026-09-10・消え方で持ち時間を決める】
        "explained", "budget",
        # 【2026-09-11・K1「記憶からの山」】上からの目的。goal未設定なら全部空。
        #   内訳（的の升と勝った升での各項）を出さないと、効かなかったとき
        #   「山が小さい」「復帰抑制に消された」「溜めが目立ちに埋もれた」の
        #   区別がつかない（仕様 後半4節）。
        "goal_id", "goal_cx", "goal_cy", "goal_repick",
        "g_at_goal", "sal_at_goal", "ior_at_goal", "acc_at_goal",
        "g_at_win", "sal_at_win", "ior_at_win", "acc_at_win",
    ]

    def setup(self, ctx):
        # 【段A・2026-09-13・設計_太郎をCoreで完結させる.md】
        #   **太郎が視覚と注意を持っていれば**（実験ファイルの `taro.visual_attention`）、
        #   ここでは作らず・呼ばず、その最後の結果を**読むだけ**にする＝道具に戻る。
        #   毎tick呼ぶのは run/trainer.py（`_visual_attention_step`）。掲示板(ctx)へ
        #   置くのもそちら。この道具は**外しても太郎の能力は落ちない**。
        #
        #   【なぜ環境ではなく太郎か】2026-09-13 に一度「環境が持つ」形を作ったが
        #   誤りだった。(1) 環境の step の中で呼ぶと位置が変わり、発話が読む
        #   「注意している物」が前tick→今tickへずれる (2) 環境は太郎の語彙を
        #   覗けないので lexicon=None になり語と物が結びつかない。
        _taro = getattr(ctx, "taro", None)
        self._taro_attn = getattr(_taro, "visual_attention", None) if _taro is not None else None
        if self._taro_attn is not None:
            self.attn = self._taro_attn
        else:
            # 後方互換：taro.visual_attention を書かない既存の実験は、
            #   今までどおりこのプラグインが自分で作って毎tick呼ぶ。
            #   ここでは taro_core/src を sys.pathへ入れてから
            #   （VisualAttention._lazy_importと同じ理由で「brain.xxx」形の
            #   importに必要）importするだけ。
            _core_src = os.path.join(_REPO_ROOT, "taro_core", "src")
            if _core_src not in sys.path:
                sys.path.insert(0, _core_src)
            from brain.cerebral_cortex.parietal_lobe.visual_attention import VisualAttention
            self.attn = VisualAttention(config=self.config)
        self.ofs = self.attn.ofs
        # 他のプラグイン・脳側の配線から読めるように置く（ctx は属性を自由に足せる）。
        # 【段A】太郎が持つときは**学習ループが置く**（trainer._build_ctx）ので、
        #   ここでは触らない（二重に置かない。置く値も順も同じ）。
        if self._taro_attn is None:
            ctx.object_files = self.ofs
            ctx.vanish_misses = self.attn.vanish_misses
            ctx.attended_object = None

        self.attend = self.attn.attend
        self.attend_out = self.attn.attend_out
        self.attend_rows = self.attn.attend_rows

        frames_out = self.config.get("frames_out")
        self.frames_out = _abs_path(frames_out) if frames_out else None
        self._frame_font = None
        if self.frames_out:
            os.makedirs(self.frames_out, exist_ok=True)

        events_out = self.config.get("events_out")
        self.events_out = _abs_path(events_out) if events_out else None

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

    def on_step(self, ctx):
        if self._taro_attn is not None:
            # 【段A・2026-09-13】太郎が毎tick見ている（run/trainer._visual_attention_step）。
            #   ここでは**呼ばない・掲示板にも書かない**。最後の結果を読んで
            #   CSV・PNGに残すだけ＝規約どおりの「測る道具」。
            result = getattr(self._taro_attn, "last_result", None)
            if result is None:
                return
            t = float(ctx.data.time)
            if not result["detected"]:
                return
        else:
            # ---- 後方互換：ctxから取り出して自分で呼ぶ ------------------------
            _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
            orienting = getattr(_u, "_orienting", None) if _u is not None else None

            # 【2026-09-13】cam_fovyの解決（ctx.envが要るのでここでだけ行う）。
            #   _ensure_efference_copy 側の「1回だけ・無ければ60度で代用し警告」
            #   という条件をそのまま外側へ持ってきただけ（値・タイミングは不変）。
            fovy = None
            if self.attn.efference_copy_cfg is not None and self.attn._ec is None:
                fovy = 60.0
                if _u is not None:
                    try:
                        cid = int(_u.model.camera("eye_left").id)
                        fovy = float(_u.model.cam_fovy[cid])
                    except Exception as e:
                        print("[efference] WARNING cam_fovy が読めない。60度で代用: %r" % (e,))
                else:
                    print("[efference] WARNING ctx.env が無く cam_fovy を読めない。60度で代用")

            taro = getattr(ctx, "taro", None)
            hearing = getattr(taro, "hearing", None) if taro is not None else None
            vocab_obj = getattr(hearing, "vocab", None) if hearing is not None else None

            t = float(ctx.data.time)
            last = getattr(ctx, "last", None) or {}
            obs = last.get("obs_out")

            result = self.attn.step(
                obs=obs, orienting=orienting, t=t, step=ctx.step, fovy=fovy,
                surprise_trace=getattr(ctx, "surprise_trace", None),
                vision_vec=getattr(ctx, "last_vision_vec", None),
                parent_target=getattr(ctx, "last_parent_utterance", None),
                vocab=vocab_obj)

            # ---- 掲示板へ置く（後方互換の経路だけ。太郎が持つときは学習ループの
            #      `_visual_attention_step` が同じ値を同じ順で置く）------------
            # 【2026-09-09・仕様_見る側の段構成_実装 1節】段0は毎tick動く
            #   （検出コマでなくても）。efference_copy=None（既定）では
            #   efferenceキーがNoneのまま＝ctx.efferenceは既定不変。
            if self.attn.efference_copy_cfg is not None:
                ctx.efference = result["efference"]

            if not result["detected"]:
                return

            # 【2026-09-10・見る側3段目】attention=None（既定）なら1行も通らない
            #   （既定不変）。attendの有無に関係なく、_detect_frame内で更新される。
            ctx.priority_map_result = result["priority_map_result"]
            ctx.priority_map_result_t = result["priority_map_result_t"]
            if self.attn._spri is not None:
                ctx.goal_point = result["goal_point"]
                ctx.salience_map = result["salience_map"]
                ctx.attention_point = result["attention_point"]
            # 【段6・段7両方から更新されうる】切り替えが一度も起きていなければ
            #   ctx.attention_switch_tは触らない（既定不変。trainer.py側の
            #   既定値-1e9を壊さないため）。
            if result["attention_switch_t"] is not None:
                ctx.attention_switch_t = result["attention_switch_t"]

            if self.attend:
                # 【M3】検出・対応づけ結果を読むだけ。
                ctx.attended_object = result["attended_object"]
                # 【M4c・2026-09-06】vanished が偽→真に変わった瞬間だけ、世界（親）へ
                #   合図する（別件・台帳「その45」に登録済み。今回の移設範囲外で
                #   触っていない。書く場所も値もタイミングも元のまま）。
                if result["noticed_gone"] and _u is not None:
                    _u._taro_noticed_gone_time = t

        # ---- ここから下は両方の経路で通る＝記録だけ ------------------------
        mode = result["mode"]
        res = result["res"]
        dt_ms = result["dt_ms"]
        onset_extra = result["onset_extra"]
        dets = result["dets"]
        img224 = result["img224"]
        prev_by_id = result["prev_by_id"]

        if self.frames_out:
            self._save_detection_frame(ctx, t, img224, dets, res, prev_by_id, mode=mode)

        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1
        self._detect_ms_sum += dt_ms
        self._detect_n += 1
        n_files = len(self.ofs.files)
        n_dets = len(dets)
        self._seg_n_files.append(n_files)
        self._seg_n_dets.append(n_dets)
        self._seg_detect_ms.append(dt_ms)

        if self.events_out is None:
            return

        import numpy as np

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
                # 【2026-09-10】遠心性コピーの予告（used）と実測（meas）。
                "ec_dx_used": onset_extra.get("ec_dx_used", ""),
                "ec_dy_used": onset_extra.get("ec_dy_used", ""),
                "ec_dx_meas": onset_extra.get("ec_dx_meas", ""),
                "ec_dy_meas": onset_extra.get("ec_dy_meas", ""),
                "ec_eye_h": onset_extra.get("ec_eye_h", ""),
                "ec_eye_v": onset_extra.get("ec_eye_v", ""),
                "attn_x": onset_extra.get("attn_x", ""),
                "attn_y": onset_extra.get("attn_y", ""),
                "ec_dx_pred": onset_extra.get("ec_dx_pred", ""),
                "ec_dy_pred": onset_extra.get("ec_dy_pred", ""),
                "img_mean": onset_extra.get("img_mean", ""),
                "img_std": onset_extra.get("img_std", ""),
                "ch_明るさ": onset_extra.get("ch_明るさ", ""),
                "ch_色": onset_extra.get("ch_色", ""),
                "ch_向き": onset_extra.get("ch_向き", ""),
                "ch_動き": onset_extra.get("ch_動き", ""),
                "sacc_n": onset_extra.get("sacc_n", ""),
                "cohesion": cohesion,
                # 【2026-09-10・消え方で持ち時間を決める】explained＝消えたことの
                #   説明のつき具合（0〜1、見失った瞬間に決まる）、budget＝残量。
                #   vanish_budget 無効時は 0.0/1.0 のまま動かないので空文字にする。
                "explained": (round(float(f_for_row.explained), 3)
                              if (self.attn.vanish_budget is not None and f_for_row is not None
                                  and getattr(f_for_row, "misses", 0) > 0) else ""),
                "budget": (round(float(f_for_row.budget), 3)
                           if (self.attn.vanish_budget is not None and f_for_row is not None) else ""),
                # 【2026-09-09・追記1「直し」3】当て付きの点の数／当てに合う
                #   大きさが無くて捨てた点の数。
                "n_points_expect": onset_extra.get("n_points_expect", ""),
                "n_reject_scale": onset_extra.get("n_reject_scale", ""),
                "n_reject_edge": onset_extra.get("n_reject_edge", ""),
                "n_reject_area": onset_extra.get("n_reject_area", ""),
                "n_reject_dedup": onset_extra.get("n_reject_dedup", ""),
                # 【2026-09-11・K1】上からの目的と、その内訳。
                #   注意：この行を作る辞書は**キーを1つずつ明示して**書く形なので、
                #   上の _EVENT_COLUMNS（見出しの一覧）に足しただけでは常に空になる。
                #   F2-129 で実際にこれをやって走行1本（9分）を無駄にした。
                **{k: onset_extra.get(k, "") for k in (
                    "goal_id", "goal_cx", "goal_cy", "goal_repick",
                    "g_at_goal", "sal_at_goal", "ior_at_goal", "acc_at_goal",
                    "g_at_win", "sal_at_win", "ior_at_win", "acc_at_win")},
            })
            # 【2026-09-11・機械で防ぐ】見出しの一覧にあるのに、行の辞書に
            #   キーが無い列は、CSV では `r.get(k, "")` で**静かに空欄**になる。
            #   F2-112pre で「ヘッダーだけ増えて行がずれる」事故を直したとき、
            #   この形（キーごと無い）は残っていた。F2-129 で再発し、走行1本
            #   （9分）を捨てた＝2回目なので、文書でなくここで止める。
            #   最初の1行で見るので、事故は走り出して数秒で分かる。
            if len(self.rows) == 1:
                _missing = [k for k in self._EVENT_COLUMNS if k not in self.rows[0]]
                if _missing:
                    raise RuntimeError(
                        "物体ファイルCSV：見出しにあるのに行に入っていない列がある "
                        "→ %s（_EVENT_COLUMNS に足したら、行を作る辞書にも足す）"
                        % _missing)

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
        if self.attend and self.attn._attended_id is not None:
            # 【M3・仕様書「描画」節】注意中のファイルを太枠(width=3)で強調し、
            #   消えていれば左上に赤字、見た目に一番近い語を四角の下に出す。
            att_f = next((f for f in self.ofs.files if f.id == self.attn._attended_id), None)
            if att_f is not None:
                x, y = att_f.pos
                dr.rectangle([x - 8, y - 8, x + 8, y + 8], outline=(255, 140, 0), width=3)
                if self.attn._last_nearest_word:
                    dr.text((x - 8, y + 9), self.attn._last_nearest_word,
                             fill=(255, 140, 0), font=self._frame_font)
            if self.attn._last_vanished:
                dr.text((4, 28), "VANISHED #%s" % self.attn._attended_id,
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
        if self.attn.gaze_from_attention:
            _u = getattr(getattr(ctx, "env", None), "unwrapped", None)
            _or = getattr(_u, "_orienting", None) if _u is not None else None
            if _or is not None:
                # 【2026-09-10・測定】サッケード1発ごとの記録を書き出す。
                _at = getattr(_or, "angle_trace", None)
                if _at:
                    import json as _json
                    _pa = os.path.join(os.path.dirname(self.events_out or "."),
                                        "眼球の角度_毎ステップ.json")
                    with open(_pa, "w", encoding="utf-8") as _fp:
                        _json.dump(_at, _fp, ensure_ascii=False)
                    out["角度の記録"] = _pa
                _sl = getattr(_or, "sacc_log", None)
                if _sl:
                    import json as _json
                    _pp = os.path.join(os.path.dirname(self.events_out or "."),
                                        "サッケード1発ごと.json")
                    with open(_pp, "w", encoding="utf-8") as _fp:
                        _json.dump(_sl, _fp, ensure_ascii=False)
                    out["サッケード記録"] = _pp
                out["gaze_map_fired"] = int(getattr(_or, "map_fired", 0))
                out["gaze_own_fired"] = int(getattr(_or, "own_fired", 0))
                out["gaze_cmd_sent"] = int(self.attn._gaze_cmd_sent)
                if self.events_out and self.attn._fire_probe:
                    import json as _json, os as _os
                    _json.dump(self.attn._fire_probe,
                               open(_os.path.join(_os.path.dirname(self.events_out),
                                                   "撃った方向の地図の値.json"),
                                    "w", encoding="utf-8"))
        self._seg_n_files, self._seg_n_dets, self._seg_detect_ms = [], [], []
        return out

    def line(self, ctx):
        return f"物体:{len(self.ofs.files)}"

    def report(self, ctx):
        if self.rows and self.events_out:
            os.makedirs(os.path.dirname(self.events_out) or ".", exist_ok=True)
            with open(self.events_out, "w", newline="", encoding="utf-8") as fp:
                # 【2026-09-10】列名は**1か所だけ**に書く。以前はヘッダーと行で
                #   別々に並べていたため、列を足したときにヘッダーだけ増えて
                #   行がずれる事故が起きた（F2-112pre で発覚）。同じ表から
                #   両方を作れば、書き忘れが起こりえない。
                w = csv.writer(fp)
                w.writerow(self._EVENT_COLUMNS)
                for r in self.rows:
                    w.writerow([r.get(k, "") for k in self._EVENT_COLUMNS])
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
