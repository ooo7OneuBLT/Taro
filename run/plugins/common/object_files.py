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
    step, sim_time, n_dets, n_files, file_id, x, y, area, event
    物体ファイル1つにつき1行。event は matched/created/lost のいずれか。
    物体ゼロ（このステップで検出も既存の追跡イベントも無い）のときは file_id 空で1行だけ書く。
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

        self._lazy_import()

        import torch
        if self.device is None:
            # load_mobilesam() の既定解決と同じ順にする（config省略時に食い違わないため）
            want = os.environ.get("TARO_VISION_DEVICE")
            self.device = want if want else ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        ).eval().to(self.device)

        self._mgen = self._load_mobilesam(device=self.device)
        if self.points_per_side is not None:
            # load_mobilesam（object_detector.py）にpoints_per_sideを渡す口が無いので、
            # 生成器の point_grids を直接作り直す（object_detector.py は変更しない）。
            from mobile_sam.utils.amg import build_all_layer_point_grids
            self._mgen.point_grids = build_all_layer_point_grids(
                int(self.points_per_side),
                self._mgen.crop_n_layers,
                self._mgen.crop_n_points_downscale_factor)

        self.ofs = self._ObjectFileSystem()
        # 他のプラグイン・今後の脳側の配線から読めるように置く（ctx は属性を自由に足せる）
        ctx.object_files = self.ofs

        self._last_t = float("-inf")
        self.rows = []
        self._detect_ms_sum = 0.0
        self._detect_n = 0
        self._seg_n_files = []
        self._seg_n_dets = []
        self._seg_detect_ms = []

    def _lazy_import(self):
        for sub in ("senses", ""):
            p = os.path.join(_REPO_ROOT, "taro_core", "src", sub) if sub else \
                os.path.join(_REPO_ROOT, "taro_core", "src")
            if p not in sys.path:
                sys.path.insert(0, p)
        # 旧パス brain.object_files は転送のみなので使わない。
        from object_detector import patch_features, detect, load_mobilesam
        from brain.cerebral_cortex.parietal_lobe.intraparietal_sulcus import ObjectFileSystem
        self._patch_features = patch_features
        self._detect = detect
        self._load_mobilesam = load_mobilesam
        self._ObjectFileSystem = ObjectFileSystem

    def on_step(self, ctx):
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

        t0 = time.perf_counter()
        p, n = self._patch_features(self._model, img224, self.device)
        dets = self._detect(p, n, img=img224, mask_generator=self._mgen)
        prev_by_id = {f.id: f for f in self.ofs.files}
        res = self.ofs.step(dets)
        dt_ms = (time.perf_counter() - t0) * 1000.0

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
        events = []
        for file_id, det_idx, residual in res["matched"]:
            f = cur_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "matched"))
        for file_id in res["created"]:
            f = cur_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "created"))
        for file_id in res["lost"]:
            f = prev_by_id.get(file_id)
            if f is not None:
                events.append((file_id, f.pos[0], f.pos[1], f.area, "lost"))
            else:
                events.append((file_id, "", "", "", "lost"))
        if not events:
            events.append(("", "", "", "", ""))

        for file_id, x, y, area, event in events:
            self.rows.append({
                "step": ctx.step, "sim_time": round(t, 3),
                "n_dets": n_dets, "n_files": n_files,
                "file_id": file_id, "x": x, "y": y, "area": area, "event": event,
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
        return out

    def line(self, ctx):
        return f"物体:{len(self.ofs.files)}"

    def report(self, ctx):
        if self.rows and self.events_out:
            os.makedirs(os.path.dirname(self.events_out) or ".", exist_ok=True)
            with open(self.events_out, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(["step", "sim_time", "n_dets", "n_files",
                           "file_id", "x", "y", "area", "event"])
                for r in self.rows:
                    w.writerow([r["step"], r["sim_time"], r["n_dets"], r["n_files"],
                               r["file_id"], r["x"], r["y"], r["area"], r["event"]])
        return {
            "検出回数": self._detect_n,
            "処理ms_平均": (round(self._detect_ms_sum / self._detect_n, 2)
                          if self._detect_n else None),
            "最終n_files": len(self.ofs.files),
        }
