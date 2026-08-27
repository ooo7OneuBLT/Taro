"""測る道具：太郎が語を言った瞬間に、その目に何が写っていたかを残す。

【なぜ要るか、2026-08-27・F2-11】産出テストで「りんご」が板と無関係に35%選ばれる
（4語均等なら25%）。原因の予想が2回続けて外れた：
  ① 「ぶうぶうの想像図が膨らんで平均を歪め、りんごが3語の中間に来るから」
     → 引き離しを外して配置が理想的（全ペアが負のコサイン）になっても偏りは残った
  ② 「板から視線が外れて背景しか写っていない瞬間に、背景に近いりんごが出るから」
     → 実測：りんごと言った20回すべてが視線10度以内・中央値2.8度＝見ていた
「測定が予想と2回食い違ったら、3回目の前に必ず絵にして見る」（CLAUDE.md 必ず守る
ルール4）に従い、数値ではなく**その瞬間の絵**を残すための道具。

【役割】読むだけ（run/plugins/base.py の規約）。太郎も環境も変えない。
  乱数を消費しない（DINOv2の推論は決定的）。読むのは：
    ctx.last_produce            … 産出イベント（trainer.py が判断ごとに置く）
    ctx.last["obs_out"]         … その判断の観測（eye_left / eye_right）
    ctx.taro.vision_backend     … 視覚の特徴を取り出す器（既に生成済みのものを借りる）

【出すもの】
  ① 発話ごとの中心窩画像PNG … 何と言ったか・確信度・通し番号をファイル名に入れる
     例: 0007_りんご_sim0.267.png
  ② 発話ごとの視覚ベクトル … npz（散布図を描くため。keyは画像と同じ通し番号）

  実験ファイルでの書き方:
    "plugins": {"produce_snapshot": {"out_dir": "F/logs/.../スナップ",
                                     "max_images": 60}}
    max_images を超えたぶんは画像を書かない（ベクトルは全部残す）。
"""
import os

import numpy as np

from run.plugins.base import Plugin


def _abs_path(p):
    if os.path.isabs(p):
        return p
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    return os.path.join(root, p)


class ProduceSnapshot(Plugin):
    name = "produce_snapshot"

    def setup(self, ctx):
        self.out_dir = _abs_path(self.config.get("out_dir", "F/logs/produce_snapshot"))
        self.max_images = int(self.config.get("max_images", 60))
        os.makedirs(self.out_dir, exist_ok=True)
        self.n = 0
        self.vecs = []       # [(通し番号, 語, sim, ベクトル)]
        self._crop = None
        self._imwrite = None

    def _lazy_import(self):
        if self._crop is None:
            from vision_backends import fovea_crop     # taro_core/src/senses
            self._crop = fovea_crop
        if self._imwrite is None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self._imwrite = plt

    def on_step(self, ctx):
        ev = getattr(ctx, "last_produce", None)
        if not ev:
            return
        last = getattr(ctx, "last", None) or {}
        obs = last.get("obs_out")
        if obs is None or "eye_left" not in obs:
            return
        taro = getattr(ctx, "taro", None)
        backend = getattr(taro, "vision_backend", None) if taro is not None else None
        self._lazy_import()

        self.n += 1
        word = ev.get("target_word", "?")
        sim = float(ev.get("sim", 0.0))

        # ① 中心窩の切り出し（親の名付け判定・逆引きと同じ幅を使う）
        fovea_px = getattr(backend, "fovea_px", None)
        img = np.asarray(obs["eye_left"])
        crop = self._crop(img, fovea_px)

        # ② 視覚ベクトル（散布図用）。backendが無ければ切り出しを平坦化して代用する
        if backend is not None:
            vec = np.asarray(backend.encode(obs["eye_left"], obs["eye_right"]),
                             dtype=np.float32).reshape(-1)
        else:
            vec = crop.astype(np.float32).reshape(-1)
        self.vecs.append((self.n, word, sim, vec))

        if self.n <= self.max_images:
            path = os.path.join(self.out_dir, "%04d_%s_sim%.3f.png" % (self.n, word, sim))
            plt = self._imwrite
            fig = plt.figure(figsize=(2.2, 2.2), dpi=110)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(np.clip(crop, 0, 255).astype(np.uint8), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(path)
            plt.close(fig)

    def report(self, ctx):
        if not self.vecs:
            return {"発話スナップ": 0}
        npz = os.path.join(self.out_dir, "視覚ベクトル.npz")
        np.savez_compressed(
            npz,
            idx=np.array([v[0] for v in self.vecs], dtype=np.int32),
            word=np.array([v[1] for v in self.vecs], dtype=object),
            sim=np.array([v[2] for v in self.vecs], dtype=np.float32),
            vec=np.stack([v[3] for v in self.vecs]))
        return {"発話スナップ": len(self.vecs),
                "画像": min(self.n, self.max_images),
                "保存先": os.path.relpath(self.out_dir)}
