# -*- coding: utf-8 -*-
"""語彙地図：太郎が知っている語同士の見た目の近さ（コサイン類似度）を、
checkpointごとに自動で図・数値として保存する常設プラグイン。

【役割は読み取り専用】run/plugins/base.py の規約どおり、ctx を読むだけ。
太郎の学習・行動には一切影響しない（乱数を消費しない・taro/envを一切書き換えない）。

【なぜ要るか、2026-09-04】f68〜f71で「語のベクトル同士の近さを見る」作業を
4回スクリプトで書き直した。特にF2-70/71で、おわんの見た目が元々コップに近い
ことが型の応用テストの結果を左右すると分かった。この「近さ」を、特別な調査を
しなくてもいつでも・走行のたびに自動で見られるようにする。
仕様書：F/docs/仕様_語彙地図プラグイン_2026-09-04.md

読むもの：
    ctx.taro.lexicon.proto   … {チャンク(タプル): 見た目ベクトル}
                                （taro_core/src/brain/cerebral_cortex/temporal_lobe/lexicon.py
                                 のproperty。中身はDEFAULT_CHANNEL["proto"]）
    ctx.taro.hearing.vocab   … チャンクを文字列に戻す語彙表
                                （taro_core/src/senses/hearing.py の Vocabulary。
                                 .decode(chunk) が PAD/BOS/EOS を除いた文字列を返す）

出すもの（out_dir指定時のみ。checkpointのstep番号ごとに連番で全部残す＝上書きしない）：
    地図_チェックポイント{step}.png  … 類似度行列のヒートマップ
    地図_チェックポイント{step}.json … 語のリストと生の行列

実験ファイルでの書き方:
    "plugins": {"word_similarity_map": {"out_dir": "F/logs/.../語彙地図",
                                        "min_chunk_len": 1}}
    out_dir を書かなければ何も保存しない（安全側の既定）。
"""
import json
import os

import numpy as np

from run.plugins.base import Plugin


def _abs_path(p):
    if os.path.isabs(p):
        return p
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    return os.path.join(root, p)


def _cos(a, b):
    """F/scripts/f68_word_expectation_probe.py の _cos と同じ実装（1個も変えない）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float((a / na) @ (b / nb))


class WordSimilarityMap(Plugin):
    name = "word_similarity_map"

    def setup(self, ctx):
        out_dir = self.config.get("out_dir")
        self.out_dir = _abs_path(out_dir) if out_dir else None
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
        self.min_chunk_len = int(self.config.get("min_chunk_len", 1))
        self._font_ready = False
        self._last_n_words = 0
        self._last_max_pair_sim = None
        self._last_max_pair = None

    def _setup_font(self):
        """日本語が豆腐（□）にならないようにする（f_gen_f49.py/f67系のmeiryo指定を踏襲）。"""
        if self._font_ready:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        fp = "C:/Windows/Fonts/meiryo.ttc"
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        self._font_ready = True

    def on_checkpoint(self, ctx):
        taro = getattr(ctx, "taro", None)
        lexicon = getattr(taro, "lexicon", None) if taro is not None else None
        proto = getattr(lexicon, "proto", None) if lexicon is not None else None
        if not proto:
            # 学習の最初期はlexiconが空。何もしないで正常終了する。
            return
        hearing = getattr(taro, "hearing", None)
        vocab = getattr(hearing, "vocab", None) if hearing is not None else None
        if vocab is None:
            return

        words, vecs = [], []
        for chunk, v in proto.items():
            if len(chunk) < self.min_chunk_len:
                continue
            s = vocab.decode(chunk)
            if not s:
                continue
            words.append(s)
            vecs.append(np.asarray(v, dtype=np.float64))
        if not words:
            self._last_n_words = 0
            self._last_max_pair_sim = None
            self._last_max_pair = None
            return

        n = len(words)
        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                sim[i, j] = 1.0 if i == j else _cos(vecs[i], vecs[j])

        # 対角線を除いた中で一番類似度が高いペアを1つ記録する（次のmetricsで使う）
        best_i = best_j = None
        best_v = -2.0
        for i in range(n):
            for j in range(n):
                if i != j and sim[i, j] > best_v:
                    best_i, best_j, best_v = i, j, sim[i, j]

        self._last_n_words = n
        if best_i is not None:
            self._last_max_pair_sim = float(best_v)
            self._last_max_pair = "%s-%s" % (words[best_i], words[best_j])
        else:
            self._last_max_pair_sim = None
            self._last_max_pair = None

        if self.out_dir:
            step = int(getattr(ctx, "step", 0))
            self._setup_font()
            import matplotlib.pyplot as plt

            side = min(24.0, 0.55 * n + 2.0)
            fig, ax = plt.subplots(figsize=(side, side * 0.85 + 0.5))
            im = ax.imshow(sim, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
            ax.set_xticks(range(n))
            ax.set_xticklabels(words, rotation=45, ha="right")
            ax.set_yticks(range(n))
            ax.set_yticklabels(words)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, "%.2f" % sim[i, j], ha="center", va="center", fontsize=8)
            ax.set_title("語彙地図（見た目の近さ・コサイン類似度） checkpoint step=%d" % step)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            png_path = os.path.join(self.out_dir, "地図_チェックポイント%d.png" % step)
            fig.savefig(png_path, dpi=110)
            plt.close(fig)

            json_path = os.path.join(self.out_dir, "地図_チェックポイント%d.json" % step)
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump({
                    "step": step,
                    "words": words,
                    "sim": sim.tolist(),
                    "max_pair": self._last_max_pair,
                    "max_pair_sim": self._last_max_pair_sim,
                }, fp, ensure_ascii=False, indent=2)

    def metrics(self, ctx):
        out = {"word_map_n_words": self._last_n_words}
        if self._last_max_pair_sim is not None:
            out["word_map_max_pair_sim"] = self._last_max_pair_sim
            out["word_map_max_pair"] = self._last_max_pair
        return out

    def report(self, ctx):
        if not self._last_n_words:
            return None
        out = {"語数": self._last_n_words}
        if self._last_max_pair is not None:
            out["一番近いペア"] = self._last_max_pair
            out["類似度"] = round(self._last_max_pair_sim, 4)
        if self.out_dir:
            out["保存先"] = os.path.relpath(self.out_dir)
        return out
