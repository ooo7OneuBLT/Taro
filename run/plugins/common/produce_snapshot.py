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
  ① 発話ごとの「脳に渡した画像」PNG … 何と言ったか・確信度・通し番号をファイル名に入れる
     例: 0007_りんご_sim0.267.png
     lexicon_vision.source="wide" なら周辺60度まるごと（無クロップ・元の画素のまま）、
     中心窩カメラありなら中心窩15度、それ以外は周辺から fovea_px で中央切り出し
     （trainer._vision_backend_encode と同じ選び方・2026-09-05修正）。
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
    # 【2026-09-13・名乗り】太郎の視覚（vision_backend.fovea_px）を一時的に
    #   書き換えて「中心窩の切り出しをやめた絵」を撮り、直後に元へ戻す
    #   （153行・159行）。同じtick内で復元するので学習の数値は変わらないが、
    #   **太郎の部品に書いている**ことに変わりはないので名乗る。
    intervenes = "撮影のため vision_backend.fovea_px を一時変更し同tick内で復元する"
    name = "produce_snapshot"

    def setup(self, ctx):
        self.out_dir = _abs_path(self.config.get("out_dir", "F/logs/produce_snapshot"))
        self.max_images = int(self.config.get("max_images", 60))
        os.makedirs(self.out_dir, exist_ok=True)
        self.full_dir = self.config.get("full_dir")
        if self.full_dir:
            self.full_dir = _abs_path(self.full_dir)
            os.makedirs(self.full_dir, exist_ok=True)
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
            # 日本語が豆腐（□）にならないようにする
            matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]
            import matplotlib.pyplot as plt
            self._imwrite = plt

    @staticmethod
    def _is_wide_source(ctx, taro):
        """lexicon_vision.source == "wide" か（trainer._vision_backend_encode と同じ判定）。

        取り方も trainer に倣う：`taro.cfg.lexicon_vision`（trainer の self.cfg と
        同一の Config オブジェクト）。cfg が無い／属性が無いときは実験ファイルの
        taro 欄（ctx.spec["taro"]["lexicon_vision"]）から読む。どちらにも無ければ False
        ＝従来どおり（fovea 経路の挙動は1ビットも変わらない）。
        """
        lv = None
        cfg = getattr(taro, "cfg", None) if taro is not None else None
        if cfg is not None:
            lv = getattr(cfg, "lexicon_vision", None)
        if lv is None:
            spec = getattr(ctx, "spec", None)
            if isinstance(spec, dict):
                lv = (spec.get("taro") or {}).get("lexicon_vision")
        lv = lv or {}
        return (lv.get("source") == "wide") if isinstance(lv, dict) else False

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

        # ① DINOv2 が実際に受け取る画像を、trainer._vision_backend_encode と
        #    まったく同じ選び方で取る（2026-08-28訂正、2026-09-05再訂正）。
        #    中心窩カメラ（eye_left_fovea・視野15度）があればそれを**切り出さずに**
        #    そのまま渡すのが本体の挙動。周辺カメラ(eye_left)から32px切り出すのは
        #    fovea_camera=False のシーンだけ。ここを取り違えると、見せる絵が
        #    「太郎が実際に見ているもの」でなくなる。
        #
        # 【2026-09-05・F2-74cで判明した不整合の修正】lexicon_vision.source=="wide"
        #   （F2-49系以降の全実験）のとき、本体は中心窩カメラを使わず
        #   **周辺60度の eye_left/eye_right をまるごと・無クロップで** encode に渡す
        #   （trainer は encode() の間だけ backend.fovea_px を 10**9 にして crop を
        #   no-op化）。ここは has_fovea を「観測に eye_left_fovea があるか」だけで
        #   決めていたので、wide のときも設定値 fovea_px=32 で中央を切り出して
        #   保存・符号化していた＝脳が語の選択に使った画像・ベクトルと別物だった。
        #   設定の取り方は trainer と同じ（taro.cfg は Taro.__init__ が受け取る
        #   trainer と同一の Config）。cfg が取れないときは実験ファイルの taro 欄
        #   （ctx.spec["taro"]）から読む。
        wide_src = self._is_wide_source(ctx, taro)
        has_fovea = ((not wide_src) and "eye_left_fovea" in obs
                     and "eye_right_fovea" in obs)
        if has_fovea:
            img = np.asarray(obs["eye_left_fovea"])
            crop = img                      # 撮影時点で視野15度に切り出し済み
            fovea_px = None
        elif wide_src:
            img = np.asarray(obs["eye_left"])
            crop = img                      # 周辺60度まるごと・無クロップ（本体と同じ）
            fovea_px = None
        else:
            fovea_px = getattr(backend, "fovea_px", None)
            img = np.asarray(obs["eye_left"])
            crop = self._crop(img, fovea_px)
        wide = np.asarray(obs["eye_left"])   # 視界の全体（視野60度）は常に周辺カメラ

        # ② 視覚ベクトル（散布図用）。backendが無ければ切り出しを平坦化して代用する
        if backend is not None:
            if has_fovea or wide_src:
                # 本体（trainer._vision_backend_encode）と同じく、encode の間だけ
                # fovea_px を no-op 値にして、渡した画像をそのまま符号化させる
                enc_l, enc_r = ((obs["eye_left_fovea"], obs["eye_right_fovea"])
                                if has_fovea else (obs["eye_left"], obs["eye_right"]))
                _old = getattr(backend, "fovea_px", None)
                if _old is not None:
                    backend.fovea_px = 10 ** 9      # 本体と同じくcropをno-op化
                try:
                    vec = np.asarray(backend.encode(enc_l, enc_r),
                                     dtype=np.float32).reshape(-1)
                finally:
                    if _old is not None:
                        backend.fovea_px = _old
            else:
                vec = np.asarray(backend.encode(obs["eye_left"], obs["eye_right"]),
                                 dtype=np.float32).reshape(-1)
        else:
            vec = crop.astype(np.float32).reshape(-1)
        self.vecs.append((self.n, word, sim, vec))

        if self.n <= self.max_images:
            path = os.path.join(self.out_dir, "%04d_%s_sim%.3f.png" % (self.n, word, sim))
            plt = self._imwrite
            if wide_src:
                # 脳に渡した画像そのもの（周辺60度・無クロップ）を**元の画素のまま**
                # 残す（拡大・再標本化しない）。full_dir の「左目・周辺(60度)」区画と
                # 画素単位で突き合わせられるようにするため。
                plt.imsave(path, np.clip(crop, 0, 255).astype(np.uint8))
            else:
                fig = plt.figure(figsize=(2.2, 2.2), dpi=110)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(np.clip(crop, 0, 255).astype(np.uint8), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                fig.savefig(path)
                plt.close(fig)

        # 【2026-08-28】両目ぶんの視界を1枚にまとめて残す（full_dir 指定時のみ）。
        #   上段＝周辺カメラ（視野60度・左右）、下段＝中心窩カメラ（視野15度・左右）。
        #   周辺カメラには「中心窩の視野15度がどこにあたるか」を赤枠で重ねる。
        if self.full_dir and self.n <= self.max_images:
            import math
            plt = self._imwrite
            wl = np.asarray(obs["eye_left"]); wr = np.asarray(obs.get("eye_right", wl))
            if has_fovea:
                fl = np.asarray(obs["eye_left_fovea"])
                fr = np.asarray(obs.get("eye_right_fovea", fl))
                r = (math.tan(math.radians(15.0 / 2))
                     / math.tan(math.radians(60.0 / 2)))
                lo_ttl = ("左目・中心窩(15度)", "右目・中心窩(15度)")
            elif wide_src:
                # source="wide"：符号化に渡すのは周辺60度そのもの（中心窩は使わない）。
                # 下段は「脳に渡した画像」＝上段と同じ絵になる。赤枠（中心窩15度）は描かない
                fl, fr = wl, wr
                r = None
                lo_ttl = ("左目・符号化入力(60度・無クロップ)",
                          "右目・符号化入力(60度・無クロップ)")
            else:
                fl = fr = crop
                r = None
                lo_ttl = ("左目・中心窩(15度)", "右目・中心窩(15度)")
            fig, axes = plt.subplots(2, 2, figsize=(5.0, 5.2))
            panels = ((wl, "左目・周辺(60度)"), (wr, "右目・周辺(60度)"),
                      (fl, lo_ttl[0]), (fr, lo_ttl[1]))
            for ax, (im, ttl) in zip(axes.ravel(), panels):
                ax.imshow(np.clip(im, 0, 255).astype(np.uint8), interpolation="nearest")
                ax.set_title(ttl, fontsize=8.5, pad=2)
                ax.set_xticks([]); ax.set_yticks([])
                if r is not None and "周辺" in ttl:
                    h, w = im.shape[0], im.shape[1]
                    fp = max(1, int(round(w * r)))
                    ax.add_patch(plt.Rectangle(((w - fp) / 2 - 0.5, (h - fp) / 2 - 0.5),
                                               fp, fp, fill=False,
                                               edgecolor="#ff1744", lw=1.6))
            fig.suptitle("%d回目 「%s」 確信 %.2f" % (self.n, word, sim), fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(os.path.join(
                self.full_dir, "%04d_%s_sim%.3f.png" % (self.n, word, sim)), dpi=115)
            plt.close(fig)
            return

        # 【旧】左目の視界だけを残す経路（full_dir 未指定時は通らない）。
        #   中心窩の切り出し枠を重ねて描くので「目全体のどこを見ているか」が分かる。
        if self.full_dir and self.n <= self.max_images:
            h, w = wide.shape[0], wide.shape[1]
            if has_fovea:
                # 視野15度 ÷ 視野60度 の比を、画角の tan 比で画素に直す
                import math
                r = math.tan(math.radians(15.0 / 2)) / math.tan(math.radians(60.0 / 2))
                fp = max(1, int(round(w * r)))
            else:
                fp = min(int(fovea_px) if fovea_px else max(1, h // 2), h, w)
            y0, x0 = (h - fp) // 2, (w - fp) // 2
            plt = self._imwrite
            fig = plt.figure(figsize=(3.0, 3.0), dpi=110)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(np.clip(wide, 0, 255).astype(np.uint8), interpolation="nearest")
            ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), fp, fp,
                                       fill=False, edgecolor="#ff1744", lw=2.0))
            ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(
                self.full_dir, "%04d_%s_sim%.3f.png" % (self.n, word, sim)))
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
