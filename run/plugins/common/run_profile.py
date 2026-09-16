# -*- coding: utf-8 -*-
"""測る道具：走行そのものの重さ（速度・メモリ・GPU）を1歩ごとに記録する。

【なぜ要るか・2026-09-16】走行に何分かかるかを知る手段が無く、見積もるたびに
**100歩と400歩を別々に走らせて引き算する**という手作業をしていた（2026-09-16）。
また過去に、描画用メモリが積み上がってプロセスが 13.6GB で強制停止した事故がある
（`run/tools/memory_guard.py` の説明・落とし穴 項82）。どちらも「走行中の重さを
誰も記録していない」ことが原因。走行のたびに勝手に残るようにする。

【この道具がやらないこと】割り算をしない。「1歩あたり何秒」「何倍遅い」といった
まとめは**測定の側**でやる（ユーザー指示 2026-09-16「出せる値は plugin で出力して、
その出力をもとに測定する」）。ここは観測できた値を列として出すだけ。

【読むだけ】太郎にも環境にも触らない。乱数を消費しない。

【重さの配慮】`perf_counter` は毎歩でも無視できる（マイクロ秒）が、メモリ・CPU・GPU
の読み取りは OS への問い合わせなので **N歩に1回**だけにする（既定20歩）。
この道具自身が使った時間も列に出す（`計測の秒`）ので、あとから差し引ける。

【実験ファイルでの書き方】
    "plugins": {"run_profile": {"out": "F/logs/<実験>/走行の負荷.csv"}}
    "plugins": {"run_profile": {"out": "...", "every": 20}}   # 重い読み取りの間隔

【出る列】
    step / 経過秒 / 1歩の秒 / 計測の秒
    メモリMB（実メモリ）/ 仮想メモリMB / CPU% / スレッド数
    システムメモリ使用% / システム空きMB
    GPU割当MB / GPU予約MB / GPU最大割当MB / GPU使用率%
    海馬の件数 / 物体ファイル数    ← 「なぜ重くなったか」を突き合わせるため
"""
import csv
import os
import time

from run.plugins.base import Plugin


def _abs_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)), p)


class RunProfile(Plugin):
    """走行の速度・メモリ・GPU を記録する（読むだけ）。"""

    name = "run_profile"
    intervenes = None

    def setup(self, ctx):
        """出力先と計測の間隔を読み、時計と各種ハンドルを用意する。戻り値は無い。"""
        self.out = self.config.get("out")
        self.every = max(1, int(self.config.get("every", 20)))
        self.rows = []
        self._t0 = time.perf_counter()
        self._prev = self._t0
        self._計測秒合計 = 0.0
        self._重い = {}          # N歩に1回だけ更新し、その間は同じ値を書く
        self._proc = None
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process(os.getpid())
            self._proc.cpu_percent(None)      # 1回目は0を返す規約なので捨てておく
        except Exception:       # noqa: BLE001  psutil が無くても走行は止めない
            self._psutil = None
        try:
            import torch
            self._torch = torch if torch.cuda.is_available() else None
        except Exception:       # noqa: BLE001
            self._torch = None
        # 【2026-09-16】最初の1回はここで取る。取らないと、次の重い読み取りが
        #   来るまで（既定20歩）メモリ・GPUの列が空欄になる（試し走行で
        #   181/200行しか埋まらなかった）。走り出しの値は「学習前の基準」として
        #   一番知りたい行でもある。
        self._重い = self._重い値を取る()

    def _重い値を取る(self):
        """OSやGPUへの問い合わせ。N歩に1回だけ呼ぶ。"""
        d = {}
        if self._proc is not None:
            try:
                mi = self._proc.memory_info()
                d["メモリMB"] = round(mi.rss / 1048576.0, 1)
                d["仮想メモリMB"] = round(getattr(mi, "vms", 0) / 1048576.0, 1)
                d["CPU%"] = round(self._proc.cpu_percent(None), 1)
                d["スレッド数"] = self._proc.num_threads()
            except Exception:   # noqa: BLE001
                pass
            try:
                vm = self._psutil.virtual_memory()
                d["システムメモリ使用%"] = round(vm.percent, 1)
                d["システム空きMB"] = round(vm.available / 1048576.0, 1)
            except Exception:   # noqa: BLE001
                pass
        if self._torch is not None:
            try:
                t = self._torch
                d["GPU割当MB"] = round(t.cuda.memory_allocated() / 1048576.0, 1)
                d["GPU予約MB"] = round(t.cuda.memory_reserved() / 1048576.0, 1)
                d["GPU最大割当MB"] = round(t.cuda.max_memory_allocated() / 1048576.0, 1)
            except Exception:   # noqa: BLE001
                pass
            try:
                # utilization は pynvml が要る。無ければ黙って諦める。
                d["GPU使用率%"] = int(self._torch.cuda.utilization())
            except Exception:   # noqa: BLE001
                pass
        return d

    def _中身の数(self, ctx):
        """重くなる原因になりそうな「溜まっているもの」の数。無ければ空欄。"""
        d = {}
        taro = getattr(ctx, "taro", None) or getattr(ctx, "brain", None)
        h = getattr(taro, "language_hippocampus", None) if taro is not None else None
        if h is not None:
            try:
                d["海馬の件数"] = len(h)
            except Exception:   # noqa: BLE001
                pass
        of = getattr(ctx, "object_files", None)
        if of is not None:
            try:
                d["物体ファイル数"] = len(of)
            except Exception:   # noqa: BLE001
                pass
        return d

    def on_step(self, ctx):
        """毎ステップ。前の歩からの経過を記録し、N歩に1回だけ重い値を取り直す。戻り値は無い。"""
        t = time.perf_counter()
        一歩 = t - self._prev
        self._prev = t
        計測開始 = time.perf_counter()
        if ctx.step % self.every == 0:
            self._重い = self._重い値を取る()
            self._重い.update(self._中身の数(ctx))
        行 = {"step": ctx.step,
              "経過秒": round(t - self._t0, 3),
              "1歩の秒": round(一歩, 5)}
        行.update(self._重い)
        計測秒 = time.perf_counter() - 計測開始
        self._計測秒合計 += 計測秒
        行["計測の秒"] = round(計測秒, 6)
        self.rows.append(行)

    def metrics(self, ctx):
        """区切りごとの値。直近の1歩の秒と実メモリだけを run.csv へ混ぜる。"""
        if not self.rows:
            return None
        r = self.rows[-1]
        out = {"run_profile.1歩の秒": r.get("1歩の秒")}
        if "メモリMB" in r:
            out["run_profile.メモリMB"] = r["メモリMB"]
        return out

    def report(self, ctx):
        """out が指定されていれば列をCSVへ書き出し、生の事実だけを辞書で返す。

        割り算（1歩あたり・何倍）はここでやらない。測定の側の仕事。
        """
        if self.rows and self.out:
            p = _abs_path(self.out)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            見出し = ["step", "経過秒", "1歩の秒", "計測の秒",
                      "メモリMB", "仮想メモリMB", "CPU%", "スレッド数",
                      "システムメモリ使用%", "システム空きMB",
                      "GPU割当MB", "GPU予約MB", "GPU最大割当MB", "GPU使用率%",
                      "海馬の件数", "物体ファイル数"]
            # 実際に出た列だけに絞る（環境によって GPU や psutil が無いことがある）
            ある = [c for c in 見出し if any(c in r for r in self.rows)]
            with open(p, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow(ある)
                for r in self.rows:
                    w.writerow([r.get(c, "") for c in ある])
        最後 = self.rows[-1] if self.rows else {}
        out = {
            "記録した歩数": len(self.rows),
            "総経過秒": round(time.perf_counter() - self._t0, 2),
            "この道具が使った秒": round(self._計測秒合計, 3),
        }
        if "メモリMB" in 最後:
            out["終わりのメモリMB"] = 最後["メモリMB"]
        if "GPU割当MB" in 最後:
            out["終わりのGPU割当MB"] = 最後["GPU割当MB"]
            out["GPU最大割当MB"] = 最後.get("GPU最大割当MB")
        if self.out:
            out["出力"] = self.out
        return out
