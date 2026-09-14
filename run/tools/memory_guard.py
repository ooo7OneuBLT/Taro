# -*- coding: utf-8 -*-
"""繰り返し環境を作り直す検証ツール向けの、メモリ上限を見張る道具。

仕様：作業記録（非公開）

【なぜ要るか、2026-08-13】`run/tools/check_joint_compliance.py`が`scene_io.build()`を
16回呼ぶ際、`vision=False`を渡し忘れていた。1回あたり顔・服のテクスチャ（約977MB、
`vision=True`のとき既定でLeanMimoEnv.strip_texturesが発動しない）を持つ環境が
積み上がり、実プロセスが13.6GBまで膨らんで強制停止された（項82「体を作り直す実験は
描画用メモリが積み上がって落ちる」と同じ型。`env.close()`を毎回呼んでいても、
描画用のリソースは`close()`では返らないことがある）。

この道具は「そもそもメモリを食わない設定にする」（個々の呼び出しにvision=Falseを
明示すること）の**代わりにはならない**。あくまで、それでも想定外にメモリが
膨らんだとき（新しい検証ツールでの指定忘れ、未知のリーク等）に、13.6GBのような
実害（プロセス強制停止）に至る前に、警告または停止で気づけるようにする保険。

【使い方】

    from memory_guard import MemoryGuard
    guard = MemoryGuard(warn_mb=2000, stop_mb=5000)
    for i in range(16):
        env, _ = scene_io.build(scene, vision=False)
        ...
        env.close()
        guard.check(f"build#{i}")   # 閾値を超えたら print で警告、stop_mb 超で例外

引数を省略すると既定値（warn_mb=2000, stop_mb=5000）が使われる。

【閾値の根拠】
    [Tier3・工学的判断] 通常の実験1本のメモリ使用量が390〜500MB（ユーザーの実測、
    2026-08-13）であることを基準に、
        警告  約4倍(2000MB)   ＝「明らかに1本ぶんより多い」と気づける水準
        停止  約10倍(5000MB)  ＝ 13.6GBという実際の事故に至るはるか手前で
                                  確実に止まる水準（13.6GB ÷ 5000MB ≈ 2.7倍の余裕）
    この2つの数値は文献的根拠を持たない恣意的な目安であり、対象のマシンの
    搭載メモリや、そのスクリプトが本来必要とする量に応じて呼び出し側で
    上書きしてよい。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

try:
    import psutil
except ImportError as e:      # pragma: no cover - 環境に無い場合は使えないことを明示する
    raise ImportError(
        "memory_guard.py には psutil が要ります（.venv には既に入っています）。\n"
        "  .venv/Scripts/python.exe を使っているか確認してください。") from e


class MemoryExceeded(RuntimeError):
    """stop_mb を超えたときに送出する例外。呼び出し側が捕まえて後始末してよい。"""


class MemoryGuard:
    """自分のプロセスの常駐メモリ（RSS）を見張る。

    Args:
        warn_mb: これを超えたら標準出力に警告を出す（処理は止めない）。既定2000MB。
        stop_mb: これを超えたら MemoryExceeded を送出して止める。既定5000MB。
        label: ログに出す見出し（複数のガードを使い分けるとき用）。
    """

    def __init__(self, warn_mb=2000.0, stop_mb=5000.0, label="memory_guard"):
        if stop_mb <= warn_mb:
            raise ValueError(
                f"stop_mb({stop_mb})はwarn_mb({warn_mb})より大きくしてください"
                "（停止しきい値が警告しきい値以下だと、警告が意味を持たない）。")
        self.warn_mb = float(warn_mb)
        self.stop_mb = float(stop_mb)
        self.label = label
        self._proc = psutil.Process(os.getpid())
        self._warned = False
        self.history = []      # [(where, rss_mb), ...] 呼び出しのたびに記録する

    def rss_mb(self):
        """今この瞬間の常駐メモリ（MB）。"""
        return self._proc.memory_info().rss / (1024.0 * 1024.0)

    def check(self, where=""):
        """今のメモリを測り、しきい値と比べる。

        Returns:
            float: 測った時点のRSS（MB）
        Raises:
            MemoryExceeded: stop_mb を超えたとき
        """
        rss = self.rss_mb()
        self.history.append((where, rss))
        if rss >= self.stop_mb:
            print(f"[{self.label}] 停止：{where} 時点でメモリ{rss:.1f}MB "
                  f"（しきい値{self.stop_mb:.1f}MB超）", flush=True)
            raise MemoryExceeded(
                f"{where}: メモリ{rss:.1f}MBがstop_mb={self.stop_mb:.1f}MBを超えました。"
                "処理を止めます（vision=Falseの指定漏れ等、想定外の積み上がりが疑われます）。")
        if rss >= self.warn_mb and not self._warned:
            print(f"[{self.label}] 注意：{where} 時点でメモリ{rss:.1f}MB "
                  f"（しきい値{self.warn_mb:.1f}MB超。停止は{self.stop_mb:.1f}MB）", flush=True)
            self._warned = True   # 同じしきい値超えを毎回連呼しない。stop_mbは別枠で毎回判定する
        return rss

    def summary(self):
        """これまでの記録から、最大値とその時点を返す（文字列）。"""
        if not self.history:
            return f"[{self.label}] 記録なし"
        where, peak = max(self.history, key=lambda t: t[1])
        return (f"[{self.label}] 最大メモリ={peak:.1f}MB（{where}時点）"
                f"  記録件数={len(self.history)}")


# ============================================================================
# 単体動作の自己テスト（わざとメモリを食うダミーコードで警告・停止が発動するか）
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(" memory_guard.py の単体動作確認")
    print("=" * 78)

    # 【なぜ小さいしきい値でテストするか】既定値(2000/5000MB)のまま確認しようとすると、
    #   実際に数GBのメモリを確保する必要があり、このテスト自体が重い・危険になる。
    #   しきい値を小さく設定し、少量のメモリ確保だけで警告・停止の発動を確認する。
    print("\n[1] 警告が出ることの確認（warn_mb=起動直後のRSSに近い小さな値）")
    baseline = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    print(f"    起動直後のRSS={baseline:.1f}MB")
    guard = MemoryGuard(warn_mb=baseline + 5, stop_mb=baseline + 500, label="test1")
    junk = []
    warned_at = None
    for i in range(20):
        junk.append(bytearray(2 * 1024 * 1024))     # 2MBずつ確保
        rss = guard.check(f"iter{i}")
        if guard._warned and warned_at is None:
            warned_at = i
    ok1 = warned_at is not None
    print(f"    警告が出た反復={warned_at}  合格={ok1}")

    print("\n[2] 停止（例外）が発動することの確認（stop_mbを低く設定）")
    guard2 = MemoryGuard(warn_mb=baseline + 5, stop_mb=baseline + 50, label="test2")
    junk2 = []
    stopped = False
    stop_at = None
    try:
        for i in range(200):
            junk2.append(bytearray(2 * 1024 * 1024))
            guard2.check(f"iter{i}")
    except MemoryExceeded as e:
        stopped = True
        stop_at = len(junk2)
        print(f"    期待通り MemoryExceeded で停止: {e}")
    ok2 = stopped
    print(f"    停止した反復（確保回数）={stop_at}  合格={ok2}")

    print("\n[3] stop_mb <= warn_mb だと ValueError で作れないことの確認")
    ok3 = False
    try:
        MemoryGuard(warn_mb=100, stop_mb=100)
    except ValueError as e:
        ok3 = True
        print(f"    期待通り ValueError: {e}")
    print(f"    合格={ok3}")

    print("\n[4] summary() が最大値を返すことの確認")
    print(f"    {guard.summary()}")
    ok4 = guard.summary().startswith("[test1] 最大メモリ=")
    print(f"    合格={ok4}")

    del junk, junk2
    print()
    print("=" * 78)
    all_ok = ok1 and ok2 and ok3 and ok4
    print("全項目OK" if all_ok else "一部NG（上記参照）")
    print("=" * 78)
    sys.exit(0 if all_ok else 1)
