"""
シミュレーション実行時の負荷設定。各run_*.pyの先頭で `import perf_setup` するだけで、
①BLAS(MKL/OpenMP)のスレッド数をtorch.set_num_threads相当に絞り、
②プロセス優先度を下げて他の作業(ブラウザ等)を優先させる。

torch.set_num_threads()はPyTorch自身の演算スレッドしか制限しないため、その下で動く
BLASライブラリのスレッドが環境変数未設定だと全コアに広がりうる（実際にタスクマネージャで
24コア全部が薄く busy になる現象として観測された）。BLAS初期化前に環境変数を設定する
必要があるため、torchをimportする前にこのモジュールをimportすること。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    # ③論理コアの上半分だけを使うように固定する（CPUアフィニティ）。優先度DOWNだけでは、
    # シミュレーションのスレッドが特定コアを100%占有してしまい、たまたま同じコアに
    # 乗ったブラウザ等の処理が順番待ちで固まる現象が実際に観測された（平均CPU使用率は
    # 低く見えても体感が重い）。下半分のコアを常に空けておくことで、優先度を落とさずに
    # 他の作業と共存できる。コア数はマシン依存で決め打ちせず動的に計算する。
    n = os.cpu_count() or 1
    if n >= 4:
        upper_half = list(range(n // 2, n))
        p.cpu_affinity(upper_half)
except ImportError:
    pass
except Exception:
    pass   # cpu_affinity未対応環境（一部OS）でも他の設定は活かす
