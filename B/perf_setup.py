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
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except ImportError:
    pass
