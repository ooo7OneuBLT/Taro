"""
診断：太郎のシミュレーション1本が使う約2.7GBの内訳を、起動の段階ごとに実測する。

【なぜ必要か】2026-07-15
6本並列で物理メモリ31.7GBのうち空きが773MBになり、コミットが上限の99%に達した。
太郎はコア律速ではなく**メモリ律速**（コアは18個空いていた）＝メモリを削れば本数が増え、
シード数が増える。今日いちばん痛かったのは n=1 でノイズに埋もれたことなので、ここは効く。

【憶測で潰した候補（どちらも実測でシロ）】
  - 触覚のメッシュ  → 触覚なし2,745MB vs 触覚あり2,785MB＝差は**40MBだけ**
  - 海馬のリプレイ  → 上限3600件 × detach済みテンソル7個 ≒ **27MB**
どちらも2桁足りない。よって「何となく重そうなもの」を疑うのをやめ、段階ごとに測る。

使い方: python mem_profile.py [--touch]
"""
import os, sys, gc, warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))


def rss_mb():
    """このプロセスが今つかんでいる物理メモリ(MB)。psutilが無ければWindows APIで取る。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1024 / 1024


_prev = [rss_mb()]
_marks = []


def mark(label):
    gc.collect()
    now = rss_mb()
    delta = now - _prev[0]
    _marks.append((label, delta, now))
    print(f"{label:38s} +{delta:8.1f} MB   （累計 {now:8.1f} MB）", flush=True)
    _prev[0] = now


def main():
    touch = "--touch" in sys.argv
    print(f"=== 1本あたりのメモリ内訳（触覚{'あり' if touch else 'なし'}）===")
    print(f"{'段階':38s} {'増加':>11s}   {'累計':>16s}")
    mark("python起動直後")

    import numpy  # noqa
    mark("import numpy")
    import torch
    torch.set_num_threads(1)
    mark("import torch")

    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))  # paths は taro_core へ移設
    import paths
    paths.setup_brain_path()
    sys.path.insert(0, paths.MIMO_DIR)
    sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
    import gymnasium as gym
    import mimoEnv  # noqa
    mark("import gym / mimoEnv")

    from hybrid_env import HybridEnv
    from taro_brain_motor import TaroBrainWithMotor
    from cerebellum_motor import MotorCerebellum
    from test_phase8_motor_learning import to_tensor
    mark("import 太郎の脳")

    tp = None
    if touch:
        from d_supine_env import infant_touch_params
        tp = infant_touch_params(2.0)
    env = HybridEnv(gym.make("MIMoBenchV2-v0", vision_params=None, touch_params=tp))
    mark("gym.make（MuJoCoモデル構築）")

    obs, _ = env.reset()
    mark("env.reset")

    n_act = env.action_space.shape[0]
    sdim = 192
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    mark("脳＋小脳の構築")

    import numpy as np
    for i in range(200):
        env.step(env.action_space.sample() * 0)
    mark("env.step × 200")

    print("\n=== 大きい順 ===")
    for label, delta, _ in sorted(_marks, key=lambda x: -x[1])[:5]:
        print(f"  {label:38s} {delta:8.1f} MB")
    total = _marks[-1][2]
    print(f"\n合計 {total:.0f} MB")
    print(f"※ 脳のパラメータ実体は {sum(p.numel() for p in brain.parameters()) * 4 / 1024 / 1024:.1f} MB "
          f"（GRU64次元・ヘッド128次元なので当然小さい）")
    env.close()


if __name__ == "__main__":
    main()
