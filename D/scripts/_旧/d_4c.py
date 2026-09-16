"""
④c（最小）：AがBの行動を"触覚"で認識できるか。
相手Bを近くに置き、Bに複数の固定"仕草"(gesture)をさせる。各仕草はAを異なる触られ方で触る。
Aは受けた触覚パターン(56次元)から「どの仕草か」を当てる（最近傍centroid分類）。
＝相手を"知覚(触覚)"して、その行動を認識できるか＝理解の第一歩（対応問題の入口）。

対照：shuffle（仕草ラベルを混ぜる→チャンス1/K）。
判定：認識率がチャンス(1/K)を明確に超え、shuffleを上回れば「Aは触覚でBの行動を認識できた」。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from two_agent_env import TwoAgentMIMo


def run(seed=0, K_gestures=6, trials_per=25, T=15, sep=0.12):
    rng = np.random.RandomState(seed)
    env = TwoAgentMIMo(sep=sep)
    m = env.model
    lo = np.array([m.actuator_ctrlrange[bi, 0] for bi in env.bid])
    hi = np.array([m.actuator_ctrlrange[bi, 1] for bi in env.bid])
    # B の固定仕草（各仕草＝別々のランダム行動ベクトル）
    gestures = [lo + rng.rand(env.nb) * (hi - lo) for _ in range(K_gestures)]
    a_passive = np.zeros(env.na)

    X, y = [], []
    for _ in range(K_gestures * trials_per):
        g = rng.randint(K_gestures)
        env.reset()
        touch = np.zeros(env.n_touch); contact_steps = 0
        for t in range(T):
            obs = env.step(a_passive, gestures[g], K=5)
            tb = obs["touch_of_B"]
            if tb.sum() > 0:
                touch += tb; contact_steps += 1
        if contact_steps > 0:
            X.append(touch / contact_steps); y.append(g)
    X = np.array(X); y = np.array(y)
    n = len(y)
    if n < K_gestures * 4:
        print(f"接触データ不足 (n={n})。sepを詰めるかTを増やす必要。", flush=True); return
    # 正規化
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    # split
    idx = rng.permutation(n); S = n // 2
    tr, te = idx[:S], idx[S:]

    def nearest_centroid_acc(ytr):
        cents = {}
        for g in range(K_gestures):
            xs = X[tr][ytr == g]
            if len(xs) > 0:
                cents[g] = xs.mean(0)
        correct = 0
        for i in te:
            dists = {g: np.linalg.norm(X[i] - c) for g, c in cents.items()}
            pred = min(dists, key=dists.get)
            if pred == y[i]:
                correct += 1
        return correct / len(te) * 100

    acc = nearest_centroid_acc(y[tr])
    yshuf = y[tr].copy(); rng.shuffle(yshuf)
    shuf = nearest_centroid_acc(yshuf)
    print(f"=== ④c 触覚でBの行動認識 (seed={seed}, K={K_gestures}仕草, n={n}接触trial) ===", flush=True)
    print(f"チャンス = {100/K_gestures:.1f}%", flush=True)
    print(f"認識率   = {acc:.1f}%", flush=True)
    print(f"shuffle  = {shuf:.1f}%", flush=True)
    print(f"判定: 認識率がチャンス({100/K_gestures:.0f}%)とshuffleを明確に超えれば「Aは触覚でBの行動を認識」成立", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
