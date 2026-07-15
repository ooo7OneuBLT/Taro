"""
D0（自己接触）の環境を録画して目で確認する。
MIMo公式の MIMoSelfBody-v0（座位・weldで固定＝倒れない・触覚ON・腕が自由）を、
腕をランダムに動かしながら描画し、mp4に保存する。

＝「太郎アルファが座って自分の体を触る」姿が意図どおりかを確認するための可視化。
使い方: python d_record.py [n_step] [env_id]
出力: D/logs/video/<env>_<日時>.mp4
"""
import os, sys, datetime, warnings
warnings.filterwarnings("ignore")
import numpy as np
import gymnasium as gym

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)
import mimoEnv  # noqa  (gym登録)
import cv2


def record(env_id="MIMoSelfBody-v0", n_step=300, every=2, scale=0.6, seed=0):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames, touch_trace = [], []
    for t in range(n_step):
        a = rng.uniform(-1, 1, env.action_space.shape[0]) * scale   # 腕を動かす(運動性喃語)
        obs, r, te, tr, info = env.step(a)
        if isinstance(obs, dict) and "touch" in obs:
            touch_trace.append(float(np.abs(obs["touch"]).sum()))
        if t % every == 0:
            f = env.render()
            if f is not None:
                frames.append(f)
        if te or tr:
            obs, _ = env.reset()
    env.close()
    if not frames:
        print("描画フレームが取得できなかった"); return None

    outdir = os.path.join(_HERE, os.pardir, "logs", "video")
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(outdir, f"{env_id.replace('-', '_')}_{stamp}.mp4")
    h, w, _ = frames[0].shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    print(f"=== 録画完了 ===")
    print(f"環境      : {env_id}（座位・weld固定・触覚ON）")
    print(f"フレーム  : {len(frames)}枚 ({w}x{h}) / {n_step}step")
    if touch_trace:
        nz = sum(1 for x in touch_trace if x > 0)
        print(f"触覚      : 立ったstep = {nz}/{len(touch_trace)} ({nz/len(touch_trace)*100:.0f}%)  平均総和={np.mean(touch_trace):.1f}")
    print(f"保存先    : {os.path.abspath(path)}")
    return path


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    eid = sys.argv[2] if len(sys.argv) > 2 else "MIMoSelfBody-v0"
    record(env_id=eid, n_step=n)
