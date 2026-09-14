# -*- coding: utf-8 -*-
"""小脳にあたる層（サッケード適応）が、誤差だけで正しい倍率へ寄るかを測る。

【設計・2026-09-12・ユーザー承認】
  わざと間違った倍率（0.7 / 1.3）から始め、着地の誤差だけを頼りに太郎が自分で直す。
  **正しい値はこちらが書かない。**「着地が狙いと一致する」という物理が決める。

【走行前に固定した合否】
  ① 精度：**元の命令**に対する誤差が、適応なしの 0.31度 より小さくなる
     （ずらした狙いに対してではない。ここを間違えると良くなったように見えるだけ）
  ② 機構：0.7 からも 1.3 からも**同じあたり**へ寄る（片方だけならただの流れ）
  ③ 速さ：何発で寄ったか。人間は163〜827回。**合否には入れない・記録するだけ**

【使い方】
  E_SACC_ADAPT=1 E_SACC_ADAPT_INIT=0.7 python F/scripts/f_saccade_adapt_probe.py --out x.json
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
import numpy as np

EXP = 'F/experiments/F2-135pre_眼球の利得を3倍に_短い走行_2026-09-12.json'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--commands', type=int, default=400, help='出す命令の総数')
    ap.add_argument('--settle', type=float, default=1.6)
    a = ap.parse_args()

    from run.plugins.common import scene as scene_mod
    import taro_core.src.brain.midbrain.orienting as OR
    spec = json.load(open(EXP, encoding='utf-8'))
    env, sc, _ = scene_mod.build(spec['scene'], taro=spec['taro'], seed=0, verbose=False)
    u = env.unwrapped
    _or = u._orienting
    print('適応=%s 初期倍率=%.2f 学習率=%.4f / 小分け=%.2f 利得=%.2f'
          % (OR.USE_SACC_ADAPT, OR.SACC_ADAPT_INIT, OR.SACC_ADAPT_RATE,
             OR.SACCADE_FRAC, OR.EYE_FB_GAIN))
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    dt_env = float(u.model.opt.timestep) * int(u.frame_skip)
    n_wait = max(1, int(round(a.settle / dt_env)))

    dirs = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows, seen, rng = [], 0, np.random.default_rng(0)
    for i in range(a.commands):
        hd = float(dirs[rng.integers(0, len(dirs))])
        h_before = _or._version_h_deg()
        _or.set_map_target(hd, 0.0)
        for _ in range(n_wait):
            env.step(zero)
        new = _or.sacc_log[seen:]; seen = len(_or.sacc_log)
        if not new: continue
        s = new[0]
        want = OR.EYE_SIGN_H * OR.SACCADE_FRAC * hd * 30.0   # 本来動かしたかった量[度]
        act = float(s['h1']) - float(h_before)
        rows.append({'番号': i, '命令h_dir': hd,
                     '本来の狙い_度': round(want, 3), '実際に動いた_度': round(act, 3),
                     '元の命令に対する誤差_度': round(abs(want - act), 3),
                     '倍率': round(float(_or._adapt_gain), 5),
                     '撃つ前_度': round(float(h_before), 3),
                     '着地_度': round(float(s['h1']), 3),
                     '狙った目標角_度': round(float(s['tgt_h']), 3)})
    json.dump({'設定': {'USE_SACC_ADAPT': OR.USE_SACC_ADAPT,
                        'SACC_ADAPT_INIT': OR.SACC_ADAPT_INIT,
                        'SACC_ADAPT_RATE': OR.SACC_ADAPT_RATE,
                        'SACCADE_FRAC': OR.SACCADE_FRAC, 'EYE_FB_GAIN': OR.EYE_FB_GAIN},
               '最終倍率': float(_or._adapt_gain),
               '可動域_度': _or._eye_h_limit_deg,
               '学習に使わなかった発': int(getattr(_or, 'adapt_skipped', 0)), '行': rows},
              open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('記録 %d 件 / 最終倍率 %.4f / 可動域 %s度 / 学習に使わなかった発 %d → %s'
          % (len(rows), _or._adapt_gain, _or._eye_h_limit_deg,
             getattr(_or, 'adapt_skipped', 0), a.out))
    return 0

if __name__ == '__main__':
    sys.exit(main())
