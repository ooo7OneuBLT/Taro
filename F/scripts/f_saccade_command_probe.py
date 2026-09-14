# -*- coding: utf-8 -*-
"""命令を**こちらから**出して、眼球がどこへ着くかを測る（2026-09-12・ユーザー指示）。

【なぜ】走行任せだと命令の大きさを選べず、注意の都合で偏る。こちらで
  「視野の方向 h_dir」を決めて何度も出せば、命令の大きさごとの着地が測れる。

【体】本番と同じ組み立て（run.plugins.common.scene.build）を使う。
  古い測定スクリプトが独立に環境を組み立てて「別の体で測る」事故を起こした
  前例があるため（E/docs/実行基盤_設計.md）。

【使い方】
  E_SACC_FRAC=1.0 python F/scripts/f_saccade_command_probe.py --out <保存先.json>
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
    ap.add_argument('--repeats', type=int, default=6, help='同じ命令を何回出すか')
    ap.add_argument('--settle', type=float, default=1.6, help='1発あたり待つ秒数')
    a = ap.parse_args()

    from run.plugins.common import scene as scene_mod
    import taro_core.src.brain.midbrain.orienting as OR
    spec = json.load(open(EXP, encoding='utf-8'))
    env, sc, _ = scene_mod.build(spec['scene'], taro=spec['taro'], seed=0, verbose=False)
    u = env.unwrapped
    _or = getattr(u, '_orienting', None)
    if _or is None:
        print('反射が無い'); return 1
    print('利得 EYE_FB_GAIN=%.3f / 小分け SACCADE_FRAC=%.2f / 打ち切り %.2f秒'
          % (OR.EYE_FB_GAIN, OR.SACCADE_FRAC, OR.SACCADE_DURATION))
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if float(env.action_space.low[0]) >= 0.0: zero[:] = 0.0
    dt_env = float(u.model.opt.timestep) * int(u.frame_skip)
    n_wait = max(1, int(round(a.settle / dt_env)))

    # 命令：視野の方向[-1,1]。1.0 = 30度。左右両方向を振る。
    dirs = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows, seen = [], 0
    for rep in range(a.repeats):
        for hd in dirs:
            h_before = _or._version_h_deg()
            _or.set_map_target(hd, 0.0)
            for _ in range(n_wait):
                env.step(zero)
            new = _or.sacc_log[seen:]; seen = len(_or.sacc_log)
            if not new: continue
            s = new[0]                      # この命令で最初に撃たれた1発
            rows.append({
                '命令h_dir': hd,
                '命令角度_度': hd * 30.0,
                '撃つ前の眼球角_度': round(float(h_before), 3),
                '狙った目標角_度': round(float(s['tgt_h']), 3),
                '着地_度': round(float(s['h1']), 3),
                '残り誤差_度': round(abs(float(s['tgt_h'] - s['h1'])), 3),
                '偏心_度': round(abs(float(s['h1'])), 3),
                '長さ_秒': round(float(s['t_end'] - s['t']), 3),
                'この命令での発数': len(new),
            })
    json.dump({'設定': {'EYE_FB_GAIN': OR.EYE_FB_GAIN, 'SACCADE_FRAC': OR.SACCADE_FRAC,
                        'SACCADE_DURATION': OR.SACCADE_DURATION},
               '行': rows}, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('記録 %d 件 → %s' % (len(rows), a.out))
    return 0

if __name__ == '__main__':
    sys.exit(main())
