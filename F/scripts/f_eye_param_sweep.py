# -*- coding: utf-8 -*-
"""眼球の定数を**1つずつ**振って、命令どおりに動くかを採点する（2026-09-12・ユーザー指示）。

【肝】命令の並びは**全条件で同じ**。違いが定数のせいだと言えるようにするため。
  環境の組み立ては1回だけで、定数はモジュールの値を走行中に書き換える
  （実測：組み立て3.3秒／1命令0.11秒）。

【採点】可動域(±45度)の内側・振幅2度以上の発だけを使い、
  **元の命令に対する誤差の中央値**を見る。到達率・時間切れも一緒に記録する。

【使い方】python F/scripts/f_eye_param_sweep.py --out F/logs/... --commands 120
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import numpy as np

EXP = 'F/experiments/F2-135pre_眼球の利得を3倍に_短い走行_2026-09-12.json'
DIRS = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]

SWEEPS = [
    ('EYE_FB_GAIN',      [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], '筋の強さ'),
    ('SACCADE_DURATION', [0.10, 0.20, 0.30, 0.45, 0.60],   '打ち切りの時間[秒]'),
    ('SACCADE_DONE_DEG', [0.10, 0.25, 0.50, 0.75, 1.00],   'どこまで近づいたら終えるか[度]'),
    ('SACCADE_FRAC',     [0.8, 0.9, 1.0, 1.1, 1.2],        '1発でどれだけ狙うか'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--commands', type=int, default=120)
    ap.add_argument('--settle', type=float, default=0.5)
    ap.add_argument('--only', default='', help='この定数だけ振る（試し用。カンマ区切り）')
    ap.add_argument('--limit', type=int, default=0, help='各定数の値を先頭N個だけ（試し用）')
    a = ap.parse_args()

    from run.plugins.common import scene as scene_mod
    import taro_core.src.brain.midbrain.orienting as OR
    spec = json.load(open(EXP, encoding='utf-8'))
    env, sc, _ = scene_mod.build(spec['scene'], taro=spec['taro'], seed=0, verbose=False)
    u = env.unwrapped; _or = u._orienting
    # 【2026-09-12・重大な直し】orienting は**二重に import されている**
    #   （`taro_core.src.brain.midbrain.orienting` と `midbrain.orienting`）。
    #   実際に動いている反射が属するのは後者なので、前者の定数を書き換えても
    #   **一度も効かない**。最初の掃引21条件はこれで全部無効になった
    #   （既定の設定を21回測っていただけ。数字が3桁まで一致したのが証拠）。
    #   → 実体側のモジュールを取り、そちらを書き換える。
    LIVE = sys.modules[type(_or).__module__]
    if LIVE is not OR:
        print('注意：orienting が二重 import（%s と %s）。実体側 %s を書き換える'
              % (OR.__name__, LIVE.__name__, LIVE.__name__))
    OR = LIVE
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    dt_env = float(u.model.opt.timestep) * int(u.frame_skip)
    n_wait = max(1, int(round(a.settle / dt_env)))
    # 全条件で同じ命令の並び
    rng = np.random.default_rng(12345)
    seq = [float(DIRS[i]) for i in rng.integers(0, len(DIRS), a.commands)]
    defaults = {k: getattr(OR, k) for k, _, _ in SWEEPS}
    print('既定値:', {k: round(v, 3) for k, v in defaults.items()})
    print('適応は%s／命令 %d 本・待ち %.1f秒'
          % ('ON（注意）' if OR.USE_SACC_ADAPT else 'OFF', a.commands, a.settle))

    def measure():
        """1条件を測る。命令ごとに『そのサッケードが撃たれて終わるまで』待つ。

        【2026-09-12・最初の試作の不具合2つを直した】
          ① `sacc_log` は400件で打ち止めなので、条件ごとに**空にする**
             （4条件目で400に達し、以降の測定が全部 nan になった）
          ② 固定の待ち時間では、ある命令のサッケードが次の命令の窓へ食い込む
             （地図の潜時0.25秒＋サッケード0.30秒＝0.55秒 > 待ち0.5秒）。
             対応づけが崩れるので、**終わるのを見てから**次へ進む。
        """
        env.reset(seed=0)
        _or.sacc_log = []                      # ①
        es, rs, acts, to, out, n, miss = [], [], [], 0, 0, 0, 0
        max_steps = int(round(2.5 / dt_env))   # 1命令あたりの上限
        for hd in seq:
            # 【2026-09-12・直し3】命令を出す前に**撃ち終わっているか**を確かめる。
            #   確かめないと、前の命令の撃ち残りを今の命令の結果として拾い、
            #   誤差が見かけ上4倍に膨らんだ（0.3度→1.2度）。
            for _ in range(int(round(1.5 / dt_env))):
                if _or._sacc_remaining <= 0.0: break
                env.step(zero)
            for _ in range(int(round(0.3 / dt_env))):   # 落ち着くまで少し置く
                env.step(zero)
            _or.sacc_log = []
            h0 = _or._version_h_deg()
            _or.set_map_target(hd, 0.0)
            got = None
            for _ in range(max_steps):         # ②撃たれて終わるまで待つ
                env.step(zero)
                if _or.sacc_log:
                    got = _or.sacc_log[0]; break
            if got is None:
                miss += 1
                continue
            _or.sacc_log = []
            n += 1
            if abs(float(got['tgt_h'])) > 45.0:
                out += 1; continue
            # 【2026-09-12・直し4】期待値に SACCADE_FRAC を掛けてはいけない。
            #   掛けると「自分で下げた目標に届いたか」を測ることになり、FRAC を
            #   変えても分母が一緒に動いて比が不変＝FRAC の効果が見えなくなる
            #   （自己検査で見つけたのと同じ穴を、本体でも踏んでいた）。
            #   命令は「視野のこの方向（=hd×30度）を見ろ」なので、期待値はそれ。
            want = OR.EYE_SIGN_H * hd * 30.0
            if abs(want) < 2.0: continue
            act = float(got['h1']) - h0
            es.append(abs(want - act)); rs.append(abs(act) / abs(want))
            acts.append(abs(act))
            if (float(got['t_end']) - float(got['t'])) >= OR.SACCADE_DURATION - 0.02: to += 1
        return {'動いた量_中央': float(np.median(acts)) if acts else float('nan'),
                '誤差_中央': float(np.median(es)) if es else float('nan'),
                '誤差_75%点': float(np.percentile(es, 75)) if es else float('nan'),
                '到達率': float(np.median(rs)) if rs else float('nan'),
                '時間切れ率': to / max(len(es), 1), '可動域外率': out / max(n, 1),
                '使えた発': len(es), '撃てた発': n, '撃てなかった命令': miss}

    sweeps = SWEEPS
    if a.only:
        keep = {x.strip() for x in a.only.split(',')}
        sweeps = [x for x in sweeps if x[0] in keep]
    if a.limit:
        sweeps = [(n, v[:a.limit], l) for n, v, l in sweeps]
    # 【自己検査】定数が本当に効いているかを、極端な値で確かめてから始める。
    #   これが無いと「書き換えたつもりで効いていない」を見逃す（実際に見逃した）。
    base = measure()
    setattr(OR, 'SACCADE_FRAC', 0.2)          # 1発でずれの2割しか狙わない＝必ず大きく届かない
    chk = measure()
    setattr(OR, 'SACCADE_FRAC', defaults['SACCADE_FRAC'])
    print('自己検査：SACCADE_FRAC 既定%.2f → 実際に動いた量 %.2f度 ／ 0.2 → %.2f度'
          '（比 %.2f。0.2/%.2f=%.2f になるはず）'
          % (defaults['SACCADE_FRAC'], base['動いた量_中央'], chk['動いた量_中央'],
             chk['動いた量_中央'] / max(base['動いた量_中央'], 1e-9),
             defaults['SACCADE_FRAC'], 0.2 / defaults['SACCADE_FRAC']))
    if not (chk['動いた量_中央'] < base['動いた量_中央'] * 0.5):
        print('**自己検査に落ちた**：定数を変えても挙動が変わっていない。'
              '書き換えが効いていないので中止する。')
        return 1
    print('自己検査：合格（定数の書き換えが効いている）')

    rows, t0 = [], time.time()
    for name, values, label in sweeps:
        for v in values:
            for k, d in defaults.items(): setattr(OR, k, d)   # 他は既定へ戻す
            setattr(OR, name, v)
            r = measure(); r.update({'定数': name, '意味': label, '値': v,
                                     '既定か': abs(v - defaults[name]) < 1e-9})
            rows.append(r)
            print('  %-17s = %-5s 誤差 %.3f度 / 到達率 %.1f%% / 時間切れ %.0f%% （%.0f秒経過）'
                  % (name, v, r['誤差_中央'], 100 * r['到達率'],
                     100 * r['時間切れ率'], time.time() - t0)
                  + ' 使えた発 %d / 撃てず %d' % (r['使えた発'], r['撃てなかった命令']))
    for k, d in defaults.items(): setattr(OR, k, d)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'既定値': defaults, '命令数': a.commands, '待ち秒': a.settle,
               '適応': bool(OR.USE_SACC_ADAPT), '行': rows},
              open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n%d 条件 / %.0f 秒 → %s' % (len(rows), time.time() - t0, a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
