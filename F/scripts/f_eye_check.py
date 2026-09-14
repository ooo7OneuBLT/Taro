# -*- coding: utf-8 -*-
"""眼球が命令どおり動くかを、ユーザー自身が走らせて確かめるための道具（2026-09-12）。

【なぜ】ユーザー指摘「今はあなたの報告からしか判断材料がない」。
  Claude が選んだ数字を見せるのでは足りない。**本人が走らせて、生の2つの数を見る**。
  出力は「命令した角度」と「実際に動いた角度」だけ。平均・割合・統計は一切出さない。

【使い方】
    python F/scripts/f_eye_check.py
    python F/scripts/f_eye_check.py --n 60          # 回数を増やす
    python F/scripts/f_eye_check.py --frac 0.30     # 直す前の設定で比べる
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
# Windows の既定の文字コードだと罫線や日本語が化けるので、ここで直す
#   （利用者に PYTHONIOENCODING を付けさせないため）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import numpy as np

EXP = 'F/experiments/F2-135pre_眼球の利得を3倍に_短い走行_2026-09-12.json'
DIRS = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30, help='命令を何回出すか（既定30）')
    ap.add_argument('--frac', type=float, default=None,
                    help='1発でどれだけ狙うか。既定は今の設定(1.0)。0.30 で直す前と比べられる')
    ap.add_argument('--gain', type=float, default=None, help='筋の強さ。既定は今の設定(0.75)')
    a = ap.parse_args()

    # 【2026-09-12】この道具もログを残す。走った結果がおかしいとき、
    #   エラー.log に何が出ていたかを後から見られるようにする。
    from run.log_setup import setup_logging, log_tail, get_logger, warn_if
    _paths = setup_logging(os.path.join('F', 'logs', '_ログ', '眼球の確認'))
    log = get_logger('眼球の確認')

    print('準備しています（MuJoCo の組み立てに数秒かかります）...')
    from run.plugins.common import scene as scene_mod
    spec = json.load(open(EXP, encoding='utf-8'))
    env, sc, _ = scene_mod.build(spec['scene'], taro=spec['taro'], seed=0, verbose=False)
    u = env.unwrapped
    _or = u._orienting
    # 定数は「実際に動いている側」のモジュールを書き換える（二重 import 対策）
    OR = sys.modules[type(_or).__module__]
    if a.frac is not None:
        OR.SACCADE_FRAC = a.frac
    if a.gain is not None:
        OR.EYE_FB_GAIN = a.gain
    limit = _or._eye_h_limit_deg or 45.0

    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    dt = float(u.model.opt.timestep) * int(u.frame_skip)
    rng = np.random.default_rng(0)

    print('')
    print('  設定： 1発でどれだけ狙うか = %.2f ／ 筋の強さ = %.2f ／ 眼球の可動域 = ±%.0f度'
          % (OR.SACCADE_FRAC, OR.EYE_FB_GAIN, limit))
    print('  出てくるのは2つの数だけです。左が命令、右が実際。並んでいれば命令どおり動いています。')
    print('')
    print('   回  │ 命令した角度 │ 実際に動いた角度 │  差   │')
    print('  ─────┼──────────────┼──────────────────┼───────┤')

    near, total, blocked = 0, 0, 0
    for i in range(a.n):
        # 前の1発が終わるまで待つ（終わっていないと前の動きを混ぜてしまう）
        for _ in range(int(round(1.5 / dt))):
            if _or._sacc_remaining <= 0.0:
                break
            env.step(zero)
        for _ in range(int(round(0.3 / dt))):
            env.step(zero)
        _or.sacc_log = []
        h0 = _or._version_h_deg()
        hd = float(DIRS[rng.integers(0, len(DIRS))])
        _or.set_map_target(hd, 0.0)
        for _ in range(int(round(2.5 / dt))):
            env.step(zero)
            if _or.sacc_log:
                break
        if not _or.sacc_log:
            print('  %4d │ 撃たれませんでした' % (i + 1))
            continue
        s = _or.sacc_log[0]
        cmd = abs(hd) * 30.0
        moved = abs(float(s['h1']) - h0)
        out = abs(float(s['tgt_h'])) > limit
        tag = '  ← 可動域に当たった（体の限界）' if out else ''
        print('  %4d │ %9.1f 度 │ %13.1f 度 │ %5.1f │%s'
              % (i + 1, cmd, moved, abs(cmd - moved), tag))
        if out:
            blocked += 1
        else:
            total += 1
            if abs(cmd - moved) <= 1.0:
                near += 1

    print('  ─────┴──────────────┴──────────────────┴───────┘')
    print('')
    print('  1度以内で届いた： %d / %d 発' % (near, total))
    if blocked:
        print('  可動域に当たった： %d 発（上の数には入れていません）' % blocked)
    print('')
    print('  ※ 上の表の数字以外は使っていません。平均も割合も出していません。')

    # 【2026-09-12・これが一番効く】例外を投げないミスを捕まえる見張り。
    #   2026-09-12 に起きた5件のミスは全部「落ちない」ミスだった。
    #   測定の結果を出す前に、前提が崩れていないかをここで確かめて警告にする。
    warn_if(log, total < max(5, a.n // 3),
            '使えた発が %d 本しかない（命令 %d 本のうち）。この数では判断できない',
            total, a.n)
    warn_if(log, blocked > 0,
            '可動域に当たった発が %d 本ある。眼球は機械的に止まるので、'
            'この発は制御の良し悪しと無関係', blocked)
    warn_if(log, total > 0 and near == 0,
            '1度以内で届いた発が0本。設定（FRAC=%.2f／GAIN=%.2f）が効いていないか、'
            '符号が逆の疑い', OR.SACCADE_FRAC, OR.EYE_FB_GAIN)
    log.debug('命令 %d 本 / 使えた %d 本 / 1度以内 %d 本 / 可動域に当たった %d 本',
              a.n, total, near, blocked)
    log_tail(_paths)
    return 0


if __name__ == '__main__':
    sys.exit(main())
