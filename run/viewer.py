"""★目視（等倍速で太郎を見る）。`run.type = "view"`。

【なぜ目視が要るか】太郎の指標は「行動の退化」を構造的に高評価することがあるので、
数字だけでは騙される（[[feedback-watch-dont-just-measure]]）。実際に：
  ・「足が根元からすごい動く」→ 体型補正が学習に効いていなかった（2026-07-25）
  ・「視線誘導反射の実験の時とは動きが全然違う」→ ★学習と測定で別の駆動モードだった
どちらも数字では気づけず、目で見て見つかった。

【なぜ切り出したか、2026-07-30】これは `E/scripts/e_growth_train.py` の
747〜850行を**そのまま写した**もの。実体（画面・キー操作・実時間追従）は
`taro_core/tools/motor_viewer.py` にあり、ここは太郎を繋ぐだけ。

⚠️★`E/scripts/e_viewer.py`（1687行・編集ウィンドウ付き）は**別物**。
  あちらはシーンを作る/直すための道具で、学習済みの太郎を動かすものではない。
  統合は第4段階（`E/docs/実行基盤_設計.md` §7）。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
for _p in (os.path.join(_ROOT, "taro_core", "tools"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                            # noqa: E402

from run.taro_setup import rescale_action               # noqa: E402
from run.trainer import Trainer                         # noqa: E402


def view(cfg, *, plugins=(), verbose=True):
    """学習済みの太郎を等倍速で動かして見る。★学習しない。"""
    from motor_viewer import run_viewer

    # ★プラグインは受け取らない。目視のループは `motor_viewer.run_viewer` が回すので、
    #   on_step / on_checkpoint が**一度も呼ばれない**。
    #   ⚠️黙って「0件」のまま終わると「測ったつもり」になるので、ここで止める。
    #     （2026-07-30 の点検で発覚。統合は第4段階＝E/docs/実行基盤_設計.md §7）
    if plugins:
        raise ValueError(
            "★run.type=view では測る道具（plugins）は使えません。\n"
            f"  指定されたもの: {[p.name for p in plugins]}\n"
            "  目視のループは motor_viewer が回すので on_step が呼ばれず、\n"
            "  **黙って何も測らないまま終わる**ため止めています。\n"
            "  測るなら run.type=train を使ってください（統合は第4段階）。")

    tr = Trainer(cfg, plugins=(), verbose=verbose).build()
    t, env = tr.taro, tr.env
    if not cfg.model:
        print("⚠️[view] ★モデルを指定していない＝白紙の脳を見ています"
              "（taro.model にチェックポイントのパスを書く）", flush=True)
    if cfg.view_explore:
        print(f"[view] 自発運動ON（探索）: std={cfg.view_std}"
              f"（学習初期の実効値≒0.174／学習中は ne で変動）", flush=True)

    # ★可変にしておく＝実行中に切り替えて往復できる。
    #   別プロセスで2回起動して比べると乱数も姿勢も違うので、「どちらが速いか」の
    #   ような主観の比較が当てにならない（同じ状態で切り替える）。
    gb = [bool(cfg.view_goal_babbling)]
    gbuf, stat = [], {"goal": 0, "explore": 0}

    def _toggle():
        gb[0] = not gb[0]
        print(f"  [目標指向] {'★ON' if gb[0] else 'OFF'}"
              f"（目標指向 {stat['goal']} 回 / 探索 {stat['explore']} 回）", flush=True)

    def _status():
        # ★画面右上に出す文字。Viewerだけ見ていても条件と発動回数が分かるように。
        #   （print だけだと「コンソールなんてない」＝機能しているか分からない）
        if not gb[0]:
            return "GoalBabbling OFF"
        if len(gbuf) < 64:
            return f"GoalBabbling ON (warmup {len(gbuf)}/64)"
        n = max(stat["goal"], 1)
        return (f"GoalBabbling ON goal={stat['goal']} "
                f"dAction avg={stat.get('dsum', 0.0) / n:.4f} "
                f"max={stat.get('dmax', 0.0):.3f}")

    def policy_fn(obs, prev_a, hidden, *, recompute, frac):
        # 連続制御の frac は今のモデルでは使わない（policy は境界でのみ再計算）
        if not recompute:
            return prev_a, hidden
        sv = t.fusion.encode(obs)
        cf = t.target_fusion.encode(obs).detach()
        z, _kl, _rc, hn = t.infer_latent(sv, prev_a, cf, hidden)
        z = z.detach()
        mean = t.act_mean(z)
        # ⚠️★経験のバッファは**常に**溜める（ON/OFF に関係なく）。
        #   ONのときだけ溜める実装にしたため、切り替えるたび64件たまる前にOFFになり
        #   **一度も発動しなかった**（2026-07-30、ユーザーの「g押してもなんも変わってない」）
        clp = t.encode_target(obs).detach()
        gbuf.append(clp)
        if len(gbuf) > 2000:
            gbuf.pop(0)
        if gb[0]:
            if len(gbuf) >= 64 and torch.rand(1).item() < cfg.view_gb_rate:
                g = gbuf[torch.randint(len(gbuf), (1,)).item()]
                before = mean.detach().clone()
                mean = t.infer_goal_action(z, clp, mean, g)
                d = float((mean - before).abs().mean())
                stat["dsum"] = stat.get("dsum", 0.0) + d
                stat["dmax"] = max(stat.get("dmax", 0.0), d)
                stat["goal"] += 1
            else:
                stat["explore"] += 1
        if cfg.view_explore:
            a, _lp = t.brain.explore(mean, torch.full_like(mean, cfg.view_std))
            return a.detach(), hn.detach()
        return torch.clamp(mean, -1.0, 1.0).detach(), hn.detach()

    banner = f"model={os.path.basename(cfg.model) if cfg.model else '(白紙)'}"
    # ⚠️★g は使えない：MuJoCo の組み込みキーと衝突して「世界が暗くなる」。
    #   記号キーは組み込みで使われていないので安全側に3つ用意する。
    from run.trainer import close_env
    try:
        run_viewer(env, t.brain, policy_fn, rescale_action,
                   K=cfg.K, n_act=t.n_act, banner=banner,
                   extra_keys={";": _toggle, "'": _toggle, "/": _toggle},
                   status_fn=_status)
    finally:
        close_env(env)      # ⚠️窓を閉じても例外が出ても必ず片づける
    return {"note": "目視（学習していない）"}
