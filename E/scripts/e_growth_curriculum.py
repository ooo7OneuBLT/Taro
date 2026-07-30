"""目標E・成長カリキュラム学習：体を少しずつ育てながら、自己モデルと運動を同時に学習する。

【なぜ】これまでのEは、目標Cで(18ヶ月児相当の体で)学習し終えた脳に、新生児らしい反射や
喃語を"本能として後付け"していた。しかしユーザーとの議論で、自己モデル(自分の体がどう動くと
感覚がどう変わるか)は運動性喃語という探索行動そのものを通じて形成されるはずで、"すでに
動ける体で学んだ自己モデル"を後から新生児の動きに被せるのは順序が逆、と判断した。

そこで、白紙の脳＋新生児の体(age=0)から始め、E_SAVEMODEL/E_LOADMODELで段階ごとに
チェックポイントを引き継ぎながら、体を少しずつ大きくして学習を続ける。

【「週」との対応について】太郎の学習量(勾配ステップ数)と人間の神経発達(生後経過日数)は
単位の違う別プロセスで、両者を対応づける裏付けは無い[ユーザー合意、2026-07-23]。
DevelopmentalClock(taro_core/src/brain/developmental_clock.py)の考え方
「発達は時計でなく経験で進む」に従い、**「週」でなく学習ステップ数そのものを発達の物差しに
する**。STAGESの各段階が「生後およそ何週相当」かは参考程度の目安であり、検証済みの
対応関係ではない[Tier3・ARBITRARY]。

【技術的な注意】run_c_metrics_ac_lr.py は環境変数をモジュール読み込み時に1回だけ読む
構造なので、同一プロセス内でC_AGE等を変えて使い回すことはできない。段階ごとに
**新しいサブプロセスとして起動**する。

使い方:
  python e_growth_curriculum.py smoke   # 動作確認用の極小テスト（数十ステップ×2段階）
  python e_growth_curriculum.py run     # 本番のスケジュール（STAGESを編集して使う）
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
# 学習ループはCのオリジナルでなく、E側のコピー(e_growth_train.py)を使う（Cを無傷に保つ）。
_TRAIN_SCRIPT = os.path.join(_HERE, "e_growth_train.py")
_OUT_DIR = os.path.join(_HERE, os.pardir, "models", "growth_curriculum")

# 動作確認用：極小ステップ数で「引き継ぎの配管」だけを検証する。
SMOKE_STAGES = [
    {"age": 0.0, "n_train": 80, "label": "stage0_scratch"},
    {"age": 0.5, "n_train": 80, "label": "stage1_grown"},
]

# 【本番の叩き台・2026-07-23、⚠️目視後に確定】1段階目(age=0・色付き・白紙)で自己モデルが
# margin+51に到達し**累積約4800stepで飽和**、引き継ぎ後は+51再到達まで約1200stepと判明した
# ことに基づく。各値の根拠：
#  - stage0 の n_train=4800：白紙からの飽和点（実測）。
#  - stage1以降の n_train=3600：引き継ぎなら飽和が速い（実測1200で+51再到達）ので余裕を見て3600。
#  - 月齢の刻み（1ヶ月ずつ）：[Tier3・ARBITRARY]。細かすぎると学習回数増、粗いと体変化が大きく
#    再学習コスト増。叩き台として1ヶ月刻み。E1(リーチ)の前提＝首すわり3-4ヶ月まで並べた。
#  - β：本来は月齢で太郎自身が決めるべき（やることリスト項目11）。今は暫定で全段階β=0.7固定
#    （8週実測の流用）。将来 developmental_schedule で月齢→β/シナジーを接続する。
# ⚠️まだ実行しない。1段階目の動きのViewer目視で writhing GM らしさを確認してから、刻み・
#   step数・βスケジュールを確定する。合計約5.6時間の計算になる点も要確認。
RUN_STAGES = [
    {"age": 0.0, "n_train": 4800, "label": "stage0_age0"},
    {"age": 1.0, "n_train": 3600, "label": "stage1_age1"},
    {"age": 2.0, "n_train": 3600, "label": "stage2_age2"},
    {"age": 3.0, "n_train": 3600, "label": "stage3_age3"},
    {"age": 4.0, "n_train": 3600, "label": "stage4_age4"},
]

# 運動性喃語（脊髄CPG）の設定。全段階で色付き＋シナジー。βは暫定固定（上記コメント参照）。
# ⚠️【2026-07-25 E_SYNERGY を 1→0 に変更】理由は「効かなかったから」ではない。
#   ★`synergy` の実装は**探索ノイズを相関させるだけ**で、人間のシナジー
#   （脊髄回路が**運動出力そのもの**を制約する）とは別物だと判明した。
#   決定的な行動（act_mean）の測定ではシナジーは一切効かない＝人間の機構を再現していない。
#   → チェックリスト項42（名前が実装と合っていないものを疑う）。
#   ★正しい実装（方策の出力を低次元のシナジー空間に通す）は やることリストの課題。
#   実測（K=10, 7000step, 真のうつ伏せ>162度）: OFF 7.2% / ON 22.5%（ただし
#   シード間 0/39/29% でばらつきが大きく「改善は確認できず」までが正確）。
BABBLE = {"E_NOISE": "colored", "E_BETA": "0.7", "E_SYNERGY": "0"}


def run_stage(seed, stage, prev_ckpt, ckpt_interval=600):
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, f"{stage['label']}_seed{seed}.pt")
    env = os.environ.copy()
    env["E_AGE"] = str(stage["age"])
    env["E_SUPINE"] = "1"          # 新生児は仰向けから（既存のE方針と同じ）
    env["E_SAVEMODEL"] = out_path
    env["E_RECORD"] = out_path.replace(".pt", ".mp4")   # 各段階の動きを録画（目視用）
    env["E_CKPT"] = str(ckpt_interval)
    env.update(BABBLE)             # 色付き＋シナジー探索（太郎の中の脊髄CPGを有効化）
    if prev_ckpt:
        env["E_LOADMODEL"] = prev_ckpt
    else:
        env.pop("E_LOADMODEL", None)   # 明示的に消す＝白紙から（ゼロから開始）
    cmd = [sys.executable, _TRAIN_SCRIPT, str(seed), str(stage["n_train"])]
    print(f"\n=== {stage['label']} (age={stage['age']}mo, n_train={stage['n_train']}, "
          f"引き継ぎ元={'なし(ゼロから)' if not prev_ckpt else os.path.basename(prev_ckpt)}) ===")
    subprocess.run(cmd, env=env, check=True)
    return out_path


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    stages = {"smoke": SMOKE_STAGES, "run": RUN_STAGES}.get(mode, SMOKE_STAGES)
    seed = int(os.environ.get("E_SEED", "0"))
    prev_ckpt = None
    for stage in stages:
        prev_ckpt = run_stage(seed, stage, prev_ckpt)
    print(f"\n完了。最終チェックポイント: {prev_ckpt}")


if __name__ == "__main__":
    main()
