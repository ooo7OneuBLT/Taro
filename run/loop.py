"""古い経路（`E/scripts/e_growth_train.py`）を呼ぶ橋。**比較のためだけに残す**。

【役目は終わった、2026-07-30】脳の構築・学習ループ・目視は run/ へ移した：
    run/config.py       設定（実験ファイルから）
    run/taro_setup.py   太郎の中身（脳・学習器・神経調節・小脳）
    run/trainer.py      学習ループ本体
    run/viewer.py       等倍速の目視
`run/main.py` はもうこのファイルを呼ばない。

【それでも残す理由】新しい経路が古い経路と同じ結果を出すかを確かめるため。
  同じ設定で両方を回して数値を並べる、という検証はこれからも必要になる
  （落とし穴チェックリスト 項3：乱数を消費する順序が1つ違うだけで別の学習になる）。

注意：これは「実験ファイル → 環境変数 → e_growth_train」という橋渡しである。
  環境変数は新しい設計では使わない（実験ファイルが唯一の指定手段）。
  新しい実験でこれを使わない。
"""
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# 実験ファイルの `taro` 欄 → e_growth_train の環境変数への対応表。
# 注意：これは橋渡しのための一時的な対応表。第2段階で不要になる。
_TARO_TO_ENV = {
    "goal_babbling": ("E_GOALBABBLE", lambda v: "1" if v else "0"),
    "goal_switch":   ("E_GB_SWITCH", str),
    "closed_loop_reach": ("E_CLTRAIN", lambda v: "1" if v else "0"),
    "effort_cost":   ("E_EFFORT", str),
    "replay":        ("E_REPLAY", lambda v: "1" if v else "0"),
    "cerebellum":    ("E_CEREB", lambda v: "1" if v else "0"),
    "touch":         ("E_TOUCH", lambda v: "1" if v else "0"),
    "somatosensory": ("E_SOMATOSENSORY", lambda v: "1" if v else "0"),
    "age_months":    ("E_AGE", lambda v: repr(float(v))),
    "age_to":        ("E_AGE_TO", lambda v: repr(float(v))),
    "age_start":     ("E_AGE_START", lambda v: str(int(v))),
    "age_ramp":      ("E_AGE_RAMP", lambda v: str(int(v))),
    "age_every":     ("E_AGE_EVERY", lambda v: str(int(v))),
    "model":         ("E_LOADMODEL", str),
    "save":          ("E_SAVEMODEL", str),
    "vision":        ("E_E1_VISION", lambda v: "1" if v else "0"),
    "noise":         ("E_NOISE", str),
    "beta":          ("E_BETA", str),
}


def _env_from_spec(spec, *, steps, view=False):
    """実験ファイルから、e_growth_train に渡す環境変数を作る（橋渡し）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["E_E1"] = "1"                     # おもちゃ環境（目標E）
    env["E_SCENE"] = str(spec["scene"])   # 環境はシーンが決める
    env.setdefault("E_E1_VISION", "1")

    taro = spec.get("taro", {})
    # 駆動モード。既定（未指定）＝筋肉モード。関節モードは明示のときだけ
    mode = str(taro.get("actuation", "muscle")).lower()
    env["E_MUSCLE"] = "0" if mode in ("joint", "spring", "springdamper", "torque") else "1"
    for key, (name, conv) in _TARO_TO_ENV.items():
        if key in taro and taro[key] is not None:
            env[name] = conv(taro[key])

    r = spec.get("run", {})
    if "checkpoint" in r:
        env["E_CKPT"] = str(int(r["checkpoint"]))
    if "K" in r:
        env["E_K"] = str(int(r["K"]))
    if view:
        env["E_VIEW"] = "1"
        env["E_REALTIME"] = "1"
        env.setdefault("E_VIEW_EXPLORE", "1")
    return env


def run_train(spec, *, steps, seed=0, view=False, log_path=None):
    """学習（または目視）を回す。注意既存の e_growth_train を子プロセスで呼ぶ。

    Returns: 終了コード
    """
    env = _env_from_spec(spec, steps=steps, view=view)
    cmd = [os.path.join(_ROOT, ".venv", "Scripts", "python.exe"), "-u",
           os.path.join(_ROOT, "E", "scripts", "e_growth_train.py"),
           str(int(seed)), str(int(steps))]
    print("  注意[loop] これは古い経路（E/scripts/e_growth_train.py）を呼ぶ橋です。"
          "run/main.py はもうこれを使いません（比較のためだけに残しています）",
          flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fp:
            p = subprocess.run(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT,
                               cwd=_ROOT)
        print(f"  → ログ: {os.path.relpath(log_path, _ROOT)}", flush=True)
    else:
        p = subprocess.run(cmd, env=env, cwd=_ROOT)
    return p.returncode
