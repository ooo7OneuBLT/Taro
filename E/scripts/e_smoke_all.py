"""【全スクリプトの動作確認】E/scripts の全部を短時間だけ実行し、落ちないかを機械的に確かめる。

【なぜ静的なコード読みでなく"動かす"のか】
2026-07-20 の1日で見つかったバグ9件は、**ほぼ全部が「動かして出力を見る」ことでしか
見つからなかった**：
  ・視覚と触覚が脳に繋がっていなかった（コードは読める形で正しかった。渡していないだけ）
  ・視力フィルタが無効（`if acuity:` が 0.0 を弾く。読んでも気づけない）
  ・`render()` の戻り値が内部バッファ（型も文法も正しい）
  ・`Renderer.close()` が後続の描画を壊す（副作用）
  ・`pc_latent.infer` が確率的（呼ぶたびに違う値）
＝**「文法エラーがない」と「意図どおり動く」は全く別**。だから機械的に実行する。

【やること】
 各スクリプトを**最小のtick数**で実行し、
   ・例外で落ちないか
   ・想定した出力（判定行・数値）が出るか
 を確認する。長い測定はしない（バグ検出が目的で、結果を得るのが目的ではない）。

⚠️これで見つかるのは「落ちる／繋がっていない」まで。**計算の中身が正しいかは別**
（それは e_wiring_check.py と、結果の妥当性チェックで見る）。

使い方: python e_smoke_all.py [timeout_sec]
"""
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")

# (スクリプト, 引数, 出力に含まれてほしい語)  ※Noneなら「落ちなければOK」
CASES = [
    ("e_wiring_check.py",  [],          "結果:"),
    ("e_hand_in_view.py",  ["3"],       "両目の視野内"),
    ("e_baseline2.py",     ["5", "0"],  "hand_in_view"),
    ("e_prone_check.py",   ["3", "0"],  "腹側の向き"),
    ("e_gaze_geometry.py", ["3"],       "視線からのずれ"),
    ("e_eye_input.py",     ["2", "0"],  "動画:"),
    ("e_clip_handview.py", ["30", "0"], None),      # イベント0回もありうる
    ("e_vision_signal.py", ["5", "0"],  "予測対象:"),
    ("e_latent_probe.py",  ["1"],       None),      # 場面が集まらない可能性
    ("e_encoder_probe.py", ["1"],       None),
    ("e_head_motion.py",   ["3"],       None),
    ("e_reach_space.py",   [],          None),
    ("e_toy_check.py",     ["3"],       None),
    ("e_limb_strength.py", [],          None),
    ("e_contact_signal.py", ["3"],      None),
    ("e_contact_diag.py",  ["3"],       None),
]

ENV = dict(os.environ)
ENV.update({"PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "E_TOY": "1", "E_VOR": "1", "C5_AGE": "0",
            "E_TOY_OBJ": "0", "E_FENCE": "1", "E_PLAIN": "1"})


def main():
    tmo = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    print(f"\n=== E/scripts 全{len(CASES)}本の動作確認（各{tmo}秒まで）===\n")
    ok = ng = 0
    for name, args, want in CASES:
        path = os.path.join(_HERE, name)
        if not os.path.exists(path):
            print(f"  [skip] {name}（存在しない）"); continue
        t0 = time.time()
        try:
            r = subprocess.run([PY, path] + args, env=ENV, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=tmo)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {name:22s} {tmo}秒で終わらず"); ng += 1; continue
        dt = time.time() - t0
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            last = [l for l in out.strip().splitlines() if l.strip()][-1:] or ["(出力なし)"]
            print(f"  [FAIL] {name:22s} exit={r.returncode} {dt:5.0f}s")
            print(f"         → {last[0][:110]}")
            ng += 1
        elif want is not None and want not in out:
            print(f"  [WARN] {name:22s} 落ちないが期待する出力『{want}』が無い {dt:5.0f}s")
            ng += 1
        else:
            print(f"  [ok]   {name:22s} {dt:5.0f}s")
            ok += 1
    print(f"\n=== 結果: OK {ok} / 問題 {ng} ===")


if __name__ == "__main__":
    main()
