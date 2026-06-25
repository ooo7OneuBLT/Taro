"""
自動ログ・メトリクス・学習曲線

毎ターンのデータをJSONLに記録し、
定期的に学習曲線（PNG）を生成する。
"""

import os
import json
import csv
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger:

    def __init__(self, log_dir="logs", run_name=None):
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.log_dir, "turns.jsonl")
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "turn", "sim_seconds", "parent", "taro",
                    "r_imit", "r_pred", "r_social", "R", "delta",
                    "p_loss", "a_loss", "temperature",
                ])

    def save_run_info(self, description, config, phrases=None):
        """実験の説明・設定・経緯をメタデータとして保存する。"""
        info = {
            "run_name": self.run_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "config": config,
        }
        if phrases is not None:
            info["phrases"] = phrases
        path = os.path.join(self.log_dir, "run_info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        self.history = {
            "turns": [], "r_imit": [], "r_pred": [], "R": [],
            "delta": [], "p_loss": [], "temperature": [],
        }

    def log_turn(self, turn, sim_seconds, parent_text, taro_text,
                 r_imit, r_pred, r_social, R, delta,
                 p_loss, a_loss, temperature):
        """1ターン分のデータを記録する。"""
        record = {
            "turn": turn,
            "sim_seconds": sim_seconds,
            "timestamp": datetime.now().isoformat(),
            "parent": parent_text,
            "taro": taro_text,
            "r_imit": round(r_imit, 4),
            "r_pred": round(r_pred, 4),
            "r_social": round(r_social, 4),
            "R": round(R, 4),
            "delta": round(delta, 4),
            "p_loss": round(p_loss, 4),
            "a_loss": round(a_loss, 4),
            "temperature": round(temperature, 4),
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                turn, sim_seconds, parent_text, taro_text,
                record["r_imit"], record["r_pred"], record["r_social"],
                record["R"], record["delta"],
                record["p_loss"], record["a_loss"], record["temperature"],
            ])

        self.history["turns"].append(turn)
        self.history["r_imit"].append(r_imit)
        self.history["r_pred"].append(r_pred)
        self.history["R"].append(R)
        self.history["delta"].append(delta)
        self.history["p_loss"].append(p_loss)
        self.history["temperature"].append(temperature)

    def plot_learning_curve(self):
        """学習曲線をPNGで出力する。"""
        if len(self.history["turns"]) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Taro A1 Learning Curve", fontsize=14)
        turns = self.history["turns"]

        axes[0, 0].plot(turns, self.history["r_imit"], label="r_imit", alpha=0.7)
        axes[0, 0].plot(turns, self.history["r_pred"], label="r_pred", alpha=0.7)
        axes[0, 0].plot(turns, self.history["R"], label="R (total)", linewidth=2)
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Rewards")
        axes[0, 0].legend()
        axes[0, 0].set_ylim(-0.05, 1.05)

        axes[0, 1].plot(turns, self.history["delta"], alpha=0.7, color="purple")
        axes[0, 1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        axes[0, 1].set_ylabel("delta")
        axes[0, 1].set_title("Dopamine (RPE)")

        axes[1, 0].plot(turns, self.history["p_loss"], alpha=0.7, color="red")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_xlabel("Turn")
        axes[1, 0].set_title("Perception Loss")

        axes[1, 1].plot(turns, self.history["temperature"], alpha=0.7, color="orange")
        axes[1, 1].set_ylabel("tau")
        axes[1, 1].set_xlabel("Turn")
        axes[1, 1].set_title("Temperature (babbling -> stable)")

        plt.tight_layout()
        path = os.path.join(self.log_dir, "learning_curve.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path
