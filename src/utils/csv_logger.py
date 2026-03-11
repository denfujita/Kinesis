import csv
import os
from pathlib import Path


class CSVTrainingLogger:
    """Logs per-epoch training metrics to a CSV file."""

    FIELDS = [
        "epoch", "avg_reward", "avg_episode_reward", "avg_episode_len",
        "min_reward", "max_reward",
        "mpjpe", "joint_angle_rmse", "activation_volume", "frame_coverage",
        "pos_reward", "vel_reward", "upright_reward", "energy_reward",
        "t_sample", "t_update",
    ]

    def __init__(self, output_dir: str, exp_name: str):
        self.log_dir = Path(output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{exp_name}_training_log.csv"
        self._write_header()

    def _write_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def log_epoch(self, epoch, loggers, info):
        row = {
            "epoch": epoch,
            "avg_reward": getattr(loggers, "avg_reward", 0),
            "avg_episode_reward": loggers.avg_episode_reward,
            "avg_episode_len": loggers.avg_episode_len,
            "min_reward": loggers.min_reward,
            "max_reward": loggers.max_reward,
            "t_sample": info.get("T_sample", 0),
            "t_update": info.get("T_update", 0),
        }
        info_dict = loggers.info_dict
        for key in ["mpjpe", "joint_angle_rmse", "activation_volume", "frame_coverage"]:
            if key in info_dict and len(info_dict[key]) > 0:
                import numpy as np
                row[key] = float(np.mean(info_dict[key]))
            else:
                row[key] = ""
        for key in ["pos_reward", "vel_reward", "upright_reward", "energy_reward"]:
            if key in info_dict and len(info_dict[key]) > 0:
                import numpy as np
                row[key] = float(np.mean(info_dict[key]))
            else:
                row[key] = ""

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(row)
