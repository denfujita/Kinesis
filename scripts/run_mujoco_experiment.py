#!/usr/bin/env python3
"""
MuJoCo-level experiment: DEP vs Random exploration on real MyoLeg physics.

Runs the actual MuJoCo MyoLeg model (80 muscles) and compares:
  1. Random exploration (isotropic Gaussian - simulates PPO without policy)
  2. DEP exploration (structured sensorimotor)

Metrics collected per episode:
  - Distance traveled (root displacement)
  - Muscle activation efficiency (mean a^2)
  - State space coverage (unique body configurations visited)
  - Action coordination (cross-correlation of muscle activations)
  - Upright maintenance (root height over time)

No SMPL or motion data needed -- uses raw MuJoCo physics only.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_experiment(
    xml_path="data/xml/legs/myolegs.xml",
    n_episodes=20,
    episode_length=200,
    seed=42,
):
    import mujoco
    from src.exploration.dep_controller import DEP
    from src.exploration.dep_exploration import DEPExploration

    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    nu = model.nu

    methods = {
        "Random": None,
        "DEP": DEPExploration(
            action_dim=nu, dep_mix=3,
            dep_intervention_proba=1.0,  # always DEP for comparison
            dep_intervention_length=9999,
            dep_alpha=0.01,
        ),
    }

    all_results = {}

    for method_name, dep_expl in methods.items():
        print(f"\n--- {method_name} Exploration ({n_episodes} episodes x {episode_length} steps) ---")
        ep_metrics = []

        for ep in range(n_episodes):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

            if dep_expl is not None:
                dep_expl.reset()

            initial_root_pos = data.xpos[1].copy()
            root_heights = []
            activations = []
            body_positions = []
            actions_history = []

            for step in range(episode_length):
                muscle_len = np.nan_to_num(data.actuator_length.copy()).astype(np.float32)
                muscle_force = np.nan_to_num(data.actuator_force.copy()).astype(np.float32)

                if dep_expl is not None:
                    muscle_states = dep_expl.get_muscle_states(muscle_len, muscle_force)
                    action = dep_expl.get_dep_action(muscle_states)
                else:
                    action = rng.uniform(-1, 1, size=nu).astype(np.float32)

                ctrl = np.clip((action + 1.0) / 2.0, 0, 1)
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)

                root_heights.append(data.xpos[1][2])
                activations.append(ctrl.copy())
                body_positions.append(data.xpos[1:].flatten().copy())
                actions_history.append(action.copy())

            final_root_pos = data.xpos[1].copy()
            displacement = np.linalg.norm(final_root_pos[:2] - initial_root_pos[:2])

            activations = np.array(activations)
            act_volume = np.mean(np.sum(activations**2, axis=1))

            actions_arr = np.array(actions_history)
            if actions_arr.shape[0] > 10:
                corr_matrix = np.corrcoef(actions_arr.T)
                off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                action_coordination = float(np.mean(np.abs(off_diag)))
            else:
                action_coordination = 0.0

            body_pos_arr = np.array(body_positions)
            state_variance = float(np.mean(np.var(body_pos_arr, axis=0)))

            root_heights = np.array(root_heights)
            upright_frac = float(np.mean(root_heights > 0.5))
            mean_height = float(np.mean(root_heights))

            ep_metrics.append({
                "displacement": float(displacement),
                "activation_volume": float(act_volume),
                "action_coordination": action_coordination,
                "state_coverage": float(state_variance),
                "upright_fraction": upright_frac,
                "mean_root_height": mean_height,
                "final_height": float(root_heights[-1]),
            })

            if ep % 5 == 0:
                print(f"  Ep {ep:3d}: disp={displacement:.4f}m, "
                      f"act_vol={act_volume:.2f}, "
                      f"coord={action_coordination:.4f}, "
                      f"height={mean_height:.3f}m")

        metrics_arr = {k: [m[k] for m in ep_metrics] for k in ep_metrics[0]}
        summary = {}
        for k, vals in metrics_arr.items():
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        all_results[method_name] = {"episodes": ep_metrics, "summary": summary}

        print(f"\n  {method_name} Summary:")
        for k, v in summary.items():
            print(f"    {k}: {v['mean']:.4f} +/- {v['std']:.4f}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Random vs DEP")
    print("=" * 60)
    comparison = {}
    for metric in all_results["Random"]["summary"]:
        r = all_results["Random"]["summary"][metric]["mean"]
        d = all_results["DEP"]["summary"][metric]["mean"]
        if r != 0:
            change = (d - r) / abs(r) * 100
        else:
            change = 0
        comparison[metric] = {"random": r, "dep": d, "change_pct": change}
        print(f"  {metric:25s}: Random={r:.4f}, DEP={d:.4f} ({change:+.1f}%)")

    all_results["comparison"] = comparison
    all_results["config"] = {
        "xml_path": xml_path,
        "n_episodes": n_episodes,
        "episode_length": episode_length,
        "n_muscles": nu,
        "seed": seed,
    }

    out_path = Path("reports/mujoco_experiment.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_experiment()
