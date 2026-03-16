#!/usr/bin/env python3
"""
Extended MuJoCo experiment across multiple morphologies.
Compares Random vs DEP exploration on real MyoLeg physics.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_single(xml_path, n_episodes, episode_length, seed, use_dep):
    import mujoco
    from src.exploration.dep_exploration import DEPExploration

    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    nu = model.nu

    dep_expl = None
    if use_dep:
        dep_expl = DEPExploration(
            action_dim=nu, dep_mix=3,
            dep_intervention_proba=1.0,
            dep_intervention_length=9999,
            dep_alpha=0.01,
        )

    ep_metrics = []
    for ep in range(n_episodes):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        if dep_expl is not None:
            dep_expl.reset()

        initial_root_pos = data.xpos[1].copy()
        root_heights = []
        activations = []
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
            actions_history.append(action.copy())

        final_root_pos = data.xpos[1].copy()
        displacement = np.linalg.norm(final_root_pos[:2] - initial_root_pos[:2])

        activations = np.array(activations)
        act_volume = np.mean(np.sum(activations**2, axis=1))

        actions_arr = np.array(actions_history)
        corr_matrix = np.corrcoef(actions_arr.T)
        off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        action_coordination = float(np.nanmean(np.abs(off_diag)))

        action_std = float(np.mean(np.std(actions_arr, axis=0)))

        root_heights = np.array(root_heights)
        upright_frac = float(np.mean(root_heights > 0.5))
        mean_height = float(np.mean(root_heights))

        ep_metrics.append({
            "displacement": float(displacement),
            "activation_volume": float(act_volume),
            "action_coordination": action_coordination,
            "action_diversity": action_std,
            "upright_fraction": upright_frac,
            "mean_root_height": mean_height,
        })

    summary = {}
    for k in ep_metrics[0]:
        vals = [m[k] for m in ep_metrics]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary, nu


def main():
    os.chdir(str(Path(__file__).resolve().parents[1]))

    morphologies = [
        ("MyoLeg (80)", "data/xml/legs/myolegs.xml"),
        ("MyoLeg+Abs (86)", "data/xml/legs_abs/myolegs_abdomen.xml"),
        ("MyoLeg+Back (290)", "data/xml/legs_back/leg_soccer/myolegs_soccer.xml"),
    ]

    n_episodes = 15
    episode_length = 300
    all_results = {}

    for morph_name, xml_path in morphologies:
        if not os.path.exists(xml_path):
            print(f"SKIP {morph_name}: {xml_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"{morph_name}")
        print(f"{'='*60}")

        for method, use_dep in [("Random", False), ("DEP", True)]:
            print(f"  Running {method}...", end=" ", flush=True)
            summary, nu = run_single(xml_path, n_episodes, episode_length, seed=42, use_dep=use_dep)
            key = f"{morph_name}_{method}"
            all_results[key] = {"summary": summary, "n_muscles": nu, "method": method, "morphology": morph_name}
            print(f"nu={nu}, disp={summary['displacement']['mean']:.4f}, "
                  f"coord={summary['action_coordination']['mean']:.4f}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Morphology':<20} {'Method':<8} {'Muscles':>7} {'Displ':>8} {'ActVol':>8} {'Coord':>8} {'ActDiv':>8} {'Height':>8}")
    print("-" * 80)
    for key, r in all_results.items():
        s = r["summary"]
        print(f"{r['morphology']:<20} {r['method']:<8} {r['n_muscles']:>7d} "
              f"{s['displacement']['mean']:>8.4f} "
              f"{s['activation_volume']['mean']:>8.2f} "
              f"{s['action_coordination']['mean']:>8.4f} "
              f"{s['action_diversity']['mean']:>8.4f} "
              f"{s['mean_root_height']['mean']:>8.3f}")

    out_path = Path("reports/mujoco_full_experiment.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
