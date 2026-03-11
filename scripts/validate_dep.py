#!/usr/bin/env python3
"""
DEP Controller and Exploration Wrapper Validation.

Runs unit tests to verify:
1. DEP controller produces correct action shapes for different morphologies
2. DEP self-organization: controller matrix C evolves from zero
3. Stochastic and deterministic switching logic
4. Morphology scaling (80, 86, 290 muscles)
5. Muscle state computation
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.exploration.dep_controller import DEP
from src.exploration.dep_exploration import DEPExploration


def test_dep_action_shapes():
    """Test that DEP outputs correct action shapes for various morphologies."""
    print("=" * 60)
    print("TEST 1: Action shape correctness across morphologies")
    print("=" * 60)
    morphologies = {"MyoLeg (legs)": 80, "MyoLeg+Abs": 86, "MyoLeg+Back": 290}
    results = {}
    for name, n_muscles in morphologies.items():
        dep = DEP()
        dep.initialize(n_muscles)
        obs = np.random.randn(n_muscles).astype(np.float32) * 0.1
        action = dep.step(obs)
        shape_ok = action.shape == (n_muscles,)
        range_ok = np.all(np.abs(action) <= 1.0 + 1e-6)
        results[name] = {"n_muscles": n_muscles, "shape": action.shape,
                         "shape_ok": shape_ok, "range_ok": range_ok}
        status = "PASS" if (shape_ok and range_ok) else "FAIL"
        print(f"  [{status}] {name}: {n_muscles} muscles -> action shape {action.shape}, "
              f"range [{action.min():.4f}, {action.max():.4f}]")
    return results


def test_dep_self_organization():
    """Test that DEP's controller matrix C evolves from zero over time."""
    print("\n" + "=" * 60)
    print("TEST 2: Self-organization (C matrix evolution)")
    print("=" * 60)
    dep = DEP()
    dep.initialize(80)
    c_norms = []
    actions_std = []
    for t in range(100):
        obs = np.sin(np.linspace(0, 2 * np.pi, 80) + t * 0.1).astype(np.float32) * 0.3
        action = dep.step(obs)
        c_norm = float(torch.linalg.norm(dep.C).item())
        c_norms.append(c_norm)
        actions_std.append(float(np.std(action)))

    c_grew = c_norms[-1] > c_norms[10] > 0
    actions_diverse = actions_std[-1] > 0.01
    status = "PASS" if (c_grew and actions_diverse) else "FAIL"
    print(f"  [{status}] C norm: t=0 -> {c_norms[0]:.6f}, t=10 -> {c_norms[10]:.6f}, "
          f"t=99 -> {c_norms[-1]:.6f}")
    print(f"         Action std: t=0 -> {actions_std[0]:.6f}, t=99 -> {actions_std[-1]:.6f}")
    print(f"         C matrix grew: {c_grew}, Actions diverse: {actions_diverse}")
    return {"c_norms": c_norms, "actions_std": actions_std,
            "c_grew": c_grew, "actions_diverse": actions_diverse}


def test_stochastic_switching():
    """Test StochSwitchDep intervention logic."""
    print("\n" + "=" * 60)
    print("TEST 3: Stochastic switching (StochSwitchDep)")
    print("=" * 60)
    np.random.seed(42)
    expl = DEPExploration(action_dim=80, dep_mix=3,
                          dep_intervention_proba=0.1,
                          dep_intervention_length=4)
    n_steps = 1000
    dep_count = 0
    for _ in range(n_steps):
        if expl.should_use_dep(greedy=False):
            dep_count += 1
    dep_frac = dep_count / n_steps
    in_range = 0.01 < dep_frac < 0.5
    status = "PASS" if in_range else "FAIL"
    print(f"  [{status}] DEP used {dep_count}/{n_steps} steps ({dep_frac:.1%})")

    expl2 = DEPExploration(action_dim=80, dep_mix=3)
    greedy_used = any(expl2.should_use_dep(greedy=True) for _ in range(100))
    status2 = "PASS" if not greedy_used else "FAIL"
    print(f"  [{status2}] Greedy mode: DEP never used = {not greedy_used}")
    return {"dep_fraction": dep_frac, "in_range": in_range,
            "greedy_blocked": not greedy_used}


def test_deterministic_switching():
    """Test DetSwitchDep intervention logic."""
    print("\n" + "=" * 60)
    print("TEST 4: Deterministic switching (DetSwitchDep)")
    print("=" * 60)
    expl = DEPExploration(action_dim=80, dep_mix=2,
                          dep_intervention_length=4, dep_rl_length=2)
    pattern = []
    for _ in range(20):
        pattern.append(expl.should_use_dep(greedy=False))
    has_both = any(pattern) and not all(pattern)
    status = "PASS" if has_both else "FAIL"
    print(f"  [{status}] Pattern (first 20): {''.join(['D' if x else 'R' for x in pattern])}")
    print(f"         Has both DEP and RL phases: {has_both}")
    return {"pattern": pattern, "has_both": has_both}


def test_muscle_state_computation():
    """Test muscle_states = muscle_len + alpha * muscle_force."""
    print("\n" + "=" * 60)
    print("TEST 5: Muscle state computation")
    print("=" * 60)
    expl = DEPExploration(action_dim=80, dep_alpha=0.01)
    muscle_len = np.random.rand(80).astype(np.float32)
    muscle_force = np.random.rand(80).astype(np.float32) * 100
    states = expl.get_muscle_states(muscle_len, muscle_force)
    expected = muscle_len + 0.01 * muscle_force
    match = np.allclose(states, expected)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] muscle_states = len + 0.01*force, shape={states.shape}, match={match}")
    return {"shape": states.shape, "match": match}


def test_dep_reset():
    """Test that DEP resets properly between episodes."""
    print("\n" + "=" * 60)
    print("TEST 6: Episode reset")
    print("=" * 60)
    expl = DEPExploration(action_dim=80, dep_mix=3)
    muscle_states = np.random.randn(80).astype(np.float32) * 0.1
    for _ in range(50):
        expl.get_dep_action(muscle_states)
    c_before = float(torch.linalg.norm(expl.dep.C).item())
    expl.reset()
    c_after = float(torch.linalg.norm(expl.dep.C).item())
    reset_ok = c_after == 0.0 and expl.dep.t == 0
    status = "PASS" if reset_ok else "FAIL"
    print(f"  [{status}] C norm before reset: {c_before:.6f}, after: {c_after:.6f}")
    print(f"         Time counter reset: {expl.dep.t == 0}")
    return {"c_before": c_before, "c_after": c_after, "reset_ok": reset_ok}


def test_morphology_scaling():
    """Test DEP exploration end-to-end for each morphology size."""
    print("\n" + "=" * 60)
    print("TEST 7: Morphology scaling (end-to-end)")
    print("=" * 60)
    morphologies = {"legs (80)": 80, "legs_abs (86)": 86, "legs_back (290)": 290}
    results = {}
    for name, n in morphologies.items():
        expl = DEPExploration(action_dim=n, dep_mix=3,
                              dep_intervention_proba=0.1,
                              dep_intervention_length=4, dep_alpha=0.01)
        muscle_len = np.random.rand(n).astype(np.float32) * 0.5
        muscle_force = np.random.rand(n).astype(np.float32) * 50
        states = expl.get_muscle_states(muscle_len, muscle_force)
        action = expl.get_dep_action(states)
        ok = action.shape == (n,) and states.shape == (n,)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: states {states.shape}, action {action.shape}")
        results[name] = {"n": n, "ok": ok}
    return results


def main():
    print("DEP Controller & Exploration Wrapper Validation")
    print("=" * 60)

    all_results = {}
    all_results["action_shapes"] = test_dep_action_shapes()
    all_results["self_organization"] = test_dep_self_organization()
    all_results["stoch_switching"] = test_stochastic_switching()
    all_results["det_switching"] = test_deterministic_switching()
    all_results["muscle_states"] = test_muscle_state_computation()
    all_results["reset"] = test_dep_reset()
    all_results["morphology_scaling"] = test_morphology_scaling()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    total = 7
    checks = [
        ("Action shapes", all(v["shape_ok"] and v["range_ok"]
                              for v in all_results["action_shapes"].values())),
        ("Self-organization", all_results["self_organization"]["c_grew"]),
        ("Stochastic switching", all_results["stoch_switching"]["in_range"]),
        ("Deterministic switching", all_results["det_switching"]["has_both"]),
        ("Muscle states", all_results["muscle_states"]["match"]),
        ("Episode reset", all_results["reset"]["reset_ok"]),
        ("Morphology scaling", all(v["ok"]
                                   for v in all_results["morphology_scaling"].values())),
    ]
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {name}")
    print(f"\n  {passed}/{total} tests passed")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    summary = {
        "tests_passed": passed, "tests_total": total,
        "self_org_c_norms": all_results["self_organization"]["c_norms"],
        "self_org_action_std": all_results["self_organization"]["actions_std"],
        "stoch_dep_fraction": all_results["stoch_switching"]["dep_fraction"],
    }
    with open(out_dir / "dep_validation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir / 'dep_validation.json'}")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
