#!/usr/bin/env python3
"""
Pipeline integration tests for DEP-enhanced Kinesis.

Tests run in stages -- later stages require data that may not be present.
Stages:
  A: Module imports
  B: MuJoCo model loading (XML only, no SMPL/motion data)
  C: MuJoCo step with random actions
  D: DEP integration with real MuJoCo env
  E: Full environment (requires SMPL + motion data)
  F: Single PPO epoch (requires E)
"""

import os
import sys
import traceback
import json
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

results = {}


def run_test(name, fn):
    """Run a test function and record result."""
    try:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        results[name] = {"status": status, "detail": detail}
        print(f"  [{status}] {name}: {detail}")
        return ok
    except Exception as e:
        results[name] = {"status": "ERROR", "detail": str(e)}
        print(f"  [ERROR] {name}: {e}")
        traceback.print_exc()
        return False


# ============================================================
# Test A: Module imports
# ============================================================
def test_imports():
    import torch
    import mujoco
    import gymnasium
    from omegaconf import OmegaConf
    from src.exploration.dep_controller import DEP
    from src.exploration.dep_exploration import DEPExploration
    from src.env.myolegs_base_env import BaseEnv
    return True, f"torch={torch.__version__}, mujoco={mujoco.__version__}"


# ============================================================
# Test B: MuJoCo model loading
# ============================================================
def test_mujoco_load():
    import mujoco
    xml_path = "data/xml/legs/myolegs.xml"
    if not os.path.exists(xml_path):
        return False, f"{xml_path} not found"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return True, f"bodies={model.nbody}, actuators={model.nu}, qpos={model.nq}"


# ============================================================
# Test C: MuJoCo step with random actions
# ============================================================
def test_mujoco_step():
    import mujoco
    xml_path = "data/xml/legs/myolegs.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    nu = model.nu
    for step in range(10):
        action = np.random.uniform(0, 1, size=nu).astype(np.float32)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

    pos = data.xpos[1].copy()
    muscle_len = data.actuator_length.copy()
    muscle_force = data.actuator_force.copy()
    return True, f"10 steps OK, root_pos={pos}, muscle_len_mean={muscle_len.mean():.4f}"


# ============================================================
# Test D: DEP with real MuJoCo muscle states
# ============================================================
def test_dep_with_mujoco():
    import mujoco
    from src.exploration.dep_controller import DEP
    from src.exploration.dep_exploration import DEPExploration

    xml_path = "data/xml/legs/myolegs.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    nu = model.nu
    expl = DEPExploration(
        action_dim=nu, dep_mix=3,
        dep_intervention_proba=0.1,
        dep_intervention_length=4, dep_alpha=0.01
    )

    dep_steps = 0
    policy_steps = 0
    for step in range(50):
        muscle_len = np.nan_to_num(data.actuator_length.copy()).astype(np.float32)
        muscle_force = np.nan_to_num(data.actuator_force.copy()).astype(np.float32)
        muscle_states = expl.get_muscle_states(muscle_len, muscle_force)

        if expl.should_use_dep(greedy=False):
            action = expl.get_dep_action(muscle_states)
            dep_steps += 1
        else:
            action = np.random.uniform(-1, 1, size=nu).astype(np.float32)
            policy_steps += 1

        ctrl = (action + 1.0) / 2.0
        data.ctrl[:] = np.clip(ctrl, 0, 1)
        mujoco.mj_step(model, data)

    return True, f"50 steps: {dep_steps} DEP, {policy_steps} policy, action_shape=({nu},)"


# ============================================================
# Test E: Full MyoLegsIm environment
# ============================================================
def test_full_env():
    smpl_path = "data/smpl/SMPL_NEUTRAL.pkl"
    motion_path = "data/kit_train_motion_dict.pkl"

    if not os.path.exists(smpl_path):
        return False, f"SMPL model not found at {smpl_path} -- download from smpl.is.tue.mpg.de"
    if not os.path.exists(motion_path):
        return False, f"Motion data not found at {motion_path} -- run convert_kit.py on AMASS KIT-ML data"

    from omegaconf import OmegaConf
    from src.env.myolegs_im import MyoLegsIm

    cfg = OmegaConf.load("cfg/config_legs.yaml")
    run_cfg = OmegaConf.load("cfg/run/train_run_legs_dep.yaml")
    env_cfg = OmegaConf.load("cfg/env/env_im.yaml")
    learn_cfg = OmegaConf.load("cfg/learning/im_mlp.yaml")
    cfg = OmegaConf.merge(cfg, {"run": run_cfg, "env": env_cfg, "learning": learn_cfg})
    cfg.run.headless = True
    cfg.run.num_threads = 1
    cfg.run.num_motions = 1
    cfg.run.random_start = False
    cfg.no_log = True

    env = MyoLegsIm(cfg)
    env.sample_motions()
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    return True, f"obs={obs.shape}, reward={reward:.4f}, nu={env.mj_model.nu}"


# ============================================================
# Test F: Single PPO epoch
# ============================================================
def test_single_epoch():
    smpl_path = "data/smpl/SMPL_NEUTRAL.pkl"
    motion_path = "data/kit_train_motion_dict.pkl"

    if not os.path.exists(smpl_path) or not os.path.exists(motion_path):
        return False, "Requires SMPL + motion data (see Test E)"

    import torch
    from omegaconf import OmegaConf
    from src.agents import agent_dict

    cfg = OmegaConf.load("cfg/config_legs.yaml")
    run_cfg = OmegaConf.load("cfg/run/train_run_legs_dep.yaml")
    env_cfg = OmegaConf.load("cfg/env/env_im.yaml")
    learn_cfg = OmegaConf.load("cfg/learning/im_mlp.yaml")
    cfg = OmegaConf.merge(cfg, {"run": run_cfg, "env": env_cfg, "learning": learn_cfg})
    cfg.run.headless = True
    cfg.run.num_threads = 1
    cfg.run.num_motions = 1
    cfg.no_log = True
    cfg.learning.max_epoch = 1
    cfg.learning.min_batch_size = 128
    cfg.learning.actor_type = "gauss"
    cfg.learning.mlp = {"units": [256, 128], "activation": "silu", "initializer": {"name": "default"}, "regularizer": {"name": "None"}}
    cfg.learning.use_dep_exploration = True
    cfg.output_dir = "data/trained_models/test_pipeline"

    os.makedirs(cfg.output_dir, exist_ok=True)

    dtype = torch.float32
    device = torch.device("cpu")

    agent = agent_dict[cfg.learning.agent_name](
        cfg, dtype, device, training=True, checkpoint_epoch=0
    )

    batch, loggers = agent.sample(cfg.learning.min_batch_size)
    agent.update_params(batch)

    return True, (
        f"1 epoch OK: {loggers.num_steps} steps, "
        f"avg_reward={loggers.avg_reward:.4f}, "
        f"avg_ep_len={loggers.avg_episode_len:.1f}"
    )


# ============================================================
# Main
# ============================================================
def main():
    os.chdir(str(Path(__file__).resolve().parents[1]))

    print("=" * 60)
    print("Kinesis Pipeline Integration Tests")
    print("=" * 60)

    print("\n--- Stage A: Module Imports ---")
    a_ok = run_test("imports", test_imports)

    print("\n--- Stage B: MuJoCo Model Loading ---")
    b_ok = run_test("mujoco_load", test_mujoco_load) if a_ok else False

    print("\n--- Stage C: MuJoCo Step ---")
    c_ok = run_test("mujoco_step", test_mujoco_step) if b_ok else False

    print("\n--- Stage D: DEP + MuJoCo Integration ---")
    d_ok = run_test("dep_mujoco", test_dep_with_mujoco) if c_ok else False

    print("\n--- Stage E: Full MyoLegsIm Environment ---")
    e_ok = run_test("full_env", test_full_env)

    print("\n--- Stage F: Single PPO Epoch ---")
    f_ok = run_test("single_epoch", test_single_epoch) if e_ok else False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v["status"] == "PASS")
    failed = sum(1 for v in results.values() if v["status"] == "FAIL")
    errors = sum(1 for v in results.values() if v["status"] == "ERROR")
    total = len(results)

    for name, r in results.items():
        print(f"  [{r['status']:5s}] {name}")
    print(f"\n  {passed} passed, {failed} failed, {errors} errors out of {total} tests")

    if failed > 0 or errors > 0:
        print("\nTo fix failing tests:")
        for name, r in results.items():
            if r["status"] != "PASS":
                print(f"  {name}: {r['detail']}")

    out_path = Path("reports/pipeline_tests.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return passed, total


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed >= 4 else 1)
