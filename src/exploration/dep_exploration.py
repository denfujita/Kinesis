# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# DEP exploration wrapper for Kinesis PPO agent.
# Implements StochSwitchDep and DetSwitchDep patterns from DEP-RL.

import numpy as np
from typing import Optional

from src.exploration.dep_controller import DEP


class DEPExploration:
    """
    Wrapper for DEP exploration compatible with Kinesis agent rollout loop.
    Supports deterministic (DetSwitchDep) and stochastic (StochSwitchDep) switching.
    """

    def __init__(
        self,
        action_dim: int,
        act_high: Optional[np.ndarray] = None,
        dep_params: Optional[dict] = None,
        dep_mix: int = 3,
        dep_intervention_proba: float = 0.1,
        dep_intervention_length: int = 4,
        dep_rl_length: int = 1,
        dep_alpha: float = 0.01,
    ):
        """
        Args:
            action_dim: Number of muscle actuators.
            act_high: Action space high bound for DEP scaling.
            dep_params: DEP controller parameters (kappa, tau, etc.).
            dep_mix: 0=no DEP, 2=DetSwitch, 3=StochSwitch (paper default).
            dep_intervention_proba: Probability of using DEP (StochSwitch).
            dep_intervention_length: Steps to use DEP when switched (DetSwitch).
            dep_rl_length: Steps to use RL when switched (DetSwitch).
            dep_alpha: Weight for muscle_force in muscle_states = len + alpha * force.
        """
        self.action_dim = action_dim
        self.dep_mix = dep_mix
        self.dep_intervention_proba = dep_intervention_proba
        self.dep_intervention_length = dep_intervention_length
        self.dep_rl_length = dep_rl_length
        self.dep_alpha = dep_alpha

        self.dep = None
        if dep_mix > 0:
            self.dep = DEP(params=dep_params)
            self.dep.initialize(action_dim, act_high)

        self.since_switch = 500 if dep_mix == 3 else 1  # Policy starts first (StochSwitch)
        self.switch = 0  # 0=RL, 1=DEP
        self.step_count = 0

    def reset(self):
        """Reset DEP and switching state (call at episode start)."""
        if self.dep is not None:
            self.dep._reset((self.action_dim,))
        self.since_switch = 500 if self.dep_mix == 3 else 1
        self.switch = 0

    def get_muscle_states(
        self,
        muscle_len: np.ndarray,
        muscle_force: np.ndarray,
    ) -> np.ndarray:
        """
        Compute muscle_states for DEP: muscle_length + alpha * muscle_force.
        Normalize to similar scale (DEP expects sensor-motor coupling).
        """
        # Normalize: use length and force as-is; alpha scales force contribution
        return muscle_len.astype(np.float32) + self.dep_alpha * muscle_force.astype(
            np.float32
        )

    def should_use_dep(self, greedy: bool = False) -> bool:
        """Determine whether to use DEP or policy this step."""
        if self.dep is None or self.dep_mix == 0:
            return False
        if greedy:
            return False

        if self.dep_mix == 2:  # DetSwitchDep
            if self.switch == 1 and self.since_switch % self.dep_intervention_length == 0:
                self.switch = 0
                self.since_switch = 1
            if self.switch == 0 and self.since_switch % self.dep_rl_length == 0:
                self.switch = 1
                self.since_switch = 1
            self.since_switch += 1
            return self.switch == 1

        if self.dep_mix == 3:  # StochSwitchDep
            if self.since_switch > self.dep_intervention_length:
                if np.random.uniform() < self.dep_intervention_proba:
                    self.since_switch = 0
            self.since_switch += 1
            return self.since_switch <= self.dep_intervention_length

        return False

    def get_dep_action(self, muscle_states: np.ndarray) -> np.ndarray:
        """Get action from DEP controller."""
        if self.dep is None:
            raise RuntimeError("DEP not initialized")
        return self.dep.step(muscle_states)
