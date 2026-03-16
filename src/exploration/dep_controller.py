# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# DEP (Differential Extrinsic Plasticity) controller.
# Ported from DEP-RL: https://github.com/martius-lab/deprl
# Original: Der et al. (2015) - Self-organizing motor primitives

from collections import deque
from typing import Optional

import numpy as np
import torch

torch.set_default_dtype(torch.float32)


class DEP:
    """
    DEP Implementation from Der et al. (2015).
    PyTorch implementation for self-organizing exploration in overactuated
    musculoskeletal systems. Expects muscle_length + alpha * muscle_force as input.
    """

    def __init__(self, params: Optional[dict] = None):
        """
        Initialize DEP controller with parameters.

        Args:
            params: Dict of DEP parameters. If None, uses defaults from DEP-RL paper.
        """
        self.params = params or self._default_params()
        for k, v in self.params.items():
            setattr(self, k, v)

        self.num_sensors = None
        self.num_motors = None
        self.n_env = 1
        self.act_scale = None
        self.action_space = None
        self.obs_spec = None
        self.M = None
        self.C = None
        self.C_norm = None
        self.Cb = None
        self.obs_smoothed = None
        self.buffer = None
        self.t = 0

    def _default_params(self) -> dict:
        """Default parameters from deprl param_files/default_agents.json"""
        return {
            "kappa": 100.0,
            "tau": 20,
            "bias_rate": 0.00002,
            "time_dist": 5,
            "normalization": "independent",
            "s4avg": 2,
            "buffer_size": 200,
            "sensor_delay": 1,
            "regularization": 32,
            "with_learning": True,
            "q_norm_selector": "l2",
        }

    def initialize(self, action_dim: int, act_high: Optional[np.ndarray] = None):
        """
        Initialize DEP for given action space.

        Args:
            action_dim: Number of actuators (muscles). num_sensors = num_motors.
            act_high: High bound of action space for scaling. Defaults to ones.
        """
        self.num_sensors = action_dim
        self.num_motors = action_dim
        self.n_env = 1
        if act_high is not None:
            self.act_scale = torch.tensor(act_high, dtype=torch.float32)
        else:
            self.act_scale = torch.ones(action_dim, dtype=torch.float32)
        self.obs_spec = (action_dim,)
        self._reset((action_dim,))

    def _reset(self, obs_shape=None):
        """Reset controller buffers and matrices."""
        if obs_shape:
            self.n_env = obs_shape[0] if len(obs_shape) > 1 else 1
        # Identity model matrix (negative for DEP convention)
        self.M = torch.broadcast_to(
            -torch.eye(self.num_motors, self.num_sensors),
            (self.n_env, self.num_motors, self.num_sensors),
        )
        self.C = torch.zeros((self.n_env, self.num_motors, self.num_sensors))
        self.C_norm = torch.zeros((self.n_env, self.num_motors, self.num_sensors))
        self.Cb = torch.zeros((self.n_env, self.num_motors))
        self.obs_smoothed = torch.zeros((self.n_env, self.num_sensors))
        self.buffer = deque(maxlen=self.buffer_size)
        self.t = 0

    def step(self, observations: np.ndarray) -> np.ndarray:
        """
        Main step function. Takes muscle_states (muscle_length + alpha * muscle_force).

        Args:
            observations: muscle_states array, shape (num_sensors,) or (n_env, num_sensors)

        Returns:
            actions: DEP-generated actions, shape matching input
        """
        observations = torch.tensor(observations, dtype=torch.float32)
        if observations.shape != self.obs_spec and len(observations.shape) == 1:
            self.obs_spec = observations.shape
            self._reset(observations.shape)
        if len(observations.shape) == 1:
            observations = observations[None, :]
        y = self._get_action(observations)
        return y.squeeze(0).numpy() if y.shape[0] == 1 else y.numpy()

    def _q_norm(self, q: torch.Tensor) -> torch.Tensor:
        """Normalization for intermediate action q = C @ x."""
        reg = 10.0 ** (-self.regularization)
        if self.q_norm_selector == "l2":
            q_norm = 1.0 / (torch.linalg.norm(q, dim=-1, keepdim=True) + reg)
        elif self.q_norm_selector == "max":
            q_norm = 1.0 / (torch.abs(q).max(dim=-1, keepdim=True)[0] + reg)
        elif self.q_norm_selector == "none":
            q_norm = 1.0
        else:
            raise NotImplementedError(
                f"q normalization {self.q_norm_selector} not implemented."
            )
        return q_norm.squeeze(-1) if q_norm.dim() > 1 else q_norm

    def _compute_action(self) -> torch.Tensor:
        """Compute DEP action from current C matrix."""
        q = torch.einsum("ijk,ik->ij", self.C_norm, self.obs_smoothed)
        q_norm = self._q_norm(q)
        if q_norm.dim() == 0:
            q_norm = q_norm.unsqueeze(0)
        q = q * q_norm.unsqueeze(-1) if q_norm.dim() == 1 else q * q_norm
        y = torch.clamp(torch.tanh(q * self.kappa + self.Cb), -1.0, 1.0)
        y = y * self.act_scale.unsqueeze(0)
        return y

    def _learn_controller(self):
        """Update DEP by one learning step."""
        self.C = self._compute_C()
        R = torch.einsum("ijk,imk->ijm", self.C, self.M)
        reg = 10.0 ** (-self.regularization)
        if self.normalization == "independent":
            factor = self.kappa / (torch.linalg.norm(R, dim=-1) + reg)
            self.C_norm = self.C * factor.unsqueeze(-1)
        elif self.normalization == "none":
            self.C_norm = self.C
        elif self.normalization == "global":
            norm = torch.linalg.norm(R)
            self.C_norm = self.C * self.kappa / (norm + reg)
        else:
            raise NotImplementedError(
                f"Controller normalization {self.normalization} not implemented."
            )
        if self.bias_rate >= 0:
            yy = self.buffer[-2][1]
            self.Cb -= torch.clamp(yy * self.bias_rate, -0.05, 0.05) + self.Cb * 0.001
        else:
            self.Cb *= 0

    def _compute_C(self) -> torch.Tensor:
        """Recompute controller matrix C from buffer of recent transitions."""
        C = torch.zeros_like(self.C)
        for s in range(2, min(self.t - self.time_dist, self.tau)):
            x = self.buffer[-s][0][:, : self.num_sensors]
            xx = self.buffer[-s - 1][0][:, : self.num_sensors]
            xx_t = (
                x
                if self.time_dist == 0
                else self.buffer[-s - self.time_dist][0][:, : self.num_sensors]
            )
            xxx_t = self.buffer[-s - 1 - self.time_dist][0][:, : self.num_sensors]
            chi = x - xx
            v = xx_t - xxx_t
            mu = torch.einsum("ijk,ik->ij", self.M, chi)
            C += torch.einsum("ij,ik->ijk", mu, v)
        return C

    def _get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Smoothing and DEP learning step."""
        if self.s4avg > 1 and self.t > 0:
            self.obs_smoothed = self.obs_smoothed + (obs - self.obs_smoothed) / self.s4avg
        else:
            self.obs_smoothed = obs.clone()

        self.buffer.append([self.obs_smoothed.detach().clone(), None])
        if self.with_learning and len(self.buffer) > (2 + self.time_dist):
            self._learn_controller()
        y = self._compute_action()
        self.buffer[-1][1] = y.detach().clone()
        self.t += 1
        return y
