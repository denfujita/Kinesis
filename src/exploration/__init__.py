# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# DEP exploration module for musculoskeletal motion imitation.
#
# Ported from DEP-RL: https://github.com/martius-lab/deprl

from src.exploration.dep_controller import DEP
from src.exploration.dep_exploration import DEPExploration

__all__ = ["DEP", "DEPExploration"]
