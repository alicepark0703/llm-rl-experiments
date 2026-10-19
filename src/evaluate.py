import numpy as np

from src.environment import MathEnvironment
from src.baselines import (
    always_direct,
    always_calculator,
    random_policy,
    oracle_policy,
)

def evaluate_policy(policy, episodes=10_000, tool_cost=0.15, seed=42):
    """Evaluate a policy with signature: policy(state, env, rng)
    Each policy receives the same sequence of problems"""