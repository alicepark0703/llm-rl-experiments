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

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    #keep policy randomness separate from environment randomness
    rng = np.random.default_rng([seed, 1])

    total_reward = 0.0
    total_correct = 0
    total_tool_calls = 0

    for episode in range(episodes):
        #fresh environment seed per episode to ensure that actions
        #cannot change which problems subsequent episodes receive.
        env = MathEnvironment(
            tool_cost=tool_cost,
            seed = seed + episode,
        )

        state = env.reset()
        action = policy(state, env, rng)

        _, reward, _, info = env.step(action)

        total_reward += reward
        total_correct += int(info["correct"])
        total_tool_calls += int(info["used_tool"])

    return{
        "average_reward": total_reward / episodes,
        "accuracy": total_correct / episodes,
        "tool_usage": total_tool_calls / episodes,
    }

if __name__ == "__main__":
    # Adapt the existing functions to a common interface.
    policies = {
        "Always direct": lambda state, env, rng: always_direct(state),
        "Always calculator": lambda state, env, rng: always_calculator(state),
        "Random": lambda state, env, rng: random_policy(state, rng),
        "Oracle": lambda state, env, rng: oracle_policy(env),
    }

    for name, policy in policies.items():
        metrics = evaluate_policy(policy)

        print(f"\n{name}")
        print(f"  Average reward: {metrics['average_reward']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']:.2%}")
        print(f"  Tool usage:     {metrics['tool_usage']:.2%}")