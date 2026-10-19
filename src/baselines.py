import numpy as np

def always_direct(state):
    """Always answer directly"""
    return 0

def always_calculator(state):
    """Always use the calculator"""
    return 1

def random_policy(state, rng): 
    """Choose either action with equal probability"""
    return int(rng.integers(0, 2))

def oracle_policy(env):
    """Choose the action with the highest expected reward.
    
    Uses the simulator's true accuracy, so this is an evaluation benchmark,
    not a learning policy.
    Call env.reset() before using it."""

    direct_reward = env.direct_answer_accuracy()
    calculator_reward = 1.0 - env.tool_cost

    #choose direct answering when expected rewards are equal
    return int(calculator_reward > direct_reward)