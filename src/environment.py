import numpy as np

class MathEnvironment:
    """
    Contextual bandit environment for learning tool-use decisions.
    
    Actions:
        0: Answer directly (simulated model)
        1: Use calculator (perfect but costly)
    
    Reward:
        1.0 if correct, otherwise 0.0
        Minus tool cost if calculator is used.
    """
    def __init__(
            self, 
            tool_cost = 0.15, 
            base_accuracy = 0.95,
            difficulty_penalty = 0.75,
            seed = None):

        self.tool_cost = tool_cost
        self.base_accuracy = base_accuracy
        self.difficulty_penalty = difficulty_penalty
        
        self.rng = np.random.default_rng(seed)

        self.a = None
        self.b = None
        self.answer = None

    def reset(self):
        """Generate a new multiplication problem."""

        self.a = int(self.rng.integers(1, 101))
        self.b = int(self.rng.integers(1, 101))
        self.answer = self.a * self.b

        #normalize state for the policy network
        state = np.array(
            [self.a / 100.0, self.b / 100.0],
            dtype=np.float32
        )

        return state

    def direct_answer_accuracy(self):
        """
        Simulate model accuracy as a function of difficulty.
        """
        assert self.answer is not None
        difficulty = self.answer / 10000.0
        accuracy = self.base_accuracy - self.difficulty_penalty * difficulty

        return float(np.clip(accuracy, 0.0, 1.0))


    def generate_wrong_answer(self):
        """Generate a simulated incorrect answer to the current multiplication problem.
        
        Randomly selects one of three arithmetic error patterns:
        - small_error: Small numerical deviations from the correct answer.
        - place_value: Errors involving tens or hundreds.
        - rounding: Rounding the answer to the nearest hundred.

        The generated answer is guaranteed to be positive and incorrect.

        Returns:
            int: A simulated incorrect prediction.
        """
        if self.answer is None:
            raise RuntimeError("Call reset() before generating an answer")

        error_type = self.rng.choice(
            ["small_error", "place_value", "rounding"]
        )

        if error_type == "small_error":
            offset = int(
                self.rng.choice([-5, -3, -2, 2, 3, 5])
            )
            prediction = self.answer + offset

        elif error_type == "place_value":
            offset = int(
                self.rng.choice([-100, -10, 10, 100])
            )
            prediction = self.answer + offset

        else:
            prediction = round(self.answer / 100) * 100


        # Ensure the prediction is positive and incorrect.
        if prediction <= 0 or prediction == self.answer:
            prediction = self.answer + 2

        return int(prediction)

    def step(self, action):
        """
        Execute the agent's selected action & return solution

        Returns:
        tuple: (next_state, reward, done, info)

        """
        correct: bool | None = None

        if self.answer is None:
            raise RuntimeError("Call reset() before step().")

        if action == 0:
            # simulated direct answering
            accuracy = self.direct_answer_accuracy()
            simulated_success = self.rng.random() < accuracy

            #generate prediction based on simulated outcome
            if simulated_success:
                prediction = self.answer
            else:
                prediction = self.generate_wrong_answer()

            correct = prediction == self.answer
            cost = 0.0

        elif action == 1:
            #calculator always produces the correct answer but with a cost
            prediction = self.answer
            correct = True
            cost = self.tool_cost

        else:
            raise ValueError(f"Invalid action: {action}")

        assert correct is not None

        #reward correctness while penalizing tool use by its cost
        reward = float(correct) - cost

        #additional information for debugging and evaluation
        info = {
            "problem": (self.a, self.b),
            "correct_answer": self.answer,
            "prediction": prediction,
            "correct": bool(correct),
            "used_tool": action == 1,
            "tool_cost": cost,
            "direct_accuracy": self.direct_answer_accuracy()
        }

        #each episode contains exactly one decision
        done = True
        next_state = None

        return next_state, reward, done, info

if __name__ == "__main__":
    env = MathEnvironment(tool_cost=0.15, seed = 42)

    state = env.reset()
    print("State:", state)

    next_state, reward, done, info = env.step(0)

    print("Reward:", reward)
    print("Prediction:", info["prediction"])
    print("Correct answer:", info["correct_answer"])
    print("Correct:", info["correct"])

        
