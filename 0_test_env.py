"""Test a random solution.

Author: Elie KADOCHE.
"""

from numpy.random import default_rng

from src.envs.cartpole import CartpoleEnv

if __name__ == "__main__":
    # Create environment
    env = CartpoleEnv()

    # Create random generator
    generator = default_rng(seed=None)

    # Reset it
    total_reward = 0.0
    state, _ = env.reset(seed=None)

    # While the episode is not finished
    terminated = False
    while not terminated:

        # Select a random action
        action = generator.integers(0, 2)

        # One step forward
        state, reward, terminated, _, _ = env.step(action)

        # Render (or not) the environment
        total_reward += reward
        env.render()

    # Print reward
    print("total_reward = {}".format(total_reward))
