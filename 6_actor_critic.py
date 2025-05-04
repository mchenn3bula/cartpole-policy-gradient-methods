"""Actor-critic algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical

from src.envs.cartpole import CartpoleEnv
from src.models.actor import ActorModel
from src.models.critic import CriticModel

# Policy and critic model path
ACTOR_PATH = "models/actor_2.pt"
CRITIC_PATH = "models/critic_2.pt"

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.8

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.1

if __name__ == "__main__":
    # Create policy
    actor = ActorModel()
    actor.train()

    # Create critic
    critic = CriticModel()
    critic.train()

    # Create the environment
    env = CartpoleEnv()

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # Create optimizer with the critic parameters
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # ---> TODO: based on the REINFORCE script, create the actor-critic script
