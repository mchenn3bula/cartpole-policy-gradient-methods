# CartPole Policy Gradient Methods 🤖

This repository implements various policy gradient reinforcement learning algorithms to solve the classic CartPole balancing problem. The project provides a step-by-step approach to understanding and implementing modern RL techniques using PyTorch.

## Project Overview 📋

The CartPole problem is a classic control task in reinforcement learning where an agent must learn to balance a pole attached to a moving cart. This project implements several policy gradient methods of increasing complexity to solve this problem:

1. **REINFORCE** - A fundamental Monte Carlo policy gradient method
2. **REINFORCE with Baseline** - Adding a value function to reduce variance
3. **Actor-Critic** - Combining policy and value function estimation for better learning

## Features ✨

- **Modular Implementation**: Clear separation between environment, algorithms, and models
- **Multiple Algorithms**: Implementations of REINFORCE, REINFORCE with baseline, and Actor-Critic
- **Visualized Training**: Live rendering of the agent's progress during training
- **Hyperparameter Tuning**: Explore the effects of learning rates and discount factors
- **Markov Property Analysis**: Investigation of state representation and its effects on learning

## Requirements 🛠️

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- SciPy

Install all dependencies with:
```
pip install -r requirements.txt
```

## Project Structure 📁

```
.
├── src/
│   ├── envs/
│   │   └── cartpole.py      # CartPole environment implementation
│   └── models/
│       ├── actor.py         # Policy network (actor)
│       └── critic.py        # Value function network (critic)
├── 0_test_env.py            # Script to test environment with random policy
├── 1_test_model.py          # Script to test untrained policy network
├── 2_reinforce.py           # REINFORCE implementation
├── 3_test_model.py          # Script to test trained REINFORCE policy
├── 4_reinforce_baseline.py  # REINFORCE with baseline implementation
├── 5_test_model.py          # Script to test trained baseline policy
├── 6_actor_critic.py        # Actor-Critic implementation
├── 7_test_model.py          # Script to test trained actor-critic policy
└── requirements.txt         # Project dependencies
```

## Algorithms Implemented 🧠

### 1. REINFORCE

The REINFORCE algorithm, also known as Monte Carlo Policy Gradient, works by:
- Collecting complete episodes of experience
- Computing discounted returns at each timestep
- Updating the policy parameters by gradient ascent using the policy gradient theorem

### 2. REINFORCE with Baseline

This algorithm reduces the variance of policy gradient estimates by:
- Adding a value function (critic) that estimates the expected return from each state
- Subtracting this baseline from the returns before updating the policy
- Training both the policy and value function simultaneously

### 3. Actor-Critic

The Actor-Critic method combines policy-based and value-based methods by:
- Using the value function to bootstrap returns (instead of waiting for episode end)
- Allowing for online updates during episode execution
- Reducing variance while maintaining reasonable bias in estimates

## Usage 🚀

### Testing the Environment

To see how the CartPole environment works with a random policy:

```bash
python 0_test_env.py
```

### Training with REINFORCE

To train a policy using the REINFORCE algorithm:

```bash
python 2_reinforce.py
```

### Testing a Trained Policy

To evaluate a policy trained with REINFORCE:

```bash
python 3_test_model.py
```

### Training with REINFORCE with Baseline

```bash
python 4_reinforce_baseline.py
```

### Training with Actor-Critic

```bash
python 6_actor_critic.py
```

## Implementation Details 📝

### CartPole Environment

The environment provides:
- **State**: Position and velocity of cart, angle and angular velocity of pole
- **Actions**: Push cart left (0) or right (1)
- **Reward**: +1 for each timestep the pole remains upright
- **Terminal conditions**: Pole falling past 12 degrees or cart moving out of bounds

### Policy Network

A neural network that maps states to action probabilities:
- Input layer: State features (4 neurons)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: Action probabilities (2 neurons with softmax)

### Value Network (for baseline and actor-critic)

A neural network that estimates the value function:
- Input layer: State features (4 neurons)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: Value estimate (1 neuron)

## Learning Outcomes 🎓

This project demonstrates several key reinforcement learning concepts:
- Policy gradient methods
- Function approximation with neural networks
- Variance reduction techniques
- The importance of the Markov property in RL
- Hyperparameter tuning for RL algorithms

## Inspirations and References 📚

This project is inspired by the foundational work in reinforcement learning:
- Sutton and Barto's "Reinforcement Learning: An Introduction"
- The classic paper "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems" by Barto, Sutton, and Anderson
- Modern deep reinforcement learning implementations

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.
