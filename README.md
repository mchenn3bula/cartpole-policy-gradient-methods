# CartPole policy-gradient learning exercises

Coursework exploring policy networks, value functions and the CartPole control problem with PyTorch and a custom environment.

**Status: incomplete learning exercises.** The current repository contains scaffolding for REINFORCE, REINFORCE with a baseline and actor-critic. It does not yet contain completed, benchmarked implementations of all three methods.

## Repository guide

| Files | Purpose and current status |
| --- | --- |
| `0_test_env.py` | Environment exploration. |
| `1_test_model.py`, `3_test_model.py`, `5_test_model.py`, `7_test_model.py` | Model inspection / rollout scripts; some expect saved checkpoints. |
| `2_reinforce.py` | Partial training loop; discounted returns, policy loss and stopping logic still contain TODOs. |
| `4_reinforce_baseline.py` | Baseline-method exercise scaffold. |
| `6_actor_critic.py` | Actor-critic exercise scaffold. |
| `src/envs/cartpole.py` | Environment implementation. |
| `src/models/actor.py`, `src/models/critic.py` | Policy and value model definitions. |
| `documents/cartpole.pdf` | Original exercise material. |

## Dependencies

The existing dependency list is at `documents/requirements.txt`:

```bash
python -m pip install -r documents/requirements.txt
```

The dependencies are not version-pinned. Inspect the environment and scripts before running them; the training exercises need completion first.

## Completion plan

1. Implement discounted returns and the REINFORCE objective.
2. Add value-function learning and advantage estimation.
3. Complete actor-critic updates and explicit stopping criteria.
4. Compare learning curves over multiple random seeds with a fixed evaluation protocol.

No solved-environment or comparative-performance claim is made for the current code.
