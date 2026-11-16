# Reinforcement Learning: Q-Learning and SARSA on CliffWalking

This repository contains two notebook-based implementations of classic reinforcement learning algorithms using Gymnasium's `CliffWalking-v1` environment:

- `Q_Learning.ipynb`
- `SARSA.ipynb`

Both notebooks focus on understanding how an agent learns a policy in a grid world by interacting with the environment, receiving rewards, and updating a value table over time. The goal is not just to make the agent move, but to teach it how to make better decisions under uncertainty, balancing exploration and exploitation.

The project is a practical, beginner-friendly introduction to temporal-difference learning. It demonstrates the core ideas behind value-based reinforcement learning and highlights the difference between two widely used algorithms: Q-learning and SARSA.

## Why this project matters

Reinforcement learning is a branch of machine learning where an agent learns by trial and error rather than by being explicitly told the correct action at each step. Instead of labeled data, the agent receives rewards and penalties and learns policies that maximize long-term return.

In this project, the agent must navigate a grid environment where:

- each step usually gives a small negative reward,
- stepping into the cliff gives a large penalty,
- reaching the goal gives the terminal outcome,
- and the challenge is to discover a path that avoids danger while still reaching the destination efficiently.

This is a canonical example often used to teach the strengths and weaknesses of different reinforcement learning algorithms.

## Environment: CliffWalking-v1

The environment is a classic grid-world problem from Gymnasium (formerly OpenAI Gym). The agent starts at the left side of the grid and must reach the goal at the far right without falling off the cliff.

Important environment properties:

- State space size: 48
- Action space size: 4
- Actions: up, down, left, right
- Observation: the current grid position
- Reward structure: usually a small negative reward for every step, with a much larger negative penalty if the agent falls off the cliff

This makes the task a natural case study for value-based learning because the agent can store action values for each state and gradually improve them through repeated experience.

The key challenge is that the cliff is a dangerous region, and a naive policy may fall into it often. The optimal policy must prefer a safe route along the top edge of the cliff while still reaching the terminal destination.

## Core reinforcement learning concepts used

### 1. State
A state is the agent's current location in the grid. In this project, each grid cell corresponds to a discrete state and the RL agent learns values for those states.

### 2. Action
At each state the agent chooses one of four possible actions: move up, down, left, or right.

### 3. Reward
After taking an action, the agent receives a numeric reward:

- small negative reward for moving around, encouraging efficiency,
- large negative reward for falling off the cliff,
- terminal outcome when the goal is reached or the episode ends.

### 4. Policy
A policy tells the agent which action to take in each state. In the notebook, the policy is mostly `epsilon-greedy`, meaning:

- with probability `epsilon`, the agent explores a random action,
- otherwise, it exploits the action with the best current estimate.

### 5. Q-value
The agent stores a Q-table, where each entry `Q[state, action]` estimates the expected cumulative future reward if the agent takes a particular action in a particular state.

The Q-table is initialized to zeros and updated repeatedly as the agent learns.

### 6. Discount factor
The variable `gamma = 0.99` means the agent values future rewards almost as much as immediate rewards, but still gives slightly more emphasis to near-term outcomes.

### 7. Learning rate
The variable `alpha = 0.5` controls how strongly the agent updates its value estimates after each step.

## Q_Learning.ipynb

This notebook implements the classic off-policy Q-learning algorithm.

### Purpose of the notebook
The notebook demonstrates how an agent learns a value estimate without necessarily following the same policy it is evaluating. In other words, Q-learning learns from the best possible future outcome available in the next state, even if the chosen action was exploratory.

### Important structure in the notebook
The notebook begins by importing the necessary libraries:

```python
import gymnasium as gym
import numpy as np
import random
```

Then it initializes the core learning parameters:

```python
gama = 0.99
epsilon = 0.1
alpha = 0.5
episodes = 500
```

A Q-table is created with shape `(48, 4)`:

```python
Q = np.zeros((48, 4))
```

This means:

- 48 possible states,
- 4 possible actions,
- each entry stores an estimated value for a state-action pair.

### Epsilon-greedy policy
The notebook defines an `epsilon_greedy` function:

```python
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()  # explore
    else:
        return np.argmax(Q[state])
```

This policy balances exploration and exploitation:

- exploration: occasionally take a random action,
- exploitation: otherwise choose the action with the best estimated Q-value.

### Q-learning update rule
The core learning update is based on the Bellman optimality equation:

```python
Q[state, action] = Q[state, action] + alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])
```

In words:

- the old estimate is adjusted toward the target value,
- the target value is the immediate reward plus the discounted best future value,
- this update uses the maximum value from the next state, representing the optimal future action.

This is the essence of Q-learning: it learns the value of the optimal policy, even while the agent is behaving according to a mixture of exploration and exploitation.

### Example behavior in the notebook
The notebook trains the agent across 500 episodes and occasionally renders the environment. Early episodes often show poor trajectories with large negative rewards, such as:

- episode 1: around `-900`
- episode 2: around `-977`
- episode 3: around `-303`

This is expected because the agent is still learning and exploring. Over time, the rewards improve dramatically. By the later episodes, the total rewards move much closer to a near-optimal value, with examples like `-13`, `-15`, or similar small negative totals.

This indicates that the policy has improved and the agent is successfully avoiding the cliff while reaching the goal efficiently.

### Evaluation after training
After the training loop, the notebook evaluates the greedy policy by selecting the best-known action at each state:

```python
action = np.argmax(Q[state])
```

This allows the agent to act according to the learned policy without random exploration. The result is a deterministic path that follows the best route it has discovered.

### Learning intuition
Q-learning is often described as an off-policy algorithm because it learns the value of the optimal policy while following another behavior policy. This means it can learn from exploratory actions and still converge toward the best long-term solution.

In the CliffWalking environment, this often leads to a path that prefers the safe route near the top of the cliff, because the penalty for falling is severe.

## SARSA.ipynb

This notebook implements the on-policy SARSA algorithm.

### Purpose of the notebook
SARSA stands for State-Action-Reward-State-Action. Unlike Q-learning, SARSA updates its value estimates based on the actual next action chosen by the current policy, not the maximum action available in the next state.

Because of this, SARSA is generally more conservative and often learns a policy that is more tightly aligned with the behavior currently being followed.

### Important structure in the notebook
The notebook starts with the same essential setup:

```python
import gymnasium as gym
import numpy as np
import random
```

It then configures the learning parameters:

```python
gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500
```

The Q-table is created in the same way:

```python
Q = np.zeros((48, 4))
```

### Epsilon-greedy policy
Similar to the Q-learning notebook, the agent uses an epsilon-greedy policy:

```python
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()  # random action => explore
    else:
        return np.argmax(Q[state])  # exploit best known action
```

### SARSA update rule
The main difference is in the target value used by the update:

```python
Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
```

Here, `next_action` is chosen according to the current policy, not necessarily the best future action. This means the update is tied to the agent's actual behavior.

This is why SARSA is considered on-policy: the learned policy is updated to reflect the policy it is currently following.

### Example behavior in the notebook
The SARSA notebook also trains over 500 episodes. In the observed outputs, the initial episodes are highly negative, for example:

- episode 1: about `-1421`
- episode 2: about `-1025`
- episode 3: about `-250`

These values are typical early-stage exploration, where the agent is still learning and often falls or wanders inefficiently.

Later episodes show much better outcomes. The final rewards in the outputs are significantly improved, reaching values close to `-20` or `-30` depending on the exact run and randomness. This suggests that the policy is converging toward a safer route that avoids the cliff while still making progress to the goal.

### Why SARSA behaves differently
SARSA tends to be more cautious because it only learns from the actions it actually takes under the current policy. In a risky environment like CliffWalking, this can sometimes lead to a more conservative policy that is less willing to take a risky action in pursuit of a slightly better outcome.

Q-learning, in contrast, can learn a more optimistic estimate because it assumes the agent can always act optimally in the next state. This often makes it more aggressive in choosing the best future value even when the current policy is exploratory.

## Q-learning vs SARSA in this project

The two notebooks are intentionally designed to highlight the difference between the algorithms:

### Q-learning
- Off-policy
- Updates using the best possible next action
- Tends to converge faster toward optimal value estimates
- More likely to act greedily based on the optimal future value

### SARSA
- On-policy
- Updates using the next action actually chosen by the policy
- More faithful to the current behavior policy
- Often more cautious in stochastic and risky environments

In CliffWalking, these differences matter because the cliff creates a strong penalty for taking risky moves. Q-learning may more directly learn the optimal route around the danger, while SARSA may update more conservatively based on the policy trajectory it is currently following.

## Practical notes for GitHub users

This repository is useful for:

- learning RL foundations from code,
- comparing temporal-difference algorithms side by side,
- visualizing how exploration-exploitation trade-offs affect policy quality,
- understanding value iteration concepts in a simple environment.

It is especially valuable as a teaching project because the code remains compact and readable while still covering major RL concepts.

## Setup and execution

To run the notebooks locally:

```bash
pip install gymnasium
pip install "gymnasium[toy-text]"
```

Then open the notebooks in Jupyter Notebook or VS Code and run the cells in order.

## Project files

```text
Reinforcement_Learning/
├── Q_Learning.ipynb
├── SARSA.ipynb
├── README.md
```

## Summary

This repository demonstrates the core mechanics of reinforcement learning in a simple but instructive task. It uses a grid-world environment to teach how an agent learns from rewards, updates a Q-table, balances exploration with exploitation, and improves its policy over time.

The two notebooks together provide a direct comparison of:

- Q-learning as an off-policy optimal-value approach,
- SARSA as an on-policy behavior-driven approach,
- and the broader RL intuition behind temporal-difference learning.

They are ideal for learning, presenting, and explaining reinforcement learning in a practical, easy-to-follow way.
