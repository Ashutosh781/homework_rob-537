# Homework 3: Reinforcement Learning

For this assignment our learning environment is a 5x10 gridworld. There is:

- A door on the red square (x) with a reward of 20
- A solid wall (gray) the agent cannot move across
- A reward of -1 for every time the agent is in a state other than the red door state

The agent starts at a random location and has five actions (move in four directions or stay in place), the state of the system is the location of the agent $(x,y)$, and an episode is 20 time steps.

NOTE: To avoid environment errors, we have provided a gridworld Python class in [gridWorld.py](/hw3/gridWorld.py).

![Gridworld](/hw3/results/grid_example.png)

1. Implement SARSA to solve this problem. How did the algorithms perform? Include learning curves and plots of the learned value tables.

2. Implement a Q-learning algorithm to solve this problem. How did the algorithms perform? How did solution compare to the SARSA solution? Discuss the implications of your results.

3. Now consider the environment where the red door moves randomly by 1 cell every time step. Keep the initial starting location of the door the same as before. Use the EXACT same algorithms from problems 1 and 2 to solve this problem.
    - How does the performance of the agent compare to problems 1 and 2?
    - Does the agent learn a good policy?
    - Describe your results and hypothesize why your agent performs the way it does. Speculate on how you may improve the performance of the agent. Again, plot learning curves and value tables.
