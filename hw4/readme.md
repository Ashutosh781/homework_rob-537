# Homework 4: Learning-Based Control

In this assignment, you will apply the principles you have learned thus far to create two learning-based controllers.

Install [Gymnasium](https://gymnasium.farama.org/) and familiarize yourself with the gym interface. This should look similar to the interface you used in the last assignment.

1. Create a Q-learning agent that learns to solve the "Cart Pole" environment. The agent should balance the pole for 100 time steps.

    - How will you handle the continuous state space?

2. Evolve a neural network to solve the previous task.

    - What will you use for your evaluation function?
    - What mapping should the network learn?

Your report should include:

- Answers to previous questions
- Performance over time curves for each experiment.
- A description of your algorithms.
- Similarities and differences in performance for the two algorithms + an explanation.

Extra-Credit Bonus: Use these two learning methods to solve another gym task. Choose your favorite environment from either the "Classic Control" or "Box2D" sets.

- What environment did you choose?
- What changes did you have to make to adapt your learning algorithms to this new environment?
- Describe the resulting behaviors each controller learned.

Or Implement a new learning algorithm of your choice like Deep Q-Learning Network. You may choose to implement a new algorithm from the literature or a new algorithm of your own design. Describe your algorithm and compare its performance to the two algorithms above.
