import numpy as np
import matplotlib.pyplot as plt

from sarsaLearner import sarsaLearner


class qLearner(sarsaLearner):
    """Q learning for the gridWorld environment of 10x5 cells
    Learns at every step from (s,a,r,s') tuples
    """

    def __init__(self, epochs:int, time_steps:int, alpha:float, gamma:float, epsilon:float, epsilon_decay:float, rng_door:bool=False):
        """Initialize the Q learner

        Args:
            epochs (int): number of episodes to run
            time_steps (int): number of time steps per episode
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): exploration rate
            epsilon_decay (float): exploration rate decay factor
            rng_door (bool, optional): whether to randomly move the door. Defaults to False.
        """

        # Class derived from sarsaLearner, as only the _learn_xx and run methods need to be changed
        # _choose_action and plot_xx methods are the same as sarsaLearner
        # Initialize the SARSA learner class
        super().__init__(epochs, time_steps, alpha, gamma, epsilon, epsilon_decay, rng_door)

    def _learn_q(self, s:tuple, a:str, r:float, s_prime:tuple):
        """Update the Q values based on the Q learning update rule

        Args:
            s (tuple): current state
            a (str): action taken
            r (float): reward received
            s_prime (tuple): next state
        """

        # Get Q values for current state and action
        Q_sa = self.Q_table[s][a]

        # Get Q values for next state
        Q_s_prime = self.Q_table[s_prime]

        # The only difference between SARSA and Q learning is that Q learning uses the maximum Q value for the next state
        # SARSA uses the Q value for the next state and next action
        # Get the maximum Q value for the next state
        max_Q_s_prime = max(Q_s_prime.values())

        # Update the Q value for the current state and action
        Q_sa = Q_sa + self.alpha * (r + self.gamma * max_Q_s_prime - Q_sa)
        # Update the Q table
        self.Q_table[s][a] = Q_sa

    def run(self, verbose:bool=False):
        """Run the Q learning algorithm

        Args:
            verbose (bool, optional): whether to print progress. Defaults to False.
        """

        # Run for the specified number of epochs
        for epoch in range(self.epochs):

            # Reset the environment
            current_state = tuple(self.reset())

            # Initialize the reward for this epoch
            epoch_reward_history = 0

            # Run for the specified number of time steps
            for time_step in range(self.time_steps):

                # Choose an action based on the epsilon greedy policy
                action = self._choose_action(current_state)

                # Take the action and get the next state and reward
                next_state, reward = self.step(action, self.rng_door)
                next_state = tuple(next_state)

                # Update the Q values based on the Q learning update rule
                self._learn_q(current_state, action, reward, next_state)

                # Update the current state
                current_state = next_state

                # Update the reward for this epoch
                epoch_reward_history += reward

                # If the agent reaches the door, break
                if reward == 20:
                    break

            # Update the reward history
            self.reward_history.append(epoch_reward_history)

            # Decay the exploration rate
            self.epsilon *= self.epsilon_decay

            # If verbose, print the progress
            if verbose and (epoch%100==0 or epoch==self.epochs-1):
                print(f"Epoch {epoch+1}/{self.epochs} | Reward for epoch: {epoch_reward_history}")

            # If verbose, print the Q_values for the door
            if verbose and epoch==self.epochs-1 and not self.rng_door:
                print(f"Q Values for Door: {self.Q_table[tuple(self.init_door)]}")


if __name__=="__main__":

    # Parameters
    epochs = 500
    time_steps = 200
    alpha = 0.25 # Learning rate
    gamma = 0.9 # Discount factor
    epsilon = 0.9 # Exploration factor
    epsilon_decay = 0.99 # Exploration decay factor
    rng_door = False # Move the door randomly

    # Initialize the Q learner
    qAgent = qLearner(epochs, time_steps, alpha, gamma, epsilon, epsilon_decay, rng_door)

    # Run the Q learner
    qAgent.run(verbose=True)

    # Generate plots
    qAgent.plot_reward_history()
    qAgent.plot_Q_table()
    qAgent.plot_policy()

    # Show the plots
    plt.show()