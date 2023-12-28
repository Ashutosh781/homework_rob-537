import numpy as np
import matplotlib.pyplot as plt

from gridWorld import gridWorld


class sarsaLearner(gridWorld):
    """SARSA learning for the gridWorld environment of 10x5 cells
    Learns at every step from (s,a,r,s',a') tuples
    """

    def __init__(self, epochs:int, time_steps:int, alpha:float, gamma:float, epsilon:float, epsilon_decay:float, rng_door:bool=False):
        """Initialize the SARSA learner

        Args:
            epochs (int): number of episodes to run
            time_steps (int): number of time steps per episode
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): exploration rate
            epsilon_decay (float): exploration rate decay factor
            rng_door (bool, optional): whether to randomly move the door. Defaults to False.
        """

        # Initialize the gridWorld environment class
        super().__init__()

        # Parameters
        self.epochs = epochs
        self.time_steps = time_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rng_door = rng_door

        # Actions
        self.actions = ["up", "down", "left", "right", "stay"]

        # Initialize the reward history
        self.reward_history = []

        # Initialize Q table as dictionary of dictionaries
        # Dictonaries are the fastest way to access the Q values
        # Dictionary keys are states, values are dictionaries of actions and Q values
        self.Q_table = {}
        for x in range(10):
            for y in range(5):
                self.Q_table[(x, y)] = {}
                for action in self.actions:
                    # Initialize Q values to 0
                    self.Q_table[(x, y)][action] = 0

    def _choose_action(self, state:tuple):
        """Choose an action based on the epsilon greedy policy for exploration

        Args:
            state (tuple): current state
        """

        # Explore with probability epsilon
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)

        # Exploit with probability 1-epsilon
        else:
            # Get the Q values for the current state
            Q_values = self.Q_table[state]
            # Take the action with the max Q value
            maxQ = max(Q_values.values())
            # Random choice if multiple actions have the same max Q value
            action = np.random.choice([k for k,v in Q_values.items() if v==maxQ])

        return action

    def _learn_sarsa(self, s:tuple, a:str, r:int, s_prime:tuple, a_prime:str):
        """Update the Q values based on the SARSA update rule

        Args:
            s (tuple): current state
            a (str): current action
            r (int): reward
            s_prime (tuple): next state
            a_prime (str): next action
        """

        # Get the Q values for the current state and action
        Q_sa = self.Q_table[s][a]
        # Get the Q values for the next state and action
        Q_sa_prime = self.Q_table[s_prime][a_prime]

        # Update the Q value for the current state and action
        Q_sa = Q_sa + self.alpha*(r + self.gamma*Q_sa_prime - Q_sa)
        # Update the Q table
        self.Q_table[s][a] = Q_sa

    def run(self, verbose:bool=False):
        """Run the SARSA learner

        Args:
            verbose (bool, optional): whether to print progress. Defaults to False.
        """

        # Run for the specified number of epochs
        for epoch in range(self.epochs):

            # Reset the environment
            current_state = tuple(self.reset())

            # Initialize the reward for this epoch
            epoch_reward_history = 0

            # Choose an initial action based on the epsilon greedy policy
            action = self._choose_action(current_state)

            # Run for the specified number of time steps
            for time_step in range(self.time_steps):

                # Take the action and get the next state and reward
                next_state, reward = self.step(action, self.rng_door)
                next_state = tuple(next_state)

                # Choose next action based on the epsilon greedy policy
                next_action = self._choose_action(next_state)

                # Update the Q values based on the SARSA update rule
                self._learn_sarsa(current_state, action, reward, next_state, next_action)

                # Update the current state
                current_state = next_state

                # Update the next action
                action = next_action

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
                print(f"Epoch: {epoch+1}/{self.epochs} | Reward for epoch: {epoch_reward_history}")

            # If verbose, print the Q_values for the door
            if verbose and epoch==self.epochs-1 and not self.rng_door:
                print(f"Q Values for Door: {self.Q_table[tuple(self.init_door)]}")

    def plot_reward_history(self):
        """Plot the reward history"""

        # Plot the reward history
        plt.figure("Reward History")
        plt.plot(self.reward_history)
        plt.title("Reward History")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")

    def plot_Q_table(self):
        """Plot the Q table as heatmap for each state and action"""

        # Initialize the Q table
        Q_table = np.zeros((5, 10, 5))

        # Fill in the Q table
        for x in range(10):
            for y in range(5):
                Q_values = self.Q_table[(x, y)]
                for i, action in enumerate(self.actions):
                    Q_table[y, x, i] = Q_values[action]

        # Plot the Q table for each action except stay
        plt.figure("Q Table")
        for i, action in enumerate(self.actions[:-1]):
            plt.subplot(2, 2, i+1)
            plt.imshow(Q_table[:, :, i], cmap="hot", vmin=np.min(Q_table), vmax=np.max(Q_table))
            plt.gca().invert_yaxis()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Q Table for Action: {action}")

        # Plot a single colorbar
        plt.colorbar(ax=plt.gcf().get_axes(), shrink=0.9, location="right")

    def plot_policy(self):
        """Plot the policy as arrows pointing in the direction of the action"""

        plt.figure("Policy")

        # Initialize the policy
        policy = np.zeros((5, 10))

        # Fill in the policy
        for x in range(10):
            for y in range(5):
                Q_values = self.Q_table[(x, y)]
                maxQ = max(Q_values.values())
                action = np.random.choice([k for k,v in Q_values.items() if v==maxQ])

                # For wall cells, just plot stay as the Q_values are all 0 and random.choice just picking randomly
                if x==7 and y<3:
                    action = "stay"

                # Assign a 1 if action is not stay
                # Plot corresponding arrows
                if action == "up":
                    policy[y, x] = 1
                    plt.arrow(x, y, 0, 0.4, width=0.05)
                elif action == "down":
                    policy[y, x] = 1
                    plt.arrow(x, y, 0, -0.4, width=0.05)
                elif action == "right":
                    policy[y, x] = 1
                    plt.arrow(x, y, 0.4, 0, width=0.05)
                elif action == "left":
                    policy[y, x] = 1
                    plt.arrow(x, y, -0.4, 0, width=0.05)
                elif action == "stay":
                    # Just plot stay as different color
                    policy[y, x] = 0

        # Plot the policy
        plt.imshow(policy, cmap="hot", interpolation="nearest")
        plt.gca().invert_yaxis()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Policy")


if __name__ == "__main__":

    # Parameters
    epochs = 500
    time_steps = 200
    alpha = 0.25 # Learning rate
    gamma = 0.9 # Discount factor
    epsilon = 0.9 # Exploration factor
    epsilon_decay = 0.99 # Exploration decay factor
    rng_door = False # Move the door randomly

    # Initialize the SARSA learner
    sarsaAgent = sarsaLearner(epochs, time_steps, alpha, gamma, epsilon, epsilon_decay, rng_door)

    # Run the SARSA learner
    sarsaAgent.run(verbose=True)

    # Generate plots
    sarsaAgent.plot_reward_history()
    sarsaAgent.plot_Q_table()
    sarsaAgent.plot_policy()

    # Show the plots
    plt.show()