import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class QAgent():
    """Q-learning agent for Gymnasium environments. Discretizes the continuous state space.

    Args:
        env: Gymnasium environment. Must have a discrete action space.
        episodes: Number of episodes to train. Defaults to 1000.
        max_time_steps: Number of time steps per episode. Defaults to 500.
        test_episodes: Number of episodes to test. Defaults to 100.
        num_bins: Number of bins to discretize the state space. Defaults to 10.
        mean_reward_window: Window size for the rolling mean. Defaults to 10.
        alpha: Learning rate. Defaults to 0.1.
        epsilon: Exploration rate. Defaults to 1.0.
        epsilon_decay: Decay rate for epsilon. Defaults to 0.99.
        epsilon_min: Minimum value for epsilon. Defaults to 0.01.
        gamma: Discount factor. Defaults to 0.9.
    """

    def __init__(self, env, episodes:int=1000, max_time_steps:int=500, test_episodes:int=100, num_bins:int=10,
                 mean_reward_window:int=10, alpha:float=0.1, epsilon:float=1.0, epsilon_decay:float=0.99, epsilon_min:float=0.01, gamma:float=0.9):
        """Initializes the Q-Learning agent."""

        # Check if the environment has a discrete action space.
        assert isinstance(env.action_space, gym.spaces.Discrete), "Environment has no discrete action space."
        # Check if the environment has a continuous observation space.
        assert isinstance(env.observation_space, gym.spaces.Box), "Environment has no continuous observation space."

        # Define the agent's attributes here.
        self.env = env
        self.episodes = episodes
        self.max_time_steps = max_time_steps
        self.test_episodes = test_episodes
        self.num_bins = num_bins
        self.mean_reward_window = mean_reward_window
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.training = True

        # Extract the number of actions and observations from the environment.
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.shape[0]

        # Exrtact the observation space bounds from the environment.
        self.observation_space_low = env.observation_space.low
        self.observation_space_high = env.observation_space.high

        # If the bounds are not finite, approximate only the infinite ones.
        # In gymnasium, infinite bounds are represented as large floats and not np.inf.
        # This is why we check for values larger than 1e3. And approximate them with -5.0 and 5.0.
        self.observation_space_low[self.observation_space_low < -1e3] = -5.0
        self.observation_space_high[self.observation_space_high > 1e3] = 5.0

        # Initialize the bins for each observation dimension.
        self.bins = np.linspace(self.observation_space_low, self.observation_space_high, self.num_bins).T

        # Initialize the Q-table with zeros.
        self.Q = np.zeros(([self.num_bins] * self.num_observations) + [self.num_actions])

        # Initialize the reward history.
        self.reward_train = []
        self.reward_test = []
        self.reward_train_rolling = [] # Rolling mean over every 1/10 of the train episodes.
        self.reward_test_rolling = [] # Rolling mean over every 1/10 of the test episodes.
        self.reward_train_mean = [] # Mean over every 1/10 of the train episodes.
        self.reward_test_mean = [] # Mean over every 1/10 of the test episodes.

        self.train_x_axis = []
        self.test_x_axis = []

    def set_training(self, training:bool=True):
        """Sets the training flag."""

        self.training = training

        # If training false, set epsilon and alpha to 0.0.
        if not training:
            self.epsilon = 0.0
            self.alpha = 0.0

    def get_discrete_observation(self, observation):
        """Discretizes the continuous observation space.

        Args:
            observation: Observation from the environment.

        Returns:
            Discretized observation index.
        """

        # Initialize the discrete observation.
        discrete_observation = np.zeros(self.num_observations, dtype=np.int64)

        # Digitize each observation dimension.
        for i in range(self.num_observations):
            discrete_observation[i] = np.digitize(observation[i], self.bins[i]) - 1

        return tuple(discrete_observation)

    def get_action(self, observation):
        """Returns the action to take given the observation. Implements epsilon-greedy policy with Q-table.

        Args:
            observation: Observation from the environment.

        Returns:
            Action to take.
        """

        # Get the discrete observation.
        sd = self.get_discrete_observation(observation)

        # With probability epsilon, take a random action.
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        # Otherwise, take the best action.
        else:
            action = np.argmax(self.Q[sd])

        return action

    def update(self, observation, action, reward, next_observation):
        """Updates the Q-table."""

        # Get the discrete observations.
        sd = self.get_discrete_observation(observation)
        sd_prime = self.get_discrete_observation(next_observation)

        # Get the Q-value for the current state and action.
        Q_sa = self.Q[sd][action]

        # Get the maximum Q-value for the next state.
        max_Q_s_prime = np.max(self.Q[sd_prime])

        # Update the Q-value for the current state and action.
        Q_sa = Q_sa + self.alpha * (reward + self.gamma * max_Q_s_prime - Q_sa)

        # Update the Q-table.
        self.Q[sd][action] = Q_sa

    def run(self, verbose:bool=False):
        """Train and test the Q-Learning agent.

        Args:
            verbose (bool, optional): whether to print progress. Defaults to False.
        """

        for episode in range(self.episodes + self.test_episodes):

            # Reset the environment.
            observation,_ = self.env.reset()
            episode_reward = 0

            # Run the episode for max_time_steps or until the episode is done.
            for _ in range(self.max_time_steps):

                # Get the action to take.
                action = self.get_action(observation)

                # Take the action and get the next observation and reward.
                next_observation, reward, terminated, truncated,_ = self.env.step(action)

                # Update the Q-table.
                self.update(observation, action, reward, next_observation)

                # Update episode reward.
                episode_reward += reward

                # Update the observation.
                observation = next_observation

                # If terminated, break the loop.
                if terminated:
                    break

                # If truncated, break the loop and give positive reward.
                if truncated:
                    episode_reward += 100.0
                    break

            # Update the reward history.
            if self.training:
                self.reward_train.append(episode_reward)
            else:
                self.reward_test.append(episode_reward)

            # Decay epsilon.
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

            # If verbose, print progress for training episodes.
            if verbose and (episode % (self.episodes // 10) == 0 or episode == self.episodes - 1) and self.training:
                print(f"Episode: {episode+1}/{self.episodes} | Mean Reward until now: {np.mean(self.reward_train):.2f}")

            # Set the training flag to false after training.
            if episode == self.episodes - 1:
                self.set_training(False)
                if verbose and self.test_episodes > 0:
                    print("Testing...")

            # Print progress for testing episodes at the end.
            if verbose and episode == self.episodes + self.test_episodes - 1 and self.test_episodes > 0:
                print(f"Mean Reward in Test: {np.mean(self.reward_test):.2f}")

        # Compute the rolling mean and mean over every mean reward window of the episodes for training episodes.
        for i in range(0, self.episodes, self.episodes // self.mean_reward_window):
            self.reward_train_rolling.append(np.mean(self.reward_train[0:i + self.episodes // self.mean_reward_window]))
            self.reward_train_mean.append(np.mean(self.reward_train[i:i + self.episodes // self.mean_reward_window]))
            self.train_x_axis.append(i + self.episodes // self.mean_reward_window)

        # Compute the rolling mean and mean over every mean reward window of the episodes for testing episodes.
        for i in range(0, self.test_episodes, self.test_episodes // self.mean_reward_window):
            self.reward_test_rolling.append(np.mean(self.reward_test[0:i + self.test_episodes // self.mean_reward_window]))
            self.reward_test_mean.append(np.mean(self.reward_test[i:i + self.test_episodes // self.mean_reward_window]))
            self.test_x_axis.append(i + self.test_episodes // self.mean_reward_window)

    def plot_results(self):
        """Plots the reward history."""

        # Training rewards
        plt.figure()
        plt.plot(self.reward_train)
        plt.title("Train Raw Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.figure()
        plt.plot(self.train_x_axis, self.reward_train_mean)
        plt.title("Train Mean windowed")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.figure()
        plt.plot(self.train_x_axis, self.reward_train_rolling)
        plt.title("Train Rolling mean")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Testing rewards
        plt.figure()
        plt.plot(self.reward_test)
        plt.title("Test Raw Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.figure()
        plt.plot(self.test_x_axis, self.reward_test_mean)
        plt.title("Test Mean windowed")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.figure()
        plt.plot(self.test_x_axis, self.reward_test_rolling)
        plt.title("Test Rolling mean")
        plt.xlabel("Episode")
        plt.ylabel("Reward")