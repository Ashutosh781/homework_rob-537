import os
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# NN related imports
import collections
import torch


class DeepQAgent():
    """Deep Q-learning agent for Gymnasium environments.

    Args:
        env: Gymnasium environment. Must have a discrete action space.
        model_weights: Path to the model weights file. If file does not exist, it will be created. Extensions supported: .pt
        use_prev: Use previous model weights. Defaults to False.
        episodes: Number of episodes to train. Defaults to 1000.
        test_episodes: Number of episodes to test. Defaults to 100.
        max_time_steps: Number of time steps per episode. Defaults to 500.
        sample_size: Sample batch size for replay memory. Defaults to 32.
        hidden_size: Number of neurons in the hidden layer. Defaults to 24.
        mean_reward_window: Window size for computing the mean reward. Defaults to 10.
        alpha: Learning rate. Defaults to 0.001.
        gamma: Discount factor. Defaults to 0.9.
    """

    def __init__(self, env, model_weights:str, use_prev:bool=False, episodes:int=1000, test_episodes:int=100, max_time_steps:int=500,
                sample_size:int=32, hidden_size:int=24, mean_reward_window:int=10, alpha:float=0.001, gamma:float=0.9):
        """Initializes the Deep Q-Learning agent."""

        # Check if the environment has a discrete action space.
        assert isinstance(env.action_space, gym.spaces.Discrete), "Environment has no discrete action space."
        # Check if the environment has a continuous observation space.
        assert isinstance(env.observation_space, gym.spaces.Box), "Environment has no continuous observation space."

        # Define the agent's attributes here.
        self.env = env
        self.model_weights = model_weights
        self.use_prev = use_prev
        self.episodes = episodes
        self.test_episodes = test_episodes
        self.max_time_steps = max_time_steps
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        self.mean_reward_window = mean_reward_window
        self.alpha = alpha
        self.gamma = gamma
        self.training = True

        # Memory for experience replay.
        self.memory = collections.deque(maxlen=2000)

        # Extract the number of actions and observations from the environment.
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.shape[0]

        # Initialize the neural network model.
        self.model = None
        self.model_loss = None
        self.optimizer = None
        self.model_init()

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
        """Sets the training flag for the agent."""

        self.training = training

        # If training false, set alpha to 0.0.
        if not training:
            self.alpha = 0.0

    def model_init(self):
        """Initializes the neural network model."""

        ## Define the model architecture here.
        # The model should take the observation as input and output the Q-values for each action.
        # The model has 1 hidden layer and output layer with linear activation.
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.num_actions),
        )

        ## Define the loss function and optimizer.
        # The loss function is mean squared error.
        self.model_loss = torch.nn.MSELoss()
        # The optimizer is Adam with learning rate alpha.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

        # Load previous model weights if specified and file exists.
        if self.use_prev and os.path.isfile(self.model_weights):
            self.model.load_state_dict(torch.load(self.model_weights))

    def get_action(self, observation):
        """Returns the action to take given an observation using model prediction.

        Args:
            observation: Observation from the environment.

        Returns:
            action: Action to take.
        """

        # Convert the observation to torch tensor.
        observation = torch.from_numpy(observation).float()

        # Get the model prediction for the observation.
        action = torch.argmax(self.model(observation)).item()

        return action

    def learn(self):
        """Replay buffer from experience replay for a batch of samples to update the model weights."""

        # Sample a batch of samples from the memory.
        minibatch = random.sample(list(self.memory), min(len(self.memory), self.sample_size))

        # Iterate through the batch of samples.
        for s, a, r, s_prime, _, truncated in minibatch:

            # Convert the state and next state to torch tensors.
            s = torch.from_numpy(s).float()
            s_prime = torch.from_numpy(s_prime).float()

            # Get the target for the current state and action from the model prediction for the next state.
            target = r if truncated else r + self.gamma * torch.max(self.model(s_prime))

            # Get the model prediction for the current state.
            y_pred = self.model(s)
            y_target = torch.zeros_like(y_pred)
            y_target[a] = target

            # Update the model weights.
            loss = self.model_loss(y_pred, y_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run(self, save_model:bool=False, verbose:bool=False):
        """Train the Deep Q-Learning agent.

        Args:
            save_model (bool, optional): whether to save the model weights. Defaults to False.
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

                # Take the action and get the next state, reward, and flags.
                next_observation, reward, terminated, truncated,_ = self.env.step(action)

                # Add the sample to the replay memory.
                self.memory.append((observation, action, reward, next_observation, terminated, truncated))

                # Update the observation.
                observation = next_observation

                # Update episode reward.
                episode_reward += reward

                # If terminated, break the loop and give negative reward.
                if terminated:
                    break

                # If truncated, break the loop and give positive reward.
                if truncated:
                    episode_reward += 100.0
                    break

            # Update the model weights.
            if self.training:
                self.learn()

            # Update the reward history.
            if self.training:
                self.reward_train.append(episode_reward)
            else:
                self.reward_test.append(episode_reward)

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

        # Save the model weights.
        if save_model:
            torch.save(self.model.state_dict(), self.model_weights)

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