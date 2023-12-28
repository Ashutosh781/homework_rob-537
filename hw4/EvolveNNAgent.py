import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# NN related imports
import torch


class EvolveNNAgent():
    """Evolutionary genetic algorithm Neural Network agent for Gymnasium environments.

    Args:
        env: Gymnasium environment. Must have a discrete action space.
        episodes: Number of episodes to train. Defaults to 1000.
        max_time_steps: Maximum number of time steps per episode. Defaults to 500.
        population_size: Number of individuals in the population. Defaults to 100.
        mutation_percent: Percentage of individuals to mutate. Defaults to 0.5.
        mean: Mean of the normal distribution to sample the noise for mutation. Defaults to 1.0.
        std: Standard deviation of the normal distribution to sample the noise for mutation. Defaults to 0.001.
    """

    def __init__(self, env, episodes:int=1000, max_time_steps:int=500, population_size:int=100,
                 mutation_percent:float=0.5, mean:float=1.0, std:float=0.001):
        """Initializes the Evolutionary genetic algorithm Neural Network agent."""

        # Check if the environment has a discrete action space.
        assert isinstance(env.action_space, gym.spaces.Discrete), "Environment has no discrete action space."
        # Check if the environment has a continuous observation space.
        assert isinstance(env.observation_space, gym.spaces.Box), "Environment has no continuous observation space."

        # Define the agent's attributes here.
        self.env = env
        self.episodes = episodes
        self.max_time_steps = max_time_steps
        self.population_size = population_size
        self.mutation_percent = mutation_percent
        self.mean = mean
        self.std = std

        # Extract the number of actions and observations from the environment.
        self.num_actions = env.action_space.n
        self.num_observations = env.observation_space.shape[0]

        # Number of parameters in the neural network.
        self.num_weights = self.num_observations * self.num_actions
        self.num_biases = self.num_actions
        self.num_params = self.num_weights + self.num_biases

        # Initialize the reward history.
        self.reward_history = np.zeros((self.episodes + 1, self.population_size))

        # Initalize the population.
        self.population = None

    def model_init(self, params:np.ndarray):
        """Initializes the neural network model.

        Args:
            params: Parameters of the neural network model. Shape: (num_params,).

        Returns:
            model: Neural network PyTorch model.
        """

        # Define the model architecture
        # Model has a linear layer with input size of num_observations and output size of num_actions
        # Basically linearly maps the observation to the action
        # There is a softmax layer at the end to convert the output to a probability distribution
        model = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, self.num_actions, bias=True),
            torch.nn.Softmax(dim=0)
        )

        # Extract the weights and biases from the parameters.
        weights = params[:self.num_weights].reshape(self.num_actions, self.num_observations)
        biases = params[self.num_weights:].reshape(self.num_actions)

        # Set the weights and biases of the model.
        model[0].weight.data = torch.from_numpy(weights).float()
        model[0].bias.data = torch.from_numpy(biases).float()

        return model

    def population_init(self):
        """Initializes the initial neural network model population using random parameters.

        Returns:
            population: List of neural network PyTorch models.
        """

        # Initialize the population.
        population = []

        # Iterate over all individuals in the population.
        for i in range(self.population_size):
            # Initialize the individual's parameters with random values.
            individual_params = np.random.uniform(-1.0, 1.0, self.num_params)

            # Initialize the neural network model with the current individual's parameters.
            model = self.model_init(individual_params)

            # Add the model to the population.
            population.append(model)

        return population

    def get_action(self, observation, model):
        """Returns the action to take given an observation. Implements epsilon-greedy exploration with model prediction.

        Args:
            observation: Observation from the environment.
            model: Neural network PyTorch model.

        Returns:
            action: Action to take.
        """

        # Convert the observation to a PyTorch tensor.
        observation = torch.from_numpy(observation).float()

        # Get the action with the highest probability from the model.
        action = torch.argmax(model(observation)).item()

        return action

    def get_model_reward(self, model):
        """Returns the reward of the model over one episode.

        Args:
            model: Neural network PyTorch model.

        Returns:
            reward: Reward for the model over one episode.
        """

        # Reset the environment.
        observation,_ = self.env.reset()
        episode_reward = 0.0

        # Run the episode for a maximum of max_time_steps or until the episode is done.
        for _ in range(self.max_time_steps):

            # Get the action to take.
            action = self.get_action(observation, model)

            # Take the action.
            next_observation, reward, terminated, truncated,_ = self.env.step(action)

            # Update the total reward.
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

        return episode_reward

    def get_population_reward(self, population):
        """Returns the reward of the population of models over one episode.

        Args:
            population: List of neural network PyTorch models.

        Returns:
            rewards: Numpy array of rewards for the population over one episode.
        """

        # Initialize the rewards.
        rewards = np.zeros(len(population))

        # Iterate over all individuals in the population.
        for i in range(len(population)):
            # Get the reward of the current individual.
            rewards[i] = self.get_model_reward(population[i])

        return rewards

    def mutate_model(self, model):
        """Mutates the model by adding random noise to its parameters.

        Args:
            model: Neural network PyTorch model.

        Returns:
            model_mutated: Mutated neural network PyTorch model.
        """

        # Extract the model parameters.
        weights = model[0].weight.data.numpy().reshape(self.num_weights)
        biases = model[0].bias.data.numpy().reshape(self.num_biases)

        # Add random noise to the parameters.
        weights *= np.random.normal(self.mean, self.std, self.num_weights)
        biases *= np.random.normal(self.mean, self.std, self.num_biases)

        # Concatenate the weights and biases.
        params = np.concatenate((weights, biases))

        # Initialize the mutated model.
        model_mutated = self.model_init(params)

        return model_mutated

    def run(self, verbose:bool=False):
        """Train the Evolutionary genetic algorithm Neural Network agent.

        Args:
            verbose (bool, optional): whether to print progress. Defaults to False.
        """

        # Initialize the population.
        self.population = self.population_init()

        # Get the reward of the population.
        rewards = self.get_population_reward(self.population)

        # Store the initial reward history.
        self.reward_history[0] = rewards

        # Iterate over all episodes.
        for episode in range(1, self.episodes + 1):

            # Get the indices of the top individuals in the population.
            top_indices = np.argsort(rewards)[-int(self.mutation_percent * self.population_size):]

            # Get the top individuals in the population.
            top_individuals = [self.population[i] for i in top_indices]

            # Mutate the top individuals.
            mutated_individuals = [self.mutate_model(model) for model in top_individuals]

            # Get rewards of the mutated individuals.
            mutated_rewards = self.get_population_reward(mutated_individuals)

            # Get buffer total population and mutated individuals.
            population_buffer = self.population + mutated_individuals # list of models
            rewards_buffer = np.concatenate((rewards, mutated_rewards)) # numpy array of rewards

            # Select the next generation.
            next_generation_indices = np.argsort(rewards_buffer)[-self.population_size:]
            self.population = [population_buffer[i] for i in next_generation_indices] # list of models
            rewards = rewards_buffer[next_generation_indices] # numpy array of rewards

            # Store the reward history.
            self.reward_history[episode] = rewards

            # If verbose, print progress.
            if verbose and (episode % (self.episodes // 10) == 0 or episode == self.episodes):
                print(f"Episode: {episode}/{self.episodes} | Mean Reward for episode: {np.mean(rewards)}")

    def plot_results(self):
        """Plots the reward history."""

        # Plot all the reward in each episode.
        plt.figure("Reward History All Individuals")
        for i in range(len(self.reward_history)):
            plt.scatter(np.repeat(i, len(self.reward_history[i])), self.reward_history[i], s=1)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward History of all individuals in the population")

        # Plot the mean reward in each episode.
        plt.figure("Reward History Mean")
        plt.plot(np.mean(self.reward_history, axis=1))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Mean Reward History of the population")

        # Plot the best reward in each episode.
        plt.figure("Reward History Best")
        plt.plot(np.max(self.reward_history, axis=1))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Best Reward History of the population")

        # Plot the standard deviation of the reward in each episode.
        plt.figure("Reward History Std")
        plt.plot(np.std(self.reward_history, axis=1))
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Standard Deviation Reward History of the population")