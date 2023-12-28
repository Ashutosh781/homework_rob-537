import os
import numpy as np
import matplotlib.pyplot as plt

from base_search_algorithm import BaseSearchAlgorithm

class StochasticBeamSearch(BaseSearchAlgorithm):
    """Stochastic Beam Search algorithm for Traveling Salesman Problem"""

    def __init__(self, csv_file:str, beam_width:int=25, iterations:int=1000, num_swaps:int=10, stochastic_factor:float=0.75, cooling_rate:float=0.9):
        """Initializes Stochastic Beam Search algorithm

            Args:
            csv_file (str): path to the csv file containing the cities' coordinates
            beam_width (int): beam width
            iterations (int): maximum number of iterations
            num_swaps (int): number of swaps to perform to generate new path
            stochastic_factor (float): factor to determine the number of paths to select randomly from the population with probability
            cooling_rate (float): cooling rate for the stochastic factor
        """

        # Generate the distance matrix
        self._generate_dist_matrix(csv_file)

        # Initialize the parameters
        self.beam_width = beam_width
        self.iterations = iterations
        self.num_swaps = num_swaps
        self.stochastic_factor = stochastic_factor
        self.cooling_rate = cooling_rate

        # Initalize the population
        self.population = []
        for i in range(self.beam_width):
            path = np.arange(self.dist_matrix.shape[0])
            np.random.shuffle(path)
            self.population.append(list(path))

        # Initialize the cost history
        self.population_cost = []
        self.iterational_cost = []

        # Get the cost of the initial population
        for path in self.population:
            self.population_cost.append(self._get_cost(path))

        # Store the initial population cost
        self.iterational_cost.append(self.population_cost)

        # Initialize the best path and best cost
        self.best_path = self.population[np.argmin(self.population_cost)]
        self.best_cost = np.min(self.population_cost)

    def _get_probabilities(self, population_cost:list) -> list:
        """Returns the probabilities of each member of the population

            Args:
            population_cost (list): cost of each member of the population

            Returns:
            list: probabilities of each member of the population
        """

        # Get the probabilities
        probabilities = []
        for cost in population_cost:
            probabilities.append(1 / cost)

        # Normalize the probabilities
        probabilities = probabilities / np.sum(probabilities)

        return probabilities

    def algorithm(self, verbose:bool=True):
        """Stochastic Beam Search algorithm"""

        for i in range(self.iterations):
            # Initialize the new population and new population cost
            new_population = []
            new_population_cost = []

            # Generate the new population
            for path in self.population:
                new_path = self._mutate_swap(path, self.num_swaps)
                new_population.append(new_path)
                new_population_cost.append(self._get_cost(new_path))

            # Get the new population
            population_buffer = self.population + new_population
            population_cost_buffer = self.population_cost + new_population_cost

            # Select the population from the buffer population depending on the stochastic factor
            num_random = int(self.stochastic_factor * self.beam_width)
            num_top = self.beam_width - num_random

            # Get the top population
            self.population, self.population_cost, idx_top = self._select_population(population_buffer, population_cost_buffer, num_top)

            # Remove the top population from the buffer population
            population_buffer = [population_buffer[idx] for idx in range(len(population_buffer)) if idx not in idx_top]
            population_cost_buffer = [population_cost_buffer[idx] for idx in range(len(population_cost_buffer)) if idx not in idx_top]

            # Get the random population
            probabilities = self._get_probabilities(population_cost_buffer)
            idx_random = np.random.choice(a=len(population_buffer), size=num_random, replace=False, p=probabilities)
            self.population += [population_buffer[idx] for idx in idx_random]
            self.population_cost += [population_cost_buffer[idx] for idx in idx_random]

            # Store the iteration cost
            self.iterational_cost.append(self.population_cost)

            # Update the best path and best cost
            if np.min(self.population_cost) < self.best_cost:
                self.best_path = self.population[np.argmin(self.population_cost)]
                self.best_cost = np.min(self.population_cost)

            # Update the stochastic factor every 1/100 of the iterations
            if i % (self.iterations // 100) == 0 and i != 0:
                self.stochastic_factor *= self.cooling_rate

            # Reduce the number of swaps for every 1/20th of the iterations
            if i % (self.iterations // 20) == 0 and i != 0 and self.num_swaps > 1:
                self.num_swaps -= 1

            # Print the progress
            if verbose:
                if i % (self.iterations // 10) == 0 or i == self.iterations - 1:
                    print(f"Iteration {i+1}/{self.iterations} - Cost: {self.best_cost:.5f}")

        if verbose:
            # Print the best path and best cost
            print(f"Best path: {self.best_path}")
            print(f"Best cost: {self.best_cost}")

            # Plot the best path
            self._plot_path(self.best_path)

    def plot_cost_history(self):
        """Plots the cost history of the algorithm"""

        plt.figure("Beam Search")

        # Plot cost of each member of the population in each iteration
        plt.subplot(1, 2, 1)
        for i in range(len(self.iterational_cost)):
            plt.scatter(np.repeat(i, self.beam_width), self.iterational_cost[i], s=1)

        plt.xlabel("Iteration")
        plt.ylabel("Path length")
        plt.title("Cost of each member of the population in each iteration")

        # Plot the best cost in each iteration
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(self.iterational_cost)), np.min(self.iterational_cost, axis=1))
        plt.xlabel("Iteration")
        plt.ylabel("Path length")
        plt.title("Best cost in each iteration")

        plt.show()


if __name__ == "__main__":

    # Get the path to the csv file
    csv_file = os.path.join(os.getcwd(), "hw2.csv")

    # Parameters
    beam_width = 25
    iterations = 1000
    num_swaps = 10
    stochastic_factor = 1.0
    cooling_rate = 0.9

    # Initialize the beam search algorithm
    sbs = StochasticBeamSearch(csv_file, beam_width, iterations, num_swaps, stochastic_factor, cooling_rate)

    # Run the algorithm
    sbs.algorithm()

    # Plot the cost history
    sbs.plot_cost_history()