import os
import numpy as np
import matplotlib.pyplot as plt

from base_search_algorithm import BaseSearchAlgorithm

class EvolutionarySearch(BaseSearchAlgorithm):
    """Evolutionary Search algorithm for Traveling Salesman Problem"""

    def __init__(self, csv_file:str, population_size:int=100, iterations:int=1000, num_swaps:int=10, mutation_size:int=25):
        """Initialize the evolutionary search algorithm class

            Args:
            csv_file (str): path to the csv file containing the cities' coordinates
            population_size (int): size of the population
            iterations (int): maximum number of iterations
            num_swaps (int): number of swaps to perform to generate new path
            mutation_size (int): number of paths to mutate
        """

        # Generate the distance matrix
        self._generate_dist_matrix(csv_file)

        # Initialize the parameters
        self.population_size = population_size
        self.iterations = iterations
        self.num_swaps = num_swaps
        self.mutation_size = mutation_size

        # Initialize the population
        self.population = []
        for i in range(self.population_size):
            path = np.arange(self.dist_matrix.shape[0])
            np.random.shuffle(path)
            self.population.append(list(path))

        # Initialize the cost history
        self.population_cost = []
        self.generational_cost = []

        # Get the cost of the initial population
        for path in self.population:
            self.population_cost.append(self._get_cost(path))

        # Store the initial generation cost
        self.generational_cost.append(self.population_cost)

        # Initialize the best path and best cost
        self.best_path = self.population[np.argmin(self.population_cost)]
        self.best_cost = np.min(self.population_cost)

        # Iniialize the top mutation_size population and cost
        self.top_population = []
        self.top_population_cost = []

    def algorithm(self, verbose:bool=True):
        """Evolutionary Search algorithm"""

        for i in range(self.iterations):
            # Select the top population from current population for mutation
            self.top_population, self.top_population_cost,_ = self._select_population(self.population, self.population_cost, self.mutation_size)

            # Mutate the top population
            mutated_population = []
            mutated_population_cost = []
            for path in self.top_population:
                new_path = self._mutate_swap(path, self.num_swaps)
                mutated_population.append(new_path)
                mutated_population_cost.append(self._get_cost(new_path))

            # Get the new population
            population_buffer = self.population + mutated_population
            population_cost_buffer = self.population_cost + mutated_population_cost

            # Select the next generation
            self.population, self.population_cost,_ = self._select_population(population_buffer, population_cost_buffer, self.population_size)

            # Store the cost history
            self.generational_cost.append(self.population_cost)

            # Update the best path and best cost
            if np.min(self.population_cost) < self.best_cost:
                self.best_path = self.population[np.argmin(self.population_cost)]
                self.best_cost = np.min(self.population_cost)

            # Reduce the swap number every 1/20 of the iterations
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
        """Plot the cost history of the algorithm"""

        plt.figure("Evolutionary Search")

        # Plot cost of each member of the population in each generation
        plt.subplot(1, 2, 1)
        for i in range(len(self.generational_cost)):
            plt.scatter(np.repeat(i, self.population_size), self.generational_cost[i], s=1)
        plt.xlabel("Generation")
        plt.ylabel("Path length")
        plt.title("Cost of each member of the population")

        # Plot the best cost in each generation
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(self.generational_cost)), np.min(self.generational_cost, axis=1))
        plt.xlabel("Generation")
        plt.ylabel("Path length")
        plt.title("Best cost in each generation")

        plt.show()


if __name__ == "__main__":

    # Get the path to the csv file
    csv_file = os.path.join(os.getcwd(), "hw2.csv")

    # Parameters
    population_size = 100
    iterations = 1000
    num_swaps = 10
    mutation_size = 90

    # Initialize the evolutionary search algorithm
    es = EvolutionarySearch(csv_file, population_size, iterations, num_swaps, mutation_size)

    # Run the algorithm
    es.algorithm()

    # Plot the cost history
    es.plot_cost_history()