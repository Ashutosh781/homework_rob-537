import os
import numpy as np
import matplotlib.pyplot as plt

from base_search_algorithm import BaseSearchAlgorithm

class SimulatedAnnealing(BaseSearchAlgorithm):
    """Simulated Annealing algorithm for Traveling Salesman Problem"""

    def __init__(self, csv_path:str, iterations:int=1000, temperature:float=0.9, temperature_decay:float=0.9, num_swaps:int=10):
        """Initializes the Simulated annealing algorithm class

            Args:
            csv_path (str): path to the csv file containing the cities' coordinates
            iterations (int): maximum number of iterations
            temperature (float): initial temperature parameter for probability calculation
            temperature_decay (float): decay factor for temperature update
            num_swaps (int): number of swaps to perform to generate new path
        """

        # Generate the distance matrix
        self._generate_dist_matrix(csv_path)

        # Initialize the parameters
        self.iterations = iterations
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.num_swaps = num_swaps

        # Initialize the path
        self.path = np.arange(self.dist_matrix.shape[0])
        np.random.shuffle(self.path)
        self.path = list(self.path)

        # Initialize the cost
        self.cost = self._get_cost(self.path)

        # Initialize the best path and best cost
        self.best_path = self.path
        self.best_cost = self.cost

        # Initialize the cost history
        self.cost_history = [self.cost]

    def _get_probability(self, new_cost:float) -> float:
        """Calculate the probability of accepting the new path

            Args:
            new_cost (float): cost of the new path
        """

        # Calculate the probability
        probability = np.exp(-np.divide((new_cost - self.cost), self.temperature))

        return probability

    def algorithm(self, verbose:bool=True):
        """Simulated Annealing algorithm"""

        for i in range(self.iterations):
            # Generate the new path
            new_path = self._mutate_swap(self.path, self.num_swaps)

            # Calculate the new cost
            new_cost = self._get_cost(new_path)

            # Accept the new path if it has lower cost
            if new_cost < self.cost:
                self.path = new_path
                self.cost = new_cost

                # Update the best path and best cost
                if self.cost < self.best_cost:
                    self.best_path = self.path
                    self.best_cost = self.cost

            # Accept the new path if it has higher cost with a probability
            else:
                probability = self._get_probability(new_cost)

                # Select with a probability otherwise reject the new path and keep the old path
                if np.random.rand() < probability:
                    self.path = new_path
                    self.cost = new_cost

            # Update the temperature
            self.temperature *= self.temperature_decay

            # Update the cost history
            self.cost_history.append(self.cost)

            # Update number of swaps every 1/20 of the iterations
            if i % (self.iterations // 20) == 0 and i != 0 and self.num_swaps > 1:
                self.num_swaps -= 1

            # Print the progress
            if verbose:
                if i % (self.iterations // 10) == 0 or i == self.iterations - 1:
                    print(f"Iteration {i+1}/{self.iterations} - Cost: {self.cost:.5f} - Temperature: {self.temperature}")

        if verbose:
            # Print the best path and best cost
            print(f"Best path: {self.best_path}")
            print(f"Best cost: {self.best_cost}")

            # Plot the best path
            self._plot_path(self.best_path)

    def plot_cost_history(self):
        """Plot the cost history of the algorithm"""

        plt.figure("Simulated Annealing")
        plt.plot(np.arange(self.iterations + 1), self.cost_history)
        plt.title("Cost History")
        plt.xlabel("Iteration")
        plt.ylabel("Path length")
        plt.show()


if __name__ == "__main__":

    # csv file path
    csv_path = os.path.join(os.getcwd(), "hw2.csv")

    # Parameters
    iterations = 5000
    temperature = 10.0
    temperature_decay = 0.9
    num_swaps = 10

    # Initialize the algorithm
    sa = SimulatedAnnealing(csv_path, iterations, temperature, temperature_decay, num_swaps)

    # Run the algorithm
    sa.algorithm()

    # Plot the cost history
    sa.plot_cost_history()