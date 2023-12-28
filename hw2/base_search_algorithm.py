import csv
import numpy as np
import matplotlib.pyplot as plt

class BaseSearchAlgorithm():
    """Base class for search algorithms with common functions"""

    def _generate_dist_matrix(self, csv_path:str):
        """Read the csv file and generate the distance matrix between cities

            Args:
            csv_path (str): path to the csv file
        """

        # Read the csv file
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)

        # Convert the data to numpy array
        self._data = np.array(data, dtype=np.float32)

        # Generate the distance matrix
        self.dist_matrix = np.zeros((self._data.shape[0], self._data.shape[0]))

        for i in range(self._data.shape[0]):
            for j in range(self._data.shape[0]):
                self.dist_matrix[i, j] = np.sqrt(np.sum((self._data[i] - self._data[j])**2))


    def _get_cost(self, path:list) -> float:
        """Compute the total distance of the path"""

        cost = 0.0

        # Add the distance between each city
        for i in range(len(path) - 1):
            cost += self.dist_matrix[path[i], path[i+1]]

        # Add the distance from the last city to the first city
        cost += self.dist_matrix[path[-1], path[0]]

        return cost

    def _is_valid(self, path:list) -> bool:
        """Check if the path is valid
        Should visit all cities
        Have no duplicate cities"""

        # Check if the path has visited all cities
        if len(set(path)) != self.dist_matrix.shape[0]:
            return False

        # Check if the path has duplicate cities
        if len(path) != self.dist_matrix.shape[0]:
            return False

        return True

    def _mutate_swap(self, path:list, num_swaps:int) -> list:
        """Mutate the path by performing swapping operation and return only valid path

        Args:
            path (list): path to be mutated
            num_mutations (int): number of swapping operations
        """

        # Check if the path is valid
        if not self._is_valid(path):
            print("Invalid path passed to mutate_path()")
            return path

        new_path = path.copy()

        # Perform swapping operation
        for _ in range(num_swaps):
            valid_swap = False
            while not valid_swap:
                # Randomly select two indices
                idx1 = np.random.randint(0, len(new_path))
                idx2 = np.random.randint(0, len(new_path))

                # If the two indices are the same, skip
                if idx1 == idx2:
                    continue

                # Swap the two cities
                new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
                valid_swap = True

        return new_path

    def _select_population(self, population:list, population_cost:list, num_select:int) -> (list, list, list):
        """Select the top population from the current population

            Args:
            population (list): population to select from
            population_cost (list): cost of the population
            num_select (int): number of agents to select

            Returns:
            select_population (list): selected population
            select_population_cost (list): cost of the selected population
            idx_chosen (list): index of the selected population from the original population
        """

        # Iniialize the selected population and cost list
        select_population = []
        select_population_cost = []
        idx_chosen = []

        # Get the top population
        order = np.argsort(population_cost)

        for i in range(num_select):
            select_population.append(population[order[i]])
            select_population_cost.append(population_cost[order[i]])
            idx_chosen.append(order[i])

        return (select_population, select_population_cost, idx_chosen)

    def _plot_path(self, path:list):
        """Plots the given path using the data points"""

        plt.figure("Path")
        plt.title("Plotted Path")

        # Plot the city points
        plt.scatter(self._data[:, 0], self._data[:, 1], s=10, c='b')

        # Only plot the path if it is valid
        if self._is_valid(path):
            # Plot the path
            plt.plot(self._data[path, 0], self._data[path, 1], c='r')
            # Complete the path circle
            plt.plot([self._data[path[-1], 0], self._data[path[0], 0]], [self._data[path[-1], 1], self._data[path[0], 1]], c='r')

            # Plot the starting point as green star
            plt.scatter(self._data[path[0], 0], self._data[path[0], 1], s=100, c='g', marker='*')

        plt.xlabel("x")
        plt.ylabel("y")