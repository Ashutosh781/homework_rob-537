# Homework 2: Search and Optimization

Use your favorite programming language to implement three search methods:

1. Simulated annealing
2. An evolutionary algorithm
3. A population-based search algorithm

You will use each of the search algorithms to solve the Traveling Salesperson Problem (TSP) for the 25-city case, with city locations given in the [hw2.csv](/hw2/hw2.csv). The lines in the file are the coordinates of each city on a grid where the left bottom is $(0,0)$.

Quick tip: compute the city-to-city distance matrix, to help save on computation.

For your report:

Solve the TSP problem using the three algorithms mentioned above. Precisely describe each algorithm you used and your experimental methodology. For all your experiments, record each approach's run time, solution quality, and repeatability (solve the problem at least 10 times and provide statistical results). Also, include a plot of the final path taken for each.

Discussion questions:

- How many “solutions” did your algorithms generate during their searches?
- How many solutions are there for the TSP with 25 cities?
- What percentage of all solutions did your algorithms search through?
- What are some of the benefits and difficulties of each of your search algorithms?
- Why do these algorithms not find an optimal solution every time?
