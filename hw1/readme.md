# Homework 1: Neural Networks

Use your favorite programming language to implement a one hidden-layer feed forward neural network to classify products into “pass” or “fail” categories. You cannot use autograd libraries, but you can use libraries for matrix math such as numpy. The neural network classifier will assume the role of quality control for a manufacturing plant. We use a simplified dataset for this assignment.

Four data files are in the [data](/hw1/data/) directory. Each file has 400 data points, with one data point on each line where the data points have five inputs $(x_1, x_2, x_3, x_4,x_5)$ and two outputs $(y_1, y_2)$:

- $x_1, x_2, x_3, x_4, x_5, y_1, y_2$

In this case, $(x_1, x_2, x_3, x_4,x_5)$ are features of products, such as specifications for dimensions, weight, and functionality. These features have been quantified by the values $x_1$ through $x_5$. The values $y_1$ and $y_2$ denote the classification of the product (pass or fail), where $(y_1 = 0, y_2 = 1)$ indicates the product has passed, and $(y_1 = 1, y_2 = 0)$ indicates the product has failed.

- [train1.csv](/hw1/data/train1.csv) contains 400 training patterns (200 pass and 200 fail) from a simple power plant.

- [train2.csv](/hw1/data/train2.csv) contains 400 training patterns (200 pass and 200 fail) from a more complex power plant.

- [test1.csv](/hw1/data/test1.csv) and [test2.csv](/hw1/data/test2.csv) are data sets to verify the accuracy of your models for the two power plants.

Use the gradient descent algorithm to train a five input, two output (one for each class) neural network using file train1.csv. Write a report addressing the following questions (you should run experiments to support each of your answers):

1. Describe the training performance of the network:

    - How does the number of hidden units impact the results?
    - How does the training time impact the results?
    - How does the learning rate impact the results?
    - What other critical parameters impacted the results?

2. Use train2.csv to train another neural network. Answer questions 1.1-1.4 from above for the test set. What conclusions can you draw from your results? What do you think is causing the difference in performance?

Note, this is a classification problem, meaning that each data pattern $(x_1, x_2, x_3, x_4,x_5)$ belongs to one of two classes $y_1$ or $y_2$.

Consequently, use correct classification percentage (instead of MSE) to report your results. You will still use MSE to train the neural networks; you will simply report the classification percentage (or classification error) to assess the performance of the neural networks.
