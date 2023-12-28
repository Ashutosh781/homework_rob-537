import os
import csv
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    """Class to define a simple feed forward neural network with 1 hidden layer"""

    def __init__(self, input_size:int, output_size:int, hidden_size:int, learning_rate:float, epochs:int,
                 activation_function:str='sigmoid', initialization:str='HeNormal' , save_flag:bool=True):
        """
        Initialize the parameters of the network

        Parameters:
            input_size: Number of features in the input
            output_size: Number of features in the output
            save_flag: Flag to save the model parameters and plots
        Hyperparameters:
            hidden_size: Number of neurons in the hidden layer
            learning_rate: Learning rate of the network
            epochs: Number of epochs to train the network
            activation_function: Activation function to use in the hidden layer, 'sigmoid' or 'relu'. Default is sigmoid
            initialization: Initialization method for the weights, 'HeNormal' or 'Normal'. Default is HeNormal
        """

        # Weights are initialized using initialization method specified
        if initialization == 'HeNormal':
            self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size) # (i,h)
            self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size) # (h,o)
        else:
            self.W1 = np.random.randn(input_size, hidden_size) # (i,h)
            self.W2 = np.random.randn(hidden_size, output_size) # (h,o)

        # Biases are initialized to 0
        self.b1 = np.zeros((1, hidden_size)) # (1,h)
        self.b2 = np.zeros((1, output_size)) # (1,o)

        # Hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        if activation_function == 'sigmoid' or activation_function == 'relu':
            self.activation_function = activation_function
        else:
            print("Invalid activation function. Using sigmoid.")
            self.activation_function = 'sigmoid'

        self.save_flag = save_flag

        # Performance metrics
        self.training_accuracy = []

    def sigmoid(self, z:float):
        """Sigmoid activation function"""

        a = 1 / (1 + np.exp(-z))
        return a

    def sigmoid_prime(self, z:float):
        """Derivative of the sigmoid activation function"""

        a = self.sigmoid(z)
        a = a * (1 - a)
        return a

    def relu(self, z:float):
        """ReLU activation function"""

        a = z * (z > 0)
        return a

    def relu_prime(self, z:float):
        """Derivative of the ReLU activation function"""

        a = 1 * (z > 0)
        return a

    def forward(self, X:np.ndarray):
        """
        Forward pass of the network

        Parameters:
            X: Input to the network - 1 data point - (1,i)
        """

        # Input to the hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1 # (1,h)

        # Activation function used for hidden layer
        if self.activation_function == 'relu':
            self.a1 = self.relu(self.z1) # (1,h)
        else:
            self.a1 = self.sigmoid(self.z1) # (1,h)

        # Hidden to the output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2 # (1,o)
        # Sigmoid activation function used for output as classification is binary
        self.a2 = self.sigmoid(self.z2) # (1,o)

        return

    def backward(self, X:np.ndarray, Y:np.ndarray):
        """
        Backward pass of the network

        Parameters:
            X: Input to the network - 1 data point - (1,i)
            Y: True output for the input X - (1,o)
        """

        # Compute gradient components from output to hidden layer
        self.dEda2 = 2 * (self.a2 - Y) # (1,o)
        self.da2dz2 = self.sigmoid_prime(self.z2) # (1,o)
        self.dEdz2 = self.dEda2 * self.da2dz2 # (1,o)
        self.dz2dW2 = self.a1 # (1,h)

        self.dEdW2 = np.dot(self.dz2dW2.T, self.dEdz2) # (h,o)
        self.dEdb2 = self.dEdz2 # (1,o)
        self.dEda1 = np.dot(self.dEdz2, self.W2.T) # (1,h)

        # Compute gradient components from hidden to input layer
        if self.activation_function == 'relu':
            self.da1dz1 = self.relu_prime(self.z1) # (1,h)
        else:
            self.da1dz1 = self.sigmoid_prime(self.z1) # (1,h)

        self.dEdz1 = self.dEda1 * self.da1dz1 # (1,h)
        self.dz1dW1 = X # (1,i)

        self.dEdW1 = np.dot(self.dz1dW1.T, self.dEdz1) # (i,h)
        self.dEdb1 = self.dEdz1 # (1,h)

        # Update weights and biases
        self.W1 -= self.learning_rate * self.dEdW1
        self.b1 -= self.learning_rate * self.dEdb1

        self.W2 -= self.learning_rate * self.dEdW2
        self.b2 -= self.learning_rate * self.dEdb2

        return

    def train(self, X:np.ndarray, Y:np.ndarray):
        """
        Train the network

        Parameters:
            X: Input to the network - All data points - (n,i)
            Y: True output for all the input X - (n,o)

        Returns:
            training_accuracy: List of accuracies for each epoch in percentage
        """

        for epoch in range(self.epochs):
            correct = 0
            for i in range(len(X)):
                # Convert the input and output to 2D arrays
                x_in = X[i].reshape(1, len(X[i]))
                y_out = Y[i].reshape(1, len(Y[i]))

                self.forward(x_in)
                self.backward(x_in, y_out)

                # Compute the number of correct predictions
                predicted = np.zeros_like(self.a2)
                predicted[np.where(self.a2 == np.max(self.a2))] = 1
                if np.array_equal(predicted, y_out):
                    correct += 1

            # Accuracy for the epoch in percentage
            self.accuracy = correct / len(X) * 100.0
            self.training_accuracy.append(self.accuracy)

        return self.training_accuracy

    def test(self, X:np.ndarray, Y:np.ndarray):
        """
        Test the network

        Parameters:
            X: Input to the network - All data points - (n,i)
            Y: True output for all the input X - (n,o)

        Returns:
            test_accuracy: Accuracy for the test data in percentage
        """

        correct = 0
        for i in range(len(X)):
            # Convert the input and output to 2D arrays
            x_in = X[i].reshape(1, len(X[i]))
            y_out = Y[i].reshape(1, len(Y[i]))

            self.forward(x_in)

            # Compute the number of correct predictions
            predicted = np.zeros_like(self.a2)
            predicted[np.where(self.a2 == np.max(self.a2))] = 1
            if np.array_equal(predicted, y_out):
                correct += 1

        # Accuracy for the test data in percentage
        self.test_accuracy = correct / len(X) * 100.0

        return self.test_accuracy

    def generate_csv(self, filename:str):
        """Save the model parameters in a csv file"""

        if not self.save_flag:
            return

        with open(filename, 'w') as f:
            writer = csv.writer(f)

            # Write the hyperparameters
            writer.writerow(['Learning rate', [self.learning_rate]])
            writer.writerow(['Epochs', [self.epochs]])
            writer.writerow(['Hidden units', [self.hidden_size]])

            # Write the weights and biases
            writer.writerow(['W1', self.W1])
            writer.writerow(['b1', self.b1])
            writer.writerow(['W2', self.W2])
            writer.writerow(['b2', self.b2])

            # Write the performance metrics
            writer.writerow(['Training accuracy', self.training_accuracy])
            writer.writerow(['Test accuracy', [self.test_accuracy]])

        print(f"Model parameters saved in {filename}")

        return

    def generate_plots(self, filename:str):
        """Generate plot for the training accuracy"""

        plt.figure()
        plt.plot(self.training_accuracy)
        plt.xlabel('Epochs')
        plt.ylabel('Training accuracy')

        plt.title(f"Accuracy with {self.hidden_size} hidden units, {self.learning_rate} learning rate and {self.epochs} epochs")
        plt.show()

        if not self.save_flag:
            return

        plt.savefig(filename)
        print(f"Plot saved in {filename}")

        return


def main(hidden_size:int=25, learning_rate:float=0.02, epochs:int=20,
         activation_function:str='sigmoid', initialization:str='HeNormal',
         save_flag:bool=False, shuffle_flag:bool=False, show_plot:bool=True, file_number:int=1):
    """
    Main function to train and test the network

    Parameters:
        hidden_size: Number of neurons in the hidden layer
        learning_rate: Learning rate of the network
        epochs: Number of epochs to train the network
        activation_function: Activation function to use in the hidden layer, 'sigmoid' or 'relu'. Default is sigmoid
        initialization: Initialization method for the weights, 'HeNormal' or 'Normal'. Default is HeNormal
        save_flag: Flag to save the model parameters and plots
        shuffle_flag: Flag to shuffle the training data
        show_plot: Flag to show the training accuracy plot
        file_number: Number of the data file to use for training and testing

    Returns:
        training_accuracy: List of accuracies for each epoch in percentage
        test_accuracy: Accuracy for the test data in percentage
    """

    # Check if the data files exist. Otherwise, use the default files
    train_file = os.path.join(os.getcwd(), f"data/train{file_number}.csv")
    test_file = os.path.join(os.getcwd(), f"data/test{file_number}.csv")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print("Invalid file number. Defaulting to 1.")
        file_number = 1
        train_file = os.path.join(os.getcwd(), f"data/train1.csv")
        test_file = os.path.join(os.getcwd(), f"data/test1.csv")

    # Read the training data
    with open(train_file, 'r') as f:
        reader = csv.reader(f)
        train_data = np.array(list(reader), dtype=np.float64)

    # Read the test data
    with open(test_file, 'r') as f:
        reader = csv.reader(f)
        test_data = np.array(list(reader), dtype=np.float64)

    # Shuffle the training data
    if shuffle_flag:
        rng = np.random.default_rng()
        rng.shuffle(train_data, axis=0)

    # Split the data into input and output
    X_train = train_data[:, :5]
    Y_train = train_data[:, 5:]
    X_test = test_data[:, :5]
    Y_test = test_data[:, 5:]

    # Input and output sizes
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]

    # Filename to save the model parameters and plots
    csv_file = os.path.join(os.getcwd(), f"results/d{file_number}_h{hidden_size}_l{learning_rate}_e{epochs}.csv")
    plot_file = os.path.join(os.getcwd(), f"results/d{file_number}_h{hidden_size}_l{learning_rate}_e{epochs}.png")

    # Create the network
    nn = NeuralNetwork(input_size, output_size, hidden_size, learning_rate, epochs,
                       activation_function, initialization, save_flag)

    # Train the network
    training_accuracy = nn.train(X_train, Y_train)

    # Test the network
    test_accuracy = nn.test(X_test, Y_test)

    # Save the model parameters
    nn.generate_csv(csv_file)

    # Generate the training accuracy plot
    if show_plot:
        nn.generate_plots(plot_file)

    return training_accuracy, test_accuracy


if __name__ == '__main__':
    main()