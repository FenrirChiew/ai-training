from utils import *


class AIModel:
    def __init__(self, input_size, hidden_size, output_size, activation_function='sigmoid', dropout_rate=0.2):
        """
        Initialize the AI model with the given parameters.

        Parameters:
        input_size (int): The number of features in the input data.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output neurons.
        activation_function (str): The activation function to use ('sigmoid', 'relu', or 'leaky_relu').
        """

        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        # Initialize weights and biases
        self.hidden_input = None
        self.hidden_output = None
        self.final_input = None
        self.final_output = None

        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)

    def forward_propagation(self, inputs, training=True):
        """
        Perform forward propagation through the network.
        """
        # Input to hidden layer
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden

        # Apply the selected activation function
        if self.activation_function == 'sigmoid':
            self.hidden_output = sigmoid(self.hidden_input)
        elif self.activation_function == 'relu':
            self.hidden_output = relu(self.hidden_input)
        elif self.activation_function == 'leaky_relu':
            self.hidden_output = leaky_relu(self.hidden_input)

        # Apply dropout during training
        if training:
            self.hidden_output = self.dropout(self.hidden_output)

        # Hidden to output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = softmax(self.final_input)
        return self.final_output

    def dropout(self, layer_output):
        """
        Apply dropout to the layer output during training.
        """
        mask = np.random.rand(*layer_output.shape) > self.dropout_rate
        return layer_output * mask

    def backward_propagation(self, inputs, expected_output, learning_rate=0.1, max_gradient_norm=5.0):
        """
        Perform backward propagation and update weights and biases.
        """
        # Calculate error
        error = expected_output - self.final_output

        # Output layer gradients
        d_output = error

        # Hidden layer error
        hidden_error = d_output.dot(self.weights_hidden_output.T)

        d_hidden = None
        if self.activation_function == 'sigmoid':
            d_hidden = hidden_error * sigmoid_derivative(self.hidden_output)
        elif self.activation_function == 'relu':
            d_hidden = hidden_error * relu_derivative(self.hidden_output)
        elif self.activation_function == 'leaky_relu':
            d_hidden = hidden_error * leaky_relu_derivative(self.hidden_output)

        # Gradient clipping: Limit gradients to a maximum norm
        grad_norm = np.linalg.norm(d_hidden)
        if grad_norm > max_gradient_norm:
            # Clip gradients to prevent explosion
            d_hidden = d_hidden * (max_gradient_norm / grad_norm)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.weights_input_hidden += inputs.T.dot(d_hidden) * learning_rate
        self.bias_output += np.sum(d_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    def train(self, training_inputs, training_outputs, validation_inputs=None, validation_outputs=None,
              iterations=10000, learning_rate=0.1, max_gradient_norm=5.0):
        """
        Train the AI model with the provided data.
        Optionally, use validation data to track the validation loss and stop early if necessary.
        """
        for i in range(iterations):
            output = self.forward_propagation(training_inputs)
            self.backward_propagation(training_inputs, training_outputs, learning_rate, max_gradient_norm)

            # Track validation loss and accuracy
            if validation_inputs is not None and validation_outputs is not None:
                val_output = self.forward_propagation(validation_inputs)
                val_loss = np.mean(np.abs(validation_outputs - val_output))
                print(f'Iteration {i}, Validation Loss: {val_loss:.4f}')

            if i % 1000 == 0:
                error = np.mean(np.abs(training_outputs - output))
                print(f'Iteration {i}, Training Error: {error:.4f}')
