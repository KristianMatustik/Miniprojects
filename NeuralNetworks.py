import numpy as np
import pandas as pd
import pickle

class NeuralNetwork:
    class Layer:
        def __init__(self, activation_function):
            self.activation = activation_function
            if activation_function == NeuralNetwork.ReLU:
                self.activation_derivative = NeuralNetwork.ReLU_derivative
            elif activation_function == NeuralNetwork.sigmoid:
                self.activation_derivative = NeuralNetwork.sigmoid_derivative
            elif activation_function == NeuralNetwork.tanh:  
                self.activation_derivative = NeuralNetwork.tanh_derivative
            elif activation_function == NeuralNetwork.softmax:
                self.activation_derivative = NeuralNetwork.softmax_derivative
            else:
                raise ValueError("Invalid activation function")

        def forward(self, input):
            pass

        def backward(self, derivative):
            pass

    class Layer_FullyConnected(Layer):
        def __init__(self, input_size, output_size, activation_function):
            super().__init__(activation_function)
            self.input_size = input_size
            self.output_size = output_size
            self.weights = np.random.randn(output_size, input_size)*np.sqrt(2 / input_size) # later edit for different activation functions than ReLU
            self.biases = np.zeros(output_size)
            self.weights_gradient = np.zeros((output_size, input_size))
            self.biases_gradient = np.zeros(output_size)

        def forward(self, input):
            self.input = input.flatten()
            self.weighted_sum = np.dot(self.weights, self.input) + self.biases
            self.output = self.activation(self.weighted_sum)
            return self.output
        
        def backward(self, cost_gradient_wrt_activation, batch_size):
            cost_gradient_wrt_wsum = cost_gradient_wrt_activation * self.activation_derivative(self.output)
            self.weights_gradient += np.outer(cost_gradient_wrt_wsum, self.input)/batch_size
            self.biases_gradient += cost_gradient_wrt_wsum/batch_size
            return np.dot(self.weights.T, cost_gradient_wrt_wsum)
        
        def update(self, learning_rate, momentum):
            self.weights = self.weights - learning_rate * self.weights_gradient
            self.biases = self.biases - learning_rate * self.biases_gradient
            self.weights_gradient = momentum * self.weights_gradient
            self.biases_gradient = momentum * self.biases_gradient
        
    class Layer_Convolutional(Layer):
        def __init__(self, input_shape, num_kernels_per_channel, kernel_size, kernel_stride, padding_type, activation_function):
            # later generalize, allow more specification: x/y for kernel_size and kernel_stride, add kernel_dilation (in x/y)
            super().__init__(activation_function)
            self.dim_input = input_shape
            self.num_channels = input_shape[0]
            self.num_rows = input_shape[1]
            self.num_columns = input_shape[2]
            self.num_kernels_per_channel = num_kernels_per_channel
            self.kernel_size = kernel_size
            self.kernel_stride = kernel_stride
            self.padding_type = padding_type           
            self.kernels = np.random.randn(self.num_channels, num_kernels_per_channel, kernel_size, kernel_size)
            self.biases = np.zeros(num_kernels_per_channel)
            self.kernels_gradient = np.zeros((self.num_channels, num_kernels_per_channel, kernel_size, kernel_size))
            self.biases_gradient = np.zeros(num_kernels_per_channel)

        def forward(self, inputs):
            self.inputs = inputs.reshape(self.num_channels, self.num_rows, self.num_columns)
            if self.padding_type == 0:
                self.inputs = np.pad(self.inputs, ((0, 0), (self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)), 'constant')
            elif self.padding_type == 1:
                self.inputs = np.pad(self.inputs, ((0, 0), (self.kernel_size//2, self.kernel_size//2), (self.kernel_size//2, self.kernel_size//2)), 'edge')
            self.dim_input = self.inputs.shape
            self.num_rows = self.dim_input[1]
            self.num_columns = self.dim_input[2]
            self.output_dim = (self.num_kernels_per_channel, (self.num_rows - self.kernel_size + 1)//self.kernel_stride, (self.num_columns - self.kernel_size + 1)//self.kernel_stride)
            self.outputs = np.zeros(self.output_dim)
            for k in range(self.num_kernels_per_channel):
                for i in range(0, self.num_rows - self.kernel_size + 1, self.kernel_stride):
                    for j in range(0, self.num_columns - self.kernel_size + 1, self.kernel_stride):
                        region = self.inputs[:, i:i+self.kernel_size, j:j+self.kernel_size]
                        self.outputs[k, i//self.kernel_stride, j//self.kernel_stride] = np.sum(region * self.kernels[:, k]) + self.biases[k]
            self.outputs = self.activation(self.outputs)
            return self.outputs

        def backward(self, cost_derivative_wrt_activation, batch_size):
            pass

        def update(self, learning_rate, momentum):
            self.kernels -= learning_rate * self.kernels_gradient
            self.biases -= learning_rate * self.biases_gradient
            self.kernels_gradient *= momentum
            self.biases_gradient *= momentum
    
    class Layer_Pooling(Layer):
        pass

    # Activation functions
    @staticmethod
    def ReLU(x):
        return np.maximum(0, x) 
    @staticmethod
    def ReLU_derivative(x):
        return (x > 0).astype(int)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    @staticmethod
    def tanh_derivative(x):
        return 1 - x**2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    @staticmethod
    def softmax_derivative(x):
        return x * (1 - x)
    
    # Cost functions
    @staticmethod
    def cost_mse(prediction, target):
        return np.sum((prediction - target)**2) / 2
    @staticmethod
    def cost_mse_derivative(prediction, target): 
        return prediction - target
    
    @staticmethod
    def cost_crossEntropy(prediction, target):
        return -np.sum(target * np.log(prediction))
    @staticmethod
    def cost_crossEntropy_derivative(prediction, target):
        return prediction - target
    
    @staticmethod
    def cost_binaryCrossEntropy(prediction, target):
        return -np.sum(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
    @staticmethod
    def cost_binaryCrossEntropy_derivative(prediction, target):
        return prediction - target

    # Methods
    def __init__(self):
        self.layers = []
        pass

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def add_Layer_FullyConnected(self, n_input, n_output, activation):
        self.layers.append(self.Layer_FullyConnected(n_input, n_output, activation))

    def add_Layer_Convolutional(self, input_shape, num_kernels_per_channel, kernel_size, kernel_stride, padding_type, activation):
        self.layers.append(self.Layer_Convolutional(input_shape, num_kernels_per_channel, kernel_size, kernel_stride, padding_type, activation))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def trainBatch(self, inputs, targets, learning_rate, momentum, cost_function, cost_function_derivative):
        batch_size = inputs.shape[0]
        cost = 0
        accuracy = 0
        for i in range(batch_size):
            sample = inputs[i]
            target = targets[i]
            prediction = self.forward(sample)
            cost += cost_function(prediction, target)
            accuracy += np.argmax(prediction) == np.argmax(target)
            derivative = cost_function_derivative(prediction, target)
            for layer in reversed(self.layers):
                derivative = layer.backward(derivative, batch_size)

        for layer in self.layers:
            layer.update(learning_rate, momentum)

        return cost/batch_size, accuracy/batch_size

    def train(self, inputs, targets, batch_size, epochs, learning_rate, momentum, cost_function):
        inputs = np.array(inputs)
        targets = np.array(targets)
        n_examples, n_input = inputs.shape
        n_batches = n_examples // batch_size

        if cost_function == NeuralNetwork.cost_mse:
            cost_function_derivative = NeuralNetwork.cost_mse_derivative
        elif cost_function == NeuralNetwork.cost_crossEntropy:
            cost_function_derivative = NeuralNetwork.cost_crossEntropy_derivative
        elif cost_function == NeuralNetwork.cost_binaryCrossEntropy:
            cost_function_derivative = NeuralNetwork.cost_binaryCrossEntropy_derivative
        else:
            raise ValueError("Invalid cost function")
        
        for epoch in range(epochs):
            permutation = np.random.permutation(n_examples)
            inputs = inputs[permutation]
            targets = targets[permutation]
            onehot_targets = np.zeros((targets.size, targets.max()+1))
            onehot_targets[np.arange(targets.size), targets] = 1
            cost = 0
            accuracy = 0
            for i in range(n_batches):
                batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
                batch_targets = onehot_targets[i*batch_size:(i+1)*batch_size]
                c, ac = self.trainBatch(batch_inputs, batch_targets, learning_rate, momentum, cost_function, cost_function_derivative)
                cost += c
                accuracy += ac
                # print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{n_batches}, Cost: {c}, Accuracy: {ac}")
            print(f"Epoch {epoch+1}/{epochs}, Cost: {cost/n_batches}, Accuracy: {accuracy/n_batches}")


# testing
l = NeuralNetwork.Layer_Convolutional((3,28,28), 2, 3, 1, 1, NeuralNetwork.ReLU)
l.forward(np.random.randn(3,28,28))