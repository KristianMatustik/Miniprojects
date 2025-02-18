import math
import numpy as np
import pandas as pd
import pickle

eps=1e-6

class NeuralNetwork:
    class Layer:
        def __init__(self, activation_function, shape_input=None, shape_output=None):
            self.shape_input = shape_input
            self.shape_output = shape_output
            self.activation = activation_function
            if activation_function == NeuralNetwork.ReLU:
                self.activation_derivative = NeuralNetwork.ReLU_derivative
            elif activation_function == NeuralNetwork.sigmoid:
                self.activation_derivative = NeuralNetwork.sigmoid_derivative
            elif activation_function == NeuralNetwork.tanh:  
                self.activation_derivative = NeuralNetwork.tanh_derivative
            elif activation_function == NeuralNetwork.softmax:
                self.activation_derivative = NeuralNetwork.softmax_derivative
            elif activation_function == None:
                self.activation_derivative = None
            else:
                raise ValueError("Invalid activation function")

        def forward(self, input):
            pass

        def backward(self, derivative):
            pass

    class Layer_FullyConnected(Layer):
        def __init__(self, input_size, output_size, activation_function):
            super().__init__(activation_function, (input_size,), (output_size,))
            self.weights = np.random.randn(output_size, input_size)*np.sqrt(2 / input_size) # later edit for different activation functions than ReLU
            self.biases = np.zeros(output_size)
            self.weights_gradient = np.zeros((output_size, input_size))
            self.biases_gradient = np.zeros(output_size)

        def forward(self, input):
            self.input = input.flatten()
            self.weighted_sum = np.dot(self.weights, self.input) + self.biases
            self.output = self.activation(self.weighted_sum)
            return self.output
        
        def backward(self, cost_gradient_wrt_output, batch_size):
            if self.activation == NeuralNetwork.softmax:
                activation_derivative = self.activation_derivative(self.output)
                cost_gradient_wrt_wsum = np.dot(activation_derivative, cost_gradient_wrt_output)
            else:
                cost_gradient_wrt_wsum = cost_gradient_wrt_output * self.activation_derivative(self.output)            
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
            input_shape = (self.num_channels, self.num_rows, self.num_columns)
            if padding_type == 0 or padding_type == 1:
                self.num_rows += kernel_size - 1
                self.num_columns += kernel_size - 1
            self.shape_padded_input = (self.num_channels, self.num_rows, self.num_columns)
            output_shape = (self.num_kernels_per_channel, 
                               (self.num_rows - self.kernel_size + 1) // self.kernel_stride, 
                               (self.num_columns - self.kernel_size + 1) // self.kernel_stride)
            super().__init__(activation_function, input_shape, output_shape)

        def forward(self, input):
            self.input = input.reshape(self.shape_input)
            if self.padding_type == 0:
                self.input = np.pad(self.input, ((0, 0), (self.kernel_size//2, self.kernel_size//2 + self.kernel_size % 2 - 1), (self.kernel_size//2, self.kernel_size//2 + self.kernel_size % 2 - 1)), 'constant')
            elif self.padding_type == 1:
                self.input = np.pad(self.input, ((0, 0), (self.kernel_size//2, self.kernel_size//2 + self.kernel_size % 2 - 1), (self.kernel_size//2, self.kernel_size//2 + self.kernel_size % 2 - 1)), 'edge')
            self.output = np.zeros(self.shape_output)
            for k in range(self.num_kernels_per_channel):
                for i in range(0, self.num_rows - self.kernel_size + 1, self.kernel_stride):
                    for j in range(0, self.num_columns - self.kernel_size + 1, self.kernel_stride):
                        region = self.input[:, i:i+self.kernel_size, j:j+self.kernel_size]
                        self.output[k, i//self.kernel_stride, j//self.kernel_stride] = np.sum(region * self.kernels[:, k]) + self.biases[k]
            self.output = self.activation(self.output)
            return self.output

        def backward(self, cost_derivative_wrt_output, batch_size):
            cost_derivative_wrt_output = cost_derivative_wrt_output.reshape(self.shape_output)
            cost_derivative_wrt_wsum = cost_derivative_wrt_output * self.activation_derivative(self.output)
            for k in range(self.num_kernels_per_channel):
                for i in range(0, self.num_rows - self.kernel_size + 1, self.kernel_stride):
                    for j in range(0, self.num_columns - self.kernel_size + 1, self.kernel_stride):
                        region = self.input[:, i:i+self.kernel_size, j:j+self.kernel_size]
                        self.kernels_gradient[:, k] += region * cost_derivative_wrt_wsum[k, i//self.kernel_stride, j//self.kernel_stride]/batch_size
                        self.biases_gradient[k] += cost_derivative_wrt_wsum[k, i//self.kernel_stride, j//self.kernel_stride]/batch_size
            cost_derivative_wrt_input = np.zeros(self.shape_padded_input)
            for k in range(self.num_kernels_per_channel):
                for i in range(0, self.num_rows - self.kernel_size + 1, self.kernel_stride):
                    for j in range(0, self.num_columns - self.kernel_size + 1, self.kernel_stride):
                        region = self.input[:, i:i+self.kernel_size, j:j+self.kernel_size]
                        cost_derivative_wrt_input[:, i:i+self.kernel_size, j:j+self.kernel_size] += self.kernels[:, k] * cost_derivative_wrt_wsum[k, i//self.kernel_stride, j//self.kernel_stride]
            if self.padding_type != -1:
                cost_derivative_wrt_input = cost_derivative_wrt_input[:, self.kernel_size//2:self.num_rows-self.kernel_size//2, self.kernel_size//2:self.num_columns-self.kernel_size//2]
            return cost_derivative_wrt_input
        
        def update(self, learning_rate, momentum):
            self.kernels -= learning_rate * self.kernels_gradient
            self.biases -= learning_rate * self.biases_gradient
            self.kernels_gradient *= momentum
            self.biases_gradient *= momentum
    
    class Layer_Pooling(Layer):
        def __init__(self, input_shape, pool_size, pool_stride=-1, pool_type="max"):
            self.num_channels = input_shape[0]
            self.num_rows = input_shape[1]
            self.num_columns = input_shape[2]
            self.pool_size = pool_size
            self.pool_stride = pool_stride if pool_stride != -1 else pool_size
            self.pool_type = pool_type
            output_shape = (self.num_channels, (self.num_rows - self.pool_size)//self.pool_stride + 1, (self.num_columns - self.pool_size)//self.pool_stride + 1)
            super().__init__(None, input_shape, output_shape)

        def forward(self, input):
            self.input = input.reshape(self.num_channels, self.num_rows, self.num_columns)
            self.output = np.zeros(self.shape_output)
            for k in range(self.num_channels):
                for i in range(0, self.num_rows - self.pool_size + 1, self.pool_stride):
                    for j in range(0, self.num_columns - self.pool_size + 1, self.pool_stride):
                        region = self.input[k, i:i+self.pool_size, j:j+self.pool_size]
                        if self.pool_type == "max":
                            self.output[k, i//self.pool_stride, j//self.pool_stride] = np.max(region)
                        elif self.pool_type == "avg":
                            self.output[k, i//self.pool_stride, j//self.pool_stride] = np.mean(region)
            return self.output

        def backward(self, cost_derivative_wrt_output, batch_size):
            cost_derivative_wrt_output = cost_derivative_wrt_output.reshape(self.shape_output)
            cost_derivative_wrt_input = np.zeros(self.shape_input)
            for k in range(self.num_channels):
                for i in range(0, self.num_rows - self.pool_size + 1, self.pool_stride):
                    for j in range(0, self.num_columns - self.pool_size + 1, self.pool_stride):
                        region = self.input[k, i:i+self.pool_size, j:j+self.pool_size]
                        if self.pool_type == "max":
                            max_index = np.unravel_index(np.argmax(region, axis=None), region.shape)
                            cost_derivative_wrt_input[k, i+max_index[0], j+max_index[1]] = cost_derivative_wrt_output[k, i//self.pool_stride, j//self.pool_stride]
                        elif self.pool_type == "avg":
                            cost_derivative_wrt_input[k, i:i+self.pool_size, j:j+self.pool_size] += cost_derivative_wrt_output[k, i//self.pool_stride, j//self.pool_stride]/(self.pool_size**2)
            return cost_derivative_wrt_input
        
        def update(self, learning_rate, momentum):
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
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    @staticmethod
    def softmax_derivative(x):
        return np.diag(x) - np.outer(x, x)
    
    # Cost functions
    @staticmethod
    def cost_mse(prediction, target):
        return np.sum((prediction - target)**2) / 2
    @staticmethod
    def cost_mse_derivative(prediction, target): 
        return prediction - target
    
    @staticmethod
    def cost_crossEntropy(prediction, target):
        return -np.sum(target * np.log(prediction+eps))
    @staticmethod
    def cost_crossEntropy_derivative(prediction, target):
        return -target * (1 / (prediction+eps))
    
    @staticmethod
    def cost_binaryCrossEntropy(prediction, target):
        return -np.sum(target * np.log(prediction+eps) + (1 - target) * np.log(1 - prediction+eps))
    @staticmethod
    def cost_binaryCrossEntropy_derivative(prediction, target):
        return -target / (prediction+eps) + (1 - target) / (1 - prediction+eps)

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
        if n_input == -1:
            n_input = np.prod(self.layers[-1].shape_output)      
        self.layers.append(self.Layer_FullyConnected(n_input, n_output, activation))

    def add_Layer_Convolutional(self, input_shape, num_kernels_per_channel, kernel_size, kernel_stride, padding_type, activation):
        if input_shape == -1:
            input_shape = self.layers[-1].shape_output
        self.layers.append(self.Layer_Convolutional(input_shape, num_kernels_per_channel, kernel_size, kernel_stride, padding_type, activation))

    def add_Layer_Pooling(self, input_shape, pool_size, pool_stride=-1, pool_type="max"):
        if input_shape == -1:
            input_shape = self.layers[-1].shape_output
        self.layers.append(self.Layer_Pooling(input_shape, pool_size, pool_stride, pool_type))

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

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
            cost_derivative_wrt_activation = cost_function_derivative(prediction, target)
            for layer in reversed(self.layers):
                cost_derivative_wrt_activation = layer.backward(cost_derivative_wrt_activation, batch_size)

        for layer in self.layers:
            layer.update(learning_rate, momentum)

        return cost/batch_size, accuracy/batch_size

    def train(self, inputs, targets, batch_size, epochs, learning_rate, momentum, cost_function, printEpochs=False):
        inputs = np.array(inputs)
        targets = np.array(targets)
        n_examples = inputs.shape[0]
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
            if targets.max() == 1:
                onehot_targets = targets
            else:
                onehot_targets = np.zeros((n_examples, targets.max()+1))
                onehot_targets[np.arange(n_examples), targets] = 1
            cost = 0
            accuracy = 0
            for i in range(n_batches):
                batch_inputs = inputs[i*batch_size:(i+1)*batch_size]
                batch_targets = onehot_targets[i*batch_size:(i+1)*batch_size]
                c, ac = self.trainBatch(batch_inputs, batch_targets, learning_rate, momentum, cost_function, cost_function_derivative)
                cost += c
                accuracy += ac
                if printEpochs:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{n_batches}, Cost: {c}, Accuracy: {ac}")
            print(f"Epoch {epoch+1}/{epochs}, Cost: {cost/n_batches}, Accuracy: {accuracy/n_batches}")