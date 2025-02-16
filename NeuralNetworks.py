import numpy as np
import pandas as pd
import pickle

class NeuralNetwork:
    class Layer:
        def __init__(self, n_input, n_output, activation):
            self.n_input = n_input
            self.n_output = n_output
            self.activation = activation
            if activation == NeuralNetwork.ReLU:
                self.activation_derivative = NeuralNetwork.ReLU_derivative
            elif activation == NeuralNetwork.sigmoid:
                self.activation_derivative = NeuralNetwork.sigmoid_derivative
            elif activation == NeuralNetwork.tanh:  
                self.activation_derivative = NeuralNetwork.tanh_derivative
            elif activation == NeuralNetwork.softmax:
                self.activation_derivative = NeuralNetwork.softmax_derivative
            else:
                raise ValueError("Invalid activation function")

        def forward(self, inputs):
            pass

        def backward(self, derivative):
            pass

    class Layer_FullyConnected(Layer):
        def __init__(self, n_input, n_output, activation):
            super().__init__(n_input, n_output, activation)
            # later edit init for different activation functions
            self.weights = np.random.randn(n_output, n_input)*np.sqrt(2 / n_input)
            self.biases = np.zeros(n_output)
            self.weights_gradient = np.zeros((n_output, n_input))
            self.biases_gradient = np.zeros(n_output)

        def forward(self, inputs):
            self.inputs = inputs
            self.w_sum =np.dot(self.weights, inputs) + self.biases
            self.outputs = self.activation(self.w_sum)
            return self.outputs
        
        def backward(self, derivative, batch_size):
            derivative = derivative * self.activation_derivative(self.outputs)
            self.weights_gradient += np.outer(derivative, self.inputs)/batch_size
            self.biases_gradient += derivative/batch_size
            return np.dot(self.weights.T, derivative)
        
        def update(self, learning_rate, momentum):
            self.weights = self.weights - learning_rate * self.weights_gradient
            self.biases = self.biases - learning_rate * self.biases_gradient
            self.weights_gradient = momentum * self.weights_gradient
            self.biases_gradient = momentum * self.biases_gradient
        

    # will hopefully implement later, tried something for start, but not sure if its the right way, need to look into it
    # class Layer_Convolutional(Layer):
    #     def __init__(self, n_input, n_columns, kernel_count, kernel_size, kernel_stride, activation, activation_derivative):
    #         self.kernel_count = kernel_count
    #         self.kernel_size = kernel_size
    #         self.kernel_stride = kernel_stride
    #         self.n_input_columns = n_columns
    #         self.n_input_rows = n_input // n_columns
    #         self.n_output_columns = (self.n_input_columns - self.kernel_size) // self.kernel_stride + 1
    #         self.n_output_rows = (self.n_input_rows - self.kernel_size) // self.kernel_stride + 1
    #         super().__init__(n_input, kernel_count*self.n_output_columns*self.n_output_rows, activation, activation_derivative)
    #         self.weights = [np.random.randn(kernel_size*kernel_size) for _ in range(kernel_count)]
    #         self.biases = [np.random.randn(kernel_count, 1) for _ in range(kernel_count)]

    #     def forward(self, inputs):
    #         self.inputs = inputs
    #         self.w_sum = np.zeros((self.kernel_count, self.n_output_columns, self.n_output_rows))
    #         for k in range(self.kernel_count):
    #             for i in range(self.n_output_columns):
    #                 for j in range(self.n_output_rows):
    #                     input_patch = [
    #                         self.inputs[(j+n)*self.kernel_stride*self.n_input_columns + (i+m)*self.kernel_stride]
    #                         for m in range(self.kernel_size)
    #                         for n in range(self.kernel_size)
    #                     ]
    #                     self.w_sum[k, i, j] = np.dot(self.weights[k],input_patch) + self.biases[k]
    #         self.output = self.activation(self.w_sum)
    #         return self.
    #
    #     def backward(self, derivative):
    #         pass
    # class Layer_Pooling(Layer):
    #     pass
    # class Layer_Recurrent(Layer):
    #     pass

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

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def standardize_data(data):
        max = np.max(np.abs(data))
        return data / max

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