# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Pratap Roy Choudhury] -- [prroyc]
# reference : https://www.kaggle.com/vitorgamalemos/multilayer-perceptron-from-scratch/notebook
# reference : https://mlfromscratch.com/neural-network-tutorial/#/
# reference : http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
import random
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None


    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._X = X
        self._y = one_hot_encoding(y)

        self.classes_number = len(set(y)) #number of unique class labels
        self.OutputLayer = len(set(y)) #number of unique outputs
        self.inputLayer = X.shape[1] #total number of features
        self.BiasHiddenValue = 1
        self.BiasOutputValue = 1

        self._h_weights = np.random.normal(scale=0.01,size =(self.inputLayer,self.n_hidden))
        self._o_weights = np.random.normal(scale=0.01,size =(self.n_hidden,self.OutputLayer))
        self._h_bias = np.array([self.BiasHiddenValue for i in range(self.n_hidden)])
        self._o_bias = np.array([self.BiasOutputValue for i in range(self.OutputLayer)])

        np.random.seed(42)

        #raise NotImplementedError('This function must be implemented by the student.')
    
    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)
        
        epoch = 1

        while(epoch <= self.n_iterations):

            
            self.output = np.zeros(self.classes_number)

            #Forward Propagation
            self.h_weight_sum = np.dot(self._X, self._h_weights) + self._h_bias.T
            self.fp_1 = self.hidden_activation(self.h_weight_sum)#((np.dot(inputs, self._h_weights) + self._h_bias.T))
            self.o_weight_sum = np.dot(self.fp_1, self._o_weights) + self._o_bias.T
            self.fp_2 = self._output_activation(self.o_weight_sum)#((np.dot(self.fp_1, self._o_weights) + self._o_bias.T))
        
            #'Backpropagation : Update Weights'
            del_output = []
            #Error: OutputLayer'
            error = self._y - self.fp_2   #-self.output
            
            del_output = np.multiply(error,self._output_activation(self.o_weight_sum,derivative = True))
            #Update weights and bias of OutputLayer
            self._o_weights =self._o_weights - (self.learning_rate * np.dot(self.fp_1.T,del_output))
            self._o_bias =self._o_bias - (self.learning_rate * np.sum(del_output))
            '''
            for i in range(self.n_hidden):
                for j in range(self.OutputLayer):
                    self._o_weights[i][j] += (self.learning_rate * (del_output[j] * self.fp_1[i]))
                    self._o_bias[j] += (self.learning_rate * del_output[j])
            '''
            #Error: HiddenLayer
            del_hidden = np.dot(del_output,self._o_weights.T) * self.hidden_activation(self.h_weight_sum,derivative = True)
     
            #Update weights and bias of HiddenLayer
            self._h_weights = self._h_weights - (self.learning_rate * np.dot(self._X.T,del_hidden))
            self._h_bias = self._h_bias - (self.learning_rate * np.sum(del_hidden))
            '''
            for i in range(self.OutputLayer):
                for j in range(self.n_hidden):
                    self._h_weights[i][j] += (self.learning_rate * (del_hidden[j] * inputs[i]))
                    self._h_bias[j] += (self.learning_rate * del_hidden[j])
            '''      
            if((epoch % 20 == 0) or (epoch == 1)):
                loss = self._loss_function(self._y,self.fp_2)
                self._loss_history.append(loss)
                #print("Epoch ", epoch, "- Total Error: ",loss)
            
                
            epoch += 1

        #raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        #Returns the predictions for every element of X
        y_predict = []
        #'Forward Propagation'
        #fp = np.matmul(X,self._h_weights) + self._h_bias
        #fp = np.matmul(fp, self._o_weights) + self._o_bias

        h_weight_sum = np.dot(X, self._h_weights) + self._h_bias.T
        fp_1 = self.hidden_activation(h_weight_sum)#((np.dot(inputs, self._h_weights) + self._h_bias.T))
        o_weight_sum = np.dot(fp_1, self._o_weights) + self._o_bias.T
        fp = self._output_activation(o_weight_sum)
                                 
        for i in fp:
            y_predict.append(max(enumerate(i), key=lambda x:x[1])[0])
    
        return np.array(y_predict)

        #raise NotImplementedError('This function must be implemented by the student.')
