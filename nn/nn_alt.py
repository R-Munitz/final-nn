# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        #pseudocode for single forward pass
        # Z (l+1) = W(l) * A(l) + b(l)
        # A (l+1) = activation(Z(l+1))

        #debugging
        print(f"shapes in single forward pass, W_curr, A_prev, b_curr")
        print(f"W_curr shape: {W_curr.shape} # (output_neuron, features) (16,68)")
        print(f"A_prev shape: {A_prev.shape} # (batch_size, feature_num) (64,68)")
        print(f"b_curr shape: {b_curr.shape} # (16,1)")  # (16,1)
      

        #calculate Z_curr
        #Z_curr = np.dot(W_curr, A_prev.T) + b_curr  #transpose here

        #testing
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T  #transpose here

        # call activation function
        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu':
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError("Activation function not supported")
        
        #debugging
        print(f"shapes in single forward pass, after calculations, A_curr, Z_curr", A_curr.shape, Z_curr.shape," should be the same")
        #print(f"W_curr shape: {W_curr.shape} # (output_neuron, features) ")  # (16,64)
        print(f"A_prev shape: {A_prev.shape} # (batch_size, feature_num) ") 
        #print(f"b_curr shape: {b_curr.shape} # (16,1)")  # (16,1)

        return A_curr, Z_curr
    
        pass

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        #initialize cache
        cache = {}

        #debugging
        print(f"X shape: {X.shape} in forward func")  

        #initialize A
        A_prev = X  #input layer, transpose?

        #debugging
        print(f"A_prev shape: {A_prev.shape} in forward func, should be batch_size, num_features ")

        #store A_prev in cache as A0
        cache['A0'] = A_prev
        
        #loop through each layer
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            #get W_curr, b_curr, A_prev, activation
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)] 
            A_prev = cache['A' + str(layer_idx-1)] #missed this line somehow??
            activation = layer['activation']
            
            print(f"current layer {layer_idx}")

            #call single forward pass
            A, Z = self._single_forward(W_curr, b_curr, A_prev, activation) 

            #store Z and A in cache
            cache['Z' + str(layer_idx)] = Z
            cache['A' + str(layer_idx)] = A 

            #debugging
            print(f"Shapes of Z and A, should be the same, in forward fun {Z.shape} {A.shape}")

            #update A_prev for next layer
            A_prev = A

        #debugging
        print(f"shapes in forward pass, after all calculations")
        print(f"W_curr shape: {W_curr.shape} # (output_neuron, features)")  # (16,64)
        print(f"A_prev shape: {A_prev.shape} (batch_size, feature_num) ") 
        print(f"b_curr shape: {b_curr.shape} # (16,1)")  # (16,1)
            

        return A, cache  #transpose here to match the shape of y?
    
        pass

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        #pseudocode
        #1. gradient of loss with respect to Z_curr
        #dZ_curr = dA . f'(Z_curr) 
        #2. gradient of loss with respect to W_curr
        #dW_curr =  (dZ_curr . A_prev.T) / m
        #3. gradient of loss with respect to b_curr
        #dB_curr = sum(dZ_curr) / m
        #4. gradient of loss with respect to A_prev
        #dA_prev = W_curr.T . dZ_curr

        #testing!
        #A_prev = A_prev.T  # undoing the forward-pass transposition


        #calculate m - number of samples
        m = A_prev.shape[0] 

        #debugging
        print(f"in  single backprop: A_prev shape [0]{A_prev.shape[0]}")
        print(f"shapes in single backprop, W_curr, A_prev, b_curr, Z_curr, dA_curr")
        print(f" single backprop: W_curr shape: {W_curr.shape} # (16,64)")  # (16,64)
        print(f" single backprop: A_prev shape: {A_prev.shape} # Should be (64, batch_size), but might be (batch_size, 64)")  # Should be (64, batch_size), but might be (batch_size, 64)
        print(f" single backprop: b_curr shape: {b_curr.shape} # (16,1)")  # (16,1)
        print(f" single backprop: Z_curr shape: {Z_curr.shape} # (16, batch_size)")
        print(f" single backprop: dA_curr shape: {dA_curr.shape} # (16, batch_size)")


        #calculate dZ_curr
        if activation_curr == 'sigmoid':
            activation_backprop = self._sigmoid_backprop
        elif activation_curr == 'relu':
            activation_backprop = self._relu_backprop
        else:
            raise ValueError("Activation function not supported")
        
        dZ_curr = activation_backprop(dA_curr,Z_curr) 
        
        #calculate dW_curr
        #dW_curr = np.dot(dZ_curr, A_prev.T) / m  

        #testing - switching transposes
        dW_curr = np.dot(dZ_curr.T, A_prev) / m 

        #calculate db_curr
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m

        #calculate dA_prev
        #dA_prev = np.dot(W_curr.T, dZ_curr) 

        #testing - switching transpose
        dA_prev = np.dot(W_curr.T, dZ_curr.T) 


        #debugging
        print(f"shapes in single backprop, after calculations, dA_prev, dW_curr, db_curr, dZ_curr")
        print(f"dA_prev shape: {dA_prev.shape} # (64, batch_size)")
        print(f"dW_curr shape: {dW_curr.shape} # (16,64)")
        print(f"db_curr shape: {db_curr.shape} # (16,1)")
        print(f"dZ_curr shape: {dZ_curr.shape} # (16, batch_size)")


        return dA_prev, dW_curr, db_curr
    
        pass

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        #initialize grad_dict
        grad_dict = {}

        #debugging
        print(f"y shape in backprop: {y.shape}")
        print(f"y_hat shape in backprop: {y_hat.shape}")

        #testing
        y = np.reshape(y, (-1, 1))  # reshape y to be (32, 1)
        

        #calculate loss gradient for output layer
        if self._loss_func == 'binary_cross_entropy':
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError("Loss function not supported")
        
        #debugging
        print(f"dA_prev shape after loss gradient calculation: {dA_prev.shape}")

       

        #loop through each layer in reverse
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1

            #if outermost layer, use previously calculated dA_prev
            if layer_idx == len(self.arch):
                dA_curr = dA_prev
            else:
                #get current Z, b, prev A, and activation function
                W_curr = self._param_dict['W' + str(layer_idx)]
                b_curr = self._param_dict['b' + str(layer_idx)]
                Z_curr = cache['Z' + str(layer_idx)]
                A_prev = cache['A' + str(layer_idx-1)] 
                activation_curr = layer['activation']


                #call single backprop
                dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

                #store gradients
                grad_dict["dW" + str(layer_idx)] = dW_curr
                grad_dict["db" + str(layer_idx)] = db_curr

                dA_curr = dA_prev
        
        #debugging
        print(f"shapes in backprop, after all calculations")
        print(f"dA_prev shape: {dA_prev.shape} # (64, batch_size)")
        print(f"dW_curr shape: {dW_curr.shape} # (16,64)")
        print(f"db_curr shape: {db_curr.shape} # (16,1)")
        print(f"Z_curr shape: {Z_curr.shape} # (16, batch size)")
        print(f"W_curr shape: {W_curr.shape} # (16,64)")

              

        return grad_dict
   
        pass

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        #loop through each layer
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1

            #get current W, b, and gradients
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            dW_curr = grad_dict['dW' + str(layer_idx)]
            db_curr = grad_dict['db' + str(layer_idx)]

            #calculate updated W and b, gradient descent, use learning rate
            W_curr = W_curr - self._lr * dW_curr
            b_curr = b_curr - self._lr * db_curr

            #store updated W and b
            self._param_dict['W' + str(layer_idx)] = W_curr
            self._param_dict['b' + str(layer_idx)] = b_curr
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        #pseudocode
        #forward pass for each epoch
        #backprop for each epoch
        #update params for each epoch
        #calculate loss for each epoch
        #store loss for each epoch

        #debugging
        print(f"X_train shape in fit: {X_train.shape}")
        print(f"y_train shape in fit: {y_train.shape}")
        print(f"X_val shape in fit: {X_val.shape}")
        print(f"y_val shape in fit: {y_val.shape}")


        #initialize loss lists
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        #calculate number of samples
        m = X_train.shape[0]

        #loop through each epoch
        for epoch in range(self._epochs):
            #initialize loss variables
            epoch_loss_train = 0
            epoch_loss_val = 0

            #loop through each mini-batch
            for i in range (0, m, self._batch_size):
                #get mini batch from training set
                X_batch = X_train[i:i + self._batch_size]
                y_batch = y_train[i:i + self._batch_size]

                #call forward pass
                y_hat, cache = self.forward(X_batch) #transpose X_batch? 

                #debugging
                print("in fit, y_hat shape after forward pass", y_hat.shape)
                print("in fit, y_batch shape after forward pass", y_batch.shape)
                print("in fit, X_batch shape after forward pass", X_batch.shape)
                
                #testing - making y and y_hat numpy arrays before calculating losses
                #y_batch = np.array(y_batch)
                #y_hat = np.array(y_hat)


                #calculate loss for training set
                if self._loss_func == 'binary_cross_entropy':
                    epoch_loss_train += self._binary_cross_entropy(y_batch, y_hat)
                elif self._loss_func == 'mean_squared_error':
                    epoch_loss_train += self._mean_squared_error(y_batch, y_hat)
                else:
                    raise ValueError("Loss function not supported")
                
                #call backprop
                grad_dict = self.backprop(y_batch, y_hat, cache)

                #update params
                self._update_params(grad_dict)

            
            #debugging
            print(f"shapes in fit, after backprop,  y_batch, y_hat, cache", y_batch.shape, y_hat.shape)


            #calculate loss for epoch
            per_epoch_loss_train.append(epoch_loss_train /(m/self._batch_size))

            #call forward pass for validation set
            y_hat_val, cache_val = self.forward(X_val) #don't need cache for validation set

            #calculate loss for validation set
            if self._loss_func == 'binary_cross_entropy':
                epoch_loss_val = self._binary_cross_entropy(y_val, y_hat_val)
            elif self._loss_func == 'mean_squared_error':
                epoch_loss_val = self._mean_squared_error(y_val, y_hat_val)
            else:
                raise ValueError("Loss function not supported")
            
            #store loss for validation set
            per_epoch_loss_val.append(epoch_loss_val)

        return per_epoch_loss_train, per_epoch_loss_val

        pass

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        #call forward pass
        y_hat, _ = self.forward(X)

        return y_hat
    
        pass

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))
        pass

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sigmoid_Z = self._sigmoid(Z)
        return (dA * sigmoid_Z * (1 - sigmoid_Z))

        pass

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)
        pass

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #debugging
        print(f"shapes in relu backprop, Z, and dA", Z.shape, dA.shape)

        #test transposing Z
        #Z = Z.T

        #debugging
        print(f"shapes in relu backprop, Z, and dA,after transposing Z", Z.shape, dA.shape)

        #testing 
        dZ = dA * (Z > 0)
        return dZ
        
        return np.where(Z > 0, dA, 0)  #error, Z and dA are not the same shape, not sure why tbey aren't the same shape?
        pass

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #debugging
        print(f"in binary cross entropy, y shape: {y.shape}, y_hat shape: {y_hat.shape}")

        y_zero_loss = y * np.log(y_hat + 1e-9)
        y_one_loss = (1 - y) * np.log(1 - y_hat + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)
    
        pass

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return - ((y / y_hat) - ((1 - y) / (1 - y_hat)))

        pass

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        m = y.shape[0] # sample number #changed to 0
        return np.sum((y - y_hat) ** 2) / m 

        pass

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = y.shape[0] # sample number #changed to 0
        return (2 / m) * ( y_hat - y)
     
        pass