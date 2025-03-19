
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
                Current layer weight matrix.  (output_neuron, features-input_neuron)
            b_curr: ArrayLike
                Current layer bias matrix. (output_neuron,)
            A_prev: ArrayLike
                Previous layer activation matrix. (batch_size, feature_num)
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix. (batch_size, output_neuron)
            Z_curr: ArrayLike
                Current layer linear transformed matrix. (batch_size, output_neuron)
        """
        #pseudocode for single forward pass
        # Z (l+1) = W(l) * A(l) + b(l)
        # A (l+1) = activation(Z(l+1))

        #debugging
        print(f"shapes in single forward pass, W_curr, A_prev, b_curr")
        print(f"W_curr shape: {W_curr.shape} # (output_neuron, features/input neuron)")
        print(f"A_prev shape: {A_prev.shape} # (batch_size, feature_num/input neuron)")
        print(f"b_curr shape: {b_curr.shape} # (output_neuron, )") 


        #calculate Z_curr
        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T 

        #I
        #Z_curr = W_curr @ A_prev + b_curr 

       
        #debugging
        print(f"shapes in single forward pass, after Z_curr calc, before activation func, Z_curr")
        print(f"Z_curr shape: {Z_curr.shape} # (batch_size, output_neuron)")
       

        # call activation function
        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu':
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError("Activation function not supported")

        #debugging
        print(f"shapes in single forward pass, after Z_curr calc, and activation func, Z_curr, A_curr")
        print(f"Z_curr shape: {Z_curr.shape} # (batch_size, output_neuron)")
        print(f"A_curr shape: {A_curr.shape} # (batch_size, output_neuron)")


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

        #initialize A
        A_curr = X
        cache['A0'] = A_curr

        #debugging
        print(f"shapes in forward pass, before calling single forward pass:")
        print(f"X shape: {X.shape} # (batch_size, feature_num)")

        #loop through each layer
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            #get W_curr, b_curr, A_prev, activation
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)] 
            activation = layer['activation']

            #debugging
            print(f"current A_prev, right before single pass call {layer_idx}")
            print(f"A_prev shape: {A_curr.shape} # (batch_size, feature_num)")
            
            #call single forward pass
            A, Z = self._single_forward(W_curr, b_curr, A_curr, activation)  #testing, changed from A_prev to A_curr

            #debugging
            print(f"shapes in forward pass, after calling single forward pass, A, Z")
            print(f"A shape: {A.shape} # (batch_size, output_neuron)")
            print(f"Z shape: {Z.shape} # (batch_size, output_neuron)")

            #store Z and A in cache
            cache['Z' + str(layer_idx)] = Z
            cache['A' + str(layer_idx)] = A

            A_curr = A

        return A_curr, cache
    
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

      


        #calculate dZ_curr
        if activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError("Activation function not supported")
        
        #calculate dA_prev
        dA_prev = np.dot(dZ_curr, W_curr) 

        #I
        #dA_prev = W_curr.T @ dZ_curr


        #calculate dW_curr
        dW_curr = np.dot(dZ_curr.T, A_prev) # works, correct shape

        #I
        #dW_curr = np.dot(dZ_curr, A_prev.T) 

        #calculate db_curr
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True) 

        #I
        #db_curr = dZ_curr

        #debugging
        print(f"shapes at the end of single backprop:")
        print(f"dA_prev {dA_prev.shape} # (batch_size, feature_num)")
        print(f"dW_curr {dW_curr.shape} # (output_neuron, feature_num)")
        print(f"db_curr {db_curr.shape} # (output_neuron, )")


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
        print(f"shapes in backprop, y, y_hat")
        print(f"y shape: {y.shape} # (batch_size, output_neuron)")
        print(f"y_hat shape: {y_hat.shape} # (batch_size, output_neuron)")


        #calculate loss gradient for output layer
        if self._loss_func == 'binary_cross_entropy':
            dA_curr= self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_curr= self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError("Loss function not supported")

        #debugging
        print(f"shapes in backprop, before loop, dA_prev")
        print(f"dA_curr shape, calculated from loss gradient: {dA_curr.shape} # (batch_size, output_neuron)")

        #loop through each layer in reverse
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1

            #debugging
            print(f"current layer in backprop {layer_idx}")
           
            #get current Z, A, and activation function
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)] 
            A_prev = cache['A' + str(layer_idx-1)] 
            activation_curr = layer['activation']
                
            #debugging
            print(f"Shape of dA_curr being passed to single backprop, before call {layer_idx}")
            print(f"dA_curr shape: {dA_curr.shape} # (batch_size, output_neuron)")

            #call single backprop
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            #store gradients
            grad_dict["W" + str(layer_idx)] = dW_curr #stored as W instead of dW
            grad_dict["b" + str(layer_idx)] = db_curr #stored as b isntead of db

            dA_curr = dA_prev

            
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

            #debugging
            print(f"current layer in update params: {layer_idx}")

            #uodate params
            self._param_dict[f"W{layer_idx}"] -= self._lr * np.mean(grad_dict[f"W{layer_idx}"], axis=0)
            self._param_dict[f"b{layer_idx}"] -= self._lr * np.mean((grad_dict[f"b{layer_idx}"].reshape(-1,1)), axis=0)

            
            #I
            #self._param_dict[f"b{layer_idx}"] -= self._lr * np.mean((grad_dict[f"b{layer_idx}"]), axis=0)


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

        #initialize loss lists
        per_epoch_loss_train = []
        per_epoch_loss_val = []


        #loop through each epoch
        for epoch in range(self._epochs):
            for batch in range(X_train.shape[0] // self._batch_size):
                #get mini batch from training set
                X_train_batch = X_train[batch * self._batch_size: (batch + 1) * self._batch_size]
                y_train_batch = y_train[batch * self._batch_size: (batch + 1) * self._batch_size]
                
                #call forward pass
                y_hat, cache = self.forward(X_train_batch)


                #debugging
                print("calling backprop")
                print(f"y_train_batch shape: {y_train_batch.shape} # (batch_size, output_neuron)")
                print(f"y_hat shape: {y_hat.shape} # (batch_size, output_neuron)")

                #call backprop
                grad_dict = self.backprop(y_train_batch, y_hat, cache)

                #update params
                self._update_params(grad_dict)

            #calculate loss for epoch
            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)

            #debugging
            print("Y_train_pred")
            print(y_train_pred) 
            print("Y_val_pred")
            print(y_val_pred)

            if self._loss_func == 'binary_cross_entropy':
                epoch_loss_train = self._binary_cross_entropy(y_train, y_train_pred)
                epoch_loss_val = self._binary_cross_entropy(y_val, y_val_pred)
            elif self._loss_func == 'mean_squared_error':
                epoch_loss_train = self._mean_squared_error(y_train, y_train_pred)
                epoch_loss_val = self._mean_squared_error(y_val, y_val_pred)
            else:
                raise ValueError("Loss function not supported")
            
            #store loss for epoch
            per_epoch_loss_train.append(epoch_loss_train)
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
        return (dA * (sigmoid_Z * (1 - sigmoid_Z)))

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
        return np.maximum(Z,0)
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
        #I 
        return dA * (Z > 0).astype(np.float64)
        #return np.where(Z > 0, dA, 0)
    

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
        y_zero_loss = y * np.log(y_hat + 1e-9)
        y_one_loss = (1 - y) * np.log(1 - y_hat + 1e-9)

        #I
        return np.mean(-1 * y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).item() 
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
        #debugging
        print(f"shapes in binary cross entropy backprop, y, y_hat")
        print(f"y shape: {y.shape} # (batch_size, output_neuron)")
        print(f"y_hat shape: {y_hat.shape} # (batch_size, output_neuron)")

        #testing
        # ensure that y has the same shape as y_hat (batch_size, 1)
        if y.ndim == 1:  # If y is a 1D array, reshape it to (batch_size, 1)
            y = y[:, np.newaxis]  # Reshape y to (batch_size, 1)

        y = np.asarray(y, dtype=np.float64)
        y_hat = np.asarray(y_hat, dtype=np.float64)

        #dA =  - y / y_hat + (1 - y) / (1 - y_hat)

        return - ((y / y_hat) - ((1 - y) / (1 - y_hat)))

        #debugging
        print(f"shapes in binary cross entropy backprop, dA")
        print(f"dA shape: {dA.shape} # (batch_size, output_neuron)")

        return dA
   
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

        #debugging
        print(f"shapes in mean squared error, y, y_hat")
        print(f"y shape: {y.shape} # (batch_size, output_neuron)")
        print(f"y_hat shape: {y_hat.shape} # (batch_size, output_neuron)")

        
        #m = y_hat.shape[1] # sample number #switched to y_hat
        #return np.sum((y - y_hat) ** 2) / m 

        #could test
        #return np.mean(np.sum((y_hat - y) ** 2, axis=1)).item()


        return np.mean(np.sum((y - y_hat) ** 2, axis=1))

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
        #m = y_hat.shape[1] # sample number 
     
        #return (2 / m) * ( y_hat - y)

        print(f"shapes in mean squared error, y, y_hat")
        print(f"y shape: {y.shape} # (batch_size, output_neuron)")
        print(f"y_hat shape: {y_hat.shape} # (batch_size, output_neuron)")



        return (2 *( y_hat - y))
     
        pass
