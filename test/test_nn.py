import sklearn
import numpy as np
import pandas as pd
from nn.nn import NeuralNetwork as nn
from nn.preprocess import sample_seqs, one_hot_encode_seqs
from sklearn.preprocessing import StandardScaler
import pytest


def test_single_forward():

    nn_arch = [{"input_dim": 4, "output_dim": 3, "activation": "relu"}, {"input_dim": 3, "output_dim": 2, "activation": "sigmoid"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mean_squared_error" # loss function 
    )


    # define test parameters - sigmoid activation
    W_curr = np.array([[0.2, -0.5], [0.3, 0.8]])  # shape (2, 2)
    b_curr = np.array([0.1, -0.2])  # shape (2,)
    A_prev = np.array([[0.4, 0.6], [0.1, 0.9]])  # shape (2, 2)
    activation = "sigmoid"

    

    #forward pass
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)

    # calc expected values
    Z_expected = np.dot(A_prev, W_curr.T) + b_curr
    A_expected = 1 / (1 + np.exp(-Z_expected))  # sigmoid activation

    #assert output shapes are correct
    assert Z_curr.shape == (2, 2)  # correct dimensions
    assert A_curr.shape == (2, 2)  # correct dimensions


    # assert output matches expected values
    np.testing.assert_array_almost_equal(Z_curr, Z_expected, decimal=6)
    np.testing.assert_array_almost_equal(A_curr, A_expected, decimal=6)

    #test relu activation
    activation = "relu"

    #forward pass
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)

    # calc expected values
    Z_expected = np.dot(A_prev, W_curr.T) + b_curr
    A_expected = np.maximum(0, Z_expected)  # relu activation

    #assert output shapes are correct
    assert Z_curr.shape == (2, 2)  # correct dimensions
    assert A_curr.shape == (2, 2)  # correct dimensions

    # assert output matches expected values
    np.testing.assert_array_almost_equal(Z_curr, Z_expected, decimal=6)
    np.testing.assert_array_almost_equal(A_curr, A_expected, decimal=6)

    pass

def test_forward():
    nn_arch = [{"input_dim": 4, "output_dim": 3, "activation": "relu"}, {"input_dim": 3, "output_dim": 2, "activation": "sigmoid"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mean_squared_error" # loss function 
    )

    test_model._param_dict = {
    "W1": np.array([[0.2, -0.5], [0.3, 0.8], [-0.7, 0.1]]),  # Shape (3,2)
    "b1": np.array([0.1, -0.2, 0.3]),  # Shape (3,)
    "W2": np.array([[0.5, -0.3, 0.7]]),  # Shape (1,3)
    "b2": np.array([-0.1]),  # Shape (1,)
    }

    #intialize input (batch size =2, features=2)
    X = np.array([[0.4, 0.6], [0.1, 0.9]])

    #forward pass
    A, cache = test_model.forward(X)

    #manually calculate expected output
    #layer 1
    Z1 = np.dot(X, test_model._param_dict["W1"].T) + test_model._param_dict["b1"]
    A1 = np.maximum(0, Z1)  # relu activation

    #layer 2
    Z2 = np.dot(A1, test_model._param_dict["W2"].T) + test_model._param_dict["b2"]
    A2 = 1 / (1 + np.exp(-Z2))  # sigmoid activation

    #assert output is correct
    np.testing.assert_array_almost_equal(A2, A, decimal=6)

    #assert cache is correct
    np.testing.assert_array_almost_equal(cache["A0"], X, decimal=6)
    np.testing.assert_array_almost_equal(cache["Z1"], Z1, decimal=6)
    np.testing.assert_array_almost_equal(cache["A1"], A1, decimal=6)
    np.testing.assert_array_almost_equal(cache["Z2"], Z2, decimal=6)
    np.testing.assert_array_almost_equal(cache["A2"], A2, decimal=6)

    pass

def test_single_backprop():
    nn_arch = [{"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mean_squared_error" # loss function 
    )

    #random seed
    np.random.seed(42)


    W = np.random.rand(3, 4) #output_dim, input_dim
    b = np.random.rand(3, 1) #output_dim, 1
    Z = np.random.rand(1, 3)  #batch_size, output_dim
    A_prev = np.random.rand(1,4) #batch_size, input_dim
    dA = np.random.rand(3, 1)

    dA_prev, dW, db = test_model._single_backprop(W, b, Z, A_prev, dA, "relu")

    #assert output is correct
    assert dA.shape == (4, 1) #correct dimensions
    assert dW.shape == (3, 4) #correct dimensions
    assert db.shape == (3, 1) #correct dimensions
        
    pass

def test_predict():
    nn_arch = [
        {"input_dim": 4, "output_dim": 3, "activation": "relu"},
        {"input_dim": 3, "output_dim": 2, "activation": "sigmoid"}
    ]

    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mean_squared_error" # loss function 
    )

    X = np.random.rand(1, 4) 

    y_hat = test_model.predict(X)

    #assert output is correct
    assert y_hat.shape == (1, 2)     

    pass

def test_binary_cross_entropy():

    nn_arch = [
        {"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="binary_cross_entropy" # loss function
    )

    y_true = np.array([[1, 0, 1]])
    y_hat = np.array([[0.9, 0.1, 0.8]])

    
    loss = test_model._binary_cross_entropy(y_true, y_hat)

    #assert loss > 0
    np.testing.assert_almost_equal(loss, 0.1446, decimal=6)

    pass

def test_binary_cross_entropy_backprop():

    nn_arch = [
        {"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="binary_cross_entropy" # loss function
    )
    
    y_true = np.array([[1, 0, 1]])
    y_hat = np.array([[0.9, 0.1, 0.8]])

    dA = test_model._binary_cross_entropy_backprop(y_true, y_hat)

    #assert output is correct
    assert dA.shape == y_true.shape  #correct dimensions
    assert np.allclose(dA, np.array([[-1.11111111,  1.11111111, -1.25]])) 
    


    pass

def test_mean_squared_error():
    nn_arch = [
        {"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mean_squared_error" # loss function
    )

    y_true = np.array([[1, 0, 1]])
    y_hat = np.array([[0.9, 0.2, 0.8]])

    loss = test_model._mean_squared_error(y_true, y_hat)

    #assert loss > 0
    assert loss == pytest.approx(0.09)

    pass

def test_mean_squared_error_backprop():
    nn_arch = [
        {"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mean_squared_error" # loss function
    )

    y_true = np.array([[1, 0, 1]])
    y_hat = np.array([[0.9, 0.2, 0.8]])

    dA = test_model._mean_squared_error_backprop(y_true, y_hat)

    #assert output is correct
    assert dA.shape == y_true.shape  #correct dimensions

    pass

def test_sample_seqs():
    #create a simple dataset
    seqs = ["ATGC", "CGTA", "GATT", "CTGA", "TGAC"]
    labels = [True, False, True, False, False]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    #check that the number of samples is correct
    assert len(sampled_seqs) == len(sampled_labels)
    assert sampled_labels.count(True) == sampled_labels.count(False)


    pass

def test_one_hot_encode_seqs():
    #encode a couple of simple sequences
    sequences = ["ATGC", "CGTA"] 

    expected_output_shape = (2,16) 
   
    encodings = one_hot_encode_seqs(sequences)

    #assert output is correct
    assert encodings.shape == expected_output_shape
  
    pass