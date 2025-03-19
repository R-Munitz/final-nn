import sklearn
import numpy as np
import pandas as pd
from nn.nn import NeuralNetwork as nn
from nn.preprocess import sample_seqs, one_hot_encode_seqs
from sklearn.preprocessing import StandardScaler
import pytest


def test_single_forward():

    #create a simple neural network with one layer 
    nn_arch = [{"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mse" # loss function 
    )

    W = np.array([[0.2, 0.5, -0.3, 0.8], [0.4, -0.7, 0.1, 0.3], [-0.6, 0.2, 0.9, -0.4]])
    b = np.array([[0.1], [-0.2], [0.3]])
    A_prev = np.array([[0.5], [-0.5], [0.2], [0.1]])

    A, Z = test_model._single_forward(W, b, A_prev, "relu")


    #assert output is correct
    assert Z.shape == (3, 1) #correct dimensions
    assert A.shape == (3, 1) #correct dimensions
    assert (A>=0).all() # relu is non negative
    assert np.allclose(A, np.array([[0.0], [0.4], [0.04]])) # correct values
    
    pass

def test_forward():
    nn_arch = [{"input_dim": 4, "output_dim": 3, "activation": "relu"}, {"input_dim": 3, "output_dim": 2, "activation": "sigmoid"}]
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mse" # loss function 
    )

    X = np.random.randn(4, 5) #4 features, batch size of 5
    A_last, cache = test_model.forward(X)

    #assert output is correct
    assert A_last.shape == (2, 5) #correct dimensions
    assert (A_last >= 0).all() and (A_last <= 1).all()  # assert igmoid output is between 0 and 1

    pass

def test_single_backprop():
    nn_arch = [{"input_dim": 4, "output_dim": 3, "activation": "relu"}]
    test_model  = nn(
    nn_arch=nn_arch,
    lr=0.01,           # learning rate
    seed=42,            # random seed
    batch_size=1,      # batch size
    epochs=10,         # number of epochs
    loss_function="mse" # loss function 
    )

    W = np.random.rand(3, 4)
    b = np.random.rand(3, 1)
    Z = np.random.rand(3, 1)
    A_prev = np.random.rand(4, 1)
    dA = np.random.rand(3, 1)

    dA, dW, db = test_model._single_backprop(W, b, Z, A_prev, dA, "relu")

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
    loss_function="mse" # loss function 
    )

    X = np.random.rand(4, 5)
    y_hat = test_model.predict(X)

    #assert output is correct
    assert y_hat.shape == (2, 5) #correct dimensions    

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
    assert loss == 0.14462152638588005 #did the math

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
    assert loss == pytest.approx(0.03)

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
    assert np.allclose(dA, np.array([[-0.06666667,  0.13333333, -0.13333333]])) 
    

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

    expected_output_shape = (2 * 4 * 4,)  #flattened one-hot encoding of 2 sequences, 4 bases each, 4 nucleotides
   
    encodings = one_hot_encode_seqs(sequences)

    #assert output is correct
    assert encodings.shape == expected_output_shape

    #check explict encoding is as expected
    #TO DO

    #test that this still passes
    pass