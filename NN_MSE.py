#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:21:00 2022

@author: smilladue
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from trainNN import train_neural_net
import pandas as pd


X = pd.read_pickle('/Users/smilladue/Downloads/02450Toolbox_Python/Tools/PCA+Features+All4-OPEN.pkl')
y = X.iloc[:, -4:]
X = X.iloc[:, :-4]
X=X.to_numpy()
y=y.to_numpy()

N, M = X.shape

# Parameters for neural network classifier
n_hidden_units1 = 46
n_hidden_units2 = 35      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K = 3                   # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Define the model
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units1), #M features to n_hidden_units
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units1, n_hidden_units2), #M features to n_hidden_units
                    torch.nn.ReLU(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units2, 4), # n_hidden_units to 4 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 

np_errors=np.array([errors[0],errors[1],errors[2]])
# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
print('\nEstimated generalization error, MMSE: {0}'.format(round(np.sqrt(np.mean(np_errors[:,0])), 4)))
print('\nEstimated generalization error, ACE: {0}'.format(round(np.sqrt(np.mean(np_errors[:,1])), 4)))
print('\nEstimated generalization error, TrailMakingA: {0}'.format(round(np.sqrt(np.mean(np_errors[:,2])), 4)))
print('\nEstimated generalization error, TrailMakingB: {0}'.format(round(np.sqrt(np.mean(np_errors[:,3])), 4)))



y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

for i in range(4):
    y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
    y_est=y_est[:,i]; y_true=y_true[:,i]; 
    axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()

    plt.show()
 
    