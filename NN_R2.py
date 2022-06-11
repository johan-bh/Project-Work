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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load all data combinations
PCA_Y_CLOSED = pd.read_pickle("data/PCA+Y-CLOSED.pkl") # (328, 56)
PCA_Y_OPEN = pd.read_pickle("data/PCA+Y-OPEN.pkl") # (328, 56)
PCA_FEATS_Y_OPEN = pd.read_pickle("data/PCA+Features+Y-OPEN.pkl") # (265, 73)
PCA_FEATS_Y_CLOSED = pd.read_pickle("data/PCA+Features+Y-CLOSED.pkl") # (265, 73)
Features_Y = pd.read_pickle("data/Features+Y.pkl") # (265, 23)

# drop the indexes in PCA_Y_CLOSED, PCA_Y_OPEN and FEATURES_Y that are not in PCA_FEATS_Y_CLOSED
PCA_Y_CLOSED = PCA_Y_CLOSED.drop(PCA_Y_CLOSED.index.difference(PCA_FEATS_Y_CLOSED.index))
PCA_Y_OPEN = PCA_Y_OPEN.drop(PCA_Y_OPEN.index.difference(PCA_FEATS_Y_OPEN.index))
Features_Y = Features_Y.drop(Features_Y.index.difference(PCA_FEATS_Y_CLOSED.index))

open_eyes_pca = {
    "PCA+Y": PCA_Y_OPEN,
    "PCA+Features+Y": PCA_FEATS_Y_OPEN,
    "Features+Y": Features_Y}
closed_eyes_pca = {
    "PCA+Y": PCA_Y_CLOSED,
    "PCA+Features+Y": PCA_FEATS_Y_CLOSED,
    "Features+Y": Features_Y}



def NeuralNetwork(key,data):
    X = data
    y = X.iloc[:, -6:]
    X = X.iloc[:, :-6]
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    
    N, M = X.shape
    
    # Parameters for neural network classifier
    n_hidden_units1 = round(M*(2/3))
    n_hidden_units2 = round(n_hidden_units1*(2/3))      # number of hidden units
    n_replicates = 10       # number of networks trained in each k-fold
    max_iter = 10000
    

    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units1), #M features to n_hidden_units
                        torch.nn.ReLU(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units1, n_hidden_units2), #M features to n_hidden_units
                        torch.nn.ReLU(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units2, 6), # n_hidden_units to 4 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    

        
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    
    #standardization of response variables
    y_train_mean = sum(y_train)/len(y_train)
    y_train_std=y_train.std(axis=0)
    y_train=(y_train-y_train_mean)/y_train_std
   
    y_test_mean = sum(y_test)/len(y_test)
    y_test_std=y_test.std(axis=0)
    y_test=(y_test-y_test_mean)/y_test_std

    """
    #Standardization of inputs, so that ADAM learns faster
    X_train_mean = sum(X_train)/len(X_train)
    X_std=X_train.std(axis=0)
    X_train=(X_train-X_train_mean)/X_std
    X_test=(X_test-X_train_mean)/X_std
    """
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
        
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    # Determine the R-squared error
    y_train_mean = sum(y_train)/len(y_train)
    enumerator=sum((y_test.float()-y_test_est.float())**2)
    denumerator=sum((y_test.float()-y_train_mean.float())**2)
    np_errors=(1-(enumerator/denumerator)).data.numpy()

        
    return np_errors

    
# run the NN for all inputs with closed eyes, and store in one matrix
pca_scores_closed = pd.DataFrame()
for key, data in closed_eyes_pca.items():
    errors = NeuralNetwork(key,data)
    errors = np.append(errors, np.mean(errors))
    pca_scores_closed=pd.concat([pca_scores_closed, pd.DataFrame(errors)], axis=1)
# rename columns of dataframe
pca_scores_closed.columns = ["PCA", "PCA + Subj_Info", "Subj_Info"]
# rename index of the dataframe
pca_scores_closed.index = ["MMSE", "ACE", "TMT A", "TMT B", "DigitSymbol", "Retention", "All"]

# run the NN for all inputs with open eyes, and store in one matrix
pca_scores_open = pd.DataFrame()
for key, data in open_eyes_pca.items():
    errors = NeuralNetwork(key,data)
    errors = np.append(errors, np.mean(errors))
    pca_scores_open=pd.concat([pca_scores_open, pd.DataFrame(errors)], axis=1)
# rename the columns of dataframe
pca_scores_open.columns = ["PCA", "PCA + Subj_Info", "Subj_Info"]
# rename index of the dataframe
pca_scores_open.index = ["MMSE", "ACE", "TMT A", "TMT B", "DigitSymbol", "Retention", "All"]

# print the dataframes to latex
print(pca_scores_closed.to_latex(index=True))
print(pca_scores_open.to_latex(index=True))

        
"""

        
    pca_scores_closed = pd.concat([pca_scores_closed, scores], axis=1)
# rename the columns to "PCA", "PCA + Health" and "Health"
pca_scores_closed.columns = ["PCA", "PCA + Health", "Health"]
# rename the last index of the dataframe to "All Response Vars"
pca_scores_closed.rename(index={"Y":"All Response Vars"}, inplace=True)

pca_scores_open = pd.DataFrame()
for key, data in open_eyes_pca.items():
    scores = NeuralNetwork(key,data)
    pca_scores_open = pd.concat([pca_scores_open, scores], axis=1)
# rename the columns to "PCA", "PCA + Health" and "Health"
pca_scores_open.columns = ["PCA", "PCA + Health", "Health"]
# rename the last index of the dataframe to "All Response Vars"
pca_scores_open.rename(index={"Y":"All Response Vars"}, inplace=True)

# print the dataframes to latex
print(pca_scores_closed.to_latex(index=False))
print(pca_scores_open.to_latex(index=False))

# Print the average classification error rate
print('\nEstimated generalization error, R-squared, All: {0}'.format(round((np.mean(errors)), 4)))
print('\nEstimated generalization error, MMSE: {0}'.format(round((np.mean(np_errors[:,0])), 4)))
print('\nEstimated generalization error, ACE: {0}'.format(round((np.mean(np_errors[:,1])), 4)))
print('\nEstimated generalization error, TrailMakingA: {0}'.format(round((np.mean(np_errors[:,2])), 4)))
print('\nEstimated generalization error, TrailMakingB: {0}'.format(round((np.mean(np_errors[:,3])), 4)))
print('\nEstimated generalization error, DigitSymbol: {0}'.format(round((np.mean(np_errors[:,4])), 4)))
print('\nEstimated generalization error, Retention: {0}'.format(round((np.mean(np_errors[:,5])), 4)))




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
 
"""