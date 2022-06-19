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

#Initialize output choices
ica = True
plot_learning_curve = False
plot_model_pred_test = False
plot_model_pred_train = True
print_latex = True

#Initialize if input datahas been preprocessed with or without ICA
if ica == False:
    # Load all data combinations
    PCA_Y_CLOSED = pd.read_pickle("data/PCA+Y-CLOSED.pkl") # (328, 56)
    PCA_Y_OPEN = pd.read_pickle("data/PCA+Y-OPEN.pkl") # (328, 56)
    PCA_FEATS_Y_OPEN = pd.read_pickle("data/PCA+Features+Y-OPEN.pkl") # (265, 73)
    PCA_FEATS_Y_CLOSED = pd.read_pickle("data/PCA+Features+Y-CLOSED.pkl") # (265, 73)
    Features_Y = pd.read_pickle("data/Features+Y.pkl") # (265, 23)
else:
    # Load all data combinations
    PCA_Y_CLOSED = pd.read_pickle("data/ICA_PCA+Y-CLOSED.pkl") # (318, 56)
    PCA_Y_OPEN = pd.read_pickle("data/ICA_PCA+Y-OPEN.pkl") # (318, 56)
    PCA_FEATS_Y_OPEN = pd.read_pickle("data/ICA_PCA+Features+Y-OPEN.pkl") # (256, 73)
    PCA_FEATS_Y_CLOSED = pd.read_pickle("data/ICA_PCA+Features+Y-CLOSED.pkl") # (256, 73)
    Features_Y = pd.read_pickle("data/Features+Y.pkl") # (265, 23)

# We use ICA_COH+FEATS+Y as the dimensionality reduction matrix because it has the lowest amount of patients - we want all the other combs to have the same dims
dim_regulator = pd.read_pickle("data/ICA_PCA+Features+Y-OPEN.pkl")
# drop the indexes in PCA_Y_CLOSED, PCA_Y_OPEN and FEATURES_Y that are not in PCA_FEATS_Y_CLOSED
PCA_Y_CLOSED = PCA_Y_CLOSED.drop(PCA_Y_CLOSED.index.difference(dim_regulator.index))
PCA_Y_OPEN = PCA_Y_OPEN.drop(PCA_Y_OPEN.index.difference(dim_regulator.index))
PCA_FEATS_Y_CLOSED = PCA_FEATS_Y_CLOSED.drop(PCA_FEATS_Y_CLOSED.index.difference(dim_regulator.index))
PCA_FEATS_Y_OPEN = PCA_FEATS_Y_OPEN.drop(PCA_FEATS_Y_OPEN.index.difference(dim_regulator.index))

open_eyes_pca = {
    "PCA": PCA_Y_OPEN,
    "PCA+Subj_Info": PCA_FEATS_Y_OPEN,
    "Subj_Info": Features_Y}
closed_eyes_pca = {
    "PCA": PCA_Y_CLOSED,
    "PCA+Subj_Info": PCA_FEATS_Y_CLOSED,
    "Subj_info": Features_Y}



def NeuralNetwork(key,data):
    X = data
    y = X.iloc[:, -6:]
    X = X.iloc[:, :-6]
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
  
    
    N, M = X.shape
    
    # Parameters for neural network classifier
    n_hidden_units1 = round((2/3*M)+6)
    #n_hidden_units2 = round(n_hidden_units1*(2/3))+6      # number of hidden units
    n_replicates = 10       # number of networks trained in each k-fold
    max_iter = 10000

    

    
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units1), #M features to n_hidden_units
                        torch.nn.ReLU(),   # 1st transfer function,
                        #torch.nn.Linear(n_hidden_units1, n_hidden_units2), #M features to n_hidden_units
                        #torch.nn.ReLU(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units1, 6), # n_hidden_units to 4 output neuron
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
    y_test=(y_test-y_train_mean)/y_train_std

    
    #Standardization of inputs, so that ADAM learns faster
    X_train_mean = sum(X_train)/len(X_train)
    X_std=X_train.std(axis=0)
    X_train=(X_train-X_train_mean)/X_std
    X_test=(X_test-X_train_mean)/X_std
    
    
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
        
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    # Determine the R-squared error for each individual output node
    y_train_mean = sum(y_train)/len(y_train)
    enumerator=sum((y_test.float()-y_test_est.float())**2)
    denumerator=sum((y_test.float()-y_train_mean.float())**2)
    testError=(1-(enumerator/denumerator)).data.numpy()
    
    #Run the model on the training data set
    y_train_est = net(X_train)
    #Calculating the training errors
    y_train_mean = sum(y_train)/len(y_train)
    enumerator=sum((y_train.float()-y_train_est.float())**2)
    denumerator=sum((y_train.float()-y_train_mean.float())**2)
    trainError=(1-(enumerator/denumerator)).data.numpy()


    # Determine the total R-squared error
    #y_train_mean_total=y_train.data.numpy().mean()
    #enumerator_total=sum(sum((y_test.float()-y_test_est.float())**2))
    #denumerator_total=sum(sum((y_test.float()-y_train_mean_total)**2))
    #error_total=(1-(enumerator_total/denumerator_total)).data.numpy()
    #np_errors=np.append(np_errors, )

        
    return testError, trainError, y_test, y_test_est, y_train, y_train_est, learning_curve


# run the NN for all inputs with closed eyes, and R-squared for test and train error in two dfferent matrices
pca_scores_closed = pd.DataFrame()
pca_scores_closed_train = pd.DataFrame()

for key, data in closed_eyes_pca.items():
    errors, trainErrors, y_test, y_test_est, y_train, y_train_est, learning_curve = NeuralNetwork(key,data)
    errors = np.append(errors, np.mean(errors))
    pca_scores_closed=pd.concat([pca_scores_closed, pd.DataFrame(errors)], axis=1)
    trainErrors = np.append(trainErrors, np.mean(trainErrors))
    pca_scores_closed_train=pd.concat([pca_scores_closed_train, pd.DataFrame(trainErrors)], axis=1)
    
    #chaning to numpy arrays for plots
    y_train=y_train.data.numpy()
    y_train_est=y_train_est.data.numpy()
    y_test=y_test.data.numpy()
    y_test_est=y_test_est.data.numpy()
    
    if plot_learning_curve == True:
        #plotting the learning curve
        plt.ylim([0, 1])
        plt.plot(learning_curve, color='tab:orange')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(f'Learning curve: {key}, closed eyes')
        plt.grid()
        #plt.savefig(f"figures/Learning_curve_closed_{key}.png", bbox_inches='tight')
        plt.show()

    
    if plot_model_pred_test == True:
        #plotting the test model
        axis_range = [np.min([y_test_est, y_test])-1,np.max([y_test_est, y_test])+1]
        plt.plot(axis_range,axis_range,'k--')
        plt.plot(y_test, y_test_est,'ob',alpha=.25)
        plt.legend(['Perfect estimation','Model estimations'])
        plt.title(f'Test predictions (Input: Closed eyes, {key})')
        plt.ylim([-4,4]); plt.xlim([-4,4])
        plt.xlabel('True value')
        plt.ylabel('Estimated value')
        plt.grid()
        plt.show()
        
    if plot_model_pred_train == True:
        #plotting the training model
        axis_range = [np.min([y_train_est, y_train])-1,np.max([y_train_est, y_train])+1]
        plt.plot(axis_range,axis_range,'k--')
        plt.plot(y_train, y_train_est,'ob',alpha=.25)
        plt.legend(['Perfect estimation','Model estimations'])
        plt.title(f'Training predictions (Input: Closed eyes, {key})')
        plt.ylim([-4,4]); plt.xlim([-4,4])
        plt.xlabel('True value')
        plt.ylabel('Estimated value')
        plt.grid()
        #plt.savefig(f"figures/Training_ICA_closed_{key}.png", bbox_inches='tight')
        plt.show()
    

# rename columns of dataframe
pca_scores_closed.columns = ["PCA", "PCA + Subj_Info", "Subj_Info"]
pca_scores_closed_train.columns = ["PCA", "PCA + Subj_Info", "Subj_Info"]
# rename index of the dataframe
pca_scores_closed.index = ["MMSE", "ACE", "TMT A", "TMT B", "DigitSymbol", "Retention", "All"]
pca_scores_closed_train.index = ["MMSE", "ACE", "TMT A", "TMT B", "DigitSymbol", "Retention", "All"]

# run the NN for all inputs with open eyes, and R-squared for test and train error in two dfferent matrices
pca_scores_open = pd.DataFrame()
pca_scores_open_train = pd.DataFrame()
for key, data in open_eyes_pca.items():
    errors, trainErrors, y_test, y_test_est, y_train, y_train_est, learning_curve = NeuralNetwork(key,data)
    errors = np.append(errors, np.mean(errors))
    pca_scores_open=pd.concat([pca_scores_open, pd.DataFrame(errors)], axis=1)
    trainErrors = np.append(trainErrors, np.mean(trainErrors))
    pca_scores_open_train=pd.concat([pca_scores_open_train, pd.DataFrame(trainErrors)], axis=1)
    
    #chaning to numpy arrays for plots
    y_train=y_train.data.numpy()
    y_train_est=y_train_est.data.numpy()
    y_test=y_test.data.numpy()
    y_test_est=y_test_est.data.numpy()
    
    if plot_learning_curve == True:
        #plotting the learning curve
        plt.ylim([0, 1])
        plt.plot(learning_curve, color='tab:orange')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(f'Learning curve: {key}, closed eyes')
        plt.grid()
        #plt.savefig(f"figures/Learning_curve_open_{key}.png",bbox_inches='tight')
        plt.show()
    
    if plot_model_pred_test == True:
        #plotting the test model
        axis_range = [np.min([y_test_est, y_test])-1,np.max([y_test_est, y_test])+1]
        plt.plot(axis_range,axis_range,'k--')
        plt.plot(y_test, y_test_est,'ob',alpha=.25)
        plt.legend(['Perfect estimation','Model estimations'])
        plt.title(f'Test predictions (Input: Open eyes, {key})')
        plt.ylim([-4,4]); plt.xlim([-4,4])
        plt.xlabel('True value')
        plt.ylabel('Estimated value')
        plt.grid()
        plt.show()
        
    if plot_model_pred_train == True:
        #plotting the training model
        axis_range = [np.min([y_train_est, y_train])-1,np.max([y_train_est, y_train])+1]
        plt.plot(axis_range,axis_range,'k--')
        plt.plot(y_train, y_train_est,'ob',alpha=.25)
        plt.legend(['Perfect estimation','Model estimations'])
        plt.title(f'Training predictions (Input: Open eyes, {key})')
        plt.ylim([-4,4]); plt.xlim([-4,4])
        plt.xlabel('True value')
        plt.ylabel('Estimated value')
        plt.grid()
        #plt.savefig(f"figures/Training_ICA_open_{key}.png", bbox_inches='tight')
        plt.show()
    
    

# rename the columns of dataframe
pca_scores_open.columns = ["PCA", "PCA + Subj_Info", "Subj_Info"]
pca_scores_open_train.columns = ["PCA", "PCA + Subj_Info", "Subj_Info"]
# rename index of the dataframe
pca_scores_open.index = ["MMSE", "ACE", "TMT A", "TMT B", "DigitSymbol", "Retention", "All"]
pca_scores_open_train.index = ["MMSE", "ACE", "TMT A", "TMT B", "DigitSymbol", "Retention", "All"]



# print the dataframes
if print_latex==True: #print in latex format
    print('Closed eyes R-square test scores')
    print(pca_scores_closed.to_latex(index=True))
    print('Closed eyes R-square train scores')
    print(pca_scores_closed_train.to_latex(index=True))
    print('Open eyes R-square test scores')
    print(pca_scores_open.to_latex(index=True))
    print('Open eyes R-square train scores')
    print(pca_scores_open_train.to_latex(index=True))
    
else:
    print('Closed eyes R-square test scores')
    print(pca_scores_closed)
    print('Closed eyes R-square train scores')
    print(pca_scores_closed_train)
    print('Open eyes R-square test scores')
    print(pca_scores_open)
    print('Open eyes R-square train scores')
    print(pca_scores_open_train)
        


        
"""
color_list = ['tab:orange', 'tab:green', 'tab:purple']
labels = ['PCA', 'PCA+Subj_info', 'Subj_info']
for i,j in range(3):

    plt.plot(y_test_true[i], y_test_pred[i],'ob',alpha=.5, color=color_list[i], label=labels[i])

axis_range = [np.min([y_test_pred, y_test_true])-1,np.max([y_test_pred, y_test_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.title(f'Estimated versus true value, test data; {key}')
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()
plt.legend()

plt.show()
        
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