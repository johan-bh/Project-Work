import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
# import timer for timing the code
import time
import numpy as np
import matplotlib.pyplot as plt
ica = False

if ica == False:
    Features_Y = pd.read_pickle("data/Features+Y.pkl")
    COHERENCE_Y_CLOSED = pd.read_pickle("data/Coherence+Y-CLOSED.pkl")
    COHERENCE_Y_OPEN = pd.read_pickle("data/Coherence+Y-OPEN.pkl")
    COHERENCE_FEATS_Y_CLOSED = pd.read_pickle("data/Coherence+Features+Y-CLOSED.pkl")
    COHERENCE_FEATS_Y_OPEN = pd.read_pickle("data/Coherence+Features+Y-OPEN.pkl")
else:
    Features_Y = pd.read_pickle("data/Features+Y.pkl")
    COHERENCE_Y_CLOSED = pd.read_pickle("data/ICA_Coherence+Y-CLOSED.pkl")
    COHERENCE_Y_OPEN = pd.read_pickle("data/ICA_Coherence+Y-OPEN.pkl")
    COHERENCE_FEATS_Y_CLOSED = pd.read_pickle("data/ICA_Coherence+Features+Y-CLOSED.pkl")
    COHERENCE_FEATS_Y_OPEN = pd.read_pickle("data/ICA_Coherence+Features+Y-OPEN.pkl")

# We use ICA_COH+FEATS+Y as the dimensionality reduction matrix because it has the lowest amount of patients - we want all the other combs to have the same dims
dim_regulator = pd.read_pickle("data/ICA_PCA+Features+Y-OPEN.pkl")
# drop the indexes in PCA_Y_CLOSED, PCA_Y_OPEN and FEATURES_Y that are not in PCA_FEATS_Y_CLOSED
COHERENCE_Y_CLOSED = COHERENCE_Y_CLOSED.drop(COHERENCE_Y_CLOSED.index.difference(dim_regulator.index))
COHERENCE_Y_OPEN = COHERENCE_Y_OPEN.drop(COHERENCE_Y_OPEN.index.difference(dim_regulator.index))
COHERENCE_FEATS_Y_OPEN = COHERENCE_FEATS_Y_OPEN.drop(COHERENCE_FEATS_Y_OPEN.index.difference(dim_regulator.index))
COHERENCE_FEATS_Y_CLOSED = COHERENCE_FEATS_Y_CLOSED.drop(COHERENCE_FEATS_Y_CLOSED.index.difference(dim_regulator.index))
Features_Y = Features_Y.drop(Features_Y.index.difference(dim_regulator.index))

# print(COHERENCE_Y_CLOSED.shape)
# print(COHERENCE_Y_OPEN.shape)
# print(COHERENCE_FEATS_Y_OPEN.shape)
# print(COHERENCE_FEATS_Y_CLOSED.shape)
# print(Features_Y.shape)

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')

closed_eyes_ridge = {
    "COHERENCE+Y": COHERENCE_Y_CLOSED,
    "COHERENCE+Features+Y": COHERENCE_FEATS_Y_CLOSED,
    "Features+Y": Features_Y}
open_eyes_ridge = {
    "COHERENCE+Y": COHERENCE_Y_OPEN,
    "COHERENCE+Features+Y": COHERENCE_FEATS_Y_OPEN,
    "Features+Y": Features_Y}

def RidgeReg(key,data, eyes="closed"):
    """This function takes a dataframe and computes ridge regression.
    It uses the last 6 columns seperately as targets, and then the last 6 columns together as a target.
    The score of each model is stored in a dictionary."""
    # Create dictionary to store scores
    scores = {}
    relative_error = {}
    # Split data into features and target
    X = data.iloc[:, :-6]
    y = data.iloc[:, -6:]


    # Get R-squared and relative error for each target
    for col in y:
        y1 = y[col]
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)
        if "Feature" in key:
            # compute mean and std for X_train
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            # Standardize X_train
            X_train = (X_train - X_train_mean) / X_train_std
            # Use X_train_mean and X_train_std to standardize X_test
            X_test = (X_test - X_train_mean) / X_train_std


        # loop through different values of alpha and find the best one and use it
        best_score = 0
        best_alpha = 0
        for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_alpha = alpha
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        TSS = np.sum((y_test - y_train.mean())**2)
        RSS = np.sum((y_test - y_pred)**2)
        r_squared = 1 - RSS/TSS
        scores[col] = r_squared

        # # Compute relative error for X_test and y_test. Add small epsilon value to avoid division by zero
        # relative_error[col] = np.mean(np.abs(y_test - model.predict(X_test)) / (y_test + 1e-10))




    # Get R-squared (all response vars)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if "Feature" in key:
        # compute mean and std for X_train
        X_train_mean = X_train.mean()
        X_train_std = X_train.std()
        # Standardize X_train
        X_train = (X_train - X_train_mean) / X_train_std
        # Use X_train_mean and X_train_std to standardize X_test
        X_test = (X_test - X_train_mean) / X_train_std
    best_score = -100
    best_alpha = -100
    # loop through different values of alpha and find the best one and use it
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    TSS = np.mean(np.sum((y_test - y_train.mean()) ** 2))
    RSS = np.mean(np.sum((y_test - y_pred) ** 2))
    r_squared = 1 - RSS / TSS
    scores["Y"] = r_squared

    # convert y_pred to numpy array
    y_test_est = np.array(y_pred)
    y_test = np.array(y_test)
    # scale y_test_est and y_test to have the same mean and std as y_test
    y_test_est = (y_test_est - y_test_est.mean()) / (y_test_est.std() + 1e-10)
    y_test = (y_test - y_test.mean()) / (y_test.std() + 1e-10)
    # Compute relative error for X_test and y_test
    # y_pred = model.predict(X_test)
    # compute relative error between the matrix y_pred and the matrix y_test
    # relative_error["Y"] = np.mean(np.mean(np.abs((y_pred-y_test))/(y_test+1e-10), axis=0))

    # convert scores to dataframe
    scores = pd.DataFrame(scores, index=[f"{key}"]).T
    # convert relative error to dataframe
    # relative_error = pd.DataFrame(relative_error, index=[f"Relative Error ({key})"]).T
    # # append relative error to scores
    # scores = pd.concat([scores, relative_error], axis=1)
    return scores, y_test_est, y_test, best_alpha

if ica == True:
    ica_flag = "_ICA"
else:
    ica_flag = ""


#loop through all_data and call RidgeReg function. Stack each dataframe column wise using the same index
#and store the result in a new dataframe. + Store data for plotting in a list.
plotting_data_closed = []
ridge_scores_closed = pd.DataFrame()
for key, data in closed_eyes_ridge.items():
    scores, y_test_est, y_test, best_alpha = RidgeReg(key,data)
    ridge_scores_closed = pd.concat([ridge_scores_closed, scores], axis=1)
    axis_range = [-4,4]
    if key == "COHERENCE+Y":
        key = "Coherence"
    if key == "COHERENCE+Features+Y":
        key = "Coherence+Subject Info"
    if key == "Features+Y":
        key = "Subject Info"
    plotting_data_closed.append([key, y_test_est, y_test, axis_range])

# rename the columns to "Coherence", "Coherence + Health" and "Health"
ridge_scores_closed.columns = ["Coherence", "Coherence + Subject Info", "Subject Info"]
# rename the last index of the dataframe to "All Response Variables"
ridge_scores_closed.rename(index={"Y":"All Response Vars"}, inplace=True)

plotting_data_open = []
ridge_scores_open = pd.DataFrame()
for key, data in open_eyes_ridge.items():
    scores, y_test_est, y_test, best_alpha  = RidgeReg(key,data, "open")
    ridge_scores_open = pd.concat([ridge_scores_open, scores], axis=1)
    axis_range = [-4,4]
    if key == "COHERENCE+Y":
        key = "Coherence"
    if key == "COHERENCE+Features+Y":
        key = "Coherence+Subject Info"
    if key == "Features+Y":
        key = "Subject Info"
    plotting_data_open.append([key, y_test_est, y_test, axis_range])

# rename the columns to "Coherence", "Coherence + Health" and "Health"
ridge_scores_open.columns = ["Coherence", "Coherence + Subject Info", "Subject Info"]
# rename the last index of the dataframe to "All Response Vars"
ridge_scores_open.rename(index={"Y":"All Response Vars"}, inplace=True)


# print the dataframes to latex
print(ridge_scores_closed.to_latex(index=True))
print(ridge_scores_open.to_latex(index=True))

if ica == False:
    # save the dataframes to pickle files
    with open("data/Ridge_scores-open.pkl", 'wb') as f:
        pickle.dump(ridge_scores_closed, f)
    with open("data/Ridge_scores-open.pkl", 'wb') as f:
        pickle.dump(ridge_scores_open, f)
else:
    with open("data/ICA_Ridge_scores-closed.pkl", 'wb') as f:
        pickle.dump(ridge_scores_closed, f)
    with open("data/ICA_Ridge_scores-open.pkl", 'wb') as f:
        pickle.dump(ridge_scores_open, f)

# create figure with 3 subplots corresponding to the 3 different data sets
fig, ax = plt.subplots(3, figsize=(10,10))
for i, data in enumerate(plotting_data_closed):
    key = data[0]
    y_test_est = data[1]
    y_test = data[2]
    axis_range = data[3]
    ax[i].plot(axis_range,axis_range, 'k--')
    ax[i].plot(y_test, y_test_est, 'ob', alpha=.25)
    ax[i].legend(['Perfect estimation', 'Model estimations'])
    ax[i].title.set_text(f'Test Predictions (Input: Closed eyes, {key})')
    ax[i].set_ylim(axis_range)
    ax[i].set_xlim(axis_range)
    ax[i].set_xlabel('True value')
    ax[i].set_ylabel('Estimated value')
    ax[i].set_aspect('equal')
    ax[i].grid(True)
    plt.subplots_adjust(hspace=0.5)
plt.savefig(f"figures/RidgePredictionPlotsClosed{ica_flag}.png",bbox_inches='tight')

# create figure with 3 subplots corresponding to the 3 different data sets
fig, ax = plt.subplots(3, figsize=(10, 10))
for i, data in enumerate(plotting_data_open):
    key = data[0]
    ax[i].plot(axis_range,axis_range, 'k--')
    ax[i].plot(y_test, y_test_est, 'ob', alpha=.25)
    ax[i].legend(['Perfect estimation', 'Model estimations'])
    ax[i].title.set_text(f'Test Predictions (Input: Open eyes, {key})')
    ax[i].set_ylim(axis_range)
    ax[i].set_xlim(axis_range)
    ax[i].set_xlabel('True value')
    ax[i].set_ylabel('Estimated value')
    ax[i].set_aspect('equal')
    ax[i].grid(True)
    plt.subplots_adjust(hspace=0.5)
plt.savefig(f"figures/RidgePredictionPlotsOpen{ica_flag}.png",bbox_inches='tight')