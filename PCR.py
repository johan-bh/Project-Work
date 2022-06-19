import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', 10)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
import pickle
ica = True

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

# print("Shape before dim. regulation")
# print(PCA_Y_CLOSED.shape)
# print(PCA_Y_OPEN.shape)
# print(PCA_FEATS_Y_OPEN.shape)
# print(PCA_FEATS_Y_CLOSED.shape)
# print(Features_Y.shape)


# We use ICA_COH+FEATS+Y as the dimensionality reduction matrix because it has the lowest amount of patients - we want all the other combs to have the same dims
dim_regulator = pd.read_pickle("data/ICA_PCA+Features+Y-OPEN.pkl")
# drop the indexes in PCA_Y_CLOSED, PCA_Y_OPEN and FEATURES_Y that are not in PCA_FEATS_Y_CLOSED
PCA_Y_CLOSED = PCA_Y_CLOSED.drop(PCA_Y_CLOSED.index.difference(dim_regulator.index))
PCA_Y_OPEN = PCA_Y_OPEN.drop(PCA_Y_OPEN.index.difference(dim_regulator.index))
PCA_FEATS_Y_CLOSED = PCA_FEATS_Y_CLOSED.drop(PCA_FEATS_Y_CLOSED.index.difference(dim_regulator.index))
PCA_FEATS_Y_OPEN = PCA_FEATS_Y_OPEN.drop(PCA_FEATS_Y_OPEN.index.difference(dim_regulator.index))

print("Shape after dim. regulation")
print(PCA_Y_CLOSED.shape)
print(PCA_Y_OPEN.shape)
print(PCA_FEATS_Y_OPEN.shape)
print(PCA_FEATS_Y_CLOSED.shape)
print(Features_Y.shape)
print(dim_regulator.shape)


open_eyes_pca = {
    "PCA+Y": PCA_Y_OPEN,
    "PCA+Features+Y": PCA_FEATS_Y_OPEN,
    "Features+Y": Features_Y}
closed_eyes_pca = {
    "PCA+Y": PCA_Y_CLOSED,
    "PCA+Features+Y": PCA_FEATS_Y_CLOSED,
    "Features+Y": Features_Y}


def PCR(key,data, eyes="closed"):
    """This function takes a dataframe and computes multilinear regression.
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
        # Create a model, fit it and score it
        model = LinearRegression()
        model.fit(X_train, y_train)
        # scores[col] = model.score(X_test, y_test)
        # calculate TSS and RSS
        y_pred = model.predict(X_test)
        TSS = np.sum((y_test - y_train.mean())**2)
        RSS = np.sum((y_test - y_pred)**2)
        r_squared = 1 - RSS/TSS
        scores[col] = r_squared
        # # Compute relative error for X_test and y_test. Add small epsilon value to avoid division by zero
        # relative_error[col] = np.mean(np.abs(y_test - model.predict(X_test)) / (y_test + 1e-10))

    # Get R-squared (all response vars)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create a model, fit it and score it
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    TSS = np.mean(np.sum((y_test - y_train.mean()) ** 2))
    RSS = np.mean(np.sum((y_test - y_pred) ** 2))
    r_squared = 1 - (RSS / TSS)
    scores["Y"] = r_squared

    y_test_est = np.array(y_pred)
    y_test = np.array(y_test)
    # scale y_test_est and y_test to have the same mean and std as y_test
    y_train_mean = np.array(y_train.mean(axis=0))
    y_train_std=np.array(y_train.std(axis=0))
    y_test_est=(y_test_est-y_train_mean)/y_train_std
    y_test=(y_test-y_train_mean)/y_train_std

    # Compute relative error for X_test and y_test
    # y_pred = model.predict(X_test)
    # # compute relative error between the matrix y_pred and the matrix y_test
    # relative_error["Y"] = np.mean(np.mean(np.abs((y_pred-y_test))/(y_test+1e-10), axis=0))

    # convert scores to dataframe
    scores = pd.DataFrame(scores, index=[f"{key}"]).T
    # # convert relative error to dataframe
    # relative_error = pd.DataFrame(relative_error, index=["Relative Error"]).T
    # # append relative error to scores
    # scores = pd.concat([scores, relative_error], axis=1)
    return scores, y_test_est, y_test,



# ICA-flag file name
if ica == True:
    ica_flag = "_ICA"
else:
    ica_flag = ""


plotting_data_closed = []
pca_scores_closed = pd.DataFrame()
for key, data in closed_eyes_pca.items():
    scores, y_test_est, y_test, = PCR(key,data)
    pca_scores_closed = pd.concat([pca_scores_closed, scores], axis=1)
    axis_range = [-4,4]

    if key == "PCA+Y":
        key = "PCA"
    if key == "PCA+Features+Y":
        key = "PCA+Subject Info"
    if key == "Features+Y":
        key = "Subject Info"

    plotting_data_closed.append([key, y_test_est, y_test, axis_range])

# rename the columns to "PCA", "PCA + Subject Info" and "Subject Info"
pca_scores_closed.columns = ["Coherence", "Coherence + Subject Info", "Subject Info"]
# rename the last index of the dataframe to "All Response Vars"
pca_scores_closed.rename(index={"Y":"All Response Vars"}, inplace=True)

plotting_data_open = []
pca_scores_open = pd.DataFrame()
for key, data in open_eyes_pca.items():
    scores, y_test_est, y_test, = PCR(key,data, eyes="open")
    pca_scores_open = pd.concat([pca_scores_open, scores], axis=1)
    axis_range = [-4,4]

    if key == "PCA+Y":
        key = "PCA"
    if key == "PCA+Features+Y":
        key = "PCA+Subject Info"
    if key == "Features+Y":
        key = "Subject Info"
    plotting_data_open.append([key, y_test_est, y_test, axis_range])

# rename the columns to "PCA", "PCA + Subject Info" and "Subject Info"
pca_scores_open.columns = ["Coherence", "Coherence + Subject Info", "Subject Info"]
# rename the last index of the dataframe to "All Response Vars"
pca_scores_open.rename(index={"Y":"All Response Vars"}, inplace=True)


# # print the dataframes to latex
# print(pca_scores_closed.to_latex(index=True))
# print(pca_scores_open.to_latex(index=True))
#
# if ica == True:
#     # save the dataframes to pickle files
#     with open("data/ICA_PCR_scores-closed.pkl", "wb") as f:
#         pickle.dump(pca_scores_closed, f)
#     with open("data/ICA_PCR_scores-open.pkl", "wb") as f:
#         pickle.dump(pca_scores_open, f)
# else:
#     # save the dataframes to pickle files
#     with open("data/PCR_scores-closed.pkl", "wb") as f:
#         pickle.dump(pca_scores_closed, f)
#     with open("data/PCR_scores-open.pkl", "wb") as f:
#         pickle.dump(pca_scores_open, f)

# create figure with 3 subplots corresponding to the 3 different data sets
fig, ax = plt.subplots(3, figsize=(20,20))
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
    #plt.subplots_adjust(hspace=0.5)
#plt.savefig(f"figures/PCR_PredictionPlotsClosed{ica_flag}.png",bbox_inches='tight')

# create figure with 3 subplots corresponding to the 3 different data sets
fig, ax = plt.subplots(3, figsize=(20, 20))
for i, data in enumerate(plotting_data_open):
    key = data[0]
    y_test_est = data[1]
    y_test = data[2]
    axis_range = data[3]
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
    #ytplt.subplots_adjust(hspace=0.5)
#plt.savefig(f"figures/PCR_PredictionPlotsOpen{ica_flag}.png",bbox_inches='tight')
