import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', 10)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

# Load all data combinations
PCA_Y_CLOSED = pd.read_pickle("data/PCA+Y-CLOSED.pkl") # (328, 56)
PCA_Y_OPEN = pd.read_pickle("data/PCA+Y-OPEN.pkl") # (328, 56)
PCA_FEATS_Y_OPEN = pd.read_pickle("data/PCA+Features+Y-OPEN.pkl") # (265, 73)
PCA_FEATS_Y_CLOSED = pd.read_pickle("data/PCA+Features+Y-CLOSED.pkl") # (265, 73)
Features_Y = pd.read_pickle("data/Features+Y.pkl") # (265, 23)


open_eyes_pca = {
    "PCA+Y": PCA_Y_OPEN,
    "PCA+Features+Y": PCA_FEATS_Y_OPEN,
    "Features+Y": Features_Y}
closed_eyes_pca = {
    "PCA+Y": PCA_Y_CLOSED,
    "PCA+Features+Y": PCA_FEATS_Y_CLOSED,
    "Features+Y": Features_Y}


# print(PCA_Y_CLOSED.shape)
# print(PCA_Y_OPEN.shape)
# print(PCA_FEATS_Y_OPEN.shape)
# print(PCA_FEATS_Y_CLOSED.shape)
# print(Features_Y.shape)

def PCR(key,data):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.15, random_state=42)
        # Create a model, fit it and score it
        model = LinearRegression()
        model.fit(X_train, y_train)
        scores[col] = model.score(X_test, y_test)
        # # Compute relative error for X_test and y_test. Add small epsilon value to avoid division by zero
        relative_error[col] = np.mean(np.abs(y_test - model.predict(X_test)) / (y_test + 1e-10))

    # Get R-squared (all response vars)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    # Create a model, fit it and score it
    model = LinearRegression()
    model.fit(X_train, y_train)
    scores["Y"] = model.score(X_test, y_test)
    # Compute relative error for X_test and y_test
    y_pred = model.predict(X_test)
    # compute relative error between the matrix y_pred and the matrix y_test
    relative_error["Y"] = np.mean(np.mean(np.abs((y_pred-y_test))/(y_test+1e-10), axis=0))

    # convert scores to dataframe
    scores = pd.DataFrame(scores, index=[f"{key}"]).T
    # # convert relative error to dataframe
    # relative_error = pd.DataFrame(relative_error, index=["Relative Error"]).T
    # # append relative error to scores
    # scores = pd.concat([scores, relative_error], axis=1)
    return scores


pca_scores_closed = pd.DataFrame()
for key, data in closed_eyes_pca.items():
    scores = PCR(key,data)
    pca_scores_closed = pd.concat([pca_scores_closed, scores], axis=1)
# rename the columns to "PCA", "PCA + Health" and "Health"
pca_scores_closed.columns = ["PCA", "PCA + Health", "Health"]
# rename the last index of the dataframe to "All Response Vars"
pca_scores_closed.rename(index={"Y":"All Response Vars"}, inplace=True)

pca_scores_open = pd.DataFrame()
for key, data in open_eyes_pca.items():
    scores = PCR(key,data)
    pca_scores_open = pd.concat([pca_scores_open, scores], axis=1)
# rename the columns to "PCA", "PCA + Health" and "Health"
pca_scores_open.columns = ["PCA", "PCA + Health", "Health"]
# rename the last index of the dataframe to "All Response Vars"
pca_scores_open.rename(index={"Y":"All Response Vars"}, inplace=True)


# print the dataframes to latex
print(pca_scores_closed.to_latex(index=True))
print(pca_scores_open.to_latex(index=True))

# save the dataframes to pickle files
pca_scores_closed.to_pickle("data/PCR_scores-closed.pkl")
pca_scores_open.to_pickle("data/PCR_scores-open.pkl")