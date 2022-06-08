import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

# Load all data combinations
PCA_Y_CLOSED = pd.read_pickle("data/PCA+Y-CLOSED.pkl")
PCA_Y_OPEN = pd.read_pickle("data/PCA+Y-OPEN.pkl")
PCA_FEATS_Y_OPEN = pd.read_pickle("data/PCA+Features+All4-OPEN.pkl")
PCA_FEATS_Y_CLOSED = pd.read_pickle("data/PCA+Features+All4-CLOSED.pkl")
PCA_FEATS_ACE_OPEN = pd.read_pickle("data/PCA+Features+ACE-OPEN.pkl")
PCA_FEATS_ACE_CLOSED = pd.read_pickle("data/PCA+Features+ACE-CLOSED.pkl")
PCA_FEATS_MMSE_OPEN = pd.read_pickle("data/PCA+Features+MMSE-OPEN.pkl")
PCA_FEATS_MMSE_CLOSED = pd.read_pickle("data/PCA+Features+MMSE-CLOSED.pkl")
PCA_FEATS_TrailA_OPEN = pd.read_pickle("data/PCA+Features+TrailMakingA-OPEN.pkl")
PCA_FEATS_TrailA_CLOSED = pd.read_pickle("data/PCA+Features+TrailMakingA-CLOSED.pkl")
PCA_FEATS_TrailB_OPEN = pd.read_pickle("data/PCA+Features+TrailMakingB-OPEN.pkl")
PCA_FEATS_TrailB_CLOSED = pd.read_pickle("data/PCA+Features+TrailMakingB-CLOSED.pkl")
Features_Y = pd.read_pickle("data/Features+Y.pkl")

# PCA and Response variables (closed)
# Train test split using last 4 columns as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_Y_CLOSED.iloc[:, :-4], PCA_Y_CLOSED.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_Y_CLOSED = model.score(X_test, y_test)
print("Score PCA+Y-CLOSED:", score_PCA_Y_CLOSED)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test.
# Take the mean of the relative error to get the mean relative error for all 4 response variables
RelErr_PCA_Y_CLOSED = np.mean(np.mean(np.abs(y_pred - y_test) / y_test))


# PCA and Response variables (open)
# Train test split using last 4 columns as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_Y_OPEN.iloc[:, :-4], PCA_Y_OPEN.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_Y_OPEN = model.score(X_test, y_test)
print("Score PCA+Y-OPEN:", score_PCA_Y_OPEN)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_Y_OPEN = np.mean(np.mean(np.abs(y_pred - y_test) / y_test))


# PCA, Features and Response variables (open)
# Train test split using last 4 columns as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_Y_OPEN.iloc[:, :-4], PCA_FEATS_Y_OPEN.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_Y_OPEN = model.score(X_test, y_test)
print("Score PCA+Features+Y-OPEN:", score_PCA_FEATS_Y_OPEN)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_FEATS_Y_OPEN = np.mean(np.mean(np.abs(y_pred - y_test) / y_test))


# PCA, Features and Response variables (closed)
# Train test split using last 4 columns as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_Y_CLOSED.iloc[:, :-4], PCA_FEATS_Y_CLOSED.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_Y_CLOSED = model.score(X_test, y_test)
print("Score PCA+Features+Y-CLOSED:", score_PCA_FEATS_Y_CLOSED)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_FEATS_Y_CLOSED = np.mean(np.mean(np.abs(y_pred - y_test) / y_test))


# PCA, Features and ACE variables (open)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_ACE_OPEN.iloc[:, :-1], PCA_FEATS_ACE_OPEN.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_ACE_OPEN = model.score(X_test, y_test)
print("Score PCA+Features+ACE-OPEN:", score_PCA_FEATS_ACE_OPEN)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_FEATS_ACE_OPEN = np.mean(np.abs(y_pred - y_test) / y_test)


# PCA, Features and ACE variables (closed)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_ACE_CLOSED.iloc[:, :-1], PCA_FEATS_ACE_CLOSED.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_ACE_CLOSED = model.score(X_test, y_test)
print("Score PCA+Features+ACE-CLOSED:", score_PCA_FEATS_ACE_CLOSED)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_FEATS_ACE_CLOSED = np.mean(np.abs(y_pred - y_test) / y_test)

# PCA, Features and MMSE variables (open)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_MMSE_OPEN.iloc[:, :-1], PCA_FEATS_MMSE_OPEN.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_MMSE_OPEN = model.score(X_test, y_test)
print("Score PCA+Features+MMSE-OPEN:", score_PCA_FEATS_MMSE_OPEN)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_FEATS_MMSE_OPEN = np.mean(np.abs(y_pred - y_test) / y_test)

# PCA, Features and MMSE variables (closed)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_MMSE_CLOSED.iloc[:, :-1], PCA_FEATS_MMSE_CLOSED.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_MMSE_CLOSED = model.score(X_test, y_test)
print("Score PCA+Features+MMSE-CLOSED:", score_PCA_FEATS_MMSE_CLOSED)
y_pred = model.predict(X_test)
# Compute relative error for X_test and y_test
RelErr_PCA_FEATS_MMSE_CLOSED = np.mean(np.abs(y_pred - y_test) / y_test)


# PCA, Features and TrailA variables (open)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_TrailA_OPEN.iloc[:, :-1], PCA_FEATS_TrailA_OPEN.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_TrailA_OPEN = model.score(X_test, y_test)
print("Score PCA+Features+TrailA-OPEN:", score_PCA_FEATS_TrailA_OPEN)
y_pred = model.predict(X_test)
RelErr_PCA_FEATS_TrailA_OPEN = np.mean(np.abs(y_test - model.predict(X_test)) / y_test)

# PCA, Features and TrailA variables (closed)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_TrailA_CLOSED.iloc[:, :-1], PCA_FEATS_TrailA_CLOSED.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_TrailA_CLOSED = model.score(X_test, y_test)
print("Score PCA+Features+TrailA-CLOSED:", score_PCA_FEATS_TrailA_CLOSED)
# Compute relative error for X_test and y_test
y_pred = model.predict(X_test)
RelErr_PCA_FEATS_TrailA_CLOSED = np.mean(np.abs(y_pred - y_test) / np.abs(y_test))

# PCA, Features and TrailB variables (open)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_TrailB_OPEN.iloc[:, :-1], PCA_FEATS_TrailB_OPEN.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_TrailB_OPEN = model.score(X_test, y_test)
print("Score PCA+Features+TrailB-OPEN:", score_PCA_FEATS_TrailB_OPEN)
# Compute relative error for X_test and y_test
y_pred = model.predict(X_test)
RelErr_PCA_FEATS_TrailB_OPEN = np.mean(np.abs(y_pred - y_test) / y_test)

# PCA, Features and TrailB variables (closed)
# Train test split using last column as target
X_train, X_test, y_train, y_test = train_test_split(
    PCA_FEATS_TrailB_CLOSED.iloc[:, :-1], PCA_FEATS_TrailB_CLOSED.iloc[:, -1:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_PCA_FEATS_TrailB_CLOSED = model.score(X_test, y_test)
print("Score PCA+Features+TrailB-CLOSED:", score_PCA_FEATS_TrailB_CLOSED)
# compute relative error for X_test and y_test
y_pred = model.predict(X_test)
RelErr_PCA_FEATS_TrailB_CLOSED = np.mean(np.abs(y_pred - y_test) / y_test)


# Features and Y variables.
# Train test split using last 4 columns as target
X_train, X_test, y_train, y_test = train_test_split(
    Features_Y.iloc[:, :-4], Features_Y.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Score
score_FEATS_Y = model.score(X_test, y_test)
print("Score Features+Y:", score_FEATS_Y)
# Compute relative error for X_test and y_test
y_pred = model.predict(X_test)
RelErr_FEATS_Y = np.mean(np.mean(np.abs(y_pred - y_test) / y_test))

# Create a dataframe with the scores
scores = pd.DataFrame(
    {
        'PCA+Y-CLOSED': [score_PCA_Y_CLOSED],
        'PCA+Y-OPEN': [score_PCA_Y_OPEN],
        'PCA+Features+Y-Open': score_PCA_FEATS_Y_OPEN,
        'PCA+Features+Y-Closed': score_PCA_FEATS_Y_CLOSED,
        'PCA+Features+ACE-OPEN': score_PCA_FEATS_ACE_OPEN,
        'PCA+Features+ACE-CLOSED': [score_PCA_FEATS_ACE_CLOSED],
        'PCA+Features+MMSE-OPEN': score_PCA_FEATS_MMSE_OPEN,
        'PCA+Features+MMSE-CLOSED': [score_PCA_FEATS_MMSE_CLOSED],
        'PCA+Features+TrailA-OPEN': [score_PCA_FEATS_TrailA_OPEN],
        'PCA+Features+TrailA-CLOSED': [score_PCA_FEATS_TrailA_CLOSED],
        'PCA+Features+TrailB-OPEN': [score_PCA_FEATS_TrailB_OPEN],
        'PCA+Features+TrailB-CLOSED': [score_PCA_FEATS_TrailB_CLOSED],
        'Features+Y': [score_FEATS_Y]})
scores = scores.T
scores.columns = ['R-Squared Score']
print(f"\n\nR-Squared score for relevant combinations of PCA,Features and Response Variables (open/closed)\n{scores}")

#Create a dataframe with the relative error
RelErr = pd.DataFrame(
    {
        'PCA+Y-CLOSED': [RelErr_PCA_Y_CLOSED],
        'PCA+Y-OPEN': [RelErr_PCA_Y_OPEN],
        'PCA+Features+Y-Open': [RelErr_PCA_FEATS_Y_OPEN],
        'PCA+Features+Y-Closed': [RelErr_PCA_FEATS_Y_CLOSED],
        'PCA+Features+ACE-OPEN': [RelErr_PCA_FEATS_ACE_OPEN][0],
        'PCA+Features+ACE-CLOSED': RelErr_PCA_FEATS_ACE_CLOSED[0],
        'PCA+Features+MMSE-OPEN': RelErr_PCA_FEATS_MMSE_OPEN[0],
        'PCA+Features+MMSE-CLOSED': RelErr_PCA_FEATS_MMSE_CLOSED[0],
        'PCA+Features+TrailA-OPEN': RelErr_PCA_FEATS_TrailA_OPEN[0],
        'PCA+Features+TrailA-CLOSED': RelErr_PCA_FEATS_TrailA_CLOSED[0],
        'PCA+Features+TrailB-OPEN': RelErr_PCA_FEATS_TrailB_OPEN[0],
        'PCA+Features+TrailB-CLOSED': RelErr_PCA_FEATS_TrailB_CLOSED[0],
        'Features+Y': RelErr_FEATS_Y})
RelErr = RelErr.T
RelErr.columns = ['Relative Error']
print(f"\n\nRelative Error for relevant combinations of PCA,Features and Response Variables (open/closed)\n{RelErr}")

