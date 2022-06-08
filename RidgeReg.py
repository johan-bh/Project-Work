import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
# import timer for timing the code
import time
import numpy as np


start = time.time()
# Load Coherence+Y-CLOSED.pkl
coherence_closed = pd.read_pickle("data/Coherence+Y-closed.pkl")
coherence_open = pd.read_pickle("data/Coherence+Y-open.pkl")

# # remove 90 percent of the rows
# coherence_closed = coherence_closed.sample(frac=0.1, random_state=42)
# coherence_open = coherence_open.sample(frac=0.1, random_state=42)

# convert dataframes to numpy arrays
coherence_closed = coherence_closed.to_numpy()
coherence_open = coherence_open.to_numpy()

# Split data into train and test. Last 4 columns are the response variables
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(
    coherence_closed[:, :-4], coherence_closed[:, -4:], test_size=0.2, random_state=42)
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(
    coherence_open[:, :-4], coherence_open[:, -4:], test_size=0.2, random_state=42)
# Run Ridge Regression on the train data
ridge_closed = Ridge(alpha=0.1)
ridge_closed.fit(X_train_closed, y_train_closed)
ridge_open = Ridge(alpha=0.1)
ridge_open.fit(X_train_open, y_train_open)
# score the model
coh_y_closed_score = ridge_closed.score(X_test_closed, y_test_closed)
coh_y_open_score = ridge_open.score(X_test_open, y_test_open)
print("Score for closed eyes:", coh_y_closed_score)
print("Score for open eyes:", coh_y_open_score)

# Split data into train and test. Use last column as target variable, remove the other response variables
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(
    coherence_closed[:, :-4], coherence_closed[:, -1:], test_size=0.2, random_state=42)
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(
    coherence_open[:, :-4], coherence_open[:, -1:], test_size=0.2, random_state=42)
# Run Ridge Regression on the train data
ridge_closed = Ridge(alpha=0.1)
ridge_closed.fit(X_train_closed, y_train_closed)
ridge_open = Ridge(alpha=0.1)
ridge_open.fit(X_train_open, y_train_open)
# score the model
coh_trailB_closed_score = ridge_closed.score(X_test_closed, y_test_closed)
coh_trailB_open_score = ridge_open.score(X_test_open, y_test_open)
print("Score for closed eyes:", coh_trailB_closed_score)
print("Score for open eyes:", coh_trailB_open_score)

# Split data into train and test. Use second last column as target variable, remove the other response variables
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(
    coherence_closed[:, :-4], coherence_closed[:, -2:-1], test_size=0.2, random_state=42)
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(
    coherence_open[:, :-4], coherence_open[:, -2:-1], test_size=0.2, random_state=42)
# Run Ridge Regression on the train data
ridge_closed = Ridge(alpha=0.1)
ridge_closed.fit(X_train_closed, y_train_closed)
ridge_open = Ridge(alpha=0.1)
ridge_open.fit(X_train_open, y_train_open)
# score the model
coh_trailA_closed_score = ridge_closed.score(X_test_closed, y_test_closed)
coh_trailA_open_score = ridge_open.score(X_test_open, y_test_open)

# Split data into train and test. Use third last column as target variable, remove the other response variables
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(
    coherence_closed[:, :-4], coherence_closed[:, -3:-2], test_size=0.2, random_state=42)
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(
    coherence_open[:, :-4], coherence_open[:, -3:-2], test_size=0.2, random_state=42)
# Run Ridge Regression on the train data
ridge_closed = Ridge(alpha=0.1)
ridge_closed.fit(X_train_closed, y_train_closed)
ridge_open = Ridge(alpha=0.1)
ridge_open.fit(X_train_open, y_train_open)
# score the model
coh_ACE_closed_score = ridge_closed.score(X_test_closed, y_test_closed)
coh_ACE_open_score = ridge_open.score(X_test_open, y_test_open)
print("Score for closed eyes:", coh_ACE_closed_score)
print("Score for open eyes:", coh_ACE_open_score)

# Split data into train and test. Use fourth last column as target variable, remove the other response variables
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(
    coherence_closed[:, :-4], coherence_closed[:, -4:-3], test_size=0.2, random_state=42)
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(
    coherence_open[:, :-4], coherence_open[:, -4:-3], test_size=0.2, random_state=42)
# Run Ridge Regression on the train data
ridge_closed = Ridge(alpha=0.1)
ridge_closed.fit(X_train_closed, y_train_closed)
ridge_open = Ridge(alpha=0.1)
ridge_open.fit(X_train_open, y_train_open)
# score the model
coh_MMSE_closed_score = ridge_closed.score(X_test_closed, y_test_closed)
coh_MMSE_open_score = ridge_open.score(X_test_open, y_test_open)
print("Score for closed eyes:", coh_MMSE_closed_score)
print("Score for open eyes:", coh_MMSE_open_score)

# Create a dataframe with the scores
coh_scores = pd.DataFrame(
    {'coh_y_closed_score': [coh_y_closed_score], 'coh_y_open_score': [coh_y_open_score],
        'coh_trailB_closed_score': [coh_trailB_closed_score], 'coh_trailB_open_score': [coh_trailB_open_score],
        'coh_trailA_closed_score': [coh_trailA_closed_score], 'coh_trailA_open_score': [coh_trailA_open_score],
        'coh_ACE_closed_score': [coh_ACE_closed_score], 'coh_ACE_open_score': [coh_ACE_open_score],
        'coh_MMSE_closed_score': [coh_MMSE_closed_score], 'coh_MMSE_open_score': [coh_MMSE_open_score]})
coh_scores.index = ['R-Squared Score']
print(coh_scores)
# Save the dataframe to a csv file
coh_scores.to_csv('data/coh_scores.csv')
end = time.time()
print("Time:", end - start)


# X_train, X_test, y_train, y_test = train_test_split(df_closed, df, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# # Find the best alpha value. Store in variable best_alpha
# best_alpha = 0
# best_score = 0
# for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
#     # print progress
#     print('alpha:', alpha)
#     reg = Ridge(alpha=alpha)
#     reg.fit(X_train, y_train)
#     score = reg.score(X_test, y_test)
#     if score > best_score:
#         best_alpha = alpha
#         best_score = score
# # Print the best alpha value
# print('Best alpha value:', best_alpha)
# # Fit the model with best_alpha
# reg = Ridge(alpha=best_alpha)
# reg.fit(X_train, y_train)
# # print the score
# print(reg.score(X_test, y_test))


