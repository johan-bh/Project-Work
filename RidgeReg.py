import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle


# Current not working. Very slow because of the large data set.

# Load the processed (non PCA) data for eyes closed and eyes open
with open('data/coherence_maps_closed.pkl', 'rb') as f:
    coherence_maps_closed = pickle.load(f)
with open('data/coherence_maps_open.pkl', 'rb') as f:
    coherence_maps_open = pickle.load(f)
df_open = pd.DataFrame.from_dict(coherence_maps_open, orient='index')
df_closed = pd.DataFrame.from_dict(coherence_maps_closed, orient='index')

# get data/response_var_df.pkl
with open('data/response_var_df.pkl', 'rb') as f:
    df = pickle.load(f)

# Drop rows that dont have corresponding response variables
df.index = df.index.astype(str)
df_closed = df_closed.drop(df_closed.index[~df_closed.index.isin(df.index)])
df_open = df_open.drop(df_open.index[~df_open.index.isin(df.index)])

# fill nan falues with mean of the given column
df_closed = df_closed.fillna(df_closed.mean())
df_open = df_open.fillna(df_open.mean())

# Implement Ridge Regression on df_closed
X_train, X_test, y_train, y_test = train_test_split(df_closed, df, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Find the best alpha value. Store in variable best_alpha
best_alpha = 0
best_score = 0
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    # print progress
    print('alpha:', alpha)
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    if score > best_score:
        best_alpha = alpha
        best_score = score
# Print the best alpha value
print('Best alpha value:', best_alpha)
# Fit the model with best_alpha
reg = Ridge(alpha=best_alpha)
reg.fit(X_train, y_train)
# print the score
print(reg.score(X_test, y_test))


