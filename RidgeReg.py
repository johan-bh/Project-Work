import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

data = pd.read_pickle("data/ai_data.pkl")

# Use standard scaler to scale "data", only scale last 4 columns
scaler = StandardScaler()
scaler.fit(data.iloc[:, -4:])
data_scaled = scaler.transform(data.iloc[:, -4:])
# convert data_scaled to dataframe
data_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns[-4:])
# Remove last 4 columns from data and append data_scaled
data = data.iloc[:,:-4]
data = pd.concat([data, data_scaled], axis=1)

# Split data into train and test. Use last 4 columns as target
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-4], data.iloc[:,-4:], test_size=0.1, random_state=42)
# Find the best alpha value
alphas = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
scores = []
for alpha in alphas:
    reg_ridge = Ridge(alpha=alpha)
    reg_ridge.fit(X_train, y_train)
    scores.append(reg_ridge.score(X_test, y_test))
# return the best alpha value
print("\nBest alpha value: ", alphas[scores.index(max(scores))])
alpha = alphas[scores.index(max(scores))]
reg_ridge = Ridge(alpha=alpha)
reg_ridge.fit(X_train, y_train)
# Predict on 2 rows of test data
y_pred_2rows = reg_ridge.predict(X_test.iloc[0:3,:])
y_actual_2rows = y_test.iloc[0:3,:]

# descale y_pred_2rows
y_pred_2rows = scaler.inverse_transform(y_pred_2rows)
# descale y_actual_2rows
y_actual_2rows = scaler.inverse_transform(y_actual_2rows)
# round y_pred_2rows to 1 decimal
y_pred_2rows = y_pred_2rows.round(1)
print("\nPredicted values:\n", y_pred_2rows)
print("\nActual values:\n", y_actual_2rows)
print("\nAccuracy: \n", reg_ridge.score(X_test, y_test))
# lorte accuracy mand... :((