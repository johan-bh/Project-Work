import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read matrix with PCA + Response variables (386x54) - open and closed eyes
PCA_Y_closed = pd.read_pickle("data/PCA_and_Y_closed.pkl")
PCA_Y_open = pd.read_pickle("data/PCA_and_Y_open.pkl")


# Scale last 4 columns of PCA_y_closed
scaler = StandardScaler()
scaler.fit(PCA_Y_closed.iloc[:, -4:])
PCA_Y_closed.iloc[:, -4:] = scaler.transform(PCA_Y_closed.iloc[:, -4:])

# Train-test split
X_train_closed, X_test_closed, y_train_closed, y_test_closed = train_test_split(
    PCA_Y_closed.iloc[:, :-4], PCA_Y_closed.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model_closed = LinearRegression()
model_closed.fit(X_train_closed, y_train_closed)
# Predict
y_pred_closed = model_closed.predict(X_test_closed)
# Score
score_closed = model_closed.score(X_test_closed, y_test_closed)
print("Score closed eyes:", score_closed)

# Scale last 4 columns of PCA_y_open
scaler = StandardScaler()
scaler.fit(PCA_Y_open.iloc[:, -4:])
PCA_Y_open.iloc[:, -4:] = scaler.transform(PCA_Y_open.iloc[:, -4:])

# Train-test split
X_train_open, X_test_open, y_train_open, y_test_open = train_test_split(
    PCA_Y_open.iloc[:, :-4], PCA_Y_open.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model_open = LinearRegression()
model_open.fit(X_train_open, y_train_open)
# Predict
y_pred_open = model_open.predict(X_test_open)
# Score
score_open = model_open.score(X_test_open, y_test_open)
print("Score open eyes:", score_open)

# Make predictions for closed and open eyes on 3 rows
PCA_Y_closed_pred = model_closed.predict(PCA_Y_closed.iloc[:3, :-4])
PCA_Y_open_pred = model_open.predict(PCA_Y_open.iloc[:3, :-4])

# Get actual values for closed and open eyes on 3 rows
PCA_Y_closed_actual = PCA_Y_closed.iloc[:3, -4:]
PCA_Y_open_actual = PCA_Y_open.iloc[:3, -4:]
# Inverse transform to get actual values
PCA_Y_closed_actual = scaler.inverse_transform(PCA_Y_closed_actual)
PCA_Y_open_actual = scaler.inverse_transform(PCA_Y_open_actual)
# Inverse transform to get predicted values
PCA_Y_closed_pred = scaler.inverse_transform(PCA_Y_closed_pred)
PCA_Y_open_pred = scaler.inverse_transform(PCA_Y_open_pred)

PCA_Y_closed_pred = PCA_Y_closed_pred.round(1)
PCA_Y_open_pred = PCA_Y_open_pred.round(1)

# compare actual and predicted values
print("Actual values for closed eyes:\n", PCA_Y_closed_actual)
print("Predicted values for closed eyes:\n", PCA_Y_closed_pred)

# Compute relative error between actual and predicted values
relative_error_closed = (PCA_Y_closed_actual - PCA_Y_closed_pred) / PCA_Y_closed_actual
relative_error_open = (PCA_Y_open_actual - PCA_Y_open_pred) / PCA_Y_open_actual
print("Relative error for closed eyes:\n", relative_error_closed)
print("Relative error for open eyes:\n", relative_error_open)

















