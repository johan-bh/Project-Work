import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read matrix with PCA + Response variables (386x54) - open and closed eyes
PCA_Y_closed = pd.read_pickle("data/PCA_and_Y_closed.pkl")
PCA_Y_open = pd.read_pickle("data/PCA_and_Y_open.pkl")
Features_Y = pd.read_pickle("data/Features_and_Y.pkl")


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






#Multiple Linear Regression on only Features
Features_Y = pd.read_pickle("data/Features_and_Y.pkl")

#Changing strings to integres
#mapping = {'Samlevende': 1, 'Enke':0, 'Søskende':1}
#Features_Y=Features_Y.replace({'civilstatus': mapping, 'familiedmens': mapping})
Features_Y=Features_Y.replace(to_replace =["Samlevende", "samlevende", "Søskende"], value = 1)
Features_Y=Features_Y.replace(to_replace =["Enke", "Nej15756"], value = 0)

# Scale last 4 columns of Features_Y
scaler = StandardScaler()
scaler.fit(Features_Y.iloc[:, -4:])
Features_Y.iloc[:, -4:] = scaler.transform(Features_Y.iloc[:, -4:])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    Features_Y.iloc[:, :-4], Features_Y.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Score
score_features = model.score(X_test, y_test)
print("Score only Features:", score_features)

# Make predictions on 3 rows
Features_pred = model.predict(Features_Y.iloc[:3, :-4])

# Get actual values on 3 rows
Features_actual = Features_Y.iloc[:3, -4:]

# Inverse transform to get actual values
Features_actual = scaler.inverse_transform(Features_actual)

# Inverse transform to get predicted values
Features_pred = scaler.inverse_transform(Features_pred)
Features_pred = Features_pred.round(1)

#Compare predicted and actual
print("Actual values for Features:\n", Features_actual)
print("Predicted values for Features:\n", Features_pred)

# Compute relative error between actual and predicted values
relative_error_features = (Features_actual - Features_pred) / Features_actual
print("Relative error for only featuress:\n", relative_error_features)



#PCR on features and eyes open
#Multiple Linear Regression on only Features
PCR_Features_Y = pd.read_pickle("data/PCA_and_Y_and_features_open.pkl")

#Changing strings to integres
#mapping = {'Samlevende': 1, 'Enke':0, 'Søskende':1}
#Features_Y=Features_Y.replace({'civilstatus': mapping, 'familiedmens': mapping})
PCR_Features_Y=PCR_Features_Y.replace(to_replace =["Samlevende", "samlevende", "Søskende"], value = 1)
PCR_Features_Y=PCR_Features_Y.replace(to_replace =["Enke", "Nej15756"], value = 0)

# Scale last 4 columns of Features_Y
scaler = StandardScaler()
scaler.fit(PCR_Features_Y.iloc[:, -4:])
PCR_Features_Y.iloc[:, -4:] = scaler.transform(PCR_Features_Y.iloc[:, -4:])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    PCR_Features_Y.iloc[:, :-4], PCR_Features_Y.iloc[:, -4:], test_size=0.2, random_state=42)
# Fit model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Score
score_features = model.score(X_test, y_test)
print("Score only Features:", score_features)

# Make predictions on 3 rows
Features_pred = model.predict(PCR_Features_Y.iloc[:3, :-4])

# Get actual values on 3 rows
Features_actual = PCR_Features_Y.iloc[:3, -4:]

# Inverse transform to get actual values
Features_actual = scaler.inverse_transform(Features_actual)

# Inverse transform to get predicted values
Features_pred = scaler.inverse_transform(Features_pred)
Features_pred = Features_pred.round(1)

#Compare predicted and actual
print("Actual values for Features:\n", Features_actual)
print("Predicted values for Features:\n", Features_pred)

# Compute relative error between actual and predicted values
relative_error_features = (Features_actual - Features_pred) / Features_actual
print("Relative error for only featuress:\n", relative_error_features)


