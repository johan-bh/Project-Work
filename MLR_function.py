#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:42:41 2022

@author: smilladue
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read matrix with PCA + Response variables (386x54) - open and closed eyes
PCA_Y_closed = pd.read_pickle("data/PCA+Y-CLOSED.pkl")
PCA_Y_open = pd.read_pickle("data/PCA+Y-OPEN.pkl")
Features_Y = pd.read_pickle("data/Features+Y.pkl")
PCA_open_features_Y = pd.read_pickle("data/PCA+Features+All4-OPEN.pkl")


def MLR(A, A_navn):
    # Scale last 4 columns
    scaler = StandardScaler()
    scaler.fit(A.iloc[:, -4:])
    A.iloc[:, -4:] = scaler.transform(A.iloc[:, -4:])
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
       A.iloc[:, :-4], A.iloc[:, -4:], test_size=0.2, random_state=42)
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Score
    score_features = model.score(X_test, y_test)
    print("Score {}:".format(A_navn), score_features)

    # Make predictions on 3 rows
    A_pred = model.predict(A.iloc[:3, :-4])

    # Get actual values on 3 rows
    A_actual = A.iloc[:3, -4:]

    # Inverse transform to get actual values
    A_actual = scaler.inverse_transform(A_actual)

    # Inverse transform to get predicted values
    A_pred = scaler.inverse_transform(A_pred)
    A_pred = A_pred.round(1)

    #Compare predicted and actual
    print("Actual values for {}:\n".format(A_navn), A_actual)
    print("Predicted values for {}:\n".format(A_navn), A_pred)

    # Compute relative error between actual and predicted values
    relative_error_features = (A_actual - A_pred) / A_actual
    # Compute relative
    print("Relative error for {} :\n".format(A_navn), relative_error_features)

MLR(PCA_Y_closed, "PCA closed eyes")
MLR(PCA_Y_open, "PCR closed eyes")
MLR(Features_Y, "Only Features")
MLR(PCA_open_features_Y, "PCA for open eyes and Features")

