#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:45:24 2022

@author: smilladue
"""

import pandas as pd

X = pd.read_pickle('/Users/smilladue/Desktop/Documents/DTU/02466_Fagprojekt/Project-Work/data/PCA+Features+Y-TENSOR_CLOSED.pkl')
y = X.iloc[:, -6:]
X = X.iloc[:, :-6]

round(y.describe(),2)
y.boxplot(figsize=[10,5],fontsize='large', column=['MMSE','ACE','TrailMakingA','TrailMakingB', 'DigitSymbol', 'Retention'])