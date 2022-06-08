#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 17:45:24 2022

@author: smilladue
"""

import pandas as pd

X = pd.read_pickle('/Users/smilladue/Downloads/02450Toolbox_Python/Tools/PCA+Features+All4-OPEN.pkl')
y = X.iloc[:, -4:]
X = X.iloc[:, :-4]

round(y.describe(),2)
y.boxplot(column=['MMSE','ACE','TrailMakingA','TrailMakingB'])