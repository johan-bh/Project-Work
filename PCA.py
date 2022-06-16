import pickle
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plotting = False
ica = False

if ica == False:
    # Load the processed data for eyes closed and eyes open
    with open('data/coherence_maps_closed.pkl', 'rb') as f:
        coherence_maps_closed = pickle.load(f)
    with open('data/coherence_maps_open.pkl', 'rb') as f:
        coherence_maps_open = pickle.load(f)
if ica == True:
    # Load the processed data for eyes closed and eyes open
    with open('data/ICA_coherence_maps_closed.pkl', 'rb') as f:
        coherence_maps_closed = pickle.load(f)
    with open('data/ICA_coherence_maps_open.pkl', 'rb') as f:
        coherence_maps_open = pickle.load(f)

# read response var dataframe
with open('data/response_var_df.pkl', 'rb') as f:
    response_var_df = pickle.load(f)
# set index of response_var_df to string
response_var_df.index = response_var_df.index.astype(str)


print("Creating dataframes...")
# Convert dictionary to panda dataframe use keys as index for eyes closed
df_closed = pd.DataFrame.from_dict(coherence_maps_closed, orient='index')
# # Create index name "Patient ID"
# df_closed.index.name = 'Patient ID'
# df_closed.columns = np.arange(1,df_closed.shape[1]+1)
# col_names = [f"Coh. val {n}" for n in df_closed.columns]
# df_closed.columns = col_names

# count total number of NaN values in df_closed, determine fraction of missing values
df_closed_missing = df_closed.isnull().sum().sum()
df_closed_missing_fraction = df_closed_missing / (df_closed.shape[0]*df_closed.shape[1])
print("Number of missing values in df_closed:", df_closed_missing)
print("Fraction of missing values in df_closed:", df_closed_missing_fraction)

# # Same process for eyes open
df_open = pd.DataFrame.from_dict(coherence_maps_open, orient='index')
# # df_open.index.name = 'Patient ID'
# # df_open.columns = np.arange(1,df_open.shape[1]+1)
# # col_names = [f"Coh. val {n}" for n in df_open.columns]
# # df_open.columns = col_names

# # count total number of NaN values in df_closed, determine fraction of missing values
# df_closed_missing = df_closed.isnull().sum().sum()
# df_closed_missing_fraction = df_closed_missing / (df_closed.shape[0]*df_closed.shape[1])
# print("Number of missing values in df_closed:", df_closed_missing)
# print("Fraction of missing values in df_closed:", df_closed_missing_fraction)

# loop through all indexes and print if they have missing values
# missing_vals = {}
# for index in df_closed.index:
#     if df_closed.loc[index].isnull().sum().sum() > 0:
#         missing_vals[index] = df_closed.loc[index].isnull().sum().sum()
# print("Number of missing values in df_closed:\n", missing_vals)
# # print key with with highest value in missing_vals, and value
# print("Index with the most missing values:", max(missing_vals, key=missing_vals.get))
# # print key "16467" in missing_vals
# print("Number of missing values in df_closed:", missing_vals["16467"])

# 58 patients had missing values in their coherence maps vector. With a fraction of NaN values ranging from 13% to 92%,
# this resulted in a total NaN value fractin of 9%. We therefore chose to drop these patients from our analysis.
df_closed.dropna(inplace=True)
df_open.dropna(inplace=True)


# Delete all rows in df_closed and df_open that dont have the same index as df
# This ensures we have the same number of rows in both dataframes.
df_closed = df_closed.drop(df_closed.index[~df_closed.index.isin(response_var_df.index)])
df_open = df_open.drop(df_open.index[~df_open.index.isin(response_var_df.index)])
response_var_df = response_var_df.drop(response_var_df.index[~response_var_df.index.isin(df_closed.index)])

# pickle response_var_df
with open('data/response_var_df.pkl', 'wb') as f:
    pickle.dump(response_var_df, f)

# Transpose the dataframe and reset index
print("Transposing dataframes...")
df_closed = df_closed.transpose()
df_open = df_open.transpose()

print("Creating PCA model for eyes closed..")
pca_closed = PCA(n_components=50)
print("Creating PCA model for eyes open..")
pca_open = PCA(n_components=50)

# # Fit the PCA model to the data
print("Fitting PCA model to eyes closed data...")
pca_closed.fit(df_closed)
print("Fitting PCA model to eyes open data...")
pca_open.fit(df_open)

# print the shape of the pca components
print("Shape of PCA components for eyes closed:", pca_closed.components_.shape)
print("Shape of PCA components for eyes open:", pca_open.components_.shape)

# # Transform the data
print("Transforming data for closed eyes...")
df_closed_pca = pca_closed.transform(df_closed)
print("Transforming data for open eyes...")
df_open_pca = pca_open.transform(df_open)



print("Pickling PCA models...")
if ica == False:
    # Pickle the PCA model
    with open('data/pca_closed.pkl', 'wb') as f:
        pickle.dump(pca_closed, f)
    with open('data/pca_open.pkl', 'wb') as f:
        pickle.dump(pca_open, f)
if ica == True:
    # Pickle the PCA model
    with open('data/ICA_pca_closed.pkl', 'wb') as f:
        pickle.dump(pca_closed, f)
    with open('data/ICA_pca_open.pkl', 'wb') as f:
        pickle.dump(pca_open, f)

if plotting == True:
    print("Plotting PCA results for eyes closed...")
    # Plot the explained variance ratio for closed eyes
    plt.figure(figsize=(10,10))
    plt.plot(np.cumsum(pca_closed.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Closed Eyes')
    # Add horizontal line at 80% explained variance
    plt.axhline(y=0.8, color='r', linestyle='-')
    # Save the figure
    print("Saving PCA figure (closed eyes)...")
    plt.savefig('figures/PCA_Closed_explained_variance.png')
    plt.show()
    print("Plotting PCA results for eyes open...")
    # Plot the explained variance ratio for open eyes
    plt.figure(figsize=(10,10))
    plt.plot(np.cumsum(pca_open.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Open Eyes')
    # Add horizontal line at 80% explained variance
    plt.axhline(y=0.8, color='r', linestyle='-')
    # Save the figure
    print("Saving PCA figure (open eyes)...")
    plt.savefig('figures/PCA_Open_explained_variance.png')
    plt.show()

