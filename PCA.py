import pickle
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("Loading data...")
# Load the processed data for eyes closed and eyes open
with open('data/coherence_maps_closed.pkl', 'rb') as f:
    coherence_maps_closed = pickle.load(f)
with open('data/coherence_maps_open.pkl', 'rb') as f:
    coherence_maps_open = pickle.load(f)

file_path = "C:\\Users\\jbhan\\Desktop\\neurotest.xlsx"
# Read second tab of excel file
df = pd.read_excel(file_path,sheet_name="CESA II")
# set cmsk column as index
df.set_index('cmsk', inplace=True)
# Only include columns MMSE, ACE and TrailMakingA and TrailMakingB. Keep index as is
df = df[['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB']]
# Remove entries with missing values
df = df.dropna()

print("Creating dataframes...")
# Convert dictionary to panda dataframe use keys as index for eyes closed
df_closed = pd.DataFrame.from_dict(coherence_maps_closed, orient='index')
# # Create index name "Patient ID"
# df_closed.index.name = 'Patient ID'
# df_closed.columns = np.arange(1,df_closed.shape[1]+1)
# col_names = [f"Coh. val {n}" for n in df_closed.columns]
# df_closed.columns = col_names

# # Same process for eyes open
df_open = pd.DataFrame.from_dict(coherence_maps_open, orient='index')
# # df_open.index.name = 'Patient ID'
# # df_open.columns = np.arange(1,df_open.shape[1]+1)
# # col_names = [f"Coh. val {n}" for n in df_open.columns]
# # df_open.columns = col_names

folder_path1  = "C:\\Users\\jbhan\\Desktop\\AA_CESA-2-DATA-EEG-Resting (Anden del)\\"
folder_path2 = "C:\\Users\\jbhan\\Desktop\\AA_CESA-2-DATA-EEG-Resting\\"

# Create lists of filenames in each folder. Only use files that starts with "S" and ends with ".dat"
file_names1 = [f for f in os.listdir(folder_path1) if f.endswith('.dat') and f.startswith('S')]
file_names2 = [f for f in os.listdir(folder_path2) if f.endswith('.dat') and f.startswith('S')]
# Delete the following fiels from their respective folders. They return errors...
if 'S26_12604_R001.dat' in file_names2:
    file_names2.remove('S26_12604_R001.dat')
if "S195_11851_R001.dat" in file_names1:
    file_names1.remove("S195_11851_R001.dat")
if "S309_11498_R001.dat" in file_names1:
    file_names1.remove("S309_11498_R001.dat")

# Get id that is written between underscores (starts with underscore and ends with underscore)
id_list1 = [f.split('_')[1].split('.')[0] for f in file_names1]
id_list2 = [f.split('_')[1].split('.')[0] for f in file_names2]
# merge the two lists
id_list = id_list1 + id_list2

# Convert df index to string
df.index = df.index.astype(str)

# Delete all rows that are not in id_list.
# We only keep rows that have matchin id's and valid health data entries
df = df.drop(df.index[~df.index.isin(id_list)])

# Delete all rows in df_closed and df_open that dont have the same index as df
# This ensures we have the same number of rows in both dataframes.
df_closed = df_closed.drop(df_closed.index[~df_closed.index.isin(df.index)])
df_open = df_open.drop(df_open.index[~df_open.index.isin(df.index)])

# # Use df.fillna with interpolation to fill NaN values
df_closed = df_closed.fillna(method='ffill')
df_open = df_open.fillna(method='ffill')

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

# # Transform the data
print("Transforming data for closed eyes...")
df_closed_pca = pca_closed.transform(df_closed)
print("Transforming data for open eyes...")
df_open_pca = pca_open.transform(df_open)

print("Pickling PCA models...")
# Pickle the PCA model
with open('data/pca_closed.pkl', 'wb') as f:
    pickle.dump(pca_closed, f)
with open('data/pca_open.pkl', 'wb') as f:
    pickle.dump(pca_open, f)


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

