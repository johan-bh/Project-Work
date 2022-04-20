import pandas as pd
import os

# This function merges the PCA and the neurotest data. It excludes entries with missing values and entries not present in both datasets.

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
file_path = "C:\\Users\\jbhan\\Desktop\\neurotest.xlsx"

# Read second tab of excel file

df = pd.read_excel(file_path,sheet_name="CESA II")
# set cmsk column as index
df.set_index('cmsk', inplace=True)
# Only include columns MMSE, ACE and TrailMakingA and TrailMakingB. Keep index as is
df = df[['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB']]

# Remove entries with missing values
df = df.dropna()

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

# Delete all rows that are not in id_list
df = df.drop(df.index[~df.index.isin(id_list)])

# open "data/pca_closed.pkl"
pca_closed = pd.read_pickle("data/pca_closed.pkl")
pca_closed = pca_closed.components_.T
# convert pca_closed to dataframe
pca_closed = pd.DataFrame(pca_closed)

# Set index of pca_closed to index of df (neuro test dataframe)
pca_closed.set_index(df.index, inplace=True)

# Append df to pca_closed. This is the matrix we want to use for regression
data = pd.concat([pca_closed, df], axis=1)
# Save this dataframe to "data/ai_data.pkl"
data.to_pickle("data/ai_data.pkl")