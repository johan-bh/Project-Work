import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
file_path = "C:\\Users\\jbhan\\Desktop\\neurotest.xlsx"

# <==== Merge: PCA + Response Vars =====>
# Read second tab of excel file
df = pd.read_excel(file_path,sheet_name="CESA II")
# set cmsk column as index
df.set_index('cmsk', inplace=True)
# Only include columns MMSE, ACE and TrailMakingA and TrailMakingB. Keep index as is
df = df[['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB']]
# Remove entries with missing values (NaN)
df = df.dropna()

# load valid_ids.csv from data folder and set first column as index
valid_ids = pd.read_csv("data/valid_ids.csv", index_col=0)
# convert first column to list
valid_ids = valid_ids.iloc[:,0].tolist()
# Delete all rows that are not in valid_ids
df = df.drop(df.index[~df.index.isin(valid_ids)])
# Save to pickle file
df.to_pickle("data/response_var_df.pkl")

# open "data/pca_closed.pkl"
pca_closed = pd.read_pickle("data/pca_closed.pkl")

pca_closed = pca_closed.components_.T
# convert pca_closed to dataframe
pca_closed = pd.DataFrame(pca_closed)

# Set index of pca_closed to index of df (neuro test dataframe)
pca_closed.set_index(df.index, inplace=True)

# Append df to pca_closed. This is the matrix we want to use for regression
data = pd.concat([pca_closed, df], axis=1)

# Save the PCA +  response vars (Y) matrix to a pickle file (closed)
data.to_pickle("data/PCA_and_Y_closed.pkl")

# open "data/pca_open.pkl"
pca_open = pd.read_pickle("data/pca_open.pkl")

pca_open = pca_open.components_.T
# convert pca_open to dataframe
pca_open = pd.DataFrame(pca_open)

# set index of pca_open to index of df (neuro test dataframe)
pca_open.set_index(df.index, inplace=True)

# Append df to pca_open. This is the matrix we want to use for analysis (without features)
data = pd.concat([pca_open, df], axis=1)

# Save the PCA +  response vars (Y) matrix to a pickle file (open)
data.to_pickle("data/PCA_and_Y_open.pkl")
# < ====== END  =======>

# < ====== Merge Data: PCA + Features + Response Vars =======>
features = pd.read_pickle("data/clean_features.pkl")
# remove the rows in data if their index is not in features
data = data.drop(data.index[~data.index.isin(features.index)])

# remove the rows in features if their index is not in data
features = features.drop(features.index[~features.index.isin(data.index)])

# store last 4 columns of data in temporary df
temp_df = data.iloc[:, -4:]
# remove last 4 columns of data
data = data.iloc[:, :-4]

# Append features to data. T
data = pd.concat([data,features], axis=1)

# Append temp_df to data. This is the matrix we want to use for regression (with features)
data = pd.concat([data,temp_df], axis=1)

# Save the PCA +  response vars (Y) matrix to a pickle file (open eyes)
data.to_pickle("data/PCA_and_Y_and_features_open.pkl")

#featuresY=data
#featuresY.drop(featuresY.iloc[:, 0:50], inplace = True, axis = 1)
#Changing strings to integres
#Features_Y=Features_Y.replace(to_replace =["Samlevende", "samlevende", "Søskende"], value = 1)
#Features_Y=Features_Y.replace(to_replace =["Enke", "Nej15756"], value = 0)
#PCA_open_features_Y=PCA_open_features_Y.replace(to_replace =["Samlevende", "samlevende", "Søskende"], value = 1)
#PCA_open_features_Y=PCA_open_features_Y.replace(to_replace =["Enke", "Nej15756"], value = 0)

# Create matrix with PCA + features + response vars (Y) matrix (closed eyes)....
