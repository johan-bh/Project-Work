import pandas as pd

# # <==== Merge: PCA + Response Vars =====>
# # Read second tab of excel file
# df = pd.read_excel(file_path,sheet_name="CESA II")
# # set cmsk column as index
# df.set_index('cmsk', inplace=True)
# # Only include columns MMSE, ACE and TrailMakingA and TrailMakingB. Keep index as is
# df = df[['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB']]
# # Remove entries with missing values (NaN)
# df = df.dropna()
#
# # load valid_ids.csv from data folder and set first column as index
# valid_ids = pd.read_csv("data/valid_ids.csv", index_col=0)
# # convert first column to list
# valid_ids = valid_ids.iloc[:,0].tolist()
# # Delete all rows that are not in valid_ids
# df = df.drop(df.index[~df.index.isin(valid_ids)])
# # Save to pickle file
# df.to_pickle("data/response_var_df.pkl")

df = pd.read_pickle('data/response_var_df.pkl')


# <==== Merge: PCA + Response Vars =====>

def PCA_Feature_Y_Dims():
    """Returns a dataframe "features" which has the dimensions of the overlap between PCA, Features and Y"""
    pca_dimension = pd.DataFrame(pd.read_pickle("data/pca_open.pkl").components_.T)
    pca_dimension.set_index(df.index, inplace=True)
    pca_dimension = pd.concat([pca_dimension, df], axis=1)
    features = pd.read_pickle("data/clean_features.pkl")
    pca_dimension = pca_dimension.drop(pca_dimension.index[~pca_dimension.index.isin(features.index)])
    features = features.drop(features.index[~features.index.isin(pca_dimension.index)])
    return features

# create function to Merge PCA with Y
def Merge_PCA_Y(eyes):
    """
    Merge PCA with response vars
    :param eyes: string ("closed" or "open")
    :return: Merged dataframe (which is also saved)
    """
    if eyes == "closed":
        data = pd.DataFrame(pd.read_pickle("data/pca_closed.pkl").components_.T)
    elif eyes == "open":
        data = pd.DataFrame(pd.read_pickle("data/pca_open.pkl").components_.T)
    Y = pd.read_pickle("data/response_var_df.pkl")
    data.set_index(Y.index, inplace=True)
    data = pd.concat([data,Y], axis=1)
    # Save the PCA +  response vars (Y) matrix to a pickle file
    data.to_pickle("data/PCA+Y-"+eyes.upper()+".pkl")
    return data

# Merge PCA with Y
for n in ["closed","open"]:
    Merge_PCA_Y(n)

def Merge_PCA_Features_ResponseVar(response_var,eyes):
    """Merge PCA with Features and Response Vars
    :param response_var: string (specific var name or "All4")
    :param eyes: string ("closed" or "open")
    :return: Merged dataframe (which is also saved)
    """
    data = Merge_PCA_Y(eyes)

    if response_var == "All4":
        response_df = data.iloc[:, -4:]
    else:
        response_df = data[response_var]

    # get dimensions of PCA + Feature + Response Vars
    features = PCA_Feature_Y_Dims()
    # drop rows in data that are not in features
    data = data.drop(data.index[~data.index.isin(features.index)])
    response_df = response_df.drop(response_df.index[~response_df.index.isin(features.index)])
    pca_only = data.iloc[:, :50]
    # Append features to PCA data
    # set index of features to be the same as pca_only
    features.set_index(pca_only.index, inplace=True)
    data_merge = pd.concat([pca_only,features], axis=1)
    # Append response_df to data_merge. This is the matrix we want to use for regression (with x features)
    data_merge = pd.concat([data_merge,response_df], axis=1)
    # Save the PCA + Features + x response vars to matrix to a pickle file
    data_merge.to_pickle("data/PCA+Features+"+response_var+"-"+eyes.upper()+".pkl")
    return data_merge

# Merge PCA + Features + Response Vars
response_vars = ['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB','All4']
for n in ["closed","open"]:
    for m in response_vars:
        Merge_PCA_Features_ResponseVar(m,n)


def Merge_Features_Y():
    """Merge Features with Response Vars"""
    features = pd.DataFrame(pd.read_pickle("data/clean_features.pkl"))
    response_vars = pd.DataFrame(pd.read_pickle("data/response_var_df.pkl"))
    # drop rows in response_vars that are not in features
    response_vars = response_vars.drop(response_vars.index[~response_vars.index.isin(features.index)])
    # drop rows in features that are not in response_vars
    features = features.drop(features.index[~features.index.isin(response_vars.index)])
    # append response_vars to features
    data = pd.concat([features,response_vars], axis=1)
    # Save the Features +  response vars (Y) matrix to a pickle file
    data.to_pickle("data/Features+Y.pkl")
    return data

# Merge Features + Y
Merge_Features_Y()