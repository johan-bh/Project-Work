import pandas as pd
import pickle
import time
ica = True

df = pd.read_pickle('data/response_var_df.pkl')



# <==== Merge: PCA + Response Vars =====>

def PCA_Feature_Y_Dims():
    """Returns a dataframe "features" which has the dimensions of the overlap between PCA, Features and Y"""
    if ica == True:
        pca_dimension = pd.DataFrame(pd.read_pickle("data/ICA_pca_open.pkl").components_.T)
    else:
        pca_dimension = pd.DataFrame(pd.read_pickle("data/pca_open.pkl").components_.T)
    pca_dimension.set_index(df.index, inplace=True)
    pca_dimension = pd.concat([pca_dimension, df], axis=1)
    features = pd.read_pickle("data/clean_features.pkl")
    pca_dimension = pca_dimension.drop(pca_dimension.index[~pca_dimension.index.isin(features.index)])
    features = features.drop(features.index[~features.index.isin(pca_dimension.index)])
    return features

# print(PCA_Feature_Y_Dims())

# create function to Merge PCA with Y
def Merge_PCA_Y(eyes):
    """
    Merge PCA with response vars
    :param eyes: string ("closed" or "open")
    :return: Merged dataframe (which is also saved)
    """
    if eyes == "closed":
        if ica == False:
            data = pd.DataFrame(pd.read_pickle("data/pca_closed.pkl").components_.T)
        else:
            data = pd.DataFrame(pd.read_pickle("data/ICA_pca_closed.pkl").components_.T)

    elif eyes == "open":
        if ica == False:
            data = pd.DataFrame(pd.read_pickle("data/pca_open.pkl").components_.T)
        else:
            data = pd.DataFrame(pd.read_pickle("data/ICA_pca_open.pkl").components_.T)
    Y = pd.read_pickle("data/response_var_df.pkl")

    data.set_index(Y.index, inplace=True)

    # # concatenate data and each column of Y separately
    # for n in range(len(Y.columns)):
    #     data = data
    #     data_x = pd.concat([data,Y.iloc[:,n]], axis=1)
    #     # get column name of Y
    #     column_name = Y.columns[n]
    #     data_x.to_pickle(f"data/PCA+{column_name}-"+eyes.upper()+".pkl")

    # concate data with Y (all response vars)
    data = pd.concat([data,Y], axis=1)
    # Save the PCA + Y matrix to a pickle file
    if ica == False:
        data.to_pickle("data/PCA+Y-"+eyes.upper()+".pkl")
    else:
        data.to_pickle("data/ICA_PCA+Y-"+eyes.upper()+".pkl")
    return data

# Merge PCA with Y
for n in ["closed","open"]:
    Merge_PCA_Y(n)


def Merge_PCA_Features_ResponseVar(eyes):
    """Merge PCA with Features and Response Vars
    :param response_var: string (specific var name or "Y" - all response vars)
    :param eyes: string ("closed" or "open")
    :return: Merged dataframe (which is also saved)
    """
    data = Merge_PCA_Y(eyes)
    response_df = data.iloc[:, -6:]

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
    if ica == False:
        data_merge.to_pickle("data/PCA+Features+"+"Y"+"-"+eyes.upper()+".pkl")
    else:
        data_merge.to_pickle("data/ICA_PCA+Features+"+"Y"+"-"+eyes.upper()+".pkl")
    return data_merge
#
# Merge PCA + Features + Response Vars
for n in ["closed","open"]:
    Merge_PCA_Features_ResponseVar(n)


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
    return data.shape

# Merge Features + Y
print(Merge_Features_Y())



def Merge_Coherence_Y():
    """
    Merge coherence maps with response vars
    :return: Merged dataframe (which is also saved)
    """
    print("Loading coherence maps...")
    if ica == False:
        with open('data/coherence_maps_closed.pkl', 'rb') as f:
            coherence_maps_closed = pickle.load(f)
        with open('data/coherence_maps_open.pkl', 'rb') as f:
            coherence_maps_open = pickle.load(f)
    else:
        with open('data/ICA_coherence_maps_closed.pkl', 'rb') as f:
            coherence_maps_closed = pickle.load(f)
        with open('data/ICA_coherence_maps_open.pkl', 'rb') as f:
            coherence_maps_open = pickle.load(f)
    df_open = pd.DataFrame.from_dict(coherence_maps_open, orient='index')
    df_closed = pd.DataFrame.from_dict(coherence_maps_closed, orient='index')

    # get data/response_var_df.pkl
    with open('data/response_var_df.pkl', 'rb') as f:
        df = pickle.load(f)

    print("Dropping rows that arent in both dataframes...")
    # Drop rows that dont have corresponding response variables
    df.index = df.index.astype(str)
    df_closed = df_closed.drop(df_closed.index[~df_closed.index.isin(df.index)])
    df_open = df_open.drop(df_open.index[~df_open.index.isin(df.index)])
    # drop rows in df that are not in df_closed
    df = df.drop(df.index[~df.index.isin(df_closed.index)])
    # drop rows in df that are not in df_open
    df = df.drop(df.index[~df.index.isin(df_open.index)])


    # Append df to df_closed and df_open
    df_closed = pd.concat([df_closed,df], axis=1)
    df_open = pd.concat([df_open,df], axis=1)
    if ica == False:
        # Save the Coherence +  response vars (Y) matrix to a pickle file
        df_closed.to_pickle("data/Coherence+Y-CLOSED.pkl")
        df_open.to_pickle("data/Coherence+Y-OPEN.pkl")
    else:
        # Save the ICA Coherence +  response vars (Y) matrix to a pickle file
        df_closed.to_pickle("data/ICA_Coherence+Y-CLOSED.pkl")
        df_open.to_pickle("data/ICA_Coherence+Y-OPEN.pkl")
    return (df_closed,df_open)

# print(Merge_Coherence_Y())

def COH_Feature_Y_Dims():
    """Returns a dataframe "features" which has the dimensions of the overlap between Coherence Maps, Features and Y"""
    if ica == False:
        coherence_dims = pd.read_pickle("data/Coherence+Y-CLOSED.pkl")
    else:
        coherence_dims = pd.read_pickle("data/ICA_Coherence+Y-CLOSED.pkl")
    coherence_dims.set_index(df.index, inplace=True)
    coherence_dims = pd.concat([coherence_dims,df], axis=1)
    features = pd.read_pickle("data/clean_features.pkl")
    coherence_dims = coherence_dims.drop(coherence_dims.index[~coherence_dims.index.isin(features.index)])
    features = features.drop(features.index[~features.index.isin(coherence_dims.index)])
    return features

# print(COH_Feature_Y_Dims())



def Merge_Coherence_Feats_Y():
    """"This function merges the coherence maps with the features and response vars"""
    df_closed, df_open = Merge_Coherence_Y()
    df_closed_responseVars = df_closed.iloc[:, -6:]
    df_open_responseVars = df_open.iloc[:, -6:]
    features = COH_Feature_Y_Dims()
    df_closed = df_closed.drop(df_closed.index[~df_closed.index.isin(features.index)])
    df_open = df_open.drop(df_open.index[~df_open.index.isin(features.index)])
    df_closed_responseVars = df_closed_responseVars.drop(df_closed_responseVars.index[~df_closed_responseVars.index.isin(features.index)])
    df_open_responseVars = df_open_responseVars.drop(df_open_responseVars.index[~df_open_responseVars.index.isin(features.index)])
    coh_only_closed = df_closed.iloc[:, :-6]
    coh_only_open = df_open.iloc[:, :-6]
    features.set_index(coh_only_open.index, inplace=True)
    data_merge_closed = pd.concat([coh_only_closed,features,df_closed_responseVars], axis=1)
    data_merge_open = pd.concat([coh_only_open,features,df_open_responseVars], axis=1)
    if ica == False:
        data_merge_closed.to_pickle("data/Coherence+Features+Y-CLOSED.pkl")
        data_merge_open.to_pickle("data/Coherence+Features+Y-OPEN.pkl")
    else:
        data_merge_closed.to_pickle("data/ICA_Coherence+Features+Y-CLOSED.pkl")
        data_merge_open.to_pickle("data/ICA_Coherence+Features+Y-OPEN.pkl")
    return (data_merge_closed.shape,data_merge_open.shape)

print(Merge_Coherence_Feats_Y())