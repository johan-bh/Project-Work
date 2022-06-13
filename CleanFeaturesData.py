import pandas as pd
import os
import numpy as np
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
# pd.set_option('display.max_columns', 500)
file_path = r"C:\Users\jbhan\Desktop\data_kognition_inflammation.xlsx"
file_path2 = r"C:\Users\jbhan\Desktop\Data_Søvn og Depression_D-vit.xlsx"
file_path3 = r"C:\Users\jbhan\Desktop\neurotest.xlsx"


def clean_file1():
    # read excel file
    df = pd.read_excel(file_path)
    # set CSMK as index
    df.set_index('CSMK', inplace=True)

    # include columns TNF, IL-6, IL-8, IL-10, HSCRP, civilstatus, skoleår, alkohol, rygning, BMI, familiecerebrovaskulaer, familiemyokardieinfarkt, familiedemens
    # drop everething else
    df = df[['TNF', 'IL-6', 'IL-8', 'IL-10', 'HSCRP', 'civilstatus', 'skoleår', 'alkohol',
             'rygning', 'BMI', 'familiecerebrovaskulaer', 'familiemyokardieinfarkt', 'familiedemens',
             "familiehjertesygdom","familiehypertension","familiediabetes2","familiedepression"]]

    # < ====================== DATA CLEANING ====================== >

    # print all columns that have zero nan values
    # print(df.isnull().sum())
    # Only TNF, IL-8 have zero nan values

    # remove all instances of "<" in df["IL-6"]
    df["IL-6"] = df["IL-6"].str.replace("<","")
    # convert all rows in IL-6 to float
    df["IL-6"] = df["IL-6"].astype(float)
    # replace all nan values in df["IL-6"] with mean of IL-6
    df["IL-6"] = df["IL-6"].fillna(df["IL-6"].mean())

    # remove all instances of "<" in df["IL-10"]
    df["IL-10"] = df["IL-10"].str.replace("<","")
    # convert all rows in IL-10 to float
    df["IL-10"] = df["IL-10"].astype(float)
    # replace all nan values in df["IL-10"] with mean of IL-10
    df["IL-10"] = df["IL-10"].fillna(df["IL-10"].mean())

    # replace all nan values in df["HSCRP"] with mean of HSCRP
    df["HSCRP"] = df["HSCRP"].fillna(df["HSCRP"].mean())

    # replace "Gift" / "Samlevende" with 1 and "Enlig" / "Enk" with 0 in df["civilstatus"]
    df["civilstatus"] = df["civilstatus"].replace(["Gift", "Samlevende", "Enlig", "Enke"], [1,1,0, 0])
    # replace all nan values with most frequent value in df["civilstatus"]
    df["civilstatus"] = df["civilstatus"].fillna(df["civilstatus"].mode()[0])

    # replace all nan values with mean of skoleår
    df["skoleår"] = df["skoleår"].fillna(df["skoleår"].mean())

    # replace all nan values with mean of alkohol
    df["alkohol"] = df["alkohol"].fillna(df["alkohol"].mean())

    # replace nan values with most frequent value in df["rygning"]
    df["rygning"] = df["rygning"].fillna(df["rygning"].mode()[0])

    # replace all nan values with mean of BMI
    df["BMI"] = df["BMI"].fillna(df["BMI"].mean())


    # replace "Nej" with 0 and "Mor", "Far" or "Søskende" with 1 in df["familiecerebrovaskulaer"]
    # print(df["familiecerebrovaskulaer"].unique())
    df["familiecerebrovaskulaer"] = df["familiecerebrovaskulaer"].replace(["Nej", "Nej15756","Mor", "Far","Søskende"], [0, 0, 1, 1, 1])
    # replace all nan values with most frequent value in df["familiecerebrovaskulaer"]
    df["familiecerebrovaskulaer"] = df["familiecerebrovaskulaer"].fillna(df["familiecerebrovaskulaer"].mode()[0])

    # print(df["familiemyokardieinfarkt"].unique())
    # replace "Nej" with 0 and "Mor", "Far" or "Søskende" with 1 in df["familiemyokardieinfarkt"]
    df["familiemyokardieinfarkt"] = df["familiemyokardieinfarkt"].replace(["Nej", "Mor", "Far", "Søskende"], [0, 1, 1,1])
    # replace all nan values with most frequent value in df["familiemyokardieinfarkt"]
    df["familiemyokardieinfarkt"] = df["familiemyokardieinfarkt"].fillna(df["familiemyokardieinfarkt"].mode()[0])

    # replace "Nej" with 0 and "Mor" or "Far" with 1 in df["familiedemens"]
    df["familiedemens"] = df["familiedemens"].replace(["Nej", "Mor", "Far", "Søskende"], [0, 1, 1, 1])
    # replace all nan values with most frequent value in df["familiedemens"]
    df["familiedemens"] = df["familiedemens"].fillna(df["familiedemens"].mode()[0])


    # for col in df.iloc[:,-4:]:
    #     print(col)
    #     print(df[col].unique())
    # replace "nn" and "Nej" with 0 and "Far", "Mor" and "Søskende" with 1 in df["familiehjertesygdom"]
    df["familiehjertesygdom"] = df["familiehjertesygdom"].replace(["nn", "Nej", "Far", "Mor", "Søskende"], [0, 0, 1, 1, 1])
    # replace all nan values with most frequent value in df["familiehjertesygdom"]
    df["familiehjertesygdom"] = df["familiehjertesygdom"].fillna(df["familiehjertesygdom"].mode()[0])

    # replace "Nej" with 0 and "Mor", "Far" or "Søskende" with 1 in df["familiehypertension"]
    df["familiehypertension"] = df["familiehypertension"].replace(["Nej", "Mor", "Far", "Søskende"], [0, 1, 1, 1])
    # replace all nan values with most frequent value in df["familiehypertension"]
    df["familiehypertension"] = df["familiehypertension"].fillna(df["familiehypertension"].mode()[0])

    # replace "Nej" with 0 and "Mor", "Far" or "Søskende" with 1 in df["familiediabetes2"]
    df["familiediabetes2"] = df["familiediabetes2"].replace(["Nej", "Mor", "Far", "Søskende"], [0, 1, 1, 1])
    # replace all nan values with most frequent value in df["familiediabetes2"]
    df["familiediabetes2"] = df["familiediabetes2"].fillna(df["familiediabetes2"].mode()[0])

    # replace "Nej" with 0 and "Mor", "Far" or "Søskende" with 1 in df["familiedepression"]
    df["familiedepression"] = df["familiedepression"].replace(["Nej", "Mor", "Far", "Søskende"], [0, 1, 1, 1])
    # replace all nan values with most frequent value in df["familiedepression"]
    df["familiedepression"] = df["familiedepression"].fillna(df["familiedepression"].mode()[0])

    # remove decimal point from the indexes of df
    df.index = df.index.map(lambda x: str(x).replace(".0", ""))
    valid_ids = pd.read_pickle("data/valid_ids.pkl")
    df.index = df.index.astype(str)

    df = df.drop(df.index[~df.index.isin(valid_ids)])
    # Delete all rows that are not in valid_ids
    df = df.drop(df.index[~df.index.isin(valid_ids)])
    return df

def clean_file2():
    # read file_path2
    df_init = pd.read_excel(file_path2)
    df_init.set_index('CSMK', inplace=True)
    df2 = df_init[["Score MDI","FølerMigVeloplagt","FølerMigTræt","JegErUdhvilet","BliverNemtTræt",
               "FysiskKanJegIkkeRetMeget","FysiskKanJegOverkommeMeget","FysiskFølerJegMigIDårligForm","FysiskFølerJegMigIVældigGodForm",
               "FølerMigMegetAktiv","LaverMegtPåEnDag","LaverMegetLidtPåEnDag","FårNæstenIkkeLavetNoget",
               "LystTilAtGøreRareTing","GruerForAtLaveNoget","HarMangePlaner","IkkeLystTilAtLaveNoget",
               "KanFastholdeTankerPåDetJegLaver","KanSagtensKoncentrereMig","AnstrengeMigMegetForAtKoncentrereMigOmNoget","BliverLetAdspredt",
               "Egentlig søvn","GLOBAL PSQI SCORE, Comp 1 - 7"]]
    # sum up the last 4 columns and save result in column "Score General Fatigue"
    df2["Score Generel Fatigue"] = df2.iloc[:,1:5].sum(axis=1)
    df2["Score Fysisk Fatigue"] = df2.iloc[:,5:9].sum(axis=1)
    df2["Score Reduceret Aktivitet"] = df2.iloc[:,9:13].sum(axis=1)
    df2["Score Reduceret Motivation"] = df2.iloc[:,13:17].sum(axis=1)
    df2["Score Mental Fatigue"] = df2.iloc[:,17:21].sum(axis=1)
    # drop all columns that are not needed anymore
    df2.drop(df2.columns[1:-7], axis=1, inplace=True)

    #add minerals, vitamins and biomarkers to df2
    df2 = pd.concat([df2, df_init.iloc[:,-24:-1]], axis=1)
    # drop the columns Methylmalonat, CRP, 25-OH-Vitamin D, 25-OH-Vitamin D ny metode nov 2017, HBA1C_DCCT (not enough data)
    df2.drop(["Methylmalonat", "CRP", "25-OH-Vitamin D", "25-OH-Vitamin D ny metode nov 2017", "HBA1C_DCCT"], axis=1, inplace=True)

    valid_ids = pd.read_pickle("data/valid_ids.pkl")
    # Drop all rows that are not in valid_ids. Set index type of df to string
    df2.index = df2.index.astype(str)
    # Delete all rows that are not in valid_ids
    df2 = df2.drop(df2.index[~df2.index.isin(valid_ids)])
    # replace nan values for all columns with most frequent value that corresponds to the column
    for col in df2.columns:
        df2[col] = df2[col].fillna(df2[col].mode()[0])

    return df2

def merge_features():
    # drop the rows of df that dont have index in df2
    df = clean_file1()
    df2 = clean_file2()

    df = df.drop(df.index[~df.index.isin(df2.index)])
    # drop the rows of df2 that dont have index in df
    df2 = df2.drop(df2.index[~df2.index.isin(df.index)])
    # merge df and df2
    features = pd.concat([df, df2], axis=1)
    # # save df to pickle file data/clean_features.pkl
    features.to_pickle("data/clean_features.pkl")
    return features.columns

def get_response_vars():
    # Read second tab of excel file
    df = pd.read_excel(file_path3,sheet_name="CESA II")
    # set cmsk column as index
    df.set_index('cmsk', inplace=True)
    # Only include columns MMSE, ACE and TrailMakingA and TrailMakingB. Keep index as is
    df = df[['MMSE', 'ACE', 'TrailMakingA', 'TrailMakingB',"DigitSymbol","Retention"]]

    # Remove entries with missing values (NaN)
    df = df.dropna()
    # load "data/valid_ids.pkl" and convert to pandas series
    valid_ids = pd.read_pickle("data/valid_ids.pkl")
    # Drop all rows that are not in valid_ids. Set index type of df to string
    df.index = df.index.astype(str)
    df = df.drop(df.index[~df.index.isin(valid_ids)])
    # Save to pickle file
    df.to_pickle("data/response_var_df.pkl")
    return df.head(),df.shape

# print(clean_file1())
# print(clean_file2())
print(merge_features())
# print(get_response_vars())