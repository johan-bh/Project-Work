import pandas as pd
import os
import numpy as np

file_path = r"C:\Users\jbhan\Desktop\data_kognition_inflammation.xlsx"
# read excel file
df = pd.read_excel(file_path)
# set CSMK as index
df.set_index('CSMK', inplace=True)

# include columns TNF, IL-6, IL-8, IL-10, HSCRP, civilstatus, skoleår, alkohol, rygning, BMI, familiecerebrovaskulaer, familiemyokardieinfarkt, familiedemens
# drop everething else
df = df[['TNF', 'IL-6', 'IL-8', 'IL-10', 'HSCRP', 'civilstatus', 'skoleår', 'alkohol',
         'rygning', 'BMI', 'familiecerebrovaskulaer', 'familiemyokardieinfarkt', 'familiedemens']]

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

# # check number of nan values in df
# print(df.isnull().sum())
# Check for unique values to make sure data has been cleaned properly.
# for col in df:
#     print(col)
#     print(df[col].unique())


valid_ids = pd.read_csv("data/valid_ids.csv", index_col=0)
# convert first column to list
valid_ids = valid_ids.iloc[:,0].tolist()
# Delete all rows that are not in valid_ids
df = df.drop(df.index[~df.index.isin(valid_ids)])
# save df to pickle file data/clean_features.pkl
df.to_pickle("data/clean_features.pkl")
#
# # < ====================== DATA CLEANING DONE ====================== >
