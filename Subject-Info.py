import pandas as pd
import pickle
pd.set_option('display.max_columns', 500)

data = pd.read_pickle("data/clean_features.pkl")

general_lifestyle = data[['civilstatus', 'skoleår',
                          'alkohol', 'rygning', 'BMI','Score MDI',
                          'Egentlig søvn', 'GLOBAL PSQI SCORE, Comp 1 - 7',
                          'Score Generel Fatigue', 'Score Fysisk Fatigue',
                          'Score Reduceret Aktivitet', 'Score Reduceret Motivation',
                          'Score Mental Fatigue']]
# drop columns of data that are in general_lifestyle
data = data.drop(data.columns[data.columns.isin(general_lifestyle.columns)], axis=1)

hereditary_diseases = data[['familiecerebrovaskulaer',
       'familiemyokardieinfarkt', 'familiedemens', 'familiehjertesygdom',
       'familiehypertension', 'familiediabetes2', 'familiedepression',]]
# drop columns of data that are in hereditary_diseases
biomarkers = data.drop(data.columns[data.columns.isin(hereditary_diseases.columns)], axis=1)

# summary statistics of biomarkers (mean, std, min, max). Turn into dataframe
biomarkers_summary = biomarkers.describe().T
# drop count, 25%, 50%, 75% columns
biomarkers_summary = biomarkers_summary.drop(biomarkers_summary.columns[biomarkers_summary.columns.str.contains('count|25%|50%|75%')], axis=1)
# add column named "Data type"


df = hereditary_diseases
dd =  df['familiehjertesygdom'].value_counts().values
# convert dd to dataframe
hereditary_diseases = pd.DataFrame(dd, columns=['familiehjertesygdom'])
# do same for other columns
for col in df.columns:
    dd = df[col].value_counts().values
    hereditary_diseases[col] = pd.DataFrame(dd, columns=[col])
hereditary_diseases = hereditary_diseases.T
# rename 0 to No and 1 to Yes
hereditary_diseases.rename(columns={0: 'No', 1: 'Yes'}, inplace=True)
# add a column with the ratio of Yes to No
hereditary_diseases['Yes/No Ratio'] = hereditary_diseases['Yes'] / hereditary_diseases['No']


general_lifestyle_summary = general_lifestyle.describe().T
# drop count, 25%, 50%, 75% columns
general_lifestyle_summary = general_lifestyle_summary.drop(general_lifestyle_summary.columns[general_lifestyle_summary.columns.str.contains('count|25%|50%|75%')], axis=1)

# replace 0 with No in column "min" and 1 with Yes in column "max" in the rows "civilstatus", "alkohol", "rygning" in the columns
general_lifestyle_summary.loc[['civilstatus', 'alkohol', 'rygning'], 'min'] = general_lifestyle_summary.loc[['civilstatus', 'alkohol', 'rygning'], 'min'].replace(0, 'No')
general_lifestyle_summary.loc[['civilstatus', 'alkohol', 'rygning'], 'max'] = general_lifestyle_summary.loc[['civilstatus', 'alkohol', 'rygning'], 'max'].replace(1, 'Yes')



# use bold for the rows
print(biomarkers_summary.to_latex(index=True, bold_rows=True))
# print(hereditary_diseases.to_latex(index=True, bold_rows=True))
# print(general_lifestyle_summary.to_latex(index=True, bold_rows=True))