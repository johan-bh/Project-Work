# Project-Work
Project Work (Course 02466) - Bsc. Artificial Intelligence and Data @Technical University of Denmark

If you want to run the preproccesing run: pip install -r requirements.txt

The PCR and RidgeReg scripts are independent of most of the dependencies - for those you simply need SKLearn and pandas.

###### Jeg har kørt preprocessing på data udeen ICA. Stadig dårlige resultater.

# Info til Smilla og Sebastian:
- data/PCA_and_Y_closed.pkl og data/PCA_and_Y_open.pkl indeholder matricerne med PCA og response variables
- data/PCA_and_Y_and_features_open.pkl indeholder matrice med PCA og response variables og features. Har ikke lavet for closed eyes endnu
(Disse kan bruges til PCA og RidgeReg etc (se scripts)




Hvis det skulle være af interesse, så indeholder følgende filer:
- data/valid_ids.csv indeholder ids for de filer som ikke forsåger fejl
- data/response_var_df.pkl indeholder dataframe med response variables
- data/clean_features.pkl indeholder dataframe med features
- data/coherence_maps_open.pkl (eller closed) indeholder coherence maps uden yderlig behandling (efter preprocessing.py)
