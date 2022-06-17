# Project-Work
Project Work (Course 02466) - Bsc. Artificial Intelligence and Data @Technical University of Denmark

# Info for reproducability:
In this project we used Python version 3.8.10
Run the following command: "pip install -r requirements.txt" to install the dependencies used for this project. Note that some of the packages (i.e. Tensor) were installed for GPU - if you don't have CUDA eligible GPU we advice you to install the CPU version which should yield the same results.

## The following is a description of the order the project files should be run in:

1) We start by defining the filepaths to the folders containing the EEG files in the extract_files.py (the EEG data is using the standard european format as described in our report. We loop through the files of each folder to find the files that are valid as described in the code. The files were exported using a Curry plugin in matlab resulting in 3 files for each EEG recording:
- a .ceo file
- a .dap file
- a .dat file
The .dat file is the one that contains the data but the 2 other files need to be in the same folder for the MNE library to be able to load the content.

2) We then run the preprocessing.py script where we can control the different data types that should be computed using the flags in the top:
- the "run_ICA" flag controls wether there should be used ICA in the signal processing
- the "tensor_regression" flag control wether the data should be computed as whole coherence map (64x64) matrices (used in tensor regression) or as a vector consiting of half of each coherence map for each frequency band (used for the PCR, Ridge Regression and NN models)

3) Run the CleanFeaturesData.py file to extract the features from the excel files.

4) Run the PCA.py file to compute the PCA data before we merge the different data combinations
- Make sure to run the function with the ICA flag set to false and true to compute PCA for both datasets. 

5) Run the MergeData.py functions to merge the PCA and Coherence data with the features and response variables.
- Make sure to run the functions with the ICA flag set to both false and true so that that all the different combinations of data will be merged and saved (with and without ICA)

All the necessary data has now been saved and stored in the data folder. The 4 different ML models can now be run independent of order. All 4 ML scripts contain an ICA flag as well. The ML models can be found in the following files:
- PCR.py (Principal Component Regression Model)
- RigdeReg.py (Ridge Regression Model)
- NN_R2.py (Neural Network Model, uses the trainNN.py which was introduced to us in the ML course 02450)
- TensorRegression.py (Tensor Regression model)

TensorSim.py is used for creating simulated data based on the Tensor Regression model

Note that the following files are not a part of the ML pipeline but are simply different scripts to print summary statistics, generate additional plotting as well as latex tables etc:
- ICA_TESTING.py
- Subject-Info.py
- SummaryStat_CognitiveTest.py
