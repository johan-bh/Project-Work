import numpy as np
import matplotlib.pyplot as plt
import pickle
ica = False

if ica == False:
    # Load all data combinations
	@@ -102,6 +102,10 @@ def PCR(key,data, eyes="closed"):

    y_test_est = np.array(y_pred)
    y_test = np.array(y_test)
    # scale y_test_est and y_test to have the same mean and std as y_test
    y_test_est = (y_test_est - y_test_est.mean()) / (y_test_est.std() + 1e-10)
    y_test = (y_test - y_test.mean()) / (y_test.std() + 1e-10)

    # Compute relative error for X_test and y_test
    # y_pred = model.predict(X_test)
    # # compute relative error between the matrix y_pred and the matrix y_test
	@@ -129,15 +133,15 @@ def PCR(key,data, eyes="closed"):
for key, data in closed_eyes_pca.items():
    scores, y_test_est, y_test, = PCR(key,data)
    pca_scores_closed = pd.concat([pca_scores_closed, scores], axis=1)
    axis_range = [-4,4]

    if key == "PCA+Y":
        key = "PCA"
    if key == "PCA+Features+Y":
        key = "PCA+Subject Info"
    if key == "Features+Y":
        key = "Subject Info"

    plotting_data_closed.append([key, y_test_est, y_test, axis_range])

# rename the columns to "PCA", "PCA + Subject Info" and "Subject Info"
	@@ -150,11 +154,12 @@ def PCR(key,data, eyes="closed"):
for key, data in open_eyes_pca.items():
    scores, y_test_est, y_test, = PCR(key,data, eyes="open")
    pca_scores_open = pd.concat([pca_scores_open, scores], axis=1)
    axis_range = [-4,4]

    if key == "PCA+Y":
        key = "PCA"
    if key == "PCA+Features+Y":
        key = "PCA+Subject Info"
    if key == "Features+Y":
        key = "Subject Info"
    plotting_data_open.append([key, y_test_est, y_test, axis_range])
	@@ -165,57 +170,59 @@ def PCR(key,data, eyes="closed"):
pca_scores_open.rename(index={"Y":"All Response Vars"}, inplace=True)


# # print the dataframes to latex
# print(pca_scores_closed.to_latex(index=True))
# print(pca_scores_open.to_latex(index=True))
#
# if ica == True:
#     # save the dataframes to pickle files
#     with open("data/ICA_PCR_scores-closed.pkl", "wb") as f:
#         pickle.dump(pca_scores_closed, f)
#     with open("data/ICA_PCR_scores-open.pkl", "wb") as f:
#         pickle.dump(pca_scores_open, f)
# else:
#     # save the dataframes to pickle files
#     with open("data/PCR_scores-closed.pkl", "wb") as f:
#         pickle.dump(pca_scores_closed, f)
#     with open("data/PCR_scores-open.pkl", "wb") as f:
#         pickle.dump(pca_scores_open, f)

# create figure with 3 subplots corresponding to the 3 different data sets
fig, ax = plt.subplots(3, figsize=(10,10))
for i, data in enumerate(plotting_data_closed):
    key = data[0]
    y_test_est = data[1]
    y_test = data[2]
    axis_range = data[3]
    ax[i].plot(axis_range,axis_range, 'k--')
    ax[i].plot(y_test, y_test_est, 'ob', alpha=.25)
    ax[i].legend(['Perfect estimation', 'Model estimations'])
    ax[i].title.set_text(f'Test Predictions (Input: Closed eyes, {key})')
    ax[i].set_ylim(axis_range)
    ax[i].set_xlim(axis_range)
    ax[i].set_xlabel('True value')
    ax[i].set_ylabel('Estimated value')
    ax[i].set_aspect('equal')
    ax[i].grid(True)
    plt.subplots_adjust(hspace=0.5)
plt.savefig(f"figures/PCR_PredictionPlotsClosed{ica_flag}.png",bbox_inches='tight')

# create figure with 3 subplots corresponding to the 3 different data sets
fig, ax = plt.subplots(3, figsize=(10, 10))
for i, data in enumerate(plotting_data_open):
    key = data[0]
    y_test_est = data[1]
    y_test = data[2]
    axis_range = data[3]
    ax[i].plot(axis_range,axis_range, 'k--')
    ax[i].plot(y_test, y_test_est, 'ob', alpha=.25)
    ax[i].legend(['Perfect estimation', 'Model estimations'])
    ax[i].title.set_text(f'Test Predictions (Input: Open eyes, {key})')
    ax[i].set_ylim(axis_range)
    ax[i].set_xlim(axis_range)
    ax[i].set_xlabel('True value')
    ax[i].set_ylabel('Estimated value')
    ax[i].set_aspect('equal')
    ax[i].grid(True)
    plt.subplots_adjust(hspace=0.5)
plt.savefig(f"figures/PCR_PredictionPlotsOpen{ica_flag}.png",bbox_inches='tight')
