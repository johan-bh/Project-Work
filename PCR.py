import pickle

# # Load pickle files from data folder
# with open('data/coherence_maps_open-eyes.pkl', 'rb') as f:
#     coh_open = pickle.load(f)
#
# with open('data/coherence_maps_closed-eyes.pkl', 'rb') as f:
#     coh_closed = pickle.load(f)
# print(coh_open.shape)
# print(coh_closed.shape)

file_path = "C:\\Users\\Johan\\Desktop\\"
import glob
print(glob.glob(file_path + "*.dat"))
