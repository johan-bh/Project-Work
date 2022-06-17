import mne.io as curry
from mne.preprocessing import ICA
import mne
import os
import matplotlib.pyplot as plt

mne.utils.set_config('MNE_USE_CUDA', 'true')

folder_path1  = "C:\\Users\\jbhan\\Desktop\\AA_CESA-2-DATA-EEG-Resting (Anden del)\\"
folder_path2 = "C:\\Users\\jbhan\\Desktop\\AA_CESA-2-DATA-EEG-Resting\\"

# Create lists of filenames in each folder. Only use files that starts with "S" and ends with ".dat"
file_names1 = [f for f in os.listdir(folder_path1) if f.endswith('.dat') and f.startswith('S')]
file_names2 = [f for f in os.listdir(folder_path2) if f.endswith('.dat') and f.startswith('S')]
# Delete the following fiels from their respective folders. They return errors...
if 'S26_12604_R001.dat' in file_names2:
    file_names2.remove('S26_12604_R001.dat')
if "S195_11851_R001.dat" in file_names1:
    file_names1.remove("S195_11851_R001.dat")
if "S309_11498_R001.dat" in file_names1:
    file_names1.remove("S309_11498_R001.dat")
if "S310_16627_R001.dat" in file_names1:
    file_names1.remove("S310_16627_R001.dat")
if "S311_11302_R001.dat" in file_names1:
    file_names1.remove("S311_11302_R001.dat")

# Get id that is written between underscores (starts with underscore and ends with underscore)
id_list1 = [f.split('_')[1].split('.')[0] for f in file_names1]
id_list2 = [f.split('_')[1].split('.')[0] for f in file_names2]

# Create a dictionary with id as key and file name as value
id_dict1 = dict(zip(id_list1, file_names1))
id_dict2 = dict(zip(id_list2, file_names2))
# Concatenate dict value to respective folder path
file_names = [folder_path1 + id_dict1[id] for id in id_list1] + [folder_path2 + id_dict2[id] for id in id_list2]
# Change all double backslashes to single forward slashes
file_names = [f.replace('\\','/') for f in file_names]
# Add to dictionary with id as key and file name as value
id_dict = dict(zip(id_list1 + id_list2, file_names))

if mne.utils.get_config('MNE_USE_CUDA') == 'true':
    n_jobs = "cuda"
    print('GPU is available')
else:
    n_jobs = "1"
    print('GPU is not available')


# valid_files = []
# for key,value in id_dict.items():
#     file_path = value
#     raw_data = curry.read_raw_curry(file_path, preload=False, verbose=None)
#     # if file includes 'HEO' and 'EKG' in the channel names, then it is a valid file
#     if 'HEO' in raw_data.ch_names and 'EKG' in raw_data.ch_names:
#         valid_files.append(file_path)
#     else:
#         print(file_path + " is not a valid file")
#
file_path = "C:/Users/jbhan/Desktop/AA_CESA-2-DATA-EEG-Resting (Anden del)/S183_11123_R001.dat"
raw_data = curry.read_raw_curry(file_path, preload=True, verbose=None)
print(raw_data.ch_names)


# file_path = id_dict[id_list1[0]]
# raw_data = curry.read_raw_curry(file_path, preload=True, verbose=None)
# raw_data.crop(tmax=130)
# raw_data.set_eeg_reference(ref_channels='average',verbose=None)
# raw_data.filter(0.5, 70., fir_design='firwin', n_jobs=n_jobs,verbose=None)
# processed_data = raw_data.resample(250, npad="auto", n_jobs=n_jobs,verbose=None)
# filt_raw = processed_data.copy().filter(l_freq=1., h_freq=None,n_jobs=n_jobs,verbose=None)
# try:
#     ica = ICA(n_components=0.99, max_iter="auto", random_state=97)
#     ica.fit(filt_raw, decim=1, verbose=None)
#     ica_success = True
# except:
#     ica_success = False
# if ica_success:
#     # print(raw_data.ch_names)
#     bad_idx1, scores1 = ica.find_bads_eog(raw_data, 'VEO', threshold=2.5)
#     bad_idx2, scores2 = ica.find_bads_eog(raw_data, 'HEO', threshold=2.5)
#     bad_idx, scores = ica.find_bads_ecg(raw_data, 'EKG', threshold=0.5)
#     bad_idx_eog = bad_idx1 + bad_idx2
#     ica.exclude = bad_idx_eog
#     processed_data.plot(title="No ICA")
#     ica.apply(processed_data.copy(),exclude=ica.exclude).plot(title='ICA applied')
#     plt.show()
    #
    # print(sorted(list(set(bad_idx_eog))))
    # print(sorted(list(set(bad_idx))))
    # ica.plot_components(title='ICA components')




