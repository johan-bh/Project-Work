import mne.io as curry
from mne.preprocessing import ICA
import mne
import os
import matplotlib.pyplot as plt
import pickle

mne.utils.set_config('MNE_USE_CUDA', 'true')
if mne.utils.get_config('MNE_USE_CUDA') == 'true':
    n_jobs = "cuda"
    print('GPU is available')
else:
    n_jobs = "1"
    print('GPU is not available')


valid_files = pickle.load(open("data/valid_files.pkl", "rb"))

ids = list(valid_files.keys())

file_path = valid_files[ids[53]]
raw_data = curry.read_raw_curry(file_path, preload=True, verbose=None)
raw_data.crop(tmax=60)
raw_data.set_eeg_reference(ref_channels='average',verbose=None)
raw_data.filter(0.5, 70., fir_design='firwin', n_jobs=n_jobs,verbose=None)
processed_data = raw_data.resample(250, npad="auto", n_jobs=n_jobs,verbose=None)
filt_raw = processed_data.copy().filter(l_freq=1., h_freq=None,n_jobs=n_jobs,verbose=None)


ica = ICA(n_components=0.99, max_iter="auto", random_state=97)
ica.fit(inst=raw_data)
eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw_data,ch_name=['VEO','HEO'])
eog_components_veo, eog_scores = ica.find_bads_eog(
    inst=eog_epochs,
    ch_name='VEO',
    threshold=2)
eog_components_heo, eog_scores = ica.find_bads_eog(
    inst=eog_epochs,
    ch_name='HEO',
    threshold=2)
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw=raw_data,ch_name='EKG')
ecg_components, ecg_scores = ica.find_bads_ecg(raw_data, ch_name='EKG', threshold=0.25)
eog_components = sorted(list(set(eog_components_veo + eog_components_heo)))
print(eog_components)
print(ecg_components)

ica.plot_components(title='ICA components')


# # print(raw_data.ch_names)
# bad_idx1, scores1 = ica.find_bads_eog(raw_data, 'VEO', threshold=3)
# bad_idx2, scores2 = ica.find_bads_eog(raw_data, 'HEO', threshold=3)
# bad_idx_eog = bad_idx1 + bad_idx2
# bad_idx, scores = ica.find_bads_ecg(raw_data, 'EKG', threshold=0.25)
# print(f"EOG related indices:{bad_idx_eog}")
# print(f"ECG related indices: {bad_idx}")
# bad_idx = bad_idx1 + bad_idx2 + bad_idx
# ica.exclude = list(set(bad_idx))
# print(ica.exclude)
# processed_data.plot(title="No ICA")
# ica.apply(processed_data.copy(),exclude=ica.exclude).plot(title='ICA applied')
# plt.show()