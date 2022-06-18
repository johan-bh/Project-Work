import mne.io as curry
from mne.preprocessing import ICA
from nitime.timeseries import TimeSeries
from nitime.analysis import CoherenceAnalyzer
from nitime.viz import drawmatrix_channels
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
import os

activate_plots = False
tensor_regression = True
run_ICA = True
mne.utils.set_config('MNE_USE_CUDA', 'true')  # Use GPU for ICA etc.
# Check if GPU is available. n_jobs = 1 (CPU) n_jobs = "cuda" (GPU). n_jobs is a param for filter functions etc.
if mne.utils.get_config('MNE_USE_CUDA') == 'true':
    n_jobs = "cuda"
    print('GPU is available')
else:
    n_jobs = "1"
    print('GPU is not available')
def preprocessing(file_path):
    """Runs the preprocessing pipeline for a single file.
    :arg file_path: Path to the file to be processed.
    :return: list with coherence map (eyes closed and eyes open)
    """
    # Disable warnings
    mne.set_log_level('CRITICAL')
    # Load data
    raw_data = curry.read_raw_curry(file_path, preload=True, verbose=None)
    # only include the first 64 channels
    # raw_data.pick_channels(raw_data.ch_names[:64])
    # Filter data
    # Crop data: 60 seconds of open eyes + 60 seconds of closed eyes + 10 seconds for good measure :)
    raw_data.crop(tmax=130)
    # Set EEG reference to common average
    raw_data.set_eeg_reference(ref_channels='average',verbose=None)
    # Band-pass filtering (0.5 - 70 Hz )
    raw_data.filter(0.5, 70., fir_design='firwin', n_jobs=n_jobs,verbose=None)
    # Downsample frequency to 250Hz
    processed_data = raw_data.resample(250, npad="auto", n_jobs=n_jobs,verbose=None)
    if run_ICA == True:
        # Remove "slow drifts" before running ICA
        filt_raw = processed_data.copy().filter(l_freq=1., h_freq=None,n_jobs=n_jobs,verbose=None)
        try:
            ica = ICA(n_components=0.99, max_iter="auto", random_state=97)
            ica.fit(filt_raw, decim=1, verbose=None)
            bad_idx1, scores1 = ica.find_bads_eog(raw_data, 'VEO', threshold=2.5)
            bad_idx2, scores2 = ica.find_bads_eog(raw_data, 'HEO', threshold=2.5)
            bad_idx_eog = bad_idx1 + bad_idx2
            ica.exclude = bad_idx_eog
            processed_data = ica.apply(filt_raw.copy(), exclude=ica.exclude)
        except:
            return None
    # only include the first 64 channels
    processed_data.pick_channels(processed_data.ch_names[:64])
    # Split filtered data into eyes closed and eyes open
    open_eyes = processed_data.copy().crop(tmin=0,tmax=60)
    closed_eyes = processed_data.copy().crop(tmin=60,tmax=120)
    # dict useful for plotting
    separated = {"open-eyes":open_eyes,"closed-eyes":closed_eyes}
    # Create a coherence map for open_eyes and closed_eyes
    coh_list = []
    for name,processed_data in separated.items():
        # Extract data & time vector from processed data
        data, times = processed_data[:, :]
        # convert to cupy array for faster computation
        data = np.asarray(data)
        # Instantiate TimeSeries object on cupy array
        T = TimeSeries(data, sampling_rate=250)
        # ch_names = raw_data2.ch_names[:-5]
        # Dict for the 7 frequency bands as a cupy array
        frequency_bands = {"delta": np.array([1,3.99]),
                           "theta": np.array([4,7.99]),
                           "alpha": np.array([8,12.99]),
                           "beta": np.array([13,29.99]),
                           "beta_1": np.array([13,17.99]),
                           "beta_2": np.array([18,23.99]),
                           "beta_3": np.array([24,29.99])}
        # Apply the CoherenceAnalyzer on the timeseries object (nitime library)
        C = CoherenceAnalyzer(T)
        # Generate coherence matrix for each frequency band and then ravel+concatenate them into a single array
        if tensor_regression == True:
            # create empty matrix for coherence
            coherence_maps = []
            for key, value in frequency_bands.items():
                try:
                    # Extract frequency indices on CoherenceAnalyzer object
                    freq_idx = np.where((C.frequencies > value[0]) * (C.frequencies < value[1]))[0]
                    coherence_map = np.mean(C.coherence[:, :, freq_idx], -1)
                except:
                    return None
                # append each 2d array to coherence_maps
                coherence_maps.append(coherence_map)
            coh_list.append(coherence_maps)
        else:
            coherence_maps = np.array([])
            for key,value in frequency_bands.items():
                try:
                    # Extract frequency indices on CoherenceAnalyzer object
                    freq_idx = np.where((C.frequencies > value[0]) * (C.frequencies < value[1]))[0]
                    coherence_map = np.mean(C.coherence[:, :, freq_idx], -1)
                    coherence_map = np.array(coherence_map[np.triu_indices(64, k=1)])
                    # Concatenate coherence maps as cupy array
                    coherence_maps = np.concatenate((coherence_maps, coherence_map.ravel()))
                except:
                    return None
            coh_list.append(coherence_maps)
    return coh_list
# load "data/valid_files.pkl"
valid_files = pickle.load(open("data/valid_files.pkl", "rb"))
# file_path = "C:/Users/jbhan/Desktop/AA_CESA-2-DATA-EEG-Resting (Anden del)/S183_11123_R001.dat"
# print(preprocessing(file_path))
# Create a dictionary with id as key and coherence map as value
coherence_maps = {}
counter = 0
for id,file_name in valid_files.items():
    # Create counter that counts how many files have been processed
    print(f"{file_name}" + " is being processed...")
    print("Current progress: " + str(int(counter+1)) + "/" + str(len(valid_files)))
    result = preprocessing(file_name)
    if result == None:
        print("Error in preprocessing")
        continue
    else:
        coherence_maps[id] = result
        counter += 1
# Split coherence maps into two dicts. Use value index 0 for open eyes and value index 1 for closed eyes
coherence_maps_open = {k: v[0] for k, v in coherence_maps.items()}
coherence_maps_closed = {k: v[1] for k, v in coherence_maps.items()}
if tensor_regression == True:
    if run_ICA == True:
        with open('data/ICA_tensor_data_open.pkl', 'wb') as f:
            pickle.dump(coherence_maps_open, f)
        with open('data/ICA_tensor_data_closed.pkl', 'wb') as f:
            pickle.dump(coherence_maps_closed, f)
    else:
        with open('data/tensor_data_open.pkl', 'wb') as f:
            pickle.dump(coherence_maps_open, f)
        with open('data/tensor_data_closed.pkl', 'wb') as f:
            pickle.dump(coherence_maps_closed, f)
else:
    if run_ICA == True:
        # Save dictionaries as pickle files for later use
        with open('data/ICA_coherence_maps_open.pkl', 'wb') as f:
            pickle.dump(coherence_maps_open, f)
        with open('data/ICA_coherence_maps_closed.pkl', 'wb') as f:
            pickle.dump(coherence_maps_closed, f)
    else:
        # Save dictionaries as pickle files for later use
        with open('data/coherence_maps_open.pkl', 'wb') as f:
            pickle.dump(coherence_maps_open, f)
        with open('data/coherence_maps_closed.pkl', 'wb') as f:
            pickle.dump(coherence_maps_closed, f)
# Create a seperate file for plotting later
if activate_plots == True:
    def add_arrows(axes):
        for ax in axes:
            freqs = ax.lines[-1].get_xdata()
            psds = ax.lines[-1].get_ydata()
            for freq in (50, 100, 150):
                idx = np.searchsorted(freqs, freq)
                # get ymax of a small region around the freq. of interest
                y = psds[(idx - 4):(idx + 5)].max()
                ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                         width=0.1, head_width=3, length_includes_head=True)
    raw_downsampled = raw_data.copy().resample(sfreq=250)
    # PSD plots for Original vs downsampled
    for data, title in zip([raw_data,raw_downsampled], ['Original Data (2000 Hz)','Downsampled Data (250Hz)']):
        fig = data.plot_psd(average=True,area_mode='range', show=False)
        fig.subplots_adjust(top=0.75)
        fig.suptitle(title, size='xx-large', weight='bold')
        plt.setp(fig.axes, xlim=(0, 300))
        plt.savefig(f"figures/psd_plot_{title}.png")
    # Plot powerline artefacts
    fig = raw_data.plot_psd(fmax=250, average=True, show=False)
    add_arrows(fig.axes[:2])
    fig.subplots_adjust(top=0.75)
    fig.suptitle("Powerline artefacts", size='xx-large', weight='bold')
    plt.savefig(f"figures/powerline_artefacts.png")
    # Plot Unfiltered vs Notch filtered
    freqs = (50, 100, 150)
    raw_notch = raw_data.copy().notch_filter(freqs=freqs)
    for title, data in zip(['Un', 'Notch '], [raw_data, raw_notch]):
        fig = data.plot_psd(fmax=250, average=True, show=False)
        fig.subplots_adjust(top=0.75)
        fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
        add_arrows(fig.axes[:2])
        plt.savefig(f"figures/notch-filtering.png")
    # Band-pass filtering (0.5 - 70 Hz )
    raw_data.filter(0.5, 70., fir_design='firwin')
    fig = raw_data.plot_psd(average=True, show=False)
    fig.subplots_adjust(top=0.75)
    fig.suptitle("Band-pass filtered (0.5-70 Hz)", size='xx-large', weight='bold')
    plt.savefig(f"figures/band-pass.png")
