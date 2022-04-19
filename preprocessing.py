import mne.io as curry
from mne.preprocessing import ICA
from nitime.timeseries import TimeSeries
from nitime.analysis import CoherenceAnalyzer
from nitime.viz import drawmatrix_channels
import numpy as np
import matplotlib.pyplot as plt
import mne

###### This script is highly inspired by the MNE-Python package #######

activate_plots = False
file_path = 'C:/Users/Johan/Desktop/S94_15510_R001.dat'
raw_data = curry.read_raw_curry(file_path,preload=True, verbose=None)
# Crop data: 180 seconds of open eyes + 180 seconds of closed eyes + 5 seconds for good measure :)
raw_data.crop(tmax=365)
# Set EEG reference to common average
raw_data.set_eeg_reference(ref_channels='average')

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


if activate_plots == True:
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


# Band-pass filtering (0.5 - 70 Hz )
raw_data.filter(0.5, 70., fir_design='firwin')

# Removing power-line noise with notch filtering
raw_data = raw_data.notch_filter(np.arange(50,251,50), fir_design='firwin')

# Downsample frequency to 250Hz
raw_data.resample(250, npad="auto")

# Remove "slow drifts" before running ICA
filt_raw = raw_data.copy().filter(l_freq=1., h_freq=None)

# Instantiate ICA model
ica = ICA(n_components=15, max_iter='auto', random_state=97)

# Fit ICA model and reconstruct data
# .fit and .apply changes ica object in-place
ica.fit(filt_raw)
reconst_raw = raw_data.copy()
ica.apply(reconst_raw)
processed_data = reconst_raw


open_eyes = processed_data.copy().crop(tmin=0,tmax=170)
closed_eyes = processed_data.copy().crop(tmin=190,tmax=360)
separated = {"open-eyes":open_eyes,"closed-eyes":closed_eyes}
for name,processed_data in separated.items():
    # Extract data & time vector from processed data
    data, times = processed_data[:, :]
    T = TimeSeries(data[:-5],sampling_rate=250)
    ch_names = raw_data.ch_names[:-5]

    # Dict for the 7 frequency bands
    frequency_bands = {"delta": [1,3.99], "theta": [4,7.99], "alpha": [8,12.99], "beta": [13,29.99],"beta_1": [13,17.99],"beta_2": [18,23.99],"beta_3": [24,29.99]}

    # Apply the CoherenAnalyzer on the timeseries object (nitime library)
    C = CoherenceAnalyzer(T)


    for key,value in frequency_bands.items():
        freq_idx = np.where((C.frequencies > value[0]) * (C.frequencies < value[1]))[0]
        coherence_map = np.mean(C.coherence[:, :, freq_idx], -1)
        # print(f"coherence_map for {key}:")
        # print(coherence_map.shape)
        coherence_plot = drawmatrix_channels(coherence_map, ch_names, size=[10., 10.],title=f"Coherence map for the ${key}$ freq. band ({name})", color_anchor=[0,1])
        plt.savefig(f"figures/coherence_{key}_{name}.png")

