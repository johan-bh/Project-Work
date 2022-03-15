import mne.io as curry
from mne.preprocessing import ICA
from nitime.timeseries import TimeSeries
from nitime.analysis import CoherenceAnalyzer
from nitime.viz import drawmatrix_channels
import numpy as np
import matplotlib.pyplot as plt

file_path = 'C:/Users/Johan/Desktop/test001.dat'
raw_data = curry.read_raw_curry(file_path, preload=True,verbose=None)

# Set EEG refence to common average
raw_data1 = raw_data.copy().set_eeg_reference(ref_channels='average')

# Select time window from sample
tmin, tmax = 0, 10

# Downsample frequency to 250Hz
raw_data = raw_data1.resample(250, npad="auto")
# raw_data.plot_psd(area_mode='range', tmax=10.0)

# Band-pass filtering (0.5 - 70 Hz )
raw_data = raw_data.filter(0.5, 70., fir_design='firwin')

# Removing power-line noise with low-pass filtering
raw_data = raw_data.filter(None, 50., fir_design='firwin')
# raw_data.plot_psd(area_mode='range', tmax=10.0, average=False)

# Remove "slow drifts" before running ICA
filt_raw = raw_data.copy().filter(l_freq=1., h_freq=None)


ica = ICA(n_components=15, max_iter='auto', random_state=97)

# .fit and .apply changes ica object in-place
ica.fit(filt_raw)
ica.exclude = [0, 1]
reconst_raw = raw_data1.copy()
ica.apply(reconst_raw)

processed_data = reconst_raw
# processed_data.plot_psd(area_mode='range', tmax=10.0, average=False)

# Extract data & time vector from processed data
data, times = processed_data[:, :]
T = TimeSeries(data[:-5],sampling_rate=250)
ch_names = raw_data1.ch_names[:-5]
delta_lb = 1
delta_ub = 3.99

C = CoherenceAnalyzer(T)
freq_idx = np.where((C.frequencies > delta_lb) * (C.frequencies < delta_ub))[0]
coherence_map = np.mean(C.coherence[:, :, freq_idx], -1)
coherence_plot = drawmatrix_channels(coherence_map, ch_names, size=[10., 10.],title="Coherence map for the $delta$ freq. band (t=[0,10])", color_anchor=0)
plt.show()
