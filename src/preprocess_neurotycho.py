import numpy as np
import scipy.signal
from scipy.io import loadmat, savemat


def filter_ecog(arr, f_sample, f_high=200., f_low=2., w_stop=3., order=2, line_freq=60.):
    """Filtering method for ECoG data
    
    Uses fourth-order Butterworth forward-backward filtering.

    Parameters
    ----------
    arr : array-like, shape (n_time,)
        Data to filter
    f_sample : float
        Sample rate, in Hz
    f_high : float, optional (default = 200)
        High frequency cutoff, in Hz
    f_low : float, optional (default = 1.0)
        Low frequency cutoff, in Hz
    w_stop : float, optional (default = 2.5)
        Half-width of 60 Hz bandstop filter
    line_freq : float, optional (default = 60.)
        Line frequency, in Hz. Set to 50 for some countries.

    Returns
    -------
    arr_filt : array-like, shape (n_time,)
        Filtered data
    """
    from scipy import signal
    f_Ny = f_sample / 2. # Nyquist frequency
    # 1) bandpass filter
    b, a = signal.butter(order, [f_low / f_Ny, f_high / f_Ny], 'band')
    arr_filt = signal.filtfilt(b, a, arr)
    # 2) bandstop filter, 60 Hz line noise and harmonics
    for mult in [1, 2, 3, 4]:
        f_stop_mid = line_freq * mult
        f_stop = np.array([f_stop_mid - w_stop, f_stop_mid + w_stop])
        b, a = signal.butter(order, f_stop / f_Ny, 'bandstop')
        arr_filt = signal.filtfilt(b, a, arr_filt)
    return arr_filt


#nperseg = 96 
#noverlap = 64
window_len = 0.096
window_overlap = 0.064
## Miller et al. (2007) choices: 0.25 s window, 0.1 s overlap
#window_len = 0.25
#window_overlap = 0.1

# data_dir = '../data_full/20090611S1_FTT_A_ZenasChao_mat_ECoG32-Motion12'
# n_chan = 32

# data_dir = '../data_full/20090527S1_FTT_K1_ZenasChao_mat_ECoG64-Motion8'
# n_chan = 64

data_dir = '../data_full/20090525S1_FTT_K1_ZenasChao_mat_ECoG64-Motion8'
n_chan = 64

#data_dir = '../data/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+' + \
#    'Shimoda_mat_ECoG64-Motion6'
#n_chan = 64

# out_fn = data_dir + '/Kam_Bands_param_Miller.mat' 
out_fn = data_dir + '/Kam_Bands.mat' 


# Load data
ecog_time = loadmat(data_dir + '/ECoG_time.mat')
ecog_time = ecog_time['ECoGTime'].flatten()
ecog_data = np.zeros((n_chan, len(ecog_time)))
for chan in range(n_chan):
    fn = "%s/ECoG_ch%d.mat" % (data_dir, chan + 1)
    var = "ECoGData_ch%d" % (chan + 1)
    tmp = loadmat(fn)
    ecog_data[chan, :] = tmp[var].astype(float)

DT = ecog_time[1] - ecog_time[0]
nperseg = int(window_len / DT)
noverlap = int(window_overlap / DT)

# Filter out line noise + highpass
for chan in range(n_chan):
    ecog_data[chan, :] = filter_ecog(ecog_data[chan, :], 1./DT,  line_freq = 50.)

# Common average referencing
#ecog_data = ecog_data - np.tile(np.median(ecog_data, axis=0), (n_chan, 1))
# Standardization
#ecog_data = ecog_data - np.tile(np.median(ecog_data, axis=1)[:,np.newaxis],
#                                    (1, ecog_data.shape[1]))
# ecog_data = ecog_data / np.tile(np.std(ecog_data, 1, ddof=1)[:,np.newaxis],
#                                     (1, ecog_data.shape[1]))
#ecog_data = ecog_data / np.tile(np.median(ecog_data, 1)[:,np.newaxis],
#                                    (1, ecog_data.shape[1]))


band_edges = [2, 32, 200]
## Miller choices:
#band_edges = [8, 32, 76, 100]
# add notch filters around 60, 120 Hz OR 50, 100, 150
# compare fits to just low, high separate versus combined
# regression decoder
# look at lambdas
# straw man comparisons: PCA, windowed DMD, ICA

n_band = len(band_edges) - 1
new_ecog = np.zeros((n_chan, n_band, int(len(ecog_time)/10))) 

for chan in range(n_chan):
    f, t, Sxx = scipy.signal.spectrogram(ecog_data[chan, :], fs=1./DT, nperseg=nperseg,
                                             noverlap=noverlap)
    if chan == 0:
        new_ecog = new_ecog[:, :, range(len(t))]
    for band in range(n_band):
        f_low = band_edges[band]
        f_high = band_edges[band + 1]
        idx = (f < f_high) & (f >= f_low)
        new_ecog[chan, band, :] = np.sum(Sxx[idx, :], axis=0)

new_time = t
n_time = len(new_time)

## rearrange data
ecog_post = np.zeros((n_chan * n_band, n_time))
for chan in range(n_chan):
    for band in range(n_band):
        ecog_post[chan * n_band + band, :] = new_ecog[chan, band, :]

ecog_post = np.log10(ecog_post)

import sklearn.preprocessing as skpre
# transform = skpre.QuantileTransformer(output_distribution='normal')
transform = skpre.StandardScaler()
ecog_post = transform.fit_transform(ecog_post.T).T

savemat(out_fn,
        {'ecog_power': new_ecog,
         'ecog_power_time' : new_time,
         'ecog_power_post' : ecog_post,
         'ecog_v_filtered' : ecog_data,
         'ecog_v_time' : ecog_time
        } )
    
