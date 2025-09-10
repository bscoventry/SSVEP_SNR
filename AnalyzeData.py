#------------------------------------------------------------------------------------------------------------
# Analysis of 8 different, 10 second video stimuli. 4 videos cut at 3hz and 4 videos cut at 6hz
# Code is based on Python MNE SSVEP tutorial:
# https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
#------------------------------------------------------------------------------------------------------------

import mne
import pathlib
import matplotlib

matplotlib.use('Qt5Agg')

from mne.preprocessing import ICA
import numpy as np
import asrpy
import glob
from scipy.special import roots_hermite
from scipy.stats import ttest_rel
import os
import matplotlib.pyplot as plt
import pdb
import pandas as pd

from autoreject import AutoReject

ar = AutoReject()

data = []



#functions
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),            #original values = 2 and 1
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


save_path = 'C:/Users/cmkro/Documents/2025_Research/SSVEP_Analysis/CleanedData/cleaned-epo.fif'
#Looks in current working directory by default
epochs_clean = mne.read_epochs(save_path, preload=True)



#Compute power spectral density (PSD) and signal to noise ratio (SNR)
tmin = 1.0 #1.0 cut for transient stimulus onset response
tmax = 10.0 
fmin = 1.0   #1.0
fmax = 50  #90.0
sfreq = epochs_clean.info["sfreq"]        

spectrum = epochs_clean.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),  # No zero-padding
    n_per_seg=int(sfreq * (tmax - tmin)),  # Full trial length
    n_overlap=0,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="hann",  
    verbose=False,
)
psds, freqs = spectrum.get_data(return_freqs=True)


#Compare power at each bin with average power of the three neighboring bins (on each side) 
#and skip one bin directly next to it.
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)   
#noise_n_neighbor_freqs = Number of neighboring frequencies used to compute noise level.
#noise_skip_neighbor_freqs = set this >=1 if you want to exclude the immediately neighboring
        #frequency bins in noise level calculation

#Plotting SNR & PSD Spectra
fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(
    np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
)

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="b")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")



# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color="r")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    #ylim=[-2, 30],
    ylim=[-2, 50],
    xlim=[fmin, fmax],
)



tick_spacing = 1
axes[1].set_xticks(np.arange(fmin, fmax + 1, tick_spacing))  # Set tick locations

fig.show()
plt.show()


#For statistical analyses:

# define stimulation frequency
stim_freq_3hz = 2.7
stim_freq_6hz = 5.3

stim_freq_A4 = 2.2 #H4 = A4
stim_freq_B4 = 4.8 #F1 = B4


i_bin_3hz = np.argmin(abs(freqs - stim_freq_3hz))
i_bin_6hz = np.argmin(abs(freqs - stim_freq_6hz))
i_bin_A4_hz = np.argmin(abs(freqs - stim_freq_A4))
i_bin_B4_hz = np.argmin(abs(freqs - stim_freq_B4))

freq_list = [i_bin_3hz, i_bin_6hz, i_bin_A4_hz, i_bin_B4_hz]
freq_names_list = ["3Hz", "6Hz", "A4Hz", "B4Hz"]



#Get indices for different trial types

#annotations come from raw epochs, which keep everything even dropped info
#i_trial_H1 = np.where(epochs_clean.annotations.description == "H1")[0]
i_trial_H1 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["H1"])[0]
i_trial_H2 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["H2"])[0]
i_trial_H3 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["H3"])[0]
i_trial_H4 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["H4"])[0]

i_trial_C1 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["C1"])[0]
i_trial_D1 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["D1"])[0]
i_trial_E2 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["E2"])[0]
i_trial_F1 = np.where(epochs_clean.events[:, 2] == epochs_clean.event_id["F1"])[0]

#indices
video_list = [i_trial_H1, i_trial_H2, i_trial_H3, i_trial_H4, i_trial_C1, i_trial_D1, i_trial_E2, i_trial_F1]
video_names_list = ["H1", "H2", "H3", "H4", "C1", "D1", "E2", "F1"]


# Define different ROIs
roi_vis = [
    "Oz",
    "O1",
    "O2",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8"
]  # visual roi

# Find corresponding indices using mne.pick_types()
picks_roi_vis = mne.pick_types(
    epochs_clean.info, eeg=True, stim=False, exclude="bads", selection=roi_vis
)

#need to reset indices of clean_epochs
print(len(epochs_clean))   #this changes between one script and the next
print(f"snrs shape: {snrs.shape}")
print('verifying')


#loop through all videos, testing three frequencies (expected and harmonics)
for v_index in range(len(video_list)):
    curvid = video_list[v_index]
    #pdb.set_trace()
    #Get SNR at 3hz
    CurVidName = video_names_list[v_index]
    
    for f_index in range(len(freq_list)):
        curfreq = freq_list[f_index]
        CurFreqName = freq_names_list[f_index]
        
        #ROI
        
        #debugging
        print(f"snrs shape: {snrs.shape}")
        print(f"curvid: {curvid}")

        #reset indices from cleaned_epochs here
        #curvid = np.arange(len(epochs_clean))

        snrs_trial_target_ROI = snrs[curvid, :, curfreq][:, picks_roi_vis]
        avg_snr_roi = snrs_trial_target_ROI.mean()
        #Print current snrs
        print(f"Video:  {CurVidName}, Freq: {CurFreqName}")
        print(f"average SNR (occipital ROI): {snrs_trial_target_ROI.mean()}")
        #All electrodes
        snrs_trial_target_all = snrs[curvid, :, curfreq]
        snrs_trial_chaverage = snrs_trial_target_all.mean(axis=0)
        avg_snr_all = snrs_trial_chaverage.mean()
        print(f"average SNR (all channels): {snrs_trial_chaverage.mean()}")

        # Append to list
        data.append({
            'Video': CurVidName,
            'Frequency': CurFreqName,
            'Avg_SNR_ROI': avg_snr_roi,
            'Avg_SNR_AllChannels': avg_snr_all,
            'All_SNR_ROI': snrs_trial_target_ROI,
            'All_SNR_AllChannels': snrs_trial_target_all
        })

# Create DataFrame
df_snrs = pd.DataFrame(data)

# Display the DataFrame
print(df_snrs)


#loop through all videos, testing three frequencies
# Flatten SNR data for each trial/channel into rows
detailed_data = []

for v_index in range(len(video_list)):
    curvid = video_list[v_index]
    
    #reset indices from cleaned_epochs here
    #curvid = np.arange(len(epochs_clean))
    
    CurVidName = video_names_list[v_index]
    
    for f_index in range(len(freq_list)):
        curfreq = freq_list[f_index]
        CurFreqName = freq_names_list[f_index]
        
        # Get data
        snrs_roi = snrs[curvid, :, curfreq][:, picks_roi_vis]
        snrs_all = snrs[curvid, :, curfreq]
        
        # Flatten ROI data
        for trial_idx in range(snrs_roi.shape[0]):
            for chan_idx in range(snrs_roi.shape[1]):
                detailed_data.append({
                    'VideoVar': CurVidName + '_' + CurFreqName + '_ROI',
                    'Video': CurVidName,
                    'Frequency': CurFreqName,
                    'ROI/All': 'ROI',
                    'Trial': trial_idx,
                    'Channel': picks_roi_vis[chan_idx],
                    'SNR': snrs_roi[trial_idx, chan_idx]
                })

        # Flatten All channels data
        for trial_idx in range(snrs_all.shape[0]):
            for chan_idx in range(snrs_all.shape[1]):
                detailed_data.append({
                    'VideoVar': CurVidName + '_' + CurFreqName + '_ALL',
                    'Video': CurVidName,
                    'Frequency': CurFreqName,
                    'ROI/All': 'All',
                    'Trial': trial_idx,
                    'Channel': chan_idx,
                    'SNR': snrs_all[trial_idx, chan_idx]
                })
                

# SNR Plots of each of the 8 videos
# Topographies for each of the 8 videos
# Amplitudes for each of the 8 videos                

# Convert to DataFrame
df_detailed = pd.DataFrame(detailed_data)
# Save to CSV
df_detailed.to_csv('snr_detailed_output_Topo_Aug28_Correct.csv', index=False)

#Pivot the table so that each VideoVar becomes a column
df_pivoted = df_detailed.pivot_table(
    index=['Trial', 'Channel'],      # Rows: combination of Trial and Channel
    columns='VideoVar',              # Columns: each unique VideoVar
    values='SNR'                     # Values: SNR values
).reset_index()                      # Flatten the index for a clean table

# Save Pivoted Data file
df_pivoted.to_csv('snr_pivoted_output_Topo_Aug28_Correct.csv', index=False)



#H1
snrs_H1 = snrs[i_trial_H1, :, i_bin_3hz]
snrs_H1_chaverage = snrs_H1.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_H1_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)

#H2
snrs_H2 = snrs[i_trial_H2, :, i_bin_3hz]
snrs_H2_chaverage = snrs_H2.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_H2_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)

#H3
snrs_H3 = snrs[i_trial_H3, :, i_bin_3hz]
snrs_H3_chaverage = snrs_H3.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_H3_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)

#H4 at 3hz
snrs_H4 = snrs[i_trial_H4, :, i_bin_A4_hz]
snrs_H4_chaverage = snrs_H4.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_H4_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)


# get average SNR at 6 Hz for ALL channels
#C1
snrs_C1 = snrs[i_trial_C1, :, i_bin_6hz]
snrs_C1_chaverage = snrs_C1.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_C1_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)

#D1
snrs_D1 = snrs[i_trial_D1, :, i_bin_6hz]
snrs_D1_chaverage = snrs_D1.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_D1_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)

#E2
snrs_E2 = snrs[i_trial_E2, :, i_bin_6hz]
snrs_E2_chaverage = snrs_E2.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_E2_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)

#F1 at 6hz (5.7)
snrs_F1 = snrs[i_trial_F1, :, i_bin_B4_hz]
snrs_F1_chaverage = snrs_F1.mean(axis=0)
# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_F1_chaverage, epochs_clean.info, vlim=(1, None), axes=ax)




exit()


