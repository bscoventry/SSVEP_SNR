#------------------------------------------------------------------------------------------------------------
# Analysis of 8 different, 10 second video stimuli. 4 videos cut at 3hz and 4 videos cut at 6hz
# Code is based on Python MNE SSVEP tutorial:
# https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
#------------------------------------------------------------------------------------------------------------

import mne
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


#Get data from folder
gdf_directory = 'C:/Users/cmkro/Documents/2025_Research/SSVEP_Analysis/Participant_Data/Sorted' #Includes Runs 1-6 for all participants. Does NOT include Outro Runs (white flashes for validation).
gdf_files = [f for f in os.listdir(gdf_directory) if f.endswith('.gdf')]
raw_list = []
numLoops = 0
notch_freqs = [50,100,150]
#Preprocessing at the same time the dataset is loaded in
for gdf_file in gdf_files:       
     file_path = os.path.join(gdf_directory, gdf_file)
     print(f"Loading: {file_path}")
     raw = mne.io.read_raw_gdf(file_path, preload=True)
     #correct channel names from wrong OpenVibe labels
     raw.rename_channels({'FP1':'Fp1','FP2':'Fp2'})
     #set montage template with xy coordinates (required for scalp maps, source localization, etc.)
     montage = mne.channels.make_standard_montage("standard_1020")
     raw.set_montage(montage)
     #highpass filter to get rid of slow drifts (recommended for asr)
     raw.filter(1., 40., fir_design='firwin')     #40
     #Remove notch_freqs
     raw.notch_filter(freqs=notch_freqs,notch_widths=5)
     
     raw = raw.set_eeg_reference(ref_channels="average")
     raw_list.append(raw)

     numLoops += 1
     

     

#Concatenate all loaded GDF files into a single Raw object
if raw_list:  # Ensure the list is not empty
    combined_raw = mne.concatenate_raws(raw_list)
    print("All files concatenated successfully!")
else:
    print("No GDF files found in the directory.")

# Construct epochs
combined_raw.annotations.rename({"33031": "H2", "33030": "H1", "33032": "H3", "33033": "H4", "33027": "D1",
                                 "33026": "C1", "33028": "E2", "33029": "F1"})   
events, _ = mne.events_from_annotations(combined_raw, verbose=False)
tmin, tmax = -1, 10, #seconds
baseline = None   

epochs = mne.Epochs(
    combined_raw,
    event_id=["H2", "H1", "H3", "H4", "D1", "C1", "E2", "F1"],
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False,
)


#Compute power spectral density (PSD) and signal to noise ratio (SNR)
tmin = 1.0 #1.0 cut for transient stimulus onset response
tmax = 10.0 
fmin = 1.0   #1.0
fmax = 50  #90.0
sfreq = epochs.info["sfreq"]        

spectrum = epochs.compute_psd(
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
stim_freq_12hz = 10.6  

# find index of frequency bin closest to stimulation frequency
i_bin_3hz = np.argmin(abs(freqs - stim_freq_3hz))
i_bin_6hz = np.argmin(abs(freqs - stim_freq_6hz))
i_bin_12hz = np.argmin(abs(freqs - stim_freq_12hz))

freq_list = [i_bin_3hz, i_bin_6hz, i_bin_12hz]
freq_names_list = ["3Hz", "6Hz", "12Hz"]

#Get indices for different trial types
i_trial_H1 = np.where(epochs.annotations.description == "H1")[0]
i_trial_H2 = np.where(epochs.annotations.description == "H2")[0]
i_trial_H3 = np.where(epochs.annotations.description == "H3")[0]
i_trial_H4 = np.where(epochs.annotations.description == "H4")[0]

i_trial_C1 = np.where(epochs.annotations.description == "C1")[0]
i_trial_D1 = np.where(epochs.annotations.description == "D1")[0]
i_trial_E2 = np.where(epochs.annotations.description == "E2")[0]
i_trial_F1 = np.where(epochs.annotations.description == "F1")[0]

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
    epochs.info, eeg=True, stim=False, exclude="bads", selection=roi_vis
)

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
        #please make pandas dataframe

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
df_detailed.to_csv('snr_detailed_output_addDate.csv', index=False)

#Pivot the table so that each VideoVar becomes a column
df_pivoted = df_detailed.pivot_table(
    index=['Trial', 'Channel'],      # Rows: combination of Trial and Channel
    columns='VideoVar',              # Columns: each unique VideoVar
    values='SNR'                     # Values: SNR values
).reset_index()                      # Flatten the index for a clean table

# Save Pivoted Data file
df_pivoted.to_csv('snr_pivoted_output_addDate.csv', index=False)



exit()


