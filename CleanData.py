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


#Get gdf file data from folder
#Includes Runs 1-6 for all participants
gdf_directory = 'C:/Users/cmkro/Documents/2025_Research/SSVEP_Analysis/Participant_Data/Sorted' 
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
     #average reference 
     raw = raw.set_eeg_reference(ref_channels="average")
     #Visualize one data file
     #raw.plot(title='Raw Data', duration=10, n_channels=30, scalings='auto')

     raw_list.append(raw)

     numLoops += 1
     
#Concatenate all loaded GDF files into a single Raw object
if raw_list:  
    combined_raw = mne.concatenate_raws(raw_list)
    print("All files concatenated successfully!")
else:
    print("No GDF files found in the directory.")     


#combined_raw.plot(block=True)
#data = combined_raw.get_data()
#has_nan = np.isnan(data).any()
#print("has nan values: ", has_nan)

# Construct epochs  
#Original Video IDs (H1, H2, H3, H4, C1, D1, E2, F1)  
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
    #add rejection criteria---- not using this because we are using autoreject to clean epochs
    #add signal loss criteria
    preload=True,
    verbose=False,
)    

#use autoreject to clean epochs
epochs_clean = ar.fit_transform(epochs)


#save_path = 'cleaned-epo.fif'
save_path = 'C:/Users/cmkro/Documents/2025_Research/SSVEP_Projections_Paper_2025/CleanedData/cleaned-epo.fif'

#permanently remove bad epochs so that indices are accurate in next script

print(len(epochs_clean))  # Number of epochs before dropping
epochs_clean.drop_bad()
print('succeeded!')
print(len(epochs_clean))  # Fewer epochs after dropping
#print(epochs.selection)  # Indices of epochs still present

#epochs_clean.save(save_path, overwrite=True)

epochs_clean.save(save_path, overwrite=True)