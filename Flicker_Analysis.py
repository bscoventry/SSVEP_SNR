#------------------------------------------------------------------------------------------------------------
# Analysis of 8 different, 10 second video stimuli. 4 cut at 6hz and 4 cut at 12hz
# Code is based on Python MNE SSVEP tutorial:
# https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
#------------------------------------------------------------------------------------------------------------

import mne
import numpy as np
import asrpy
import glob
from scipy.special import roots_hermite
from scipy.stats import ttest_rel
import os
import matplotlib.pyplot as plt
import pdb


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
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
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
gdf_directory = 'C:/Users/cmkro/Documents/2025_Research/SSVEP_Analysis/Participant_Data/Sorted' #Includes Runs 1-6 for all participants. Does NOT include Intro or Outro Runs.
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
     raw.filter(1., None, fir_design='firwin')
     #Remove notch_freqs
     raw.notch_filter(freqs=notch_freqs,notch_widths=5)
     #configure asr object
     #asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
     
     #pdb.set_trace()


     #clean data
     #asr.fit(raw)
     #data= asr.transform(raw)
     #data = data.set_eeg_reference(ref_channels="average")
     #raw_list.append(data)
     
     raw = raw.set_eeg_reference(ref_channels="average")
     raw_list.append(raw)
     
     #pdb.set_trace()

     numLoops += 1
    

#Concatenate all loaded GDF files into a single Raw object
if raw_list:  # Ensure the list is not empty
    combined_raw = mne.concatenate_raws(raw_list)
    print("All files concatenated successfully!")
else:
    print("No GDF files found in the directory.")

#combined_raw holds about 360 minutes of data. This is correct.

#pdb.set_trace()

#My Questions
#Do I need baseline?    

#This was assumed:
#C1, D1, E2, F1 at 12hz
#H1, H2, H3, H4 is at 6hz

#This is the data based on the plot:
#Plots saved are before asr cleaning
#H2 is 3hz?
#H1 is 3hz and 6hz?
#H3 is 6hz, maybe 3hz but less so, most prominent stuff
#H4 is only 6hz
#D1 is 6hz
#C1 is 6hz    
#E2 is 6hz
#F1 is maybe 12hz
    


# Construct epochs
#combined_raw.annotations.rename({"33031": "H2", "33030": "H1", "33032": "H3", "33033": "H4", "33027": "D1",
                                 #"33026": "C1", "33028": "E2", "33029": "F1"})   
#combined_raw.annotations.rename({"33029": "F1"})     

combined_raw.annotations.rename({"33032": "H3"})       

events, _ = mne.events_from_annotations(combined_raw, verbose=False)
tmin, tmax = -1.0, 10.0, #seconds
baseline = None   

#pdb.set_trace()

epochs = mne.Epochs(
    combined_raw,
    #event_id=["H2", "H1", "H3", "H4", "D1", "C1", "E2", "F1"],
    event_id=["H3"],
    #event_id=["F1"],
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False,
)

#Compute power spectral density (PSD) and signal to noise ratio (SNR)
tmin = 1.0 #cut for transient stimulus onset response
tmax = 10.0 #changed to 10.0 from 20.0
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]        

spectrum = epochs.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds, freqs = spectrum.get_data(return_freqs=True)


#call function to compute SNR spectrum
#LOOK INTO THIS

#3 and 2


#Here, we want to compare power at each bin with average power of the three neighboring bins (on each side) 
#and skip one bin directly next to it.
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=4)    #2 seems to give the best results for second parameter

#---------------------------------------------------------------------------------#
#Debugging
print("snrs.shape is below")
print(snrs.shape)   #epochs, channels, frequency points (802) not time points

#epoch_duration = 802 / epochs.info['sfreq']  # Time in seconds
#print(f"epoch duration is: {epoch_duration}")
print(f"sampling freq is: {sfreq}")
print(epochs.times[-1])  # Should be 10 seconds if correct
print(epochs.get_data().shape)
print(epochs.times.shape)  # Should be 5000 if epochs are 10s
print(epochs.times[0], epochs.times[-1])  # Start and end times





#exit()
#---------------------------------------------------------------------------------#





tick_spacing = 1

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
    ylim=[-2, 30],
    xlim=[fmin, fmax],
)

axes[1].set_xticks(np.arange(fmin, fmax + 1, tick_spacing))  # Set tick locations

fig.show()
plt.show()

exit()

#--------------------------------------------------------------------------------------#
#Statistical Analyses
#Compare each video stimulus event ('H1', 'D1', etc) to each other
#Compare videos at 6hz to videos at 12hz
#during event vs during baseline before event
#occipital ROI versus other electrodes


# define stimulation frequency
stim_freq_12hz = 12.0
stim_freq_6hz = 6.0

stim_freq_5hz = 4.0

# find index of frequency bin closest to stimulation frequency
i_bin_12hz = np.argmin(abs(freqs - stim_freq_12hz))
i_bin_6hz = np.argmin(abs(freqs - stim_freq_6hz))

i_bin_5hz = np.argmin(abs(freqs - stim_freq_5hz))


# for later, we will already find the 15 Hz bin and the 1st and 2nd harmonic
# for both.
#i_bin_24hz = np.argmin(abs(freqs - 24))
#i_bin_36hz = np.argmin(abs(freqs - 36))
#i_bin_15hz = np.argmin(abs(freqs - 15))
#i_bin_30hz = np.argmin(abs(freqs - 30))
#i_bin_45hz = np.argmin(abs(freqs - 45))


#Examples
#C1, D1, E2, F1 at 12hz
#H1, H2, H3, H4 is at 6hz


#Get indices for different trial types
i_trial_C1_12 = np.where(epochs.annotations.description == "C1")[0]
i_trial_H1_6 = np.where(epochs.annotations.description == "H1")[0]

#test if H1 is actually different
i_trial_H1_5 = np.where(epochs.annotations.description == "H1")[0]

#add remaining conditions

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


#This should be the SNR of C1 epochs at 12hz, in ROI
snrs_target_C1_12 = snrs[i_trial_C1_12, :, i_bin_12hz][:, picks_roi_vis]
print("C1 trials, SNR at 12 Hz")
print(f"average SNR (occipital ROI): {snrs_target_C1_12.mean()}")

#This should be the SNR of H1 epochs at 6hz, in ROI
snrs_target_H1_6 = snrs[i_trial_H1_6, :, i_bin_6hz][:, picks_roi_vis]
print("H1 trials, SNR at 6 Hz")
print(f"average SNR (occipital ROI): {snrs_target_H1_6.mean()}")

#This should be the SNR of H1 epochs at 3hz, in ROI
snrs_target_H1_5 = snrs[i_trial_H1_5, :, i_bin_5hz][:, picks_roi_vis]
print("H1 trials, SNR at 4 Hz")
print(f"average SNR (occipital ROI): {snrs_target_H1_5.mean()}")



#exit()



# get average SNR at 6 Hz for ALL channels
snrs_6hz_H1 = snrs[i_trial_H1_6, :, i_bin_6hz]  #all channels
snrs_6hz_chaverage = snrs_6hz_H1.mean(axis=0)
#print("SNR at 6Hz for A2 epochs, all channels:")

# get average SNR at 12 Hz for ALL channels
snrs_12Hz_C1 = snrs[i_trial_C1_12, :, i_bin_12hz]
snrs_12hz_chaverage = snrs_12Hz_C1.mean(axis=0)
#print("SNR at 12Hz for A1 epochs, all channels:")

#test
snrs_5hz_H1 = snrs[i_trial_H1_5, :, i_bin_5hz]  #all channels
snrs_5hz_chaverage = snrs_5hz_H1.mean(axis=0)


# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_6hz_chaverage, epochs.info, vlim=(1, None), axes=ax)
mne.viz.plot_topomap(snrs_12hz_chaverage, epochs.info, vlim=(1, None), axes=ax)

print("C1 trials, SNR at 12 Hz")
print(f"average SNR (occipital ROI): {snrs_target_C1_12.mean()}")
print(f"average SNR (all channels) for C1 trials: {snrs_12hz_chaverage.mean()}")

print("H1 trials, SNR at 6 Hz")
print(f"average SNR (occipital ROI): {snrs_target_H1_6.mean()}")
print(f"average SNR (all channels) for H1 trials: {snrs_6hz_chaverage.mean()}")

print("H1 trials, SNR at 4 Hz")
print(f"average SNR (occipital ROI): {snrs_target_H1_5.mean()}")
print(f"average SNR (all channels) for H1 trials: {snrs_5hz_chaverage.mean()}")


exit()





#Try looking at different things (time points that are not during the event)
#Plot individual epochs in time series , plots the ten second interval


#This code is in place at the top of this script
#combined_raw.annotations.rename({"33024": "A1", "33025": "A2"})  


#HERE

#How to set events into events variable
events = mne.find_events(raw, stim_channel="O1")

event_dict = {
    "33024": "A1",
    "33025": "A2",
}

white_flashes = mne.pick_events(events, include=['A1', 'A2'])
epochs["A1"].plot(
    events=white_flashes,
    event_id="A1",
    event_color=dict(A1="blue"),
)

#no events