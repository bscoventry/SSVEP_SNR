#------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry, Claudia Krogmeier       Wisconsin Institute for Translational Neuroengineering
# Date: 01/28/2025
# Purpose: Testing SSVEP Responses, Brandon's edition (quoting Taylor Swift)
# Revision Hist: None
#------------------------------------------------------------------------------------------------------------
#Import relevant repos
import mne
import numpy as np
import asrpy
import glob
from scipy.special import roots_hermite
from scipy.stats import ttest_rel
import os
import matplotlib.pyplot as plt
import pdb
#Add any function defs here for clarity

#I'm going to steal a bit of this code from MNE. Method seems sound, but may need characterization.
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

    # The mean is not defined on the edges so we will pad it with nans. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

#Let's sort data
#Standard white flash SSVEP
gdf_directory = 'C:/Users/cmkro/Documents/2025_Research/SSVEP_Analysis/Participant_Data/Validation'
gdf_files = [f for f in os.listdir(gdf_directory) if f.endswith('.gdf')]
raw_list = []
numLoops = 0
notch_freqs = [50,100,150]
#Changes from Claud's code: Did preprocessing for each dataset loaded in at time of load.
for gdf_file in gdf_files:        #This seems to be working properly, but name is set to first file read in. Check this
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
    
    raw.notch_filter(freqs=notch_freqs,notch_widths=5)
    #configure asr object
    asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
    #clean data
    asr.fit(raw)
    data= asr.transform(raw)
    data = data.set_eeg_reference(ref_channels="average")
    raw_list.append(data)
    numLoops += 1
# Concatenate all loaded GDF files into a single Raw object
if raw_list:  # Ensure the list is not empty
    combined_raw = mne.concatenate_raws(raw_list)
    print("All files concatenated successfully!")
else:
    print("No GDF files found in the directory.")

#Data loading and preprocessing steps were completed in above loop. Everything should be good
#for analysis. Moving to epocing.

raw.annotations.rename({"33024": "A1", "33025": "A2"})    
tmin, tmax = -1.0, 10.0,  # in s, changed to 10.0 from 20.0, changed from 0 to -1.0
baseline = None     #To do: add baseline
pdb.set_trace()

events = mne.events_from_annotations(raw)
catch_trials = mne.pick_events(events[0], include=[6,7])

epochs = mne.Epochs(
    raw,
    event_id=["A1", "A2"],
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False,
)
if plot_True:
    epochs["face"].plot(events=catch_trials)
    plt.show()
#
print('How I learned to not be afraid and trust the analysis')    
#Compute power spectral density (PSD) and signal to noise ratio (SNR)

tmin = 1.0 #cut for transient stimulus onset response
tmax = 10.0 #changed to 10.0 from 20.0
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]        #This is reading different from hardware. Check 500 vs 512.
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
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1)    #2 seems to give the best results

#trials, channels, freq bins
print("snrs.shape is below")
print(snrs.shape)

#exit()




#Above, we want to compare power at each bin with average
#    power of the three neighboring bins (on each side) and skip one bin directly next to it.




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

fig.show()
plt.show()

# define stimulation frequency
stim_freq_12hz = 12.0
stim_freq_6hz = 6.0

# find index of frequency bin closest to stimulation frequency
i_bin_12hz = np.argmin(abs(freqs - stim_freq_12hz))
i_bin_6hz = np.argmin(abs(freqs - stim_freq_6hz))



# could be updated to support multiple frequencies

# for later, we will already find the 15 Hz bin and the 1st and 2nd harmonic
# for both.
#i_bin_24hz = np.argmin(abs(freqs - 24))
#i_bin_36hz = np.argmin(abs(freqs - 36))
#i_bin_15hz = np.argmin(abs(freqs - 15))
#i_bin_30hz = np.argmin(abs(freqs - 30))
#i_bin_45hz = np.argmin(abs(freqs - 45))

#i_trial_E2 = np.where(epochs.annotations.description == "E2")[0]

#Get indices for different trial types
i_trial_A1 = np.where(epochs.annotations.description == "A1")[0]
i_trial_A2 = np.where(epochs.annotations.description == "A2")[0]
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


#This should be the SNR of A1 epochs at 12hz, in ROI
snrs_target_A1 = snrs[i_trial_A1, :, i_bin_12hz][:, picks_roi_vis]
print("sub 2, A1 trials, SNR at 12 Hz")
print(f"average SNR (occipital ROI): {snrs_target_A1.mean()}")

#This should be the SNR of A2 epochs at 6hz, in ROI
snrs_target_A2 = snrs[i_trial_A2, :, i_bin_6hz][:, picks_roi_vis]
print("sub 2, A2 trials, SNR at 6 Hz")
print(f"average SNR (occipital ROI): {snrs_target_A2.mean()}")

# get average SNR at 6 Hz for ALL channels
snrs_6hz_A2 = snrs[i_trial_A2, :, i_bin_6hz]  #all channels
snrs_6hz_chaverage = snrs_6hz_A2.mean(axis=0)
#print("SNR at 6Hz for A2 epochs, all channels:")

# get average SNR at 12 Hz for ALL channels
snrs_12Hz_A1 = snrs[i_trial_A1, :, i_bin_12hz]
snrs_12hz_chaverage = snrs_12Hz_A1.mean(axis=0)
#print("SNR at 12Hz for A1 epochs, all channels:")

# plot SNR topography
#fig, ax = plt.subplots(1)
#mne.viz.plot_topomap(snrs_6hz_chaverage, epochs.info, vlim=(1, None), axes=ax)
#mne.viz.plot_topomap(snrs_12hz_chaverage, epochs.info, vlim=(1, None), axes=ax)

print("sub 2, A1 trials, SNR at 12 Hz")
print(f"average SNR (occipital ROI): {snrs_target_A1.mean()}")
print(f"average SNR (all channels) for A1 trials: {snrs_12hz_chaverage.mean()}")

print("sub 2, A2 trials, SNR at 6 Hz")
print(f"average SNR (occipital ROI): {snrs_target_A2.mean()}")
print(f"average SNR (all channels) for A2 trials: {snrs_6hz_chaverage.mean()}")