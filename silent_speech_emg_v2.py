import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import scipy
import os
import json
import sys    
import joblib #when called with same arguments, previously computed result is returned from cache
import hmmlearn
from hmmlearn import hmm


#silent_data="D:\Lauren\emg_data\closed_vocab\silent\5-19_silent"
voiced_data_path= Path(r"D:\Lauren\emg_data\closed_vocab\voiced\5-19")
emg_0=np.load(r"D:\Lauren\emg_data\closed_vocab\voiced\5-19\0_emg.npy")


'''
for file in Path('/path/to/directory').glob('*_emg.npy'): # Find all .txt files
    print(file)
'''
#load emg dataset and labels
def load_emg_dataset(voiced_data_path):
    emg_files = []
    labels = []

    for file in os.listdir(voiced_data_path):
        if file.endswith("_emg.npy"):
            emg_path = os.path.join(voiced_data_path, file)
            json_path = os.path.join(voiced_data_path, file.replace("_emg.npy", "_info.json"))
            
            if not os.path.exists(json_path):
                continue
            
            # Load EMG and label
            emg = np.load(emg_path)
            with open(json_path, 'r') as f:
                meta = json.load(f)
            label = meta["text"]
            emg_files.append(emg_path)
            labels.append(label)
    return emg_files, labels           



def preprocess_emg(emg, fs, new_fs=None, lowcut=20, highcut=450, notch_freq=60, harmonics=8, apply_tanh=True, tanh_scale=50, tanh_pre_scale=1/20):
    #Bandstop to remove AC electrical noise
    def notch(sig, freq, sample_frequency):
        b, a = signal.iirnotch(freq, 30, sample_frequency)
        return signal.filtfilt(b, a, sig)
    #apply notch filters (and harmonics)
    def notch_harmonics(sig, freq, sample_frequency, n_harmonics):
        for harmonic in range(1, n_harmonics + 1):
            f = freq * harmonic
            if f < sample_frequency / 2:  # only apply if below Nyquist
                sig = notch(sig, f, sample_frequency)
        return sig
    #remove low-frequency drift (high-pass) and DC offset
    def remove_drift(sig, sample_frequency):
        b, a = signal.butter(3, 2, 'highpass', fs=sample_frequency)
        return signal.filtfilt(b, a, sig)
    #resample to new frequency
    def resample_signal(sig, new_freq, old_freq):
        times = np.arange(len(sig)) / old_freq
        new_times = np.arange(0, times[-1], 1 / new_freq)
        return np.interp(new_times, times, sig)

    # Ensure 2D array
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]

    #processed = np.zeros_like(emg)
    processed = []
    for ch in range(emg.shape[1]):
        sig = emg[:, ch]

        #Remove DC offset
        sig = sig - np.mean(sig)

        # Notch filters + harmonics
        sig = notch_harmonics(sig, notch_freq, fs, harmonics)

        #High-pass to remove drift
        sig = remove_drift(sig, fs)

        #Bandpass filter
        nyq = 0.5 * fs
        b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')
        sig = signal.filtfilt(b, a, sig)

        #de-spiking (amplitude compression to limit outliers)
        #same values as dataset code
        if apply_tanh:
            sig = sig * tanh_pre_scale  # normalize before tanh
            sig = tanh_scale * np.tanh(sig / tanh_scale)

        # Optional resampling
        if new_fs is not None and new_fs != fs:
            sig = resample_signal(sig, new_fs, fs)

        processed.append(sig)
    processed = np.stack(processed, axis=1)
    #processed[:len(sig), ch] = sig[:len(processed)]

    return processed

#function to frame the signal into overlapping windows
def frame_signal(signal, frame_size, hop_size):
    frames = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frames.append(signal[start:start + frame_size])
    return np.array(frames)

'''
#to from multiple channels
def frame_multichannel_emg(emg, frame_size, hop_size):
    """
    Frame multichannel EMG into overlapping windows.
    Returns array of shape (n_frames, frame_size, n_channels)
    """
    n_samples, n_channels = emg.shape
    frames = []
    for start in range(0, n_samples - frame_size, hop_size):
        frames.append(emg[start:start + frame_size, :])
    return np.stack(frames)
'''
def extract_time_features(frames, fs):
    feats = []
    #compute the 5 features as per Jo et al., 2006
    for frame in frames:
        # Low-pass and high-pass versions
        b_low, a_low = signal.butter(2, 100 / (0.5 * fs), 'low')
        b_high, a_high = signal.butter(2, 100 / (0.5 * fs), 'high')

        low = signal.filtfilt(b_low, a_low, frame, axis=0)
        high = signal.filtfilt(b_high, a_high, frame, axis=0)

        # Features
        mean_low = np.mean(low, axis=0)
        rectified_mean = np.mean(np.abs(high), axis=0)
        power_low = np.mean(low ** 2, axis=0)
        power_high = np.mean(high ** 2, axis=0)

        zero_cross = np.mean([
            ((np.roll(high[:, ch], 1) * high[:, ch]) < 0).sum()
            for ch in range(high.shape[1])
        ])

        # Spectral features (optional)
        f, Pxx = signal.welch(frame, fs, nperseg=128, axis=0)
        spectral_mean = np.mean(Pxx, axis=0)

        frame_feats = np.hstack([mean_low, rectified_mean, power_low, power_high, spectral_mean, zero_cross])
        feats.append(frame_feats)

    feats = np.array(feats)
    print("Extracted features shape:", feats.shape)
    return feats

def normalize_features(features, scaler=None):
    """
    Normalize features to zero mean and unit variance.
    If scaler is provided, uses the same parameters (for validation/test).
    
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
    #can probably delete this else
    else:
        normalized = scaler.transform(features)
    return normalized, scaler

'''
def extract_frequency_features(frames, fs):
    n_frames, frame_size, n_channels = frames.shape
    feats = []

    for ch in range(n_channels):
        x = frames[:, :, ch]
        # Compute 16-point FFT
        spectrum = np.fft.rfft(x, n=16)  # shape (n_frames, 9)
        mag = np.abs(spectrum)
        power = mag ** 2
        freqs = np.fft.rfftfreq(16, d=1/fs)

        # Normalize power for probability-based features
        p_norm = power / np.sum(power, axis=1, keepdims=True)

        # Feature calculations
        mean_freq = np.sum(freqs * p_norm, axis=1)
        median_freq = np.array([
            freqs[np.searchsorted(np.cumsum(p), 0.5)] for p in p_norm
        ])
        total_power = np.sum(power, axis=1)
        peak_freq = freqs[np.argmax(power, axis=1)]
        variance = np.var(power, axis=1)
        skewness = skew(power, axis=1)
        kurt = kurtosis(power, axis=1)
        spec_entropy = entropy(p_norm, axis=1)
        spec_flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (np.mean(power, axis=1) + 1e-12)

        ch_feats = np.vstack([
            mean_freq, median_freq, total_power, peak_freq,
            variance, skewness, kurt, spec_entropy, spec_flatness
        ]).T

        feats.append(ch_feats)

    feats = np.hstack(feats)
    return feats

'''
#main
import sys
raw_emg=np.load(r"D:\Lauren\emg_data\closed_vocab\voiced\5-19\0_emg.npy")

fs=1000 # Sampling frequency in Hz
new_fs=516.76 # New sampling frequency in Hz
frame_size=16 # Frame size in samples (e.g., 33 samples for ~64ms at 516.76 Hz)
hop_size=6  # Hop size in samples (e.g., 16 samples for ~31ms at 516.76 Hz)

# --- Load and split files ---
emg_files, labels = load_emg_dataset(voiced_data_path)
train_files, temp_files, train_labels, temp_labels = train_test_split(emg_files, labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)
# ≈ 420 / 90 / 90 split

# --- Feature extraction ---
X_train_features, y_train_labels = [], []
X_val_features, y_val_labels = [], []

for emg_file, label in zip(train_files, train_labels):

    emg_file=np.load(emg_file)

    #preprocess the emg signal
    filtered_emg=preprocess_emg(emg_file, fs, new_fs)

    frames=frame_signal(filtered_emg, frame_size, hop_size)
    print("Frames shape:", frames.shape)  # Should be (n_frames, frame_size, n_channels)

    #extract time-domain features
    features=extract_time_features(frames, fs)
    print("Features shape:", features.shape)  # Should be (n_frames, n_features)

    X_train_features.append(features)
    y_train_labels.append(label)

# Fit scaler on training data only
all_train_feats = np.vstack(X_train_features)
all_train_feats_norm, scaler = normalize_features(all_train_feats)
joblib.dump(scaler, "emg_feature_scaler.pkl")   

# Now normalize each sequence separately
idx = 0
for i in range(len(X_train_features)):
    n = len(X_train_features[i])
    X_train_features[i] = all_train_feats_norm[idx:idx+n]
    idx += n

# Validation set (apply same scaler)
for file, label in zip(val_files, val_labels):
    emg = np.load(file)
    emg_proc = preprocess_emg(emg, fs=1000, new_fs=516.76)
    frames = frame_signal(emg_proc, frame_size=16, hop_size=6)
    feats = extract_time_features(frames, fs=1000)
    feats_norm, _ = normalize_features(feats, scaler=scaler)
    X_val_features.append(feats_norm)
    y_val_labels.append(label)


# --- HMM training ---
unique_labels = sorted(set(y_train_labels))
models = {}

for label in unique_labels:
    X_label = [x for x, l in zip(X_train_features, y_train_labels) if l == label]
    lengths = [len(x) for x in X_label]
    X_concat = np.vstack(X_label)
    model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=100)
    model.fit(X_concat, lengths)
    models[label] = model
    print(f"Trained HMM for label: {label}")

# --- Validation ---
def predict_label(models, features):
    scores = {label: model.score(features) for label, model in models.items()}
    return max(scores, key=scores.get)

y_pred = [predict_label(models, feats) for feats in X_val_features]
accuracy = np.mean(np.array(y_pred) == np.array(y_val_labels))
print("Validation Accuracy:", accuracy)

#frequency features using STFT
'''
#get frequency features using STFT
freq_feats = stft_features(framed_emg, fs=516.8)
print(freq_feats.shape)  # (n_frames, 9*8 = 72)

# Combine with time-domain features (40)
all_feats = np.hstack([time_feats, freq_feats])
print(all_feats.shape)  # (n_frames, 112)

'''

#feature normalizer check
'''
plt.figure(figsize=(8,3))
plt.plot(np.mean(features_norm, axis=0), label='Mean (≈0)')
plt.plot(np.std(features_norm, axis=0), label='Std (≈1)')
plt.legend()
plt.title("Feature Normalization Check")
plt.show()
'''

# Suppose X = list of [n_frames, n_features]
# and y = list of word/sentence labels
unique_labels = sorted(set(y_train_labels))
models = {}

for label in unique_labels:
    # Collect all feature sequences for this label
    X_label = [x for x, l in zip(X_train_features, y_train_labels) if l == label]
    
    # Concatenate for training (HMM expects continuous data)
    lengths = [len(x) for x in X_label]
    X_concat = np.vstack(X_label)
    
    # Train a Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=5, covariance_type='diag', n_iter=100
    )
    model.fit(X_concat, lengths)
    
    models[label] = model
    print(f"Trained HMM for label: {label}")

#//////////////////////

#validation
def predict_label(models, features):
    scores = {label: model.score(features) for label, model in models.items()}
    return max(scores, key=scores.get)

y_pred = []
for features, true_label in zip(X_val_features, y_val_labels):
    pred = predict_label(models, features)
    y_pred.append(pred)

accuracy = np.mean(np.array(y_pred) == np.array(y_val_labels)) #comparing predicted with actual labels
print("Validation Accuracy:", accuracy)




'''
# Processed signal
t_proc = (np.arange(len(filtered_emg))/new_fs)  # time in seconds
print(len(t_proc))
print(len(filtered_emg))
plt.figure(figsize=(10,4))#figsize=(100, 40))
plt.subplot(2, 1, 2)
plt.plot(t_proc, filtered_emg[:, 0], color='orange')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Processed EMG - Channel 1 - EMG 0")

t_raw = (np.arange(len(emg_0))/new_fs)  # time in seconds
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t_raw, emg_0[:, 0])
plt.xlabel("Time (s)")
plt.ylabel("EMG Channel 0 (mV)")
plt.title("Original EMG - Channel 0")
plt.show()
'''

'''
for file in Path(voiced_data_path).glob('*_emg.npy'): #for all emg files
    emg=np.load(file)
    #print(emg)
    x=emg[:,0]#0th column
    y=emg[0:,:]#1th row to end
    plt.plot(x,y)
   
    #preprocess_emg(file, fs=2000, lowcut=20, highcut=450)
    #print(file)
'''


