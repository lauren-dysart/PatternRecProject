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
from sklearn.mixture import GaussianMixture
from scipy.fftpack import dct

voiced_data_path= Path(r"D:\Lauren\emg_data\closed_vocab\voiced\5-19")


#load emg dataset and labels
def load_emg_dataset(voiced_data_path):
    emg_files = []
    json_files = []
    labels = []

    for file in os.listdir(voiced_data_path):
        if file.endswith("_emg.npy"):
            emg_path = voiced_data_path / file
            json_path = voiced_data_path / file.replace("_emg.npy", "_info.json")
            if not json_path.exists():
                continue

            with open(json_path, "r") as f:
                meta = json.load(f)
            label = meta["text"]

            emg_files.append(str(emg_path))
            json_files.append(str(json_path))
            labels.append(label)

    return emg_files, json_files, labels

#emg preprocessing
def preprocess_emg(emg, fs, new_fs=None, lowcut=10, highcut=400, notch_freq=60, harmonics=8, apply_tanh=True, tanh_scale=50, tanh_pre_scale=1/20):
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
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]

    processed = []
    nyq = 0.5 * fs
    b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype='band')

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
    return processed

#segment loading
def load_emg_segments(emg_file, json_file):
    """Load EMG and corresponding labeled word segments."""
    emg = np.load(emg_file)
    with open(json_file, 'r') as f:
        meta = json.load(f)

    chunks = meta["chunks"]
    words = meta["text"].split()  # ["Monday", "February", "12"]
    
    segments = []
    for i, (start, end, _) in enumerate(chunks[:len(words)]):
        seg = emg[start:end]
        label = words[i]
        segments.append((seg, label))
    
    return segments


#function to frame the signal into overlapping windows
def frame_signal(signal, frame_size, hop_size):
    frames = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frames.append(signal[start:start + frame_size])
    return np.array(frames)

#time-domain feature extraction
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

        #extra features, dont add any performance according to tests
        ''' 
        variance_high = np.var(high, axis=0)
        variance_low = np.var(low, axis=0)
        root_mean_square = np.sqrt(np.mean(frame ** 2, axis=0))
        mean_waveform_length = np.mean(np.abs(np.diff(frame, axis=0)), axis=0)
        '''

        zero_cross = np.mean([
            ((np.roll(high[:, ch], 1) * high[:, ch]) < 0).sum()
            for ch in range(high.shape[1])
        ])

        # Spectral features (optional)
        f, Pxx = signal.welch(frame, fs, nperseg=128, axis=0)
        spectral_mean = np.mean(Pxx, axis=0)

        frame_feats = np.hstack([mean_low, rectified_mean, power_low, power_high, spectral_mean, zero_cross]) #,variance_high, variance_low, root_mean_square, mean_waveform_length])
        feats.append(frame_feats)

    feats = np.array(feats)
    return feats

def extract_mfcc_features(frames, fs, n_mfcc=13, n_filters=26, include_deltas=True):
    """
    Compute MFCC features per EMG channel.
    Input:
        frames: (n_frames, frame_size, n_channels)
        fs: sampling frequency
    Output:
        mfcc_features: (n_frames, n_channels * n_mfcc*(1, 2, or 3))
    """

    n_frames, frame_size, n_channels = frames.shape
    
    # Window
    window = np.hamming(frame_size)

    # FFT params
    NFFT = 512

    # Mel filterbank
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(m):
        return 700 * (10**(m / 2595) - 1)

    # Mel filterbank
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(fs / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((NFFT + 1) * hz_points / fs).astype(int)

    fbank = np.zeros((n_filters, NFFT // 2 + 1))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            fbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i - 1, j] = (right - j) / (right - center)

    # Store features
    feats_all_channels = []

    for ch in range(n_channels):
        x = frames[:, :, ch] * window

        # FFT → power spectrum
        spectrum = np.fft.rfft(x, NFFT)
        power = (1.0 / NFFT) * (np.abs(spectrum) ** 2)

        # Apply Mel filterbank
        mel_energy = np.dot(power, fbank.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)

        # Log Mel energies
        log_mel = np.log(mel_energy)

        # DCT → MFCC
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]

        # Add Δ (delta) and ΔΔ (delta-delta)
        if include_deltas:
            # simple delta formula
            delta = np.vstack([mfcc[1:] - mfcc[:-1], mfcc[-1:] - mfcc[-2]])
            delta_delta = np.vstack([delta[1:] - delta[:-1], delta[-1:] - delta[-2]])
            feats = np.hstack([mfcc, delta, delta_delta])
        else:
            feats = mfcc

        feats_all_channels.append(feats)

    # Concatenate channel MFCCs
    feats_all_channels = np.hstack(feats_all_channels)
    return feats_all_channels

#extracting all features per word
def extract_features_for_segments(emg_file, json_file, fs, new_fs, frame_size, hop_size):
    segments = load_emg_segments(emg_file, json_file)
    features, labels = [], []

    for seg, label in segments:
        filtered = preprocess_emg(seg, fs, new_fs)
        frames = frame_signal(filtered, frame_size, hop_size)

        time_feats = extract_time_features(frames, fs)
        mfcc_feats = extract_mfcc_features(frames, new_fs, n_mfcc=13, include_deltas=True)

        # Ensure both have same number of frames
        min_frames = min(time_feats.shape[0], mfcc_feats.shape[0])
        time_feats = time_feats[:min_frames]
        mfcc_feats = mfcc_feats[:min_frames]
        
        # Normalize each feature group separately
        time_feats = (time_feats - time_feats.mean(axis=0)) / (time_feats.std(axis=0) + 1e-8)
        mfcc_feats = (mfcc_feats - mfcc_feats.mean(axis=0)) / (mfcc_feats.std(axis=0) + 1e-8)   
        # Concatenate along feature dimension (axis=1)
        combined_feats = np.hstack([time_feats, mfcc_feats])
        
        # Handle NaNs
        combined_feats = np.nan_to_num(combined_feats)   

        #feats = np.nan_to_num(feats)
        features.append(combined_feats)
        labels.append(label)

    return features, labels

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

def make_pca(X_all, target_dim=20):
    # Remove constant columns
    nonzero = X_all.std(axis=0) > 1e-6
    X_all = X_all[:, nonzero]

    print("Dims before PCA:", X_all.shape[1])

    # Safe PCA size
    d = min(target_dim, X_all.shape[1])
    pca = PCA(n_components=d, whiten=True)
    pca.fit(X_all)
    return pca, nonzero

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



def train_gmm_hmms(word_list, X_train_reduced, y_train, 
                   n_states=4, n_mixtures=2, min_frames=30, min_samples=3):
    """
    FIXED: Trains robust GMM-HMMs with proper initialization.
    
    Key fixes:
    1. Initialize startprob_ and transmat_ to valid values
    2. Reduce n_mixtures to 2 (was 3) for more stability
    3. Check minimum samples per word (not just frames)
    4. Add more conservative covariance regularization
    """
    gmms = {}
    
    for word in word_list:
        # Collect sequences for this word
        word_sequences = [f for f, lab in zip(X_train_reduced, y_train) if lab == word]
        
        # Check minimum samples
        if len(word_sequences) < min_samples:
            print(f"⚠ Skipping {word}: only {len(word_sequences)} samples (need {min_samples})")
            continue
        
        # Concatenate all frames
        word_frames = np.vstack(word_sequences)
        
        # Remove NaNs/Infs
        word_frames = word_frames[np.isfinite(word_frames).all(axis=1)]
        
        if word_frames.shape[0] < min_frames:
            print(f"⚠ Skipping {word}: only {word_frames.shape[0]} frames (need {min_frames})")
            continue

        print(f"Training GMM-HMM for '{word}' on {len(word_sequences)} samples, {word_frames.shape[0]} frames...")

        # Create model with proper initialization
        model = hmm.GMMHMM(
            n_components=n_states,
            n_mix=n_mixtures,
            covariance_type='diag',
            n_iter=100,  # Reduced from 200 for faster convergence
            random_state=42,
            min_covar=1e-2,  # INCREASED from 1e-3 for more regularization
            init_params="",  # Don't auto-initialize (we'll do it manually)
            params="stmc"  # Update all parameters during training
        )

        # CRITICAL FIX: Manually initialize to valid probability distributions
        model.startprob_ = np.ones(n_states) / n_states
        model.transmat_ = (np.ones((n_states, n_states)) + np.eye(n_states) * 2) / (n_states + 2)
        
        # Initialize GMM parameters with k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_states * n_mixtures, random_state=42, n_init=10)
        kmeans.fit(word_frames)
        
        # Train with concatenated frames
        try:
            # Compute lengths for each sequence
            lengths = [seq.shape[0] for seq in word_sequences]
            model.fit(word_frames, lengths=lengths)
            
            # Verify the model is valid
            if np.isnan(model.startprob_).any() or np.isnan(model.transmat_).any():
                print(f"⚠ Skipping {word}: NaN values after training")
                continue
                
            gmms[word] = model
            print(f"✔ Successfully trained '{word}'")
            
        except Exception as e:
            print(f"⚠ Failed to train '{word}': {str(e)}")
            continue

    return gmms

def make_lda(X_all, y_all, max_components=None):
    """
    LDA finds directions that best separate classes.
    Max components = min(n_features, n_classes - 1)
    """
    # Remove constant columns
    nonzero = X_all.std(axis=0) > 1e-6
    X_all = X_all[:, nonzero]
    
    n_classes = len(np.unique(y_all))
    max_dims = min(X_all.shape[1], n_classes - 1)
    
    if max_components is None:
        n_components = max_dims
    else:
        n_components = min(max_components, max_dims)
    
    print(f"LDA: {X_all.shape[1]} features → {n_components} components for {n_classes} classes")
    
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_all, y_all)
    
    return lda, nonzero

def score_word(gmms, feat):
    """Return per-word log-likelihood scores."""
    return {
        word: model.score(feat) 
        for word, model in gmms.items()
    }

def predict_word_gmmhmm(gmms, feat):
    """Return best word + score + all scores."""
    scores = {}
    for word, model in gmms.items():
        try:
            scores[word] = model.score(feat)
        except Exception as e:
            scores[word] = -np.inf
            
    if not scores or all(s == -np.inf for s in scores.values()):
        return None, -np.inf, scores
        
    best_word = max(scores, key=scores.get)
    return best_word, scores[best_word], scores

#word prediction using GMMs
def predict_word(gmms, features_scaled):
    scores = {word: gmm.score(features_scaled) for word, gmm in gmms.items()}
    return max(scores, key=scores.get)

def classify_with_unknown(gmms, feat, threshold=-15000):
    """
    Return UNKNOWN if all words score poorly.
    """
    word, score, _ = predict_word_gmmhmm(gmms, feat)
    if score < threshold:
        return "UNKNOWN"
    return word



# --- Parameters ---
fs=1000 # Sampling frequency in Hz
new_fs=516.76 # New sampling frequency in Hz
frame_size=16 # Frame size in samples (e.g., 33 samples for ~64ms at 516.76 Hz)
hop_size=6  # Hop size in samples (e.g., 16 samples for ~31ms at 516.76 Hz)

# --- Load and split files ---
emg_files, json_files, labels = load_emg_dataset(voiced_data_path)

train_emg, temp_emg, train_json, temp_json = train_test_split(emg_files, json_files, test_size=0.3, random_state=42)
val_emg, test_emg, val_json, test_json = train_test_split(temp_emg, temp_json, test_size=0.5, random_state=42)
print(f"Train: {len(train_emg)}, Val: {len(val_emg)}, Test: {len(test_emg)}")

# ≈ 420 / 90 / 90 split

# Gather features from all files
X_train, y_train = [], []

# --- Feature extraction - train ---
X_train_features, y_train_labels = [], []
X_val_features, y_val_labels = [], []

for emg_path, json_path in zip(train_emg, train_json):
    feats, labels = extract_features_for_segments(emg_path, json_path, fs, new_fs, frame_size, hop_size)
    X_train.extend(feats)
    y_train.extend(labels)

'''# Normalize PCA
all_feats = np.vstack(X_train)
scaler = StandardScaler().fit(all_feats)
X_train_scaled = [scaler.transform(f) for f in X_train]
joblib.dump(scaler, "scaler.pkl")'''

'''# PCA 
X_all_scaled = np.vstack(X_train_scaled)
pca, nz = make_pca(X_all_scaled, target_dim=75)
joblib.dump(pca, "pca.pkl")
joblib.dump(nz, "pca_nonzero_columns.pkl")

X_train_reduced = [pca.transform(f[:, nz]) for f in X_train_scaled]'''

# --- Normalize ---
all_feats = np.vstack(X_train)
scaler = StandardScaler().fit(all_feats)
X_train_scaled = [scaler.transform(f) for f in X_train]
joblib.dump(scaler, "scaler.pkl")

# --- LDA (with expanded labels) ---
X_all_scaled = []
y_all_expanded = []

for features, label in zip(X_train_scaled, y_train):
    X_all_scaled.append(features)
    y_all_expanded.extend([label] * features.shape[0])

X_all_scaled = np.vstack(X_all_scaled)
y_all_expanded = np.array(y_all_expanded)

print(f"Training LDA on {X_all_scaled.shape[0]} frames with {len(np.unique(y_all_expanded))} classes")

# Remove constant features
nonzero = X_all_scaled.std(axis=0) > 1e-6
X_filtered = X_all_scaled[:, nonzero]

# Fit LDA
n_classes = len(np.unique(y_all_expanded))
#lda_dims = min(15, n_classes - 1, X_filtered.shape[1])
lda_dims = min(232, n_classes - 1, X_filtered.shape[1])
lda = LinearDiscriminantAnalysis(n_components=lda_dims)
lda.fit(X_filtered, y_all_expanded)

print(f"LDA: {X_filtered.shape[1]} features → {lda_dims} components")

joblib.dump(lda, "lda.pkl")
joblib.dump(nonzero, "lda_nonzero_columns.pkl")

# Transform training data
X_train_reduced = [lda.transform(f[:, nonzero]) for f in X_train_scaled]


# Filter out empty feature arrays and align labels
X_train_reduced_filtered = []
y_train_filtered = []
for feats, label in zip(X_train_reduced, y_train):
    if feats.shape[0] > 0:
        X_train_reduced_filtered.append(feats)
        y_train_filtered.append(label)


print(f"Training samples after filtering: {len(X_train_reduced_filtered)}")

# Check word distribution
from collections import Counter
word_counts = Counter(y_train_filtered)
print("\nWord distribution in training set:")
for word, count in sorted(word_counts.items()):
    print(f"  {word}: {count} samples")


# Train GMM-HMMs

word_list = sorted(set(y_train_filtered))
print(f"\nTraining GMM-HMMs for {len(word_list)} words...")
gmms = train_gmm_hmms(
    word_list, 
    X_train_reduced_filtered, 
    y_train_filtered,
    n_states=3,      # REDUCED from 4 to 3
    n_mixtures=2,    # REDUCED from 3 to 2
    min_frames=20,   # REDUCED threshold
    min_samples=1    # Require at least 3 samples per word
)

print(f"\nSuccessfully trained {len(gmms)} models")

# --- Validation ---
y_pred, y_true = [], []
target_dim = X_train_reduced_filtered[0].shape[1]

print("\nValidating...")
for e, j in zip(val_emg, val_json):
    feats_list, labels_list = extract_features_for_segments(
        e, j, fs, new_fs, frame_size, hop_size)

    for f_raw, true_lab in zip(feats_list, labels_list):
        # Scale and transform
        f = scaler.transform(f_raw)
        #f = f[:, nz]
        f = f[:, 0:target_dim]  # Ensure correct dim after LDA
        if f.size == 0 or f.shape[0] < 5:
            continue
            
        #f = pca.transform(f)
        f = lda.transform(f)
        if not np.isfinite(f).all() or f.shape[1] != target_dim:
            continue

        # Predict
        pred, score, _ = predict_word_gmmhmm(gmms, f)
        
        if pred is None:
            continue
            
        y_pred.append(pred)
        y_true.append(true_lab)

# Calculate accuracy
if len(y_pred) > 0:
    acc = np.mean(np.array(y_pred) == np.array(y_true))
    print(f"\nValidation accuracy: {acc * 100:.2f}%")
    print(f"Predictions made: {len(y_pred)} / {len([item for sublist in [extract_features_for_segments(e, j, fs, new_fs, frame_size, hop_size)[1] for e, j in zip(val_emg, val_json)] for item in sublist])}")
    
    # Confusion matrix
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0)) 
else:
    print("No predictions made - check your data!")
