import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import scipy
import os
import json
import sys
import joblib
import hmmlearn
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from scipy.fftpack import dct

from google.colab import drive
drive.mount('/content/drive')

# Optional: functions used in some frequency features (if needed)
from scipy.stats import skew, kurtosis
from scipy.stats import entropy as spec_entropy_func

voiced_data_path = Path('/content/drive/My Drive/Pattern Rec/Pattern Rec Project/closed_vocab/voiced/5-19')
silent_data_path = Path('/content/drive/My Drive/Pattern Rec/Pattern Rec Project/closed_vocab/silent/5-19_silent')

# ---------------------------
# load emg dataset and labels (MODIFIED - no segmentation)
# ---------------------------
def load_emg_dataset(data_path):
    emg_files = []
    json_files = []
    labels = []

    for file in os.listdir(data_path):
        if file.endswith("_emg.npy"):
            emg_path = data_path / file
            json_path = data_path / file.replace("_emg.npy", "_info.json")
            if not json_path.exists():
                continue

            with open(json_path, "r") as f:
                meta = json.load(f)
            label = meta["text"]

            emg_files.append(str(emg_path))
            json_files.append(str(json_path))
            labels.append(label)

    return emg_files, json_files, labels


# ---------------------------
# emg preprocessing
# ---------------------------
def preprocess_emg(emg, fs, new_fs=None, lowcut=10, highcut=400, notch_freq=60, harmonics=8,
                   apply_tanh=True, tanh_scale=50, tanh_pre_scale=1/20):
    # Bandstop to remove AC electrical noise
    def notch(sig, freq, sample_frequency):
        b, a = signal.iirnotch(freq, 30, sample_frequency)
        return signal.filtfilt(b, a, sig)

    # apply notch filters (and harmonics)
    def notch_harmonics(sig, freq, sample_frequency, n_harmonics):
        for harmonic in range(1, n_harmonics + 1):
            f = freq * harmonic
            if f < sample_frequency / 2:  # only apply if below Nyquist
                sig = notch(sig, f, sample_frequency)
        return sig

    # remove low-frequency drift (high-pass) and DC offset
    def remove_drift(sig, sample_frequency):
        b, a = signal.butter(3, 2, "highpass", fs=sample_frequency)
        return signal.filtfilt(b, a, sig)

    # resample to new frequency
    def resample_signal(sig, new_freq, old_freq):
        times = np.arange(len(sig)) / old_freq
        new_times = np.arange(0, times[-1], 1 / new_freq)
        return np.interp(new_times, times, sig)

    # Ensure 2D array
    if emg.ndim == 1:
        emg = emg[:, np.newaxis]

    processed = []
    nyq = 0.5 * fs
    b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype="band")

    for ch in range(emg.shape[1]):
        sig = emg[:, ch]

        # Remove DC offset
        sig = sig - np.mean(sig)

        # Notch filters + harmonics
        sig = notch_harmonics(sig, notch_freq, fs, harmonics)

        # High-pass to remove drift
        sig = remove_drift(sig, fs)

        # Bandpass filter
        nyq = 0.5 * fs
        b, a = signal.butter(4, [lowcut / nyq, highcut / nyq], btype="band")
        sig = signal.filtfilt(b, a, sig)

        # de-spiking (amplitude compression to limit outliers)
        if apply_tanh:
            sig = sig * tanh_pre_scale  # normalize before tanh
            sig = tanh_scale * np.tanh(sig / tanh_scale)

        # Optional resampling
        if new_fs is not None and new_fs != fs:
            sig = resample_signal(sig, new_fs, fs)

        processed.append(sig)
    processed = np.stack(processed, axis=1)
    return processed


# ---------------------------
# framing function
# ---------------------------
def frame_signal(signal_arr, frame_size, hop_size):
    frames = []
    for start in range(0, len(signal_arr) - frame_size, hop_size):
        frames.append(signal_arr[start : start + frame_size])
    return np.array(frames)


# ---------------------------
# time-domain features
# ---------------------------
def extract_time_features(frames, fs):
    feats = []
    # compute the 5 features as per Jo et al., 2006
    for frame in frames:
        # Low-pass and high-pass versions (per-channel)
        b_low, a_low = signal.butter(2, 100 / (0.5 * fs), "low")
        b_high, a_high = signal.butter(2, 100 / (0.5 * fs), "high")

        low = signal.filtfilt(b_low, a_low, frame, axis=0)
        high = signal.filtfilt(b_high, a_high, frame, axis=0)

        # Features
        mean_low = np.mean(low, axis=0)
        rectified_mean = np.mean(np.abs(high), axis=0)
        power_low = np.mean(low ** 2, axis=0)
        power_high = np.mean(high ** 2, axis=0)

        zero_cross = np.mean(
            [
                ((np.roll(high[:, ch], 1) * high[:, ch]) < 0).sum()
                for ch in range(high.shape[1])
            ]
        )

        # Spectral features (optional)
        f, Pxx = signal.welch(frame, fs, nperseg=128, axis=0)
        spectral_mean = np.mean(Pxx, axis=0)

        frame_feats = np.hstack(
            [mean_low, rectified_mean, power_low, power_high, spectral_mean, zero_cross]
        )
        feats.append(frame_feats)

    feats = np.array(feats)
    return feats


# ---------------------------
# MFCC-like features on EMG frames
# ---------------------------
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

    # Mel filterbank helpers
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

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
            if center - left > 0:
                fbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right - center > 0:
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
            if mfcc.shape[0] >= 2:
                delta = np.vstack([mfcc[1:] - mfcc[:-1], mfcc[-1:] - mfcc[-2]])
            else:
                delta = np.zeros_like(mfcc)
            if delta.shape[0] >= 2:
                delta_delta = np.vstack([delta[1:] - delta[:-1], delta[-1:] - delta[-2]])
            else:
                delta_delta = np.zeros_like(delta)
            feats = np.hstack([mfcc, delta, delta_delta])
        else:
            feats = mfcc

        feats_all_channels.append(feats)

    # Concatenate channel MFCCs
    feats_all_channels = np.hstack(feats_all_channels)
    return feats_all_channels


# ---------------------------
# extracting features for ENTIRE file (no segmentation)
# ---------------------------
def extract_features_for_file(emg_file, json_file, fs, new_fs, frame_size, hop_size):
    """
    Extract features for the entire EMG file without word-level segmentation.
    Returns a single feature sequence and the label.
    """
    emg = np.load(emg_file)
    
    with open(json_file, "r") as f:
        meta = json.load(f)
    
    label = meta["text"]
    
    # Preprocess entire signal
    filtered = preprocess_emg(emg, fs, new_fs)
    frames = frame_signal(filtered, frame_size, hop_size)
    
    if frames.size == 0:
        return None, None
    
    time_feats = extract_time_features(frames, fs)
    mfcc_feats = extract_mfcc_features(frames, new_fs, n_mfcc=13, include_deltas=True)
    
    # Ensure both have same number of frames
    min_frames = min(time_feats.shape[0], mfcc_feats.shape[0])
    time_feats = time_feats[:min_frames]
    mfcc_feats = mfcc_feats[:min_frames]
    
    # Concatenate features
    combined_feats = np.hstack([time_feats, mfcc_feats])
    combined_feats = np.nan_to_num(combined_feats)
    
    return combined_feats, label


# ---------------------------
# GMM-HMM training (IMPROVED)
# ---------------------------
def train_gmm_hmms(word_list, X_train_reduced, y_train, n_states=4, n_mixtures=2, min_frames=30, min_samples=3):
    gmms = {}

    for word in word_list:
        word_sequences = [f for f, lab in zip(X_train_reduced, y_train) if lab == word]

        if len(word_sequences) < min_samples:
            print(f"⚠ Skipping {word}: only {len(word_sequences)} samples (need {min_samples})")
            continue

        word_frames = np.vstack(word_sequences)
        word_frames = word_frames[np.isfinite(word_frames).all(axis=1)]

        if word_frames.shape[0] < min_frames:
            print(f"⚠ Skipping {word}: only {word_frames.shape[0]} frames (need {min_frames})")
            continue

        # Add small random noise to prevent identical frames (degenerate covariance)
        word_frames = word_frames + np.random.randn(*word_frames.shape) * 1e-6
        
        # Ensure sufficient variance in data
        frame_std = np.std(word_frames, axis=0)
        if np.any(frame_std < 1e-4):
            print(f"⚠ Skipping {word}: insufficient variance in features")
            continue

        print(f"Training GMM-HMM for '{word}' on {len(word_sequences)} samples, {word_frames.shape[0]} frames...")

        # Adjust n_states and n_mixtures based on data availability
        avg_seq_length = word_frames.shape[0] / len(word_sequences)
        adaptive_n_states = min(n_states, max(2, int(avg_seq_length / 10)))
        adaptive_n_mixtures = min(n_mixtures, max(1, len(word_sequences) // 2))
        
        model = hmm.GMMHMM(
            n_components=adaptive_n_states,
            n_mix=adaptive_n_mixtures,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
            min_covar=1e-3,  # Increased from 1e-2 to be more permissive
            init_params="stmc",  # Initialize all parameters
            params="stmc",
        )

        # Better initialization for transition matrix (left-to-right bias with self-transitions)
        model.startprob_ = np.zeros(adaptive_n_states)
        model.startprob_[0] = 0.9
        model.startprob_[1:] = 0.1 / (adaptive_n_states - 1) if adaptive_n_states > 1 else 0
        
        # Left-to-right with strong self-transition and skip possibilities
        model.transmat_ = np.zeros((adaptive_n_states, adaptive_n_states))
        for i in range(adaptive_n_states):
            # Self-transition (stay in state)
            model.transmat_[i, i] = 0.7
            # Forward transition
            if i < adaptive_n_states - 1:
                model.transmat_[i, i + 1] = 0.25
                # Skip transition (can skip one state)
                if i < adaptive_n_states - 2:
                    model.transmat_[i, i + 2] = 0.05
            # Distribute remaining probability to stay in last state or spread
            if i == adaptive_n_states - 1:
                model.transmat_[i, i] = 0.95
            # Normalize
            row_sum = model.transmat_[i, :].sum()
            if row_sum > 0:
                model.transmat_[i, :] /= row_sum
            else:
                # Fallback: uniform distribution
                model.transmat_[i, :] = 1.0 / adaptive_n_states

        from sklearn.cluster import KMeans

        try:
            # Initialize GMM means with KMeans
            n_clusters = adaptive_n_states * adaptive_n_mixtures
            if n_clusters < word_frames.shape[0]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(word_frames)
            else:
                print(f"⚠ Adjusting clusters: {n_clusters} → {max(2, word_frames.shape[0] // 2)}")
                adaptive_n_mixtures = max(1, word_frames.shape[0] // (2 * adaptive_n_states))
                n_clusters = adaptive_n_states * adaptive_n_mixtures
                
                # Re-create model with adjusted parameters
                model = hmm.GMMHMM(
                    n_components=adaptive_n_states,
                    n_mix=adaptive_n_mixtures,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                    min_covar=1e-3,
                    init_params="stmc",
                    params="stmc",
                )
                # Re-initialize matrices
                model.startprob_ = np.zeros(adaptive_n_states)
                model.startprob_[0] = 0.9
                model.startprob_[1:] = 0.1 / (adaptive_n_states - 1) if adaptive_n_states > 1 else 0
                
                model.transmat_ = np.zeros((adaptive_n_states, adaptive_n_states))
                for i in range(adaptive_n_states):
                    model.transmat_[i, i] = 0.7
                    if i < adaptive_n_states - 1:
                        model.transmat_[i, i + 1] = 0.25
                        if i < adaptive_n_states - 2:
                            model.transmat_[i, i + 2] = 0.05
                    if i == adaptive_n_states - 1:
                        model.transmat_[i, i] = 0.95
                    row_sum = model.transmat_[i, :].sum()
                    if row_sum > 0:
                        model.transmat_[i, :] /= row_sum
                    else:
                        model.transmat_[i, :] = 1.0 / adaptive_n_states

            lengths = [seq.shape[0] for seq in word_sequences]
            
            # Fit model with error handling
            model.fit(word_frames, lengths=lengths)

            # Validate model after training
            if np.isnan(model.startprob_).any() or np.isnan(model.transmat_).any():
                print(f"⚠ Skipping {word}: NaN values after training")
                continue
            
            # Check for zero-sum rows in transmat
            row_sums = model.transmat_.sum(axis=1)
            if np.any(row_sums < 1e-6):
                print(f"⚠ Warning for {word}: fixing zero-sum transition rows")
                for i in range(len(row_sums)):
                    if row_sums[i] < 1e-6:
                        model.transmat_[i, :] = 1.0 / adaptive_n_states

            gmms[word] = model
            print(f"✔ Successfully trained '{word}' (states={adaptive_n_states}, mixtures={adaptive_n_mixtures})")

        except Exception as e:
            print(f"⚠ Failed to train '{word}': {str(e)}")
            continue

    return gmms


# ---------------------------
# prediction helper
# ---------------------------
def predict_word_gmmhmm(gmms, feat):
    scores = {}
    for word, model in gmms.items():
        try:
            scores[word] = model.score(feat)
        except Exception:
            scores[word] = -np.inf

    if not scores or all(s == -np.inf for s in scores.values()):
        return None, -np.inf, scores

    best_word = max(scores, key=scores.get)
    return best_word, scores[best_word], scores


# ---------------------------
# Confusion matrix plot utility
# ---------------------------
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", fname=None):
    if normalize:
        with np.errstate(all="ignore"):
            cmn = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cmn = cm

    plt.figure(figsize=(12, 10))
    plt.imshow(cmn, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cmn.max() / 2.0
    for i, j in np.ndindex(cmn.shape):
        val = cmn[i, j]
        if np.isfinite(val):
            plt.text(j, i, format(val, fmt), ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=6)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.show()


# ---------------------------
# --- Parameters ---
# ---------------------------
fs = 1000  # Sampling frequency in Hz
new_fs = 516.76  # New sampling frequency in Hz
frame_size = 16
hop_size = 6

# ---------------------------
# --- Load voiced (train) and silent (test) files ---
# ---------------------------
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

voiced_emg, voiced_json, voiced_labels = load_emg_dataset(voiced_data_path)
silent_emg, silent_json, silent_labels = load_emg_dataset(silent_data_path)

print(f"Voiced files (training): {len(voiced_emg)}")
print(f"Silent files (testing): {len(silent_emg)}")

# Split silent data into validation and test
val_emg, test_emg, val_json, test_json = train_test_split(
    silent_emg, silent_json, test_size=0.5, random_state=42
)
print(f"Validation: {len(val_emg)}, Test: {len(test_emg)}")

# ---------------------------
# --- Feature extraction - TRAIN (voiced) ---
# ---------------------------
print("\n" + "=" * 60)
print("EXTRACTING TRAINING FEATURES (VOICED)")
print("=" * 60)

X_train, y_train = [], []

for emg_path, json_path in zip(voiced_emg, voiced_json):
    feats, label = extract_features_for_file(emg_path, json_path, fs, new_fs, frame_size, hop_size)
    if feats is not None:
        X_train.append(feats)
        y_train.append(label)

print(f"Training sequences: {len(X_train)}")

# Flatten all frames for scaler fit
all_feats = np.vstack([f for f in X_train if f.size > 0])
scaler = StandardScaler().fit(all_feats)
joblib.dump(scaler, "scaler.pkl")
print(f"Scaler fitted on {all_feats.shape[0]} frames, {all_feats.shape[1]} features")

# Apply scaler per-sequence
X_train_scaled = [scaler.transform(f) if f.size > 0 else f for f in X_train]

# Expand labels per-frame for LDA training
X_all_scaled = []
y_all_expanded = []

for features, label in zip(X_train_scaled, y_train):
    if features.size == 0:
        continue
    X_all_scaled.append(features)
    y_all_expanded.extend([label] * features.shape[0])

X_all_scaled = np.vstack(X_all_scaled)
y_all_expanded = np.array(y_all_expanded)

print(f"Training LDA on {X_all_scaled.shape[0]} frames with {len(np.unique(y_all_expanded))} classes")

# Remove constant features and fit LDA
nonzero = X_all_scaled.std(axis=0) > 1e-6
X_filtered = X_all_scaled[:, nonzero]

n_classes = len(np.unique(y_all_expanded))
lda_dims = min(232, n_classes - 1, X_filtered.shape[1])
lda = LinearDiscriminantAnalysis(n_components=lda_dims)
lda.fit(X_filtered, y_all_expanded)

print(f"LDA: {X_filtered.shape[1]} features → {lda_dims} components")

joblib.dump(lda, "lda.pkl")
joblib.dump(nonzero, "lda_nonzero_columns.pkl")

# Transform training data
X_train_reduced = [lda.transform(f[:, nonzero]) if f.size > 0 else f for f in X_train_scaled]

# Filter out empty sequences
X_train_reduced_filtered = []
y_train_filtered = []
for feats, label in zip(X_train_reduced, y_train):
    if isinstance(feats, np.ndarray) and feats.shape[0] > 0:
        X_train_reduced_filtered.append(feats)
        y_train_filtered.append(label)

print(f"Training samples after filtering: {len(X_train_reduced_filtered)}")

# Word distribution
from collections import Counter
word_counts = Counter(y_train_filtered)
print("\nWord distribution in training set:")
for word, count in sorted(word_counts.items()):
    print(f"  {word}: {count} samples")

# ---------------------------
# --- Train GMM-HMMs ---
# ---------------------------
print("\n" + "=" * 60)
print("TRAINING GMM-HMMs")
print("=" * 60)

word_list = sorted(set(y_train_filtered))
print(f"Training GMM-HMMs for {len(word_list)} words...")
gmms = train_gmm_hmms(
    word_list,
    X_train_reduced_filtered,
    y_train_filtered,
    n_states=3,
    n_mixtures=2,
    min_frames=20,
    min_samples=1,
)

print(f"\nSuccessfully trained {len(gmms)} models")

# ---------------------------
# --- Validation on SILENT speech ---
# ---------------------------
print("\n" + "=" * 60)
print("VALIDATION ON SILENT SPEECH")
print("=" * 60)

y_pred, y_true = [], []
target_dim = X_train_reduced_filtered[0].shape[1] if len(X_train_reduced_filtered) > 0 else None

for e, j in zip(val_emg, val_json):
    f_raw, true_lab = extract_features_for_file(e, j, fs, new_fs, frame_size, hop_size)
    
    if f_raw is None or f_raw.size == 0:
        continue

    # Scale using the saved scaler
    f = scaler.transform(f_raw)

    # Apply the SAME nonzero mask used for LDA training
    f = f[:, nonzero]

    # Apply LDA transform
    try:
        f_reduced = lda.transform(f)
    except Exception:
        continue

    # Check dims and minimal length
    if not np.isfinite(f_reduced).all() or f_reduced.shape[0] < 5 or f_reduced.shape[1] != target_dim:
        continue

    pred, score, _ = predict_word_gmmhmm(gmms, f_reduced)
    if pred is None:
        continue

    y_pred.append(pred)
    y_true.append(true_lab)

# ---------------------------
# --- Validation Results ---
# ---------------------------
if len(y_pred) > 0:
    acc = np.mean(np.array(y_pred) == np.array(y_true))
    print(f"\nValidation accuracy: {acc * 100:.2f}%")
    print(f"Predictions made: {len(y_pred)} / {len(val_emg)}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    labels_sorted = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("\nConfusion matrix shape:", cm.shape)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, classes=labels_sorted, normalize=True, 
                         title="Validation Confusion Matrix (Silent Speech)", 
                         fname="confusion_matrix_validation.png")
    print("Saved confusion matrix to confusion_matrix_validation.png")

else:
    print("No predictions made - check your data!")

# ---------------------------
# --- Test on SILENT speech ---
# ---------------------------
print("\n" + "=" * 60)
print("TESTING ON SILENT SPEECH")
print("=" * 60)

y_pred_test, y_true_test = [], []

for e, j in zip(test_emg, test_json):
    f_raw, true_lab = extract_features_for_file(e, j, fs, new_fs, frame_size, hop_size)
    
    if f_raw is None or f_raw.size == 0:
        continue

    f = scaler.transform(f_raw)
    f = f[:, nonzero]

    try:
        f_reduced = lda.transform(f)
    except Exception:
        continue

    if not np.isfinite(f_reduced).all() or f_reduced.shape[0] < 5 or f_reduced.shape[1] != target_dim:
        continue

    pred, score, _ = predict_word_gmmhmm(gmms, f_reduced)
    if pred is None:
        continue

    y_pred_test.append(pred)
    y_true_test.append(true_lab)

# ---------------------------
# --- Test Results ---
# ---------------------------
if len(y_pred_test) > 0:
    acc = np.mean(np.array(y_pred_test) == np.array(y_true_test))
    print(f"\nTest accuracy: {acc * 100:.2f}%")
    print(f"Predictions made: {len(y_pred_test)} / {len(test_emg)}")

    print("\nClassification Report:")
    print(classification_report(y_true_test, y_pred_test, zero_division=0))

    # Confusion matrix
    labels_sorted = sorted(list(set(y_true_test) | set(y_pred_test)))
    cm = confusion_matrix(y_true_test, y_pred_test, labels=labels_sorted)
    print("\nConfusion matrix shape:", cm.shape)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, classes=labels_sorted, normalize=True, 
                         title="Test Confusion Matrix (Silent Speech)", 
                         fname="confusion_matrix_test.png")
    print("Saved confusion matrix to confusion_matrix_test.png")

else:
    print("No predictions made - check your data!")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
