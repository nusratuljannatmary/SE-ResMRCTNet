import numpy as np
import mne
from sklearn.preprocessing import StandardScaler


def preprocessing_func(data, channel_names, fs, epoch_length):
    # Create MNE information
    info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types=["eeg"] * len(channel_names))

    # Convert NumPy signal into MNE Raw format
    raw = mne.io.RawArray(data, info, verbose=False)

    # Apply 0.5–45 Hz zero-phase FIR band-pass filter
    raw.filter(l_freq=0.5, h_freq=45.0, fir_design="firwin", phase="zero", verbose=False)

    # Calculate the number of complete non-overlapping epochs
    number_of_epochs = int(raw.n_times // (epoch_length * fs))

    # Define the starting sample of each epoch
    events = np.array([[int(i * epoch_length * fs), 0, 1] for i in range(number_of_epochs)])

    # Perform fixed-length epoching with linear detrending
    epochs = mne.Epochs(raw, events, event_id={"epoch": 1}, tmin=0, tmax=epoch_length - (1 / fs), baseline=None, detrend=1, preload=True, verbose=False)

    # Convert from epochs × channels × samples to epochs × samples × channels
    epochs_data = np.moveaxis(epochs.get_data(), 1, 2)

    return epochs_data.astype(np.float32)


def fit_zscore(x_train):
    """Fit z-score normalization using training data only."""
    scaler = StandardScaler()
    number_of_channels = x_train.shape[-1]
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, number_of_channels))
    x_train_scaled = x_train_scaled.reshape(x_train.shape)

    return x_train_scaled.astype(np.float32), scaler


def apply_zscore(data, scaler):
    number_of_channels = data.shape[-1]
    data_scaled = scaler.transform(data.reshape(-1, number_of_channels))
    data_scaled = data_scaled.reshape(data.shape)

    return data_scaled.astype(np.float32)
