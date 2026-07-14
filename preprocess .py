"""
preprocess.py  --  STEW dataset preprocessing.

Steps (exactly as in the original notebook):
  1. Load all "high workload" (stress) and "low workload" (relax) .txt recordings.
  2. Keep the 14 EEG channels, build MNE RawArray.
  3. FIR band-pass filter 0.5-45 Hz.
  4. Cut into 30-second epochs (detrended).
  5. Label stress=1, relax=0 and save as .npy arrays for train.py.

Usage:
  python preprocess.py \
      --stress_dir data/stew/hi_txt_files \
      --relax_dir  data/stew/lo_txt_files \
      --out_dir    data
"""

import os
import argparse
import numpy as np
import mne

# ---- Acquisition / preprocessing constants (from the paper) ----
SFREQ = 128            # sampling frequency (Hz)
L_FREQ = 0.5           # band-pass low cut (Hz)
H_FREQ = 45            # band-pass high cut (Hz)
EPOCH_LENGTH = 30      # epoch length (seconds)

ALL_CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7',
                     'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SELECTED_CHANNEL_INDICES = list(range(14))   # use all 14 channels


def load_txt_data(file):
    return np.loadtxt(file)


def load_folder(folder):
    """Load every .txt recording in a folder and stack into (channels, time)."""
    files = sorted(
        os.path.join(d, f)
        for d, _, names in os.walk(folder)
        for f in names if f.endswith('.txt')
    )
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder}")
    return np.vstack([load_txt_data(f) for f in files]).T


def to_epochs(data, channel_names, event_label):
    """Filter the raw signal and cut it into fixed-length epochs."""
    info = mne.create_info(ch_names=channel_names, sfreq=SFREQ,
                           ch_types=['eeg'] * len(channel_names))
    raw = mne.io.RawArray(data, info)
    raw.filter(L_FREQ, H_FREQ, fir_design='firwin', verbose=False)

    tmin, tmax = 0, EPOCH_LENGTH - 1 / SFREQ
    n_epochs = raw.n_times // (EPOCH_LENGTH * SFREQ)
    events = np.array([[i * EPOCH_LENGTH * SFREQ, 0, event_label]
                       for i in range(n_epochs)])
    epochs = mne.Epochs(raw, events, event_id={'x': event_label},
                        tmin=tmin, tmax=tmax, baseline=None,
                        detrend=1, preload=True, verbose=False)
    return epochs.get_data()


def main(args):
    channel_names = [ALL_CHANNEL_NAMES[i] for i in SELECTED_CHANNEL_INDICES]

    stress_data = load_folder(args.stress_dir)[SELECTED_CHANNEL_INDICES, :]
    relax_data = load_folder(args.relax_dir)[SELECTED_CHANNEL_INDICES, :]

    stress_epochs = to_epochs(stress_data, channel_names, event_label=1)
    relax_epochs = to_epochs(relax_data, channel_names, event_label=0)

    if stress_epochs.shape[1:] != relax_epochs.shape[1:]:
        raise ValueError("Shape mismatch between stress and relax epochs.")

    # (epochs, channels, time) -> (epochs, time, channels) for Conv1D
    all_epochs = np.moveaxis(np.vstack((stress_epochs, relax_epochs)), 1, 2).astype(np.float32)
    labels = np.hstack((np.ones(len(stress_epochs), dtype=int),
                        np.zeros(len(relax_epochs), dtype=int)))

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "all_epochs_data.npy"), all_epochs)
    np.save(os.path.join(args.out_dir, "all_labels.npy"), labels)

    print("Saved:", os.path.join(args.out_dir, "all_epochs_data.npy"))
    print("Epochs data shape:", all_epochs.shape)   # e.g. (480, 3840, 14)
    print("Labels shape:", labels.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="STEW EEG preprocessing")
    p.add_argument("--stress_dir", required=True, help="folder with high-workload (stress) .txt files")
    p.add_argument("--relax_dir", required=True, help="folder with low-workload (relax) .txt files")
    p.add_argument("--out_dir", default="data", help="where to save the .npy outputs")
    main(p.parse_args())
