import math
from typing import Sequence, Tuple

import mne
import numpy as np


def class_dependent_sliding_window_augmentation(
    raw_recordings: Sequence[mne.io.BaseRaw],
    recording_labels: Sequence[int],
    epoch_length: float,
    samp_freq: float,
    overlap_non_stress: float = 0.20,
    overlap_stress: float = 0.40,
    output_order: str = "channels_first",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate class-dependent overlapping EEG epochs.

    This function must be applied only to training recordings or
    training-only signal portions. Validation and test data should
    remain original and non-overlapping.
    """

    if len(raw_recordings) != len(recording_labels):
        raise ValueError(
            "raw_recordings and recording_labels must have equal lengths."
        )

    if output_order not in {"channels_first", "time_first"}:
        raise ValueError(
            "output_order must be 'channels_first' or 'time_first'."
        )

    for overlap in (overlap_non_stress, overlap_stress):
        if not 0 <= overlap < 1:
            raise ValueError(
                "Each overlap ratio must satisfy 0 <= overlap < 1."
            )

    window_samples = int(epoch_length * samp_freq)

    if window_samples < 1:
        raise ValueError(
            "epoch_length and samp_freq produce an invalid window size."
        )

    all_epochs = []
    all_labels = []

    for raw, label in zip(raw_recordings, recording_labels):
        label = int(label)

        if label not in (0, 1):
            raise ValueError(
                "Labels must be 0 for non-stress or 1 for stress."
            )

        overlap = (
            overlap_stress
            if label == 1
            else overlap_non_stress
        )

        step = max(
            1,
            math.floor(
                window_samples * (1.0 - overlap)
            ),
        )

        n_samples = raw.n_times

        if n_samples < window_samples:
            continue

        n_epochs = (
            (n_samples - window_samples) // step
        ) + 1

        events = np.array(
            [
                [start_sample, 0, 1]
                for start_sample in range(
                    0,
                    n_samples - window_samples + 1,
                    step,
                )
            ],
            dtype=np.int64,
        )

        # Safety check
        if len(events) != n_epochs:
            raise RuntimeError(
                "Unexpected difference in calculated epoch count."
            )

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id={"epoch": 1},
            tmin=0.0,
            tmax=epoch_length - (1.0 / samp_freq),
            baseline=None,
            detrend=1,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )

        epoch_data = epochs.get_data().astype(
            np.float32
        )

        # MNE output:
        # epochs x channels x samples
        if output_order == "time_first":
            epoch_data = np.transpose(
                epoch_data,
                (0, 2, 1),
            )

        epoch_labels = np.full(
            epoch_data.shape[0],
            label,
            dtype=np.int32,
        )

        all_epochs.append(epoch_data)
        all_labels.append(epoch_labels)

    if not all_epochs:
        raise ValueError(
            "No complete epochs could be generated "
            "from the provided recordings."
        )

    augmented_data = np.concatenate(
        all_epochs,
        axis=0,
    )

    augmented_labels = np.concatenate(
        all_labels,
        axis=0,
    )

    return augmented_data, augmented_labels
