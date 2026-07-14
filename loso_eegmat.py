"""
loso_eegmat.py  --  EEGMAT subject-independent (Leave-One-Subject-Out) experiment.

Same fair-LOSO design as loso_stew.py (per-subject z-score, subject-disjoint
splits, per-fold scaler, class weighting, EarlyStopping, multi-seed ensemble,
window- and recording-level metrics), adapted to EEGMAT:
  - reads .edf files, picks 21 channels, 500 Hz, 8 s epochs,
  - label from filename suffix: '*_1' = relax (0), '*_2' = stress (1).

The model architecture is imported from model.py (unchanged).

Usage:
  python loso_eegmat.py --input_dir data/eegmat --out_dir outputs_loso_eegmat
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import mne
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, cohen_kappa_score,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import EarlyStopping

from model import build_transformer_model

# ======================= Configuration (from the notebook) =======================
SEED = 42
sfreq = 500
epoch_length = 8             # seconds -> seq_len = 8 * 500 = 4000
overlap_relax = 0.60
overlap_stress = 0.87
APPLY_SUBJECT_LEVEL_NORM = True

EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 25
NUM_SEEDS = 2
SEED_LIST = [SEED + s for s in [0, 7, 13, 23, 37]][:NUM_SEEDS]
VAL_SUBJECT_RATIO = 0.20

CHANNELS_TO_PICK = [
    'EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',
    'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6',
    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
    'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG A2-A1', 'ECG ECG',
]


# ============================ Preprocessing helpers ============================
def load_edf_as_raw(path, channels_to_pick=CHANNELS_TO_PICK,
                    l_freq=0.5, h_freq=45.0, apply_subject_norm=True):
    """Read EDF -> average reference -> pick channels -> filter -> per-subject z-score."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    raw.pick_channels(channels_to_pick, ordered=True)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)
    if apply_subject_norm:
        data = raw.get_data()
        mu = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-6
        raw._data = ((data - mu) / std).astype(np.float32)
    return raw


def epoch_raw_with_overlap(raw, label, epoch_length, sfreq, overlap):
    win = int(epoch_length * sfreq)
    step = max(1, int(win * (1.0 - overlap)))
    n_epochs = (raw.n_times - win) // step + 1
    if n_epochs <= 0:
        return None, None
    events = np.array([[i * step, 0, 1] for i in range(n_epochs)], dtype=int)
    epochs = mne.Epochs(raw, events, dict(epoch=1), tmin=0,
                        tmax=epoch_length - 1 / sfreq, baseline=None,
                        detrend=1, preload=True, verbose=False)
    X = epochs.get_data()
    return X, np.full((X.shape[0],), label, dtype=int)


def extract_subject_id(filename):
    """'Subject00_1.edf' -> 'subject00'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.endswith('_1') or base.endswith('_2'):
        base = base[:-2]
    return base.lower()


def get_label_from_filename(filename):
    """'*_1' -> 0 (relax/baseline), '*_2' -> 1 (stress/arithmetic)."""
    name = os.path.splitext(os.path.basename(filename))[0]
    if name.endswith('_1'):
        return 0
    if name.endswith('_2'):
        return 1
    raise ValueError(f"Cannot infer label from filename: {filename}")


def find_edf_files(root_dir):
    paths = [os.path.join(d, f) for d, _, files in os.walk(root_dir)
             for f in files if f.lower().endswith('.edf')]
    paths.sort()
    return paths


# =============================== Metric helpers ===============================
def compute_binary_metrics(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError:
        auc = np.nan
    return dict(
        acc=accuracy_score(y_true, y_pred),
        prec=precision_score(y_true, y_pred, zero_division=0),
        sens=recall_score(y_true, y_pred, zero_division=0),
        spec=tn / (tn + fp + 1e-9),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=auc, kappa=cohen_kappa_score(y_true, y_pred),
        cm=cm, y_pred=y_pred, y_prob=y_prob,
    )


def aggregate_to_recording_level(y_test_windows, y_prob_windows):
    rec_true, rec_pred, rec_prob = [], [], []
    for label_val in [0, 1]:
        mask = (y_test_windows == label_val)
        if mask.sum() == 0:
            continue
        mean_prob = y_prob_windows[mask].mean(axis=0)
        rec_true.append(label_val)
        rec_pred.append(int(np.argmax(mean_prob)))
        rec_prob.append(float(mean_prob[1]))
    return np.array(rec_true), np.array(rec_pred), np.array(rec_prob)


# ==================================== Main ====================================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    mne.set_log_level("WARNING")

    all_edf_files = find_edf_files(args.input_dir)
    print(f"Found {len(all_edf_files)} EDF files under {args.input_dir}")

    X_list, y_list, subj_list, skipped = [], [], [], []
    for fp in all_edf_files:
        try:
            sid = extract_subject_id(fp)
            label = get_label_from_filename(fp)
        except ValueError as e:
            skipped.append((fp, str(e))); continue
        overlap = overlap_stress if label == 1 else overlap_relax
        try:
            raw = load_edf_as_raw(fp, apply_subject_norm=APPLY_SUBJECT_LEVEL_NORM)
        except Exception as e:
            skipped.append((fp, f"load failed: {e}")); continue
        X, y = epoch_raw_with_overlap(raw, label, epoch_length, sfreq, overlap)
        if X is None:
            skipped.append((fp, "no epochs (too short)")); continue
        X_list.append(X); y_list.append(y)
        subj_list.append(np.full(X.shape[0], sid, dtype=object))

    if skipped:
        print(f"[!] Skipped {len(skipped)} files (showing up to 5): "
              f"{[os.path.basename(p) for p, _ in skipped[:5]]}")

    X_all = np.moveaxis(np.concatenate(X_list, axis=0), 1, 2).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0)
    subj_all = np.concatenate(subj_list, axis=0)

    unique_subjects = np.unique(subj_all)
    NUM_SUBJECTS = len(unique_subjects)
    SEQ_LEN, INPUT_DIM = X_all.shape[1], X_all.shape[2]
    NUM_CLASSES = len(np.unique(y_all))
    print(f"Total epochs: {X_all.shape[0]} | Subjects: {NUM_SUBJECTS} | shape {X_all.shape}")

    loso_results, per_seed_records, all_epoch_logs = [], [], []
    pool_win_y_true, pool_win_y_pred, pool_win_y_prob = [], [], []
    pool_rec_y_true, pool_rec_y_pred, pool_rec_y_prob = [], [], []
    GLOBAL_BEST = dict(val_acc=-1.0, weights=None, subject=None, seed=None, best_epoch=None)

    for i, test_subj in enumerate(unique_subjects, start=1):
        print(f"\n{'='*70}\n  LOSO Fold {i}/{NUM_SUBJECTS} -- test subject: {test_subj}\n{'='*70}")

        train_pool = [s for s in unique_subjects if s != test_subj]
        n_val = max(2, int(round(len(train_pool) * VAL_SUBJECT_RATIO)))
        rng = np.random.default_rng(SEED + i)
        val_subjects = sorted(rng.choice(train_pool, size=n_val, replace=False).tolist(), key=str)
        val_set = set(val_subjects)
        train_subjects = [s for s in train_pool if s not in val_set]

        train_mask = np.isin(subj_all, train_subjects)
        val_mask = np.isin(subj_all, val_subjects)
        test_mask = (subj_all == test_subj)

        X_train_raw, y_train = X_all[train_mask], y_all[train_mask]
        X_val_raw, y_val = X_all[val_mask], y_all[val_mask]
        X_test_raw, y_test = X_all[test_mask], y_all[test_mask]

        if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  ! Skipping fold {i} (a split has only one class).")
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw.reshape(-1, INPUT_DIM)).reshape(X_train_raw.shape)
        X_val = scaler.transform(X_val_raw.reshape(-1, INPUT_DIM)).reshape(X_val_raw.shape)
        X_test = scaler.transform(X_test_raw.reshape(-1, INPUT_DIM)).reshape(X_test_raw.shape)

        cw_arr = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw = {int(c): float(w) for c, w in zip(np.unique(y_train), cw_arr)}

        seed_test_probs, seed_val_accs, seed_best_epochs = [], [], []
        for seed_idx, sd in enumerate(SEED_LIST):
            print(f"  -- Seed {seed_idx + 1}/{NUM_SEEDS} (sd={sd}) --")
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(sd)

            lr_schedule = schedules.ExponentialDecay(1e-4, decay_steps=3600,
                                                     decay_rate=0.5, staircase=True)
            model = build_transformer_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM,
                                            intermediate_dim=32, num_heads=5, num_layers=1,
                                            dropout_rate=0.1, num_classes=NUM_CLASSES)
            model.compile(optimizer=Adam(learning_rate=lr_schedule),
                          loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
            es = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE,
                               restore_best_weights=True, verbose=1)
            hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(X_val, y_val), class_weight=cw,
                             callbacks=[es], verbose=2)

            val_acc_history = hist.history.get("val_accuracy", [0.0])
            best_val_acc = float(np.max(val_acc_history))
            best_epoch = int(np.argmax(val_acc_history) + 1)
            seed_val_accs.append(best_val_acc); seed_best_epochs.append(best_epoch)
            seed_test_probs.append(model.predict(X_test, verbose=0))

            if best_val_acc > GLOBAL_BEST['val_acc']:
                GLOBAL_BEST = dict(val_acc=best_val_acc, weights=model.get_weights(),
                                   subject=test_subj, seed=sd, best_epoch=best_epoch)

            ep_log = pd.DataFrame(hist.history)
            ep_log.insert(0, "epoch", np.arange(1, len(ep_log) + 1))
            ep_log.insert(0, "seed", sd)
            ep_log.insert(0, "test_subject", str(test_subj))
            ep_log.insert(0, "fold", i)
            all_epoch_logs.append(ep_log)
            per_seed_records.append(dict(fold=i, test_subject=str(test_subj), seed=sd,
                                         best_val_acc=best_val_acc, best_epoch=best_epoch,
                                         total_epochs_trained=len(val_acc_history)))

        y_test_prob_ens = np.mean(seed_test_probs, axis=0)
        win_metrics = compute_binary_metrics(y_test, y_test_prob_ens)
        rec_true, rec_pred, rec_prob = aggregate_to_recording_level(y_test, y_test_prob_ens)
        rec_acc = float((rec_true == rec_pred).mean()) if len(rec_true) else 0.0

        loso_results.append(dict(
            subject=str(test_subj), val_subjects="|".join(map(str, val_subjects)),
            train_epochs=int(X_train_raw.shape[0]), val_epochs=int(X_val_raw.shape[0]),
            test_epochs=int(X_test_raw.shape[0]), n_seeds=NUM_SEEDS,
            mean_best_val_acc=float(np.mean(seed_val_accs)),
            max_best_val_acc=float(np.max(seed_val_accs)),
            mean_best_epoch=float(np.mean(seed_best_epochs)),
            win_acc=win_metrics['acc'], win_prec=win_metrics['prec'],
            win_sens=win_metrics['sens'], win_spec=win_metrics['spec'],
            win_f1=win_metrics['f1'], win_auc=win_metrics['auc'], win_kappa=win_metrics['kappa'],
            rec_acc=rec_acc, rec_correct=int((rec_true == rec_pred).sum()) if len(rec_true) else 0,
            rec_total=int(len(rec_true))))

        pool_win_y_true.extend(y_test.tolist())
        pool_win_y_pred.extend(np.argmax(y_test_prob_ens, axis=1).tolist())
        pool_win_y_prob.extend(y_test_prob_ens[:, 1].tolist())
        pool_rec_y_true.extend(rec_true.tolist())
        pool_rec_y_pred.extend(rec_pred.tolist())
        pool_rec_y_prob.extend(rec_prob.tolist())

        print(f"  >> Fold {i}: win Acc={win_metrics['acc']:.4f} F1={win_metrics['f1']:.4f} "
              f"AUC={win_metrics['auc']:.4f} | rec acc={rec_acc:.4f}")

    print(f"\nLOSO COMPLETE | folds: {len(loso_results)}/{NUM_SUBJECTS}")

    out = lambda name: os.path.join(args.out_dir, name)
    pd.DataFrame(loso_results).to_csv(out("eegmat_fair_loso_per_subject_results.csv"), index=False)
    pd.DataFrame(per_seed_records).to_csv(out("eegmat_fair_loso_per_fold_per_seed.csv"), index=False)
    if all_epoch_logs:
        pd.concat(all_epoch_logs, ignore_index=True).to_csv(out("eegmat_fair_loso_epochwise_logs.csv"), index=False)

    print("\nHeadline (mean +/- std over folds):")
    for key in ['win_acc', 'win_prec', 'win_sens', 'win_spec', 'win_f1', 'win_auc', 'win_kappa', 'rec_acc']:
        vals = [r[key] for r in loso_results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        print(f"  {key:<10}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} median={np.median(vals):.4f}")

    pwt, pwp, pwpr = map(np.array, (pool_win_y_true, pool_win_y_pred, pool_win_y_prob))
    prt, prp, prpr = map(np.array, (pool_rec_y_true, pool_rec_y_pred, pool_rec_y_prob))

    def pooled_block(yt, yp, yprob):
        cm = confusion_matrix(yt, yp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
        try:
            auc = roc_auc_score(yt, yprob)
        except ValueError:
            auc = float('nan')
        return dict(accuracy=float(accuracy_score(yt, yp)),
                    precision=float(precision_score(yt, yp, zero_division=0)),
                    sensitivity=float(recall_score(yt, yp, zero_division=0)),
                    specificity=float(tn / (tn + fp + 1e-9)),
                    f1=float(f1_score(yt, yp, zero_division=0)),
                    auc=float(auc), kappa=float(cohen_kappa_score(yt, yp)),
                    confusion_matrix=cm.tolist()), cm

    win_pooled, cm_win = pooled_block(pwt, pwp, pwpr)
    rec_pooled, cm_rec = pooled_block(prt, prp, prpr)
    with open(out("eegmat_fair_loso_pooled_metrics.json"), "w") as f:
        json.dump({"window_level": win_pooled, "recording_level": rec_pooled,
                   "config": {"dataset": "EEGMAT", "num_seeds": NUM_SEEDS, "seed_list": SEED_LIST,
                              "overlap_relax": overlap_relax, "overlap_stress": overlap_stress,
                              "apply_subject_norm": APPLY_SUBJECT_LEVEL_NORM,
                              "val_subject_ratio": VAL_SUBJECT_RATIO, "patience": PATIENCE,
                              "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                              "sfreq": sfreq, "epoch_length_sec": epoch_length,
                              "channels": CHANNELS_TO_PICK},
                   "n_folds_completed": len(loso_results)}, f, indent=2)
    print("\nPooled window-level:", win_pooled)
    print("Pooled recording-level:", rec_pooled)

    for cm, cmap, name, title in [(cm_win, 'Blues', "eegmat_fair_loso_window_cm.png", "Window-Level"),
                                  (cm_rec, 'Greens', "eegmat_fair_loso_recording_cm.png", "Recording-Level")]:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2%", annot_kws={"size": 14}, cmap=cmap,
                    xticklabels=['Relax', 'Stress'], yticklabels=['Relax', 'Stress'], cbar=True)
        plt.xlabel('Predicted label'); plt.ylabel('True label')
        plt.title(f'Fair LOSO (EEGMAT) -- {title} Pooled Confusion Matrix')
        plt.tight_layout(); plt.savefig(out(name), dpi=200, bbox_inches='tight'); plt.close()

    sub_ids = [r['subject'] for r in loso_results]
    win_accs = [r['win_acc'] for r in loso_results]
    rec_accs = [r['rec_acc'] for r in loso_results]
    x = np.arange(len(sub_ids)); width = 0.4
    fig, ax = plt.subplots(figsize=(max(10, len(sub_ids) * 0.4), 4.5))
    ax.bar(x - width / 2, win_accs, width, label='Window-level', color='steelblue', edgecolor='black')
    ax.bar(x + width / 2, rec_accs, width, label='Recording-level', color='seagreen', edgecolor='black')
    ax.axhline(np.mean(win_accs), color='blue', ls='--', alpha=0.6, label=f'Win mean = {np.mean(win_accs):.3f}')
    ax.axhline(np.mean(rec_accs), color='green', ls='--', alpha=0.6, label=f'Rec mean = {np.mean(rec_accs):.3f}')
    ax.set_xticks(x); ax.set_xticklabels(sub_ids, rotation=75, fontsize=8)
    ax.set_ylabel('Final Blind Test Accuracy'); ax.set_xlabel('Held-out Subject')
    ax.set_title('Fair LOSO (EEGMAT) -- Per-Subject Accuracy'); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout(); plt.savefig(out("eegmat_fair_loso_per_subject_acc.png"), dpi=200, bbox_inches='tight'); plt.close()

    print(f"\nBest val-selected model: subject={GLOBAL_BEST['subject']} seed={GLOBAL_BEST['seed']} "
          f"epoch={GLOBAL_BEST['best_epoch']} val_acc={GLOBAL_BEST['val_acc']:.4f}")
    if GLOBAL_BEST['weights'] is not None:
        best_model = build_transformer_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM,
                                             intermediate_dim=32, num_heads=5, num_layers=1,
                                             dropout_rate=0.1, num_classes=NUM_CLASSES)
        best_model.set_weights(GLOBAL_BEST['weights'])
        best_model.save(out("eegmat_fair_loso_best.keras"))
        print("Saved ->", out("eegmat_fair_loso_best.keras"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EEGMAT subject-independent LOSO")
    p.add_argument("--input_dir", required=True, help="folder containing EEGMAT .edf files")
    p.add_argument("--out_dir", default="outputs_loso_eegmat")
    main(p.parse_args())
