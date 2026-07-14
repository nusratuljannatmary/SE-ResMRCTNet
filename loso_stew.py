"""
loso_stew.py  --  STEW subject-independent (Leave-One-Subject-Out) experiment.

Fair LOSO cross-validation with:
  - per-subject z-score normalization (key LOSO improvement),
  - subject-disjoint train / validation / test splits,
  - per-fold StandardScaler fitted on training subjects only,
  - class weighting, EarlyStopping on val_loss,
  - multi-seed ensembling (mean softmax over seeds),
  - window-level and recording-level metrics, plus a val-selected best model.

The model architecture is imported from model.py (unchanged).

Usage:
  python loso_stew.py --stress_dir data/stew/hi_txt_files \
                      --relax_dir  data/stew/lo_txt_files \
                      --out_dir    outputs_loso_stew
"""

import os
import re
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
sfreq = 128
epoch_length = 30            # seconds -> seq_len = 30 * 128 = 3840
overlap_relax = 0.75
overlap_stress = 0.75
APPLY_SUBJECT_LEVEL_NORM = True   # per-subject per-channel z-score on raw signal

EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 30
NUM_SEEDS = 2
SEED_LIST = [SEED + s for s in [0, 7, 13, 23, 37]][:NUM_SEEDS]
VAL_SUBJECT_RATIO = 0.20

CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


# ============================ Preprocessing helpers ============================
def load_txt_as_raw(path, info, l_freq=0.5, h_freq=45.0, apply_subject_norm=True):
    """Load one subject .txt -> filter -> average reference -> per-subject z-score."""
    x = np.loadtxt(path)
    if x.ndim != 2:
        raise ValueError(f"{path} is not 2D. Got shape: {x.shape}")
    if x.shape[1] == 14:
        x = x.T
    elif x.shape[0] == 14:
        pass
    else:
        raise ValueError(f"{path} must have 14 channels. Got shape: {x.shape}")

    x = x.astype(np.float32)
    raw = mne.io.RawArray(x, info, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose=False)

    if apply_subject_norm:
        data = raw.get_data()
        mu = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True) + 1e-6
        raw._data = ((data - mu) / std).astype(np.float32)
    return raw


def epoch_raw_with_overlap(raw, label, epoch_length, sfreq, overlap):
    """Split a Raw into overlapping epochs with the given label."""
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
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'(sub\d+|S\d+|\d+)', base, re.IGNORECASE)
    return m.group(1).lower() if m else base.lower()


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
    """One prediction per recording = argmax of mean softmax over its windows."""
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

    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq,
                           ch_types=['eeg'] * len(CHANNEL_NAMES))

    # ---- Build dataset with subject tracking ----
    stress_files = sorted(os.path.join(args.stress_dir, f)
                          for f in os.listdir(args.stress_dir) if f.endswith(".txt"))
    relax_files = sorted(os.path.join(args.relax_dir, f)
                         for f in os.listdir(args.relax_dir) if f.endswith(".txt"))

    X_list, y_list, subj_list = [], [], []
    for fp in relax_files:                                # relax = 0
        raw = load_txt_as_raw(fp, info, apply_subject_norm=APPLY_SUBJECT_LEVEL_NORM)
        X, y = epoch_raw_with_overlap(raw, 0, epoch_length, sfreq, overlap_relax)
        if X is not None:
            X_list.append(X); y_list.append(y)
            subj_list.append(np.full(X.shape[0], extract_subject_id(fp), dtype=object))
    for fp in stress_files:                               # stress = 1
        raw = load_txt_as_raw(fp, info, apply_subject_norm=APPLY_SUBJECT_LEVEL_NORM)
        X, y = epoch_raw_with_overlap(raw, 1, epoch_length, sfreq, overlap_stress)
        if X is not None:
            X_list.append(X); y_list.append(y)
            subj_list.append(np.full(X.shape[0], extract_subject_id(fp), dtype=object))

    X_all = np.moveaxis(np.concatenate(X_list, axis=0), 1, 2).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0)
    subj_all = np.concatenate(subj_list, axis=0)

    unique_subjects = np.unique(subj_all)
    NUM_SUBJECTS = len(unique_subjects)
    SEQ_LEN, INPUT_DIM = X_all.shape[1], X_all.shape[2]
    NUM_CLASSES = len(np.unique(y_all))
    print(f"Total epochs: {X_all.shape[0]} | Subjects: {NUM_SUBJECTS} | shape {X_all.shape}")

    # ---- Fair LOSO loop with multi-seed ensemble ----
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

    # ---- Save per-subject + per-seed + epoch logs ----
    out = lambda name: os.path.join(args.out_dir, name)
    pd.DataFrame(loso_results).to_csv(out("fair_loso_improved_per_subject_results.csv"), index=False)
    pd.DataFrame(per_seed_records).to_csv(out("fair_loso_improved_per_fold_per_seed.csv"), index=False)
    if all_epoch_logs:
        pd.concat(all_epoch_logs, ignore_index=True).to_csv(out("fair_loso_improved_epochwise_logs.csv"), index=False)

    print("\nHeadline (mean +/- std over folds):")
    for key in ['win_acc', 'win_prec', 'win_sens', 'win_spec', 'win_f1', 'win_auc', 'win_kappa', 'rec_acc']:
        vals = [r[key] for r in loso_results if not (isinstance(r[key], float) and np.isnan(r[key]))]
        print(f"  {key:<10}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} median={np.median(vals):.4f}")

    # ---- Pooled metrics ----
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
    with open(out("fair_loso_improved_pooled_metrics.json"), "w") as f:
        json.dump({"window_level": win_pooled, "recording_level": rec_pooled,
                   "config": {"num_seeds": NUM_SEEDS, "seed_list": SEED_LIST,
                              "overlap_relax": overlap_relax, "overlap_stress": overlap_stress,
                              "apply_subject_norm": APPLY_SUBJECT_LEVEL_NORM,
                              "val_subject_ratio": VAL_SUBJECT_RATIO, "patience": PATIENCE,
                              "epochs": EPOCHS, "batch_size": BATCH_SIZE},
                   "n_folds_completed": len(loso_results)}, f, indent=2)
    print("\nPooled window-level:", win_pooled)
    print("Pooled recording-level:", rec_pooled)

    # ---- Confusion-matrix figures ----
    for cm, cmap, name, title in [(cm_win, 'Blues', "fair_loso_improved_window_cm.png", "Window-Level"),
                                  (cm_rec, 'Greens', "fair_loso_improved_recording_cm.png", "Recording-Level")]:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".2%", annot_kws={"size": 14}, cmap=cmap,
                    xticklabels=['Relax', 'Stress'], yticklabels=['Relax', 'Stress'], cbar=True)
        plt.xlabel('Predicted label'); plt.ylabel('True label')
        plt.title(f'Fair LOSO (STEW) -- {title} Pooled Confusion Matrix')
        plt.tight_layout(); plt.savefig(out(name), dpi=200, bbox_inches='tight'); plt.close()

    # ---- Per-subject accuracy bar chart ----
    sub_ids = [r['subject'] for r in loso_results]
    win_accs = [r['win_acc'] for r in loso_results]
    rec_accs = [r['rec_acc'] for r in loso_results]
    x = np.arange(len(sub_ids)); width = 0.4
    fig, ax = plt.subplots(figsize=(max(10, len(sub_ids) * 0.35), 4.5))
    ax.bar(x - width / 2, win_accs, width, label='Window-level', color='steelblue', edgecolor='black')
    ax.bar(x + width / 2, rec_accs, width, label='Recording-level', color='seagreen', edgecolor='black')
    ax.axhline(np.mean(win_accs), color='blue', ls='--', alpha=0.6, label=f'Win mean = {np.mean(win_accs):.3f}')
    ax.axhline(np.mean(rec_accs), color='green', ls='--', alpha=0.6, label=f'Rec mean = {np.mean(rec_accs):.3f}')
    ax.set_xticks(x); ax.set_xticklabels(sub_ids, rotation=75, fontsize=8)
    ax.set_ylabel('Final Blind Test Accuracy'); ax.set_xlabel('Held-out Subject')
    ax.set_title('Fair LOSO (STEW) -- Per-Subject Accuracy'); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout(); plt.savefig(out("fair_loso_improved_per_subject_acc.png"), dpi=200, bbox_inches='tight'); plt.close()

    # ---- Save best validation-selected model ----
    print(f"\nBest val-selected model: subject={GLOBAL_BEST['subject']} seed={GLOBAL_BEST['seed']} "
          f"epoch={GLOBAL_BEST['best_epoch']} val_acc={GLOBAL_BEST['val_acc']:.4f}")
    if GLOBAL_BEST['weights'] is not None:
        best_model = build_transformer_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM,
                                             intermediate_dim=32, num_heads=5, num_layers=1,
                                             dropout_rate=0.1, num_classes=NUM_CLASSES)
        best_model.set_weights(GLOBAL_BEST['weights'])
        best_model.save(out("stew_fair_loso_improved_best.keras"))
        print("Saved ->", out("stew_fair_loso_improved_best.keras"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="STEW subject-independent LOSO")
    p.add_argument("--stress_dir", required=True, help="high-workload (stress) .txt folder")
    p.add_argument("--relax_dir", required=True, help="low-workload (relax) .txt folder")
    p.add_argument("--out_dir", default="outputs_loso_stew")
    main(p.parse_args())
