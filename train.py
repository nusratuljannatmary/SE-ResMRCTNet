"""
train.py  --  STEW training (leakage-safe, subject-dependent / intra-subject).

Pipeline (exactly as in the original notebook):
  1. Split into train+val / test FIRST (test stays untouched).
  2. 7-fold StratifiedKFold on train+val; in each fold the StandardScaler is
     fitted ONLY on that fold's training data (no leakage).
  3. Keep the best fold's weights, then retrain on the full train+val set.
  4. Evaluate on the held-out test set.
  5. Save:  the Keras model (.keras), the scaler parameters (.npz),
            the scaled test set (x_test.npy / y_test.npy) for evaluate.py,
            and a TensorFlow Lite model (.tflite) for edge deployment.

Usage:
  python train.py --data data/all_epochs_data.npy --labels data/all_labels.npy --out_dir outputs
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, schedules

from model import build_transformer_model

RANDOM_STATE = 42


def fit_transform_train_only(x_train_raw, x_other_raw=None):
    """Fit StandardScaler on training epochs only; transform another split if given."""
    scaler = StandardScaler()
    n_features = x_train_raw.shape[-1]
    x_train = scaler.fit_transform(
        x_train_raw.reshape(-1, n_features)
    ).reshape(x_train_raw.shape).astype(np.float32)
    if x_other_raw is None:
        return x_train, scaler
    x_other = scaler.transform(
        x_other_raw.reshape(-1, n_features)
    ).reshape(x_other_raw.shape).astype(np.float32)
    return x_train, x_other, scaler


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    all_epochs = np.load(args.data).astype(np.float32)
    labels = np.asarray(np.load(args.labels))
    seq_len, input_dim = all_epochs.shape[1], all_epochs.shape[2]

    # 1) Train+val / test split (test held out, unscaled for now)
    trainval_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=0.30,
        stratify=labels, random_state=RANDOM_STATE,
    )
    x_trainval_raw, y_trainval = all_epochs[trainval_idx], labels[trainval_idx]
    x_test_raw, y_test = all_epochs[test_idx], labels[test_idx]

    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=0.0001, decay_steps=3600,
        decay_rate=0.5, staircase=True,
    )

    # 2) 7-fold cross-validation, scaler fit on each fold's train only
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
    best_val_acc, best_weights = 0.0, None

    for fold, (tr, va) in enumerate(skf.split(x_trainval_raw, y_trainval)):
        print(f"Fold {fold + 1}")
        x_tr, x_va, _ = fit_transform_train_only(
            x_trainval_raw[tr], x_trainval_raw[va])
        y_tr, y_va = y_trainval[tr], y_trainval[va]

        model = build_transformer_model(seq_len=seq_len, input_dim=input_dim)
        model.compile(optimizer=Adam(learning_rate=lr_schedule),
                      loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        history = model.fit(x_tr, y_tr, epochs=200, batch_size=16,
                            validation_data=(x_va, y_va), verbose=0)

        val_acc = max(history.history['val_accuracy'])
        print(f"Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_weights = val_acc, model.get_weights()

    # 3) Final scaler fit on full train+val; transform held-out test
    x_trainval, x_test, final_scaler = fit_transform_train_only(
        x_trainval_raw, x_test_raw)

    np.savez(os.path.join(args.out_dir, "stew_train_only_standardizer.npz"),
             mean=final_scaler.mean_, scale=final_scaler.scale_, var=final_scaler.var_)

    final_model = build_transformer_model(seq_len=seq_len, input_dim=input_dim)
    final_model.compile(optimizer=Adam(learning_rate=lr_schedule),
                        loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    if best_weights is not None:
        final_model.set_weights(best_weights)
    final_model.fit(x_trainval, y_trainval, epochs=200, batch_size=16, verbose=0)

    # 4) Held-out test evaluation
    test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Model Test Accuracy: {test_acc:.4f}")

    # 5) Save artifacts
    keras_path = os.path.join(args.out_dir, "stew_model_30.keras")
    final_model.save(keras_path)
    np.save(os.path.join(args.out_dir, "x_test.npy"), x_test)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test.astype(np.int32))

    # TensorFlow Lite conversion (edge deployment)
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(final_model).convert()
    tflite_path = os.path.join(args.out_dir, "stew_model_30.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("Saved Keras model :", keras_path)
    print("Saved TFLite model:", tflite_path)
    print("Saved test set    :", os.path.join(args.out_dir, "x_test.npy"),
          "/", os.path.join(args.out_dir, "y_test.npy"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="STEW training")
    p.add_argument("--data", default="data/all_epochs_data.npy")
    p.add_argument("--labels", default="data/all_labels.npy")
    p.add_argument("--out_dir", default="outputs")
    main(p.parse_args())
