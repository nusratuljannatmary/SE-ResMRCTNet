"""
inference_tflite.py  --  Run the exported TFLite model on the edge.

Demonstrates how the .tflite model produced by train.py is loaded and used
for prediction with only the lightweight TFLite runtime (no full TensorFlow
training stack needed). This directly supports the paper's edge-deployment claim.

Usage:
  python inference_tflite.py --tflite outputs/stew_model_30.tflite --x_test outputs/x_test.npy
"""

import argparse
import numpy as np

try:
    # Lightweight runtime, typical on edge devices (Raspberry Pi, etc.)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback to the interpreter bundled with full TensorFlow
    from tensorflow.lite import Interpreter

CLASS_NAMES = {0: "Stress", 1: "Relax"}


def main(args):
    interpreter = Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x_test = np.load(args.x_test).astype(np.float32)
    n = min(args.num_samples, len(x_test))
    print(f"Running TFLite inference on {n} sample(s)...\n")

    for i in range(n):
        sample = x_test[i:i + 1]                      # shape (1, seq_len, channels)
        interpreter.set_tensor(inp['index'], sample)
        interpreter.invoke()
        probs = interpreter.get_tensor(out['index'])[0]
        pred = int(np.argmax(probs))
        print(f"Sample {i}: predicted = {CLASS_NAMES[pred]} "
              f"(prob={probs[pred]:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TFLite edge inference demo")
    p.add_argument("--tflite", default="outputs/stew_model_30.tflite")
    p.add_argument("--x_test", default="outputs/x_test.npy")
    p.add_argument("--num_samples", type=int, default=5)
    main(p.parse_args())
