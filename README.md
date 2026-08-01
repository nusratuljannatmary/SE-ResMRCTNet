# SE-ResMRCTNet for EEG Workload Detection: Design, Deployment, and Latency Evaluation on Raspberry Pi

## Abstract

Detecting cognitive workload during arithmetic tasks provides valuable insights into how the brain processes incoming stimuli. Electroencephalography (EEG) offers a non-invasive means of assessing cognitive workload; however, accurate classification remains challenging and often relies on extensive preprocessing and handcrafted feature extraction. Although recent deep learning approaches have achieved promising performance, translating these models into computationally efficient and deployable systems for edge devices remains a significant challenge.

In this paper, we propose **SE-ResMRCTNet**, a compact end-to-end architecture for EEG-based cognitive workload detection that emphasizes computational efficiency and deployment on resource-constrained platforms. The proposed network extracts short- and mid-range temporal patterns directly from raw EEG signals using multi-resolution convolutions, adaptively recalibrates feature representations through Squeeze-and-Excitation modules, and captures long-range temporal dependencies using a lightweight residual Transformer encoder followed by a residual multilayer perceptron (MLP) classification head.

This architecture effectively combines local inductive biases with global temporal modeling while maintaining low computational complexity. Experimental results demonstrate that SE-ResMRCTNet achieves **95.83% accuracy, 91.67% sensitivity, and 100% specificity on the STEW dataset**, and **97.20% accuracy, 97.99% sensitivity, and 95.38% specificity on the EEGMAT dataset**.

The model requires only **47.05 K parameters (183.79 KB) and 1.14 GFLOPs on STEW**, and **57.80 K parameters (225.78 KB) and 1.27 GFLOPs on EEGMAT**, demonstrating an excellent trade-off between classification performance and computational efficiency. Furthermore, deployment on a Raspberry Pi 3B+ confirmed identical classification performance to the PC implementation while requiring less than 250 KB of memory, demonstrating the practical feasibility of the proposed model for real-time cognitive workload monitoring on resource-constrained edge devices.

---

## Repository Structure

- `Preprocessing/` — EEG preprocessing and normalization scripts
- `Model/` — SE-ResMRCTNet architecture
- `Trained_Models/` — Trained models and weights for STEW and EEGMAT
- `Performances/` — Evaluation scripts and performance results
- `Interpretability/` — UMAP-based feature visualization
- `model.py` — Main model implementation

---

## Datasets

The proposed model was evaluated using:

- **STEW dataset**
- **EEGMAT dataset**

The preprocessing pipeline uses:

- 0.5–45 Hz zero-phase FIR band-pass filtering
- Fixed-length EEG segmentation
- 30-second epochs for STEW
- 8-second epochs for EEGMAT
- Channel-wise z-score normalization
- Training-only class-dependent overlapping-window augmentation for EEGMAT

---

## Model Architecture

SE-ResMRCTNet integrates:

- Multi-resolution temporal convolution
- Squeeze-and-Excitation channel recalibration
- Lightweight residual Transformer encoder
- Global average pooling
- Residual multilayer perceptron classification head
- Softmax output for stress and relax classification

---

## Performance

| Dataset | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|
| STEW | 95.83% | 91.67% | 100.00% |
| EEGMAT | 97.20% | 97.99% | 95.38% |

---

## Edge Deployment

The trained SE-ResMRCTNet model was converted to TensorFlow Lite and deployed on a Raspberry Pi 3B+.

The deployment preserved the same classification performance as the PC implementation while maintaining a compact model size and practical inference latency for window-based cognitive workload monitoring.

---

## Contact

For any queries, feel free to contact us.

**Contact Author**  
Dr. Mohammod Abdul Motin  
Associate Professor  
Department of Electrical & Electronic Engineering  
Rajshahi University of Engineering & Technology  
Rajshahi 6204, Bangladesh  

E-mail: [m.a.motin@ieee.org](mailto:m.a.motin@ieee.org)

---

## License

This project is licensed under the MIT License.
