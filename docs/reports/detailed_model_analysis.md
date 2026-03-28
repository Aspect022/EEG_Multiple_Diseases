# Detailed Analysis of ECG Classification Architectures

This report provides an in-depth, meticulous breakdown of the three cutting-edge paradigms developed for the PTB-XL ECG Apnea Classification framework. 

Each model—Spiking Neural Network (SNN), Vision Transformer (ViT/Swin), and Quantum Hybrid CNN—targets continuous wavelet transform (CWT) scalograms representing 12-channel 2D ECG spatial-temporal states. Below are the precise configurations, features, and mechanisms that define their implementations.

---

## 1. Vision Transformer (ViT / Swin Transformer)
*Referred to contextually as BIT/ViT.*

### Architecture Overview
The model (`SwinECGClassifier`) leverages the **Swin Transformer** (Shifted Window) architecture, wrapping the `timm` implementation for ECG sequence learning. Standard Vision Transformers (ViTs) compute global self-attention with quadratic complexity $O(N^2)$, which is computationally prohibitive for high-resolution scalograms. The Swin architecture mitigates this using **Shifted Window Attention**.

### Detailed Mechanisms
*   **Base Type:** `swin_tiny_patch4_window7_224` (28M parameters, pre-trained on ImageNet).
*   **Window Logic:** The image is divided into non-overlapping local windows of size $7 \times 7$. Self-attention is computed locally ($O(N)$ complexity). To allow cross-window communication, successive layers shift the window partition by a displacement of $(M/2, M/2)$ pixels.
*   **Patch Size:** $4 \times 4$. Input resolution is exactly 224x224.
*   **Layers & Channels:** The hierarchical architecture employs 4 stages. The embedding dimension undergoes a 2x multiplier at each stage, eventually resulting in `embed_dim * 8` features before the final classifier.
*   **Classification Head:** Custom-built via `nn.Sequential(nn.LayerNorm, nn.Dropout, nn.Linear)` mapping the deeply extracted hierarchical features to the target ECG diagnostic classes.
*   **Regularization & Training:**
    *   Dropout Rate: Configurable, default `0.0`.
    *   Stochastic Depth Rate (Drop Path): `0.1`.
    *   Linear Probing Option: Equipped with a `freeze_backbone` command to freeze all weights except the classification head.

---

## 2. Spiking Neural Network (SNN / SNL)
*Referred to contextually as SNL.*

### Architecture Overview
The SNN model (`SpikingResNet`) is built on a neuromorphic computing paradigm. Rather than continuous floating-point activations, it propagates discrete binary events (1 or 0 spikes). This mimics biological neural activity and reduces active power consumption dramatically on specialized hardware (e.g., Intel Loihi). 

### Detailed Mechanisms
*   **Backbone:** ResNet-18 skeleton, where traditional continuous `ReLU` activation functions are completely replaced by Spiking Neurons.
*   **Neuron Model (LIF):** **Leaky Integrate-and-Fire** (`snn.Leaky` from `snntorch`). 
    *   *Mechanism:* A neuron maintains a membrane potential $U(t)$. This potential "leaks" or decays over time by a factor of $\beta$. When incoming stimulus pushes $U(t)$ over a threshold $V_{th}$, the neuron emits a spike $S(t)=1$ and the potential is reset.
    *   *Decay Rate ($\beta$):* Defined as `0.9`. 
    *   *Threshold ($V_{th}$):* Standardized at `1.0`.
*   **Temporal Dynamics:** The model simulates time via a temporal execution loop. The static 2D ECG scalogram image is repeatedly presented to the network for $T=25$ discrete timesteps. The final prediction relies on the firing rate distribution over these $T=25$ steps.
*   **Surrogate Training:** Spikes are non-differentiable step functions (Heaviside). To train via standard Backpropagation Through Time (BPTT), the network uses a **Surrogate Gradient**.
    *   *Function:* `fast_sigmoid` with a slope constant of `25`. During the backward pass, the strict step function is approximated by a smooth sigmoid derivative to allow gradient flow.

---

## 3. Hybrid Quantum-Classical CNN (Fusion)

### Architecture Overview
The Quantum Hybrid model (`HybridQuantumCNN`) is a **fusion architecture**. It seamlessly integrates heavy classical Convolutional Neural Networks (CNN) with a highly parameterized Quantum Variational Circuit (QVC) leveraging the `PennyLane` and `PyTorch` frameworks.

### The Reason for Fusing
Pure quantum simulation is computationally explosive ($O(2^N)$ logic tracking). A full-scale raw input quantum neural network is intractable on a local RTX 5050 for high-resolution images. **The Fusion Strategy** dictates that classical Conv2D layers heavily downsample and extract initial dense features ($C=64 \to 128 \to 256$) *before* bottlenecking into the quantum dimension. The quantum layer acts as a uniquely correlated, entangled feature filter that classical logic cannot efficiently replicate.

### Detailed Mechanisms
*   **Classical Front-End:** Three blocks of `Conv2d` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d` (Stride 2), heavily reducing the spatial 224x224 input down to a manageable feature map.
*   **Quantum Backend (Quanv Layer):**
    *   *Number of Qubits (`n_qubits`):* 4. 
    *   *Quantum Depth (`q_depth` / Layers):* 2.
    *   *Input Encoding:* Classical float data is encoded into the quantum state using **Angle Embedding**. Specifically, the vector elements parameterize Pauli-$R_Y$ rotations: $R_Y(x_i \cdot \pi)$ across the qubits.
    *   *Parametric Rotations:* To learn, the quantum layer utilizes differentiable parametric rotations $R_X(\theta_1)$, $R_Y(\theta_2)$, and $R_Z(\theta_3)$ for each qubit.
    *   *Type of Entanglement:* A **Ring Topology** is enforced utilizing $CNOT$ gates (`qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])`). This linearly tangles $q_0 \to q_1 \to q_2 \to q_3 \to q_0$, creating strictly non-classical super-positional correlations.
    *   *Measurement:* The output of the circuit collapses via the **Expectation Value** measuring the Pauli-Z operator ($\langle Z \rangle$) across all 4 qubits.

---

## 4. Universal Pipeline Triggers (Enabled Features)
All three models are seamlessly executed inside `unified_pipeline.py`. Key automated quality-of-life and performance mechanisms strictly active are:

*   **Mixed Precision (AMP):** Standardized. Halves runtime memory by executing volatile layers in `FP16` via `torch.cuda.amp.autocast()`, preserving 6GB VRAM bounds on the RTX 5050.
*   **Gradient Accumulation:** Step count set to `4`. Mimics the stability of evaluating larger batches (effective batch size = 64) despite sequential physical hardware limits (batch size = 16).
*   **Early Stopping:** Enabled unconditionally.
    *   *Patience:* Set to `5` epochs. If Macro F1-Score does not strictly improve within 5 temporal periods, training completely stops to prevent overfitting.
*   **Scheduler Constraints:** `CosineAnnealingLR` actively curves the learning rate gradually smoothing parameter approach towards the optimal minima.
*   **Reproducibility Controls:** Full `seed_everything(42)` ensures the CUDNN backend operates deterministically when randomizers initiate.
