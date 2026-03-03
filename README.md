# 🫀 Advanced ECG Classification: Local Pipeline

Hey! This project lets you run the 3 cutting-edge Machine Learning paradigms (Spiking Neural Networks, Quantum-Classical Hybrid CNNs, and Swin Vision Transformers) on your RTX 5050 to classify ECG loops for Apnea events. All modules have been merged into a single easy-to-use research pipeline!

## 🚀 Getting Started

The pipeline contains automated optimizations specially set for your RTX 5050 (like mixed precision AMP and gradient accumulation to ensure we don't hit the 6GB VRAM limit). 

### 1. Prerequisites
Make sure you have an environment with **Python 3.10+**. 
It's recommended to create a virtual environment first.

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies
Run the following to grab all the required Deep Learning libraries (PyTorch with CUDA, PennyLane for Quantum, snntorch for neuromorphic, timm for Transformer).

```bash
pip install -r requirements.txt
```

### 3. Provide Data
You'll need the **Apnea-ECG Database** from PhysioNet. 

If you cloned this via git and the `data/` folder is included locally, you're all set! 
(If the `data/` folder appears missing or requires manual download later on, just download the Apnea-ECG datset and place the extracted files in `data/apnea-ecg-database-1.0.0`)

### 4. Run the Pipeline!
Simply execute the unified pipeline file. It will sequentially train all three models:

```bash
python unified_pipeline.py --epochs 15
```

If you only want to test to see everything compile:
```bash
python unified_pipeline.py --epochs 1
```

## 🛠 Model Highlights

1. **SNN (SpikingResNet):** A neuromorphic implementation tracking membrane voltage over time, simulating energy-efficient brain impulses for detecting sleep apnea patterns.
2. **Quantum CNN:** Processes input through a classical-quantum feature extraction mechanism mimicking the `qnode` pipeline with `PennyLane`. 
3. **Swin Transformer:** Utilizes shifted window attention directly on 12-channel scalogram representations of standard ECG waves.
