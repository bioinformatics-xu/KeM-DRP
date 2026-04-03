# KeM-DRP

A Knowledge-Enhanced Multimodal Framework with Genomic Reconstruction for Drug Response Prediction in DLBCL

KeM-DRP is a cutting-edge deep learning framework that integrates multi-modal clinical data (genomic, transcriptomic, and clinical features) to predict drug responses in Diffuse Large B-Cell Lymphoma patients. 

---

### ✔ Prerequisites

- Python ≥ 3.8  
- TensorFlow ≥ 2.4  
- Transformers ≥ 4.5  
  - BERT (TensorFlow version)  
- Recommend: Conda for environment management  
- GPU support (optional but recommended)

---

### ✔ Data

- Reactome

---

### ✔ Installation

   ```bash
   git clone https://github.com/yourname/KeM-DRP.git
   cd KeM-DRP
   ```
### ✔ Usage
Step 1: Activate the project-specific conda environment
   ```bash
    source activateKeM-DRP_env
   ```
Step 2: Navigate to the data directory and run the data splitting script
   ```bash
    cd ./data/propstate_paper
    python split_data.py
   ```
Step 3: Navigate to the training directory and launch the main training script
   ```bash
    cd ./train
    python run_me.py
   ```
###  ✔ Developor

Xiaolu Xu (lu.xu@lnnu.edu.cn)

Yulong Li (liyulong20000810@163.com)

School of Computer and Artificial Intelligence 

Liaoning Normal University
