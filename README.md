# 🧬 STEC Geographic Classification Pipeline

This project provides a **reproducible machine learning pipeline** to classify _Shiga toxin-producing E. coli_ (STEC) samples by **geographic region** and **country** using **k-mer features** and **Random Forest classifiers**.

> **Hardware Used**: All experiments were conducted on an AWS EC2 `r5.8xlarge` instance (32 vCPUs, 256 GB RAM) running **Ubuntu Server 24.04 LTS**.

---

## 🚀 Overview

This pipeline performs the following steps:

1. **Metadata Cleaning & Normalization** (country, region, strain types)
2. **Feature Preprocessing** (k-mer filtering, normalisation, scaling)
3. **Feature Selection** (variance threshold, correlation filtering, RF importance)
4. **Data Balancing** (SMOTE, oversampling, undersampling)
5. **Model Training & Evaluation** (Random Forest with hyperparameter tuning)
6. **Top K-mer Interpretation** (feature importance + BLAST)
7. **Test Prediction & Reporting**

---

## 🛠️ Tech Stack

<p align="left">
  <img src="https://github.com/user-attachments/assets/5e678fc0-9597-4252-98dd-eb9aaccc823e" alt="Python" width="60" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/4bbcf45e-d572-45e9-a16c-3ff379e72390" alt="Bash" width="65" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/805532d9-fc8b-446f-aac6-933cc4aa6185" alt="Git" width="65" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/0427f54d-9e05-4969-91d1-13af16c3fb42" alt="SQL" width="110" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/bfc30e37-cb64-4d59-8cec-52ab5c12fab7" alt="Docker" width="75" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/910424f1-59e7-40cf-bc40-2b55d0ccb7d5" alt="AWS" width="90" style="margin: 0 10px;"/>
</p>

---

## 📦 Installation

```bash
git clone https://github.com/guillermocomesanacimadevila/STEC-Classifier.git
```

```bash
cd STEC-Classifier
```

```bash
$($(find / -name nextflow -type f 2>/dev/null | head -n 1))
```

```bash
cd ~/STEC-Classifier
```

```bash
chmod +x run_pipeline.sh && ./run_pipeline.sh
```

## 🧪 Pipeline Stages
<img src="https://github.com/user-attachments/assets/50b82c27-34a7-4e31-a0c5-e0814a57be05" width="800"/>
