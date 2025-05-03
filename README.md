# STEC Pipeline Description 🧬🧬
Random Forest Classification pipeline -> Classify Shigatoxigenic E.coli samples based on Geographical Region & Country

** All model training and evaluation were performed on an AWS EC2 r5.8xlarge instance (32 vCPUs, 256 GB RAM) running Ubuntu Server 24.04 LTS. ** 

-> Features = kmers

-> Predictor = Region / Country

## Tools & Languages
<p align="left">
  <img src="https://github.com/user-attachments/assets/5e678fc0-9597-4252-98dd-eb9aaccc823e" alt="Python" width="60" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/4bbcf45e-d572-45e9-a16c-3ff379e72390" alt="Bash" width="65" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/805532d9-fc8b-446f-aac6-933cc4aa6185" alt="Git" width="65" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/0427f54d-9e05-4969-91d1-13af16c3fb42" alt="SQL" width="110" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/bfc30e37-cb64-4d59-8cec-52ab5c12fab7" alt="Docker" width="75" style="margin: 0 10px;"/>
  <img src="https://github.com/user-attachments/assets/910424f1-59e7-40cf-bc40-2b55d0ccb7d5" alt="AWS" width="90" style="margin: 0 10px;"/>
</p>

## Execute Pipeline

```bash
git clone https://github.com/guillermocomesanacimadevila/STEC-Classifier.git
```

```bash
cd STEC-Classifier
```

```bash
bash ~/tools/nextflow run /full/path/to/main.nf
```

```bash
chmod +x run_pipeline.sh && ./run_pipeline.sh
```

## Pipeline Workflow
![image](https://github.com/user-attachments/assets/47276e78-0b3b-46f2-a3fa-bd62c0d03164)
