# seizewatch_ml_pipeline
# SeizeWatch ML Pipeline
This project is a personalized seizure-detection pipeline built from wearable heart rate and accelerometer data. It combines rule-based detection, anomaly detection, and supervised calibration to classify incoming sensor windows as OK, WARN, or ALARM for real-time seizure monitoring.

---

## Features
- Personalized baseline learning from wearable sensor data
- Hybrid detection using deterministic rules and machine learning
- Autoencoder-based latent representation of normal behavior
- kNN anomaly scoring in latent space
- Supervised calibration using annotated seizure logs
- Final outputs of OK, WARN, and ALARM
- Android runtime export support

---

## Pipeline Overview
1. **Stage 1: Autoencoder**
   Learns a user-specific baseline from accelerometer and heart rate features.

2. **Stage 2: Deterministic rules**
   Scores clinically meaningful seizure-related patterns such as tonic-clonic, focal-like, and startle-like behavior.

3. **Stage 3: kNN anomaly detector**
   Measures how far a new window deviates from the learned baseline in latent space.

4. **Stage 4: Supervised calibrator**
   Uses seizure-log labels to combine earlier signals into a final seizure-likelihood score and alarm state.

---

## Current Selected Configuration
Selected training result:
- ALARM detection: **79.2%** of logged events  
- Detection at `WARN` or higher: **95.8%**  
- Tonic-clonic ALARM recall: **100%** in holdout evaluation  

---

## Repository Structure
- `main_pipeline.py` – full offline training and evaluation pipeline
- `part1_autoencoder.py` – feature extraction and autoencoder baseline learning
- `part2_deterministic.py` – rule-based seizure scoring
- `part3_knn_detector.py` – latent-space anomaly scoring
- `part4_supervised_calibrator.py` – supervised calibration layer

---

## Future Work
- Additional validation on prospective data
- Improved false-positive reduction during high activity/play
- Separation of offline training and mobile runtime inference
- Continued integration into Android wearable monitoring workflow

---

## Privacy Note
This repository contains **no real patient data**.  
All datasets used during development were either synthetic or de-identified, and no seizure logs, reports, or derived outputs are included in this repository.
