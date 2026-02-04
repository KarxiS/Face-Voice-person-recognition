# Face & Voice Person Recognition

A bimodal biometric identification system capable of recognizing individuals based on their voice and facial features. This project implements and compares two distinct machine learning pipelines: **Gaussian Mixture Models (GMM)** for audio processing and **Support Vector Machines (SVM)** with **HOG features** for image recognition.

Developed as part of the **Voice and Image Recognition (SUR)** course.

## 📌 Overview

The system identifies **31 distinct subjects** from a dataset containing:
- **Audio:** 16kHz WAV recordings (6 samples per person).
- **Images:** PNG facial crops (6 samples per person).

The project demonstrates the complete ML lifecycle: data preprocessing, feature engineering, hyperparameter tuning via GridSearch, model training, and evaluation.

## 🛠 Tech Stack

- **Language:** Python 3.x
- **Environment:** Jupyter Notebook
- **Core Libraries:**
  - `scikit-learn`: GMM, SVM, StandardScaler, GridSearchCV, Pipelines.
  - `scikit-image`: HOG feature extraction.
  - `OpenCV (cv2)`: Image preprocessing.
  - `NumPy`: Matrix operations and vectorization.
  - `Matplotlib`: Visualization of results and metrics.
  - `ikrlib`: Custom library for MFCC extraction and DSP utilities.

## 🧠 Methodology

### 1. Audio Recognition Pipeline
*Voice-based identification using spectral features.*

- **Preprocessing:**
  - Raw WAV files (16kHz) are processed to remove silence/noise based on energy thresholds.
- **Feature Extraction:**
  - **MFCC (Mel-Frequency Cepstral Coefficients):** Extracts spectral envelope features that characterize the vocal tract.
  - Features are normalized using `StandardScaler` to handle variance in signal energy.
- **Modeling:**
  - **GMM (Gaussian Mixture Models):**
    - A separate GMM is trained for each of the 31 classes (One-vs-Rest approach effectively).
    - **Configuration:** 8 components, Full Covariance Matrix, 20 EM iterations.
    - **Classification:** Maximum Log-Likelihood estimate across all 31 models.

### 2. Image Recognition Pipeline
*Face identification using structural shape analysis.*

- **Preprocessing:**
  - Conversion to grayscale.
  - Global Histogram Equalization (Contrast normalization).
  - Resizing to a fixed **64x64** resolution.
- **Feature Extraction:**
  - **HOG (Histogram of Oriented Gradients):** Captures edge directions and gradient structure (essential for facial features).
  - *Params:* 9 orientations, 8x8 pixels per cell, 2x2 cells per block.
- **Modeling:**
  - **SVM (Support Vector Machine):**
    - Trained on flattened HOG vectors.
    - **Configuration:** Linear Kernel, C=0.1. Tuned via 5-fold Cross-Validation (`GridSearchCV`).
    - **Pipeline:** `StandardScaler` -> `SVC`.

## 📊 Results

Both modalities achieved identical accuracy on the evaluation set, though they excel in different classes.

| Modality | Model | Accuracy | F1-Score |
|----------|-------|----------|----------|
| **Audio** | GMM | **72.58%** | 0.69 |
| **Image** | SVM + HOG | **72.58%** | 0.67 |

- **Audio GMM:** Higher precision (0.74) compared to Image SVM (0.66).
- **Image SVM:** Slightly more robust recall in specific classes.

## 📂 Project Structure

```
.
├── src/
│   ├── audio_MFCC_GMM.ipynb  # Audio pipeline: MFCC extraction -> GMM Training -> Eval
│   ├── image_SVM.ipynb       # Image pipeline: HOG extraction -> SVM Training -> Eval
│   ├── ikrlib.py             # Helper library for DSP and MFCC calculation
│   ├── modelsAudio/          # Serialized GMM models (.pkl)
│   └── modelsImage/          # Serialized SVM models (.pkl)
├── documentation.pdf         # Detailed project report
└── README.md
```

## 🚀 Setup & Usage

1. **Install Dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install numpy scipy matplotlib scikit-learn scikit-image opencv-python joblib
   ```

2. **Run the Notebooks:**
   The project is divided into two independent notebooks. Start Jupyter Lab or Notebook:
   ```bash
   jupyter notebook
   ```
   - Open `src/audio_MFCC_GMM.ipynb` for the audio pipeline.
   - Open `src/image_SVM.ipynb` for the image pipeline.

3. **Inference:**
   The notebooks include cells that automatically load the pretrained models from `src/modelsAudio/` and `src/modelsImage/` to evaluate the test dataset without needing to retrain.