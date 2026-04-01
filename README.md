# Human Activity Recognition (HAR) using Sensor Data 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-f7931e)
![Colab](https://img.shields.io/badge/Google-Colab-f9ab00)

##  Overview
This repository contains the code and models for a 3-day Machine Learning sprint focused on **Human Activity Recognition (HAR)**. We classify physical activities (walking, sitting, standing, laying) by analyzing time-series data from smartphone accelerometers and gyroscopes. 

This project compares traditional machine learning baselines with deep learning architectures to handle complex sensor signals.

---

## 🛠️ Tech Stack
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Baseline Modeling:** `scikit-learn` (Random Forest / SVM)
* **Deep Learning:** `PyTorch` (1D-CNN / LSTM)
* **Environment:** Google Colab, GitHub

---

## Project Workflow (3-Day Sprint)

### **Day 1: Data Foundation & EDA** 
* **Exploratory Data Analysis:** Checking for class imbalances and visualizing sensor waveforms.
* **Pipeline:** Loading the UCI HAR Dataset and verifying data integrity (null checks, shape verification).
* **Visualization:** Plotting "Active" vs. "Stationary" signals to understand sensor patterns.

### **Day 2: Model Development (In Progress)** 
* **Baseline:** Training a Random Forest model to establish a performance benchmark.
* **Deep Learning:** Designing a PyTorch architecture (LSTM/1D-CNN) to capture temporal dependencies.

### **Day 3: Integration & Evaluation** 
* Merging preprocessing pipelines with model inference.
* Evaluating performance using Accuracy scores and **Confusion Matrices**.

---

##  Dataset
We use the **[UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)**. 

> **Note:** To keep the repository lightweight, the raw dataset is ignored via `.gitignore`. The notebook contains commands to download and unzip the data automatically in the Colab environment.

---

##  Project Structure
```bash
.
├── har.ipynb                # Day 1: Data Loading & EDA (Partner)
├── README.md                # Project documentation
└── .gitignore               # Ignoring large dataset files
