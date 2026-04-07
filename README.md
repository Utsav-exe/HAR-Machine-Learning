# Human Activity Recognition (HAR) via Smartphone Sensor Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-f7931e)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458)
![Colab](https://img.shields.io/badge/Google-Colab-f9ab00)

## Project Overview
This repository contains a complete end-to-end Machine Learning pipeline designed to classify human physical activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying) using continuous time-series data captured from smartphone accelerometers and gyroscopes. 

Developed during a rapid 3-day collaborative sprint, this project demonstrates a mature Data Science workflow: starting with exploratory data analysis (EDA), establishing a traditional machine learning baseline, and ultimately engineering a custom Deep Neural Network to maximize predictive accuracy.

## The Dataset & Real-World Application
This model utilizes the **UCI HAR Dataset**, acting as the core "engine" similar to the algorithms running inside modern smartwatches (e.g., Apple Watch, Fitbit) for health monitoring and workout tracking. 
* **Input:** 561 statistical features extracted from 2.56-second sliding windows of 3-axial linear acceleration and 3-axial angular velocity.
* **Output:** Multiclass classification across 6 distinct activity states.

## Architecture & Methodology
We divided the project into a competitive architecture, pitting a traditional ML ensemble against a Deep Learning approach.

1. **Data Pipeline & EDA:** Downloaded raw signals, verified class balance, and visualized sensor waveform variance (e.g., high-variance walking waves vs. low-variance resting flatlines).
2. **The Baseline (Scikit-Learn):** Trained a **Random Forest Classifier** (100 estimators) to establish a performance floor. Random Forests are highly interpretable and robust against overfitting on tabular data.
3. **The Deep Learning Challenger (PyTorch):** Engineered a **Deep Feedforward Neural Network (DNN)** with PyTorch. 
   * **Architecture:** 561 Input Nodes → 256 Hidden (ReLU) → Dropout (0.3) → 128 Hidden (ReLU) → 6 Output Nodes.
   * **Optimization:** Adam Optimizer (lr=0.001) and CrossEntropyLoss, trained over 50 epochs.

## Results & Evaluation

| Model | Accuracy | Training Time | Complexity |
| :--- | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 92.57% | Instant | Low |
| **PyTorch Neural Network** | **94.16%** | Moderate (50 Epochs) | High |

### Key Data Science Insight: The "Static Posture" Problem
While the PyTorch DNN achieved an impressive **94.16% overall accuracy**, a deep dive into the Confusion Matrix reveals a shared vulnerability in both models: **Differentiating between `SITTING` and `STANDING`.**

* **The "Why":** Because the smartphone is positioned in a static pocket, the microscopic gravitational acceleration profiles for sitting and standing are virtually indistinguishable to the sensors once the transition movement is complete. 
* **Future Work:** To break this plateau, future iterations would require engineering specific temporal features that capture the *transition* (the act of standing up or sitting down) rather than relying solely on static 2.56-second windows.

## Repository Structure
This project was built collaboratively using branch-based version control.
* `har.ipynb`: Shakshitha's initial branch for Data Retrieval, EDA, and Baseline Modeling.
* `preprocessing.ipynb`: Utsav's initial branch for PyTorch tensor formatting and architecture design.
* **`Final_HAR_Showdown.ipynb`**: The master integration notebook. **Start here.** This notebook combines the entire pipeline from data ingestion to the final dual-model showdown.

## How to Run
This project is fully self-contained and requires zero local file downloading.
1. Open `Final_HAR_Showdown.ipynb` in Google Colab.
2. Click **Runtime > Run all**. 
3. The script will automatically fetch the UCI dataset via `wget`, preprocess the tensors, train both models, and output the final confusion matrices.

## Contributors
* **Utsav Saxena** ([@Utsav-exe](https://github.com/Utsav-exe)) - *Deep Learning Architecture & PyTorch Integration*
* **Shakshitha M** ([@shagitjams](https://github.com/shagitjams)) - *Data Pipeline, Exploratory Analysis, & Baseline Modeling*
