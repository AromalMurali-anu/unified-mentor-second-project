🩺 Lung Cancer Prediction

This repository contains a machine learning project that predicts the likelihood of lung cancer based on patient medical and lifestyle data. The notebook performs data preprocessing, exploratory analysis, model training, and evaluation using multiple classification algorithms.

📘 Project Overview

Lung cancer is one of the most serious diseases worldwide, and early prediction can significantly improve patient survival rates.
This project builds a predictive model using clinical and demographic data to classify patients as being at high or low risk of lung cancer.

The notebook includes:

Data cleaning and preprocessing

Feature encoding and scaling

Model training using multiple algorithms

Model evaluation and comparison

Model persistence using joblib

🧩 Features of the Notebook

Data Handling

Loads and cleans the dataset (dataset_med.csv)

Handles missing values and duplicates

Performs feature type inspection and descriptive statistics

Preprocessing Pipeline

Encodes categorical variables using OneHotEncoder

Scales numerical features using StandardScaler

Uses ColumnTransformer and Pipeline for efficient preprocessing

Model Training

Trains multiple classifiers:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Support Vector Machine (SVM)

Splits data using train_test_split for validation

Model Evaluation

Evaluates model performance using accuracy, precision, recall, and F1-score

Visualises results with plots using matplotlib

Model Saving

Saves the best-performing model using joblib for deployment or future use

🧠 Algorithms Used
Algorithm	Description	Library
Logistic Regression	A statistical method for binary classification	sklearn.linear_model
Random Forest	Ensemble method using decision trees	sklearn.ensemble
Gradient Boosting	Boosting-based ensemble method	sklearn.ensemble
Support Vector Machine	Margin-based classifier	sklearn.svm
📊 Dependencies

To run the notebook, install the following dependencies:

pip install pandas numpy matplotlib scikit-learn joblib


Optionally, you can use a virtual environment:

python -m venv env
source env/bin/activate   # (Linux/Mac)
env\Scripts\activate      # (Windows)

🧪 How to Run

Clone this repository:

git clone https://github.com/<your-username>/Lung_Cancer_Prediction.git
cd Lung_Cancer_Prediction


Open the Jupyter Notebook:

jupyter notebook Lung_Cancer_Prediction.ipynb


Run all cells in sequence to reproduce the results.

Ensure the dataset file (dataset_med.csv) is in the same directory as the notebook.

📈 Example Output

The notebook outputs:

Summary statistics of the dataset

Model comparison metrics

Graphs and heatmaps for feature correlations

The best-performing model saved as a .pkl file

💾 Model Export

The final trained model can be saved and reused:

import joblib
joblib.dump(best_model, "lung_cancer_model.pkl")


To load the model later:

model = joblib.load("lung_cancer_model.pkl")

🧭 Future Enhancements

Integrate with Flask or Streamlit for real-time predictions

Use deep learning models for higher accuracy

Perform feature importance and SHAP explainability analysis

Expand dataset for multi-class cancer prediction
