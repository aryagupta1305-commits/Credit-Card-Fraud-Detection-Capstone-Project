# Credit Card Fraud Detection - Capstone Project

## 📌 Project Overview
This capstone project focuses on identifying fraudulent credit card transactions using machine learning. Since fraud detection datasets inherently suffer from severe class imbalance, the primary objective is to compare different modeling approaches and resampling techniques to effectively detect the minority class (fraud) without falsely flagging legitimate transactions.

## 📊 Dataset Details
* **Features:** Contains anonymized PCA-transformed features (`V1` to `V28`), along with `Time` and `Amount`.
* **Target Variable:** `Class` (1 for fraud, 0 for non-fraud).
* **Imbalance Challenge:** The dataset is highly imbalanced, with fraudulent transactions making up only **0.1727%** of the entire data (492 frauds out of 284,807 transactions).

## ⚙️ Methodology & Workflow
1. **Exploratory Data Analysis (EDA):** Visualized class distributions and used correlation heatmaps to identify the top features correlated with the fraud class (e.g., V17, V14, V12).
2. **Data Preprocessing:** Applied standard scaling to the `Amount` and `Time` features to ensure uniformity across all independent variables.
3. **Train-Test Split:** Used a stratified split to maintain the exact fraud-to-non-fraud ratio in both training and testing sets.
4. **Handling Class Imbalance (Two Approaches):**
   * **Algorithmic Penalization:** Using `class_weight='balanced'` for Logistic Regression and Random Forest, and `scale_pos_weight` for XGBoost.
   * **Data Resampling:** Using **SMOTE** (Synthetic Minority Over-sampling Technique) exclusively on the training set to synthetically balance the classes.

## 🤖 Models Evaluated
Three machine learning algorithms were trained and compared under both imbalance-handling approaches:
* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier

## 📈 Key Results & Evaluation
Models were evaluated using metrics suited for imbalanced datasets, specifically **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

* **Best Performing Model:** **XGBoost (weighted)** achieved the best overall balance with a **ROC-AUC of 0.9772** and a **PR-AUC of 0.8655**.
* **SMOTE Observations:** While SMOTE improved recall for certain models, it struggled with precision (e.g., Random Forest with SMOTE achieved a precision of only 0.44 for the fraud class, leading to a higher false-positive rate).
* Algorithmic class weighting generally outperformed SMOTE in maintaining a high Precision-Recall Area Under Curve (PR-AUC) on the unseen test set.

## 🛠️ Tools & Libraries Used
* **Programming Language:** Python
* **Data Manipulation & Analysis:** pandas, numpy
* **Data Visualization:** matplotlib, seaborn
* **Machine Learning:** scikit-learn, xgboost, imbalanced-learn (SMOTE)

## 📂 Repository Structure
* `data/` - Placeholder for the dataset (add `.gitignore` if the file exceeds GitHub limits).
* `notebooks/` - Contains the main Jupyter Notebook (`Credit_Card_Fraud_Detection_Capstone.ipynb`) with the complete code, EDA, and model training.
* `requirements.txt` - Required Python packages and dependencies to run the notebook.
