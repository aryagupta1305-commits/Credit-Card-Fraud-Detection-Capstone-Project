Project Title: Credit Card Fraud Detection Capstone Project. 

Objective: 
- Identify fraudulent credit card transactions using machine learning.
- Compare multiple modeling approaches to handle severe class imbalance.

Dataset Overview:
- The data contains anonymized credit card transactions labeled as fraud (1) or non-fraud (0).
- The features include Time, Amount, and PCA-transformed variables V1 through V28.
- The dataset is highly imbalanced, with fraud making up only 0.1727% of all transactions.

Methodology:
- Exploratory Data Analysis (EDA): Analyzes class distributions and utilizes a correlation heatmap to identify the top features correlated with the fraud class.
- Data Preprocessing: Applies standard scaling to the Amount and Time variables before dropping the original columns.
- Train-Test Split: Utilizes a stratified split to preserve the true fraud ratio across both training and testing datasets.
- Handling Class Imbalance: Compares algorithmic penalization (class_weight='balanced' and scale_pos_weight) against SMOTE synthetic oversampling applied exclusively to the training set.
- Model Training: Trains Logistic Regression, Random Forest, and XGBoost classifiers under both imbalance approaches.

Evaluation & Results:
- Models are evaluated using metrics suited for imbalanced data, specifically Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- The XGBoost model trained with weighted classes achieved the best overall performance with a ROC-AUC of 0.9772 and a PR-AUC of 0.8655.
- The Random Forest model trained with SMOTE achieved perfect precision (1.00) for non-fraud but struggled with precision (0.44) for the fraud class.

Dependencies: The environment setup instructions should list all libraries imported in the notebook.
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost
- imbalanced-learn   
