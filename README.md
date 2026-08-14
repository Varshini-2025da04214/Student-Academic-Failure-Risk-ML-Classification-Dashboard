# Student Academic Failure Risk Classification

## 1. Problem Statement
The objective of this project is to build and compare multiple Machine Learning classification models to predict `Academic_Failure_Risk` based on student demographics, daily digital behaviors (social media and AI tool usage), physical/mental health indicators, social isolation, and academic performance metrics. Predicting academic failure risk early allows educators and counselors to deploy targeted interventions to support struggling students.

---

## 2. Dataset Description
* **Number of Rows**: 15,000
* **Number of Columns**: 14
* **Target Variable**: `Academic_Failure_Risk` (Binary: `1` indicates risk of academic failure, `0` indicates low/no risk)
* **Dataset Imbalance**: Highly imbalanced dataset with 14,100 (94.0%) low/no risk instances and 900 (6.0%) risk instances.
* **Predictor Features**:
  - `Age` (Numerical): Range 15-25 years.
  - `Gender` (Categorical): Male, Female, Non-binary.
  - `Education_Level` (Categorical): High School, College, University.
  - `Daily_Social_Media_Hours` (Numerical): Self-reported daily usage.
  - `Daily_AI_Tool_Usage_Hours` (Numerical): Self-reported daily usage.
  - `Sleep_Hours` (Numerical): Average nightly sleep hours.
  - `Physical_Activity_Hours` (Numerical): Average daily active hours.
  - `Mental_Health_Score` (Numerical): Scaled wellness indicator (0-100).
  - `Physical_Health_Score` (Numerical): Scaled wellness indicator (0-100).
  - `Social_Isolation_Score` (Numerical): Self-reported isolation index.
  - `Burnout_Level` (Categorical): Low, Moderate, High, Severe.
  - `Academic_Performance_Score` (Numerical): Grade point average scale (0-100).
* **Identifier**:
  - `Student_ID`: Dropped during preprocessing to ensure the models learn generalized patterns.
* **Source**: Cleaning was pre-applied to the provided assignment file.

---

## 3. Project Structure
The folder layout is structured as follows:

```text
student_health_ml_assignment/
│
├── app.py                     # Streamlit web application dashboard
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── test_data.csv              # Holdout test set (20%) for evaluation/predictions
│
├── data/
│   └── AI_SocialMedia_Student_Health_Dataset_clean.csv  # Cleaned source data
│
├── model/
│   ├── train_models.py          # Training, preprocessing & evaluation script
│   ├── logistic_regression.pkl  # Trained Logistic Regression pipeline
│   ├── decision_tree.pkl        # Trained Decision Tree pipeline
│   ├── knn.pkl                  # Trained KNN pipeline
│   ├── naive_bayes.pkl          # Trained Naive Bayes pipeline
│   └── random_forest.pkl        # Trained Random Forest pipeline
│
└── results/
    ├── model_comparison.csv     # Model evaluation table
    └── model_observations.csv   # Observations table
```

---

## 4. Data Preprocessing
To prevent data leakage, a strict preprocessing pipeline is constructed using scikit-learn's `ColumnTransformer` and `Pipeline` objects. The split is performed before fit operations.
* **ID Exclusion**: `Student_ID` is removed from predictors.
* **Train-Test Split**: `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` splits the data into 80% train and 20% test, preserving target balance.
* **Missing Value Handling**:
  - Numerical features are imputed using median values.
  - Categorical features are imputed using the mode (most frequent value).
* **Scaling**: Numerical features are scaled via `StandardScaler` to ensure distance-based models (KNN) and regularized models (Logistic Regression) perform correctly.
* **Encoding**: Categorical features (`Gender`, `Education_Level`, `Burnout_Level`) are transformed using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

---

## 5. Models Used
1. **Logistic Regression**: Linear classifier trained with an increased iteration limit (`max_iter=2000`) for convergence stability.
2. **Decision Tree**: Restricted to a maximum depth of 6 (`max_depth=6`) to avoid overfitting.
3. **KNN**: Distance-based voting model with $k=5$.
4. **Gaussian Naive Bayes**: Probabilistic classifier; uses a dense encoder output to avoid sparse matrix incompatibility.
5. **Random Forest (Ensemble)**: Multi-tree ensemble (`n_estimators=100`) using `class_weight='balanced'` to offset target class skewness.

---

## 6. Evaluation Metrics
For every model, 6 classification metrics are calculated on the 20% holdout test set (3,000 records):
* **Accuracy**: Proportion of overall correct predictions.
* **AUC-ROC (Area Under ROC Curve)**: Ability to distinguish between risk and no-risk groups.
* **Precision**: Percentage of predicted risks that are actual risks.
* **Recall**: Percentage of actual risks correctly identified.
* **F1-Score**: Harmonic mean of Precision and Recall.
* **Matthews Correlation Coefficient (MCC)**: High-quality metric for imbalanced classes ranging from -1 to +1.

---

## 7. Model Comparison
The actual calculated metrics on the 3,000 holdout records are:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **99.80%** | **0.9997** | **98.33%** | **98.33%** | **98.33%** | **0.9823** |
| Decision Tree | 99.50% | 0.9965 | 96.09% | 95.56% | 95.82% | 0.9556 |
| KNN | 97.83% | 0.9887 | 91.97% | 70.00% | 79.50% | 0.7919 |
| Naive Bayes | 93.50% | 0.9951 | 48.00% | 100.00% | 64.86% | 0.6684 |
| Random Forest (Ensemble) | 99.53% | 0.9997 | 96.63% | 95.56% | 96.09% | 0.9584 |

---

## 8. Observations
Performance observations based on the actual test evaluation:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Linear decision boundary model. Achieved Accuracy: 99.80%, F1: 98.33%, and AUC: 0.9997. It shows high stability and captures the classification boundary extremely well without overfitting signs. |
| **Decision Tree** | Non-linear model restricted to depth 6. Achieved Accuracy: 99.50%, F1: 95.82%, and Recall: 95.56%. It provides highly interpretable decisions, but has slightly more false predictions than the forest ensemble. |
| **KNN** | Distance-based classifier. Achieved Accuracy: 97.83%, F1: 79.50%, and MCC: 0.7919. Though numerical features were scaled, the class imbalance affects nearest-neighbor voting, yielding a lower recall (70%). |
| **Naive Bayes** | Probabilistic classifier assuming feature independence. Achieved Accuracy: 93.50%, F1: 64.86%, and Recall: 100.00%. In order to maximize recall on failure risk, it predicts a higher number of false positives, sacrificing precision. |
| **Random Forest (Ensemble)** | Ensemble method leveraging bootstrap aggregation. Achieved Accuracy: 99.53%, F1: 96.09%, and MCC: 0.9584. By incorporating 'balanced' class weights, it substantially mitigates target imbalance and yields a robust trade-off. |

---

## 9. Overall Winner
The best overall model is **Logistic Regression**. It achieves the highest scores across all metrics (F1-Score: **98.33%**, MCC: **0.9823**, AUC-ROC: **0.9997**). It classifies student academic risk with high confidence, minimal false positives, and minimal false negatives, closely followed by the Random Forest classifier.

---

## 10. Streamlit Application Features
The built interactive dashboard offers:
1. **Test CSV Upload**: User can upload any custom test CSV, or load the pre-partitioned holdout test set by default.
2. **Model Selection**: Switch dynamically between the 5 saved pipelines to see their specific predictions.
3. **Dataset Overview**: Interactive tables showing statistics, record previews, and target class distributions.
4. **Metrics Indicators**: Displays Accuracy, AUC, Precision, Recall, F1, and MCC scores using prominent color-coded metric cards.
5. **Confusion Matrix Heatmap**: Displays custom seaborn confusion matrix heatmaps.
6. **Detailed Classification Report**: View full support, precision, recall, and f1 breakdown.
7. **Global Model Comparison**: Explores the precomputed model metrics comparison and dynamically highlights the best performer.
8. **Batch Prediction Export**: Download predicted failure risk labels along with predicted failure probabilities as a CSV.

---

## 11. Installation and Running Locally
To run this project on your system:

1. Clone this repository:
   ```bash
    git clone https://github.com/Varshini-2025da04214/Student-Academic-Failure-Risk-ML-Classification-Dashboard.git
    cd Student-Academic-Failure-Risk-ML-Classification-Dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Retrain/Evaluate models (optional, models are pre-packaged):
   ```bash
   python model/train_models.py
   ```
4. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 12. GitHub Repository
[https://github.com/Varshini-2025da04214/Student-Academic-Failure-Risk-ML-Classification-Dashboard](https://github.com/Varshini-2025da04214/Student-Academic-Failure-Risk-ML-Classification-Dashboard)

---

## 13. Live Streamlit App
[https://student-academic-failure-risk-dashboard.streamlit.app/](https://student-academic-failure-risk-dashboard.streamlit.app/)
