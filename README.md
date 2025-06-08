# 📊 Term Deposit Subscription Prediction

This project involves building a classification model to predict whether a bank client will subscribe to a term deposit. The solution leverages various machine learning models such as Logistic Regression, Random Forest, XGBoost, and Naive Bayes, and includes data preprocessing, exploratory data analysis, SMOTE for class balancing, and hyperparameter tuning.

---

## 📁 Dataset Overview

- **Records**: 45,211
- **Features**: 16 independent + 1 target (`y`)
- **Target Variable**: `y` (Binary: "yes", "no")
- **Imbalanced Dataset**: ~12% "yes" and ~88% "no"

---

## 🧪 Objectives

- Understand the key factors influencing customer subscription.
- Handle data imbalance using SMOTE.
- Build and evaluate classification models.
- Tune hyperparameters for optimal performance.
- Select the best model based on accuracy, precision, recall, and F1 score.

---

## 🔧 Tech Stack

- **Language**: Python 3.x  
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`
  - `xgboost`
  - `imblearn` (SMOTE)
- **Model Evaluation**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

---

## 📊 Exploratory Data Analysis

- Visualized categorical features against target
- Analyzed distributions of numerical features
- Observed clear influence of:
  - Contact type
  - Poutcome (previous outcome)
  - Duration
  - Age groups

---

## ⚙️ Preprocessing Steps

- Removed duplicates (none found)
- Handled outliers using IQR method
- Encoded categorical variables (LabelEncoder)
- Scaled numerical values using `StandardScaler`
- Train-test split (80-20)
- Applied **SMOTE** on training data

---

## 🤖 Models Trained

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| Medium   | High      | Medium | Balanced |
| Random Forest       | High     | High      | High   | High     |
| XGBoost             | ✅ **Best** | High  | Good   | Good     |
| Gaussian Naive Bayes| Low      | Poor      | Poor   | Poor     |

✅ Final Model Selected: **XGBoost**

---

## 🔍 Hyperparameter Tuning

- Method: `GridSearchCV`
- Best Parameters for XGBoost:
  - `n_estimators`: 100
  - `max_depth`: 5
  - `learning_rate`: 0.1
- Cross-validation (5-fold)
- Scoring: F1 Score

---

## ✅ Final Model Results (XGBoost)

- **Train Accuracy**: 96.8%
- **Test Accuracy**: 90.0%

**Class-wise Performance**:
- **Yes**:
  - Precision: 0.57
  - Recall: 0.58
  - F1-Score: 0.58
- **No**:
  - Precision: 0.94
  - Recall: 0.94
  - F1-Score: 0.94

---

## 📌 Key Insights

- Customers contacted via **cellular** and with a **successful previous outcome** are more likely to subscribe.
- **Call duration** is one of the strongest predictors.
- **Imbalanced dataset** handled effectively with **SMOTE**.
- **XGBoost** outperforms others in handling imbalance and accuracy.

---

## 🚀 Future Enhancements

- Try ensemble stacking
- Integrate with a dashboard (e.g., Streamlit or Dash)
- Use SHAP for model interpretability
- Automate retraining pipeline

---
