
# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 2. LOAD DATASET
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

data_path = os.path.join(
    project_dir,
    "data",
    "bank-additional-full.csv"
)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path, sep=";")

# print("\n========== DATASET LOADED ==========")

# print("Dataset Shape:", df.shape)
# print(df.head())


# ==========================================
# 3. CHECK AND REMOVE EXACT DUPLICATES
# ==========================================



# Assumption: exact duplicate rows are redundant records.
# Set this to False if you have not verified that assumption.
DROP_EXACT_DUPLICATES = True

if DROP_EXACT_DUPLICATES:
    df = df.drop_duplicates().reset_index(drop=True)

# print("Dataset Shape After Duplicate Check:", df.shape)


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

# Create a copy
data = df.copy()

# Target variable
y = data["y"].map({
    "no": 0,
    "yes": 1
})

# Validate target values
if y.isnull().any():
    raise ValueError("Unexpected or missing target values found!")

# Features
# Remove target and duration
X = data.drop(columns=["y", "duration"]).copy()

# print("\n========== X AND Y SEPARATION ==========")

# print("Features Shape:", X.shape)
# print("Target Shape:", y.shape)

# print("\nTarget Distribution:")
# print(y.value_counts())


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

# Create an indicator for previous contact
X["was_previously_contacted"] = (
    X["pdays"] != 999
).astype(int)

# Convert special 999 values into missing values
X["pdays"] = X["pdays"].replace(999, np.nan)

# print("\n========== FEATURE ENGINEERING ==========")

# print("Duration Exists:", "duration" in X.columns)

# print("\nPdays Analysis:")
# print(X[["pdays", "was_previously_contacted"]].head())

# print("New Features Shape:", X.shape)


# ==========================================
# 6. IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# print("\n========== FEATURE TYPES ==========")

# print("Numerical Features:", numeric_features)
# print("Categorical Features:", categorical_features)

# print("Numerical Count:", len(numeric_features))
# print("Categorical Count:", len(categorical_features))


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




# ==========================================
# 8. NUMERICAL PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler())
])


# ==========================================
# 9. CATEGORICAL PREPROCESSING
# ==========================================

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("onehot", OneHotEncoder(
        handle_unknown="ignore"
    ))
])


# ==========================================
# 10. COLUMN TRANSFORMER
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 11. LOGISTIC REGRESSION PIPELINE
# ==========================================

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=None
    ))
])


# ==========================================
# 12. TRAIN MODEL
# ==========================================

# print("\n========== MODEL TRAINING ==========")
logistic_model.fit(X_train, y_train)




y_pred = logistic_model.predict(X_test)



# ==========================================
# 17. FINAL SUMMARY
# ==========================================

sample_data = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": np.nan,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": -1.8,
    "cons.price.idx": 92.893,
    "cons.conf.idx": -46.2,
    "euribor3m": 1.299,
    "nr.employed": 5099.1,
    "was_previously_contacted": 0
}

# print("\n========== TRAINING COMPLETED ==========")


# Convert dictionary into DataFrame
test_df = pd.DataFrame([sample_data])

# Generate prediction
# prediction = logistic_model.predict(test_df)[0]

# # Get probability
# probability = logistic_model.predict_proba(test_df)[0]

# # Display results
# print("\n===== CUSTOMER PREDICTION =====")

# print("Predicted Class:", prediction)

# print("No Subscription Probability:", round(probability[0] * 100, 2), "%")
# print("Yes Subscription Probability:", round(probability[1] * 100, 2), "%")

# if prediction == 1:
#     print("Prediction: Customer may subscribe.")
# else:
#     print("Prediction: Customer may not subscribe.")

customer_data = {
    "age": 52,
    "job": "technician",
    "marital": "married",
    "education": "high.school",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "telephone",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 2,
    "pdays": np.nan,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191,
    "was_previously_contacted": 0
}

# Convert dictionary into DataFrame
customer_df = pd.DataFrame([customer_data])



# Generate prediction
prediction = logistic_model.predict(customer_df)[0]

# Generate probabilities
probabilities = logistic_model.predict_proba(customer_df)[0]

# Convert prediction to readable label
predicted_result = "yes" if prediction == 1 else "no"

# Get probability of each class
classes = logistic_model.named_steps["classifier"].classes_
actual_result = "no"

probability_no = probabilities[list(classes).index(0)] * 100
probability_yes = probabilities[list(classes).index(1)] * 100

print("\n========== CUSTOMER PREDICTION ==========")

print("Predicted Result:", predicted_result)

print(f"No Subscription Probability: {probability_no:.2f}%")
print(f"Yes Subscription Probability: {probability_yes:.2f}%")

print("\n========== ACTUAL VS PREDICTED ==========")

# print("Actual Result:", actual_result)
print("Predicted Result:", predicted_result)

if actual_result == predicted_result:
    print("Prediction: Correct")
else:
    print("Prediction: Incorrect")
