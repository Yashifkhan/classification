import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

data_path = os.path.join(
    project_dir,
    "data",
    "mobilePriceData.csv"
)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    5, 3,
    figsize=(16, 20)
)

numeric_cols = [
    "battery_power",
    "clock_speed",
    "fc",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time"
]
binary_cols = [
    "blue",
    "dual_sim",
    "four_g",
    "three_g",
    "touch_screen",
    "wifi"
]

# Input features
X = df.drop(columns=["price_range"])

# Target
y = df["price_range"]

print("Input shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget classes:")
print(y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    # "Decision Tree": DecisionTreeClassifier(
    #     random_state=42
    # ),

    # "Random Forest": RandomForestClassifier(
    #     n_estimators=300,
    #     random_state=42,
    #     n_jobs=-1
    # )
}

models["Logistic Regression"].fit(X_train, y_train)


mobile_data = {
    "battery_power": 1800,
    "blue": 1,
    "clock_speed": 2.5,
    "dual_sim": 1,
    "fc": 8,
    "four_g": 1,
    "int_memory": 64,
    "m_dep": 0.5,
    "mobile_wt": 140,
    "n_cores": 8,
    "pc": 16,
    "px_height": 1800,
    "px_width": 1920,
    "ram": 3500,
    "sc_h": 16,
    "sc_w": 9,
    "talk_time": 18,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

# Get trained model
lr_model = models["Logistic Regression"]

# Predict test data
y_pred = lr_model.predict(X_test)

# print("Actual values:")
# print(y_test.values[:10])

# print("\nPredicted values:")
# print(y_pred[:10])



import pandas as pd

# Convert dictionary into DataFrame
input_df = pd.DataFrame([mobile_data])

# Keep exactly the same feature order as training
input_df = input_df[X_train.columns]

# Get trained Logistic Regression model
lr_model = models["Logistic Regression"]

# Predict price range
prediction = lr_model.predict(input_df)

# Get probabilities
probabilities = lr_model.predict_proba(input_df)

print("Predicted Price Range:", prediction[0])

print("\nPrediction Probabilities:")

classes = lr_model.named_steps["classifier"].classes_

# for class_name, probability in zip(classes, probabilities[0]):
    # print(f"Class {class_name}: {probability * 100:.2f}%")

print("test commit :")