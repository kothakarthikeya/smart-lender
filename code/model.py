import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -------------------------------
# Create folders
# -------------------------------
os.makedirs("static/images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("dataset.csv")

print("=" * 60)
print("SMART LENDER - LOAN PREDICTION")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------
# Drop Loan_ID
# -------------------------------
df.drop("Loan_ID", axis=1, inplace=True)

# -------------------------------
# Separate Features and Target
# -------------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Convert target to numeric
y = y.map({"N": 0, "Y": 1})

# -------------------------------
# Column Types
# -------------------------------
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Features")
print(categorical_features)

print("\nNumerical Features")
print(numerical_features)

# -------------------------------
# Preprocessing Pipelines
# -------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])
from sklearn.metrics import accuracy_score

# -------------------------------
# Models
# -------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

results = []

best_pipeline = None
best_model_name = ""
best_accuracy = 0

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append([
        name,
        round(train_acc,4),
        round(test_acc,4)
    ])

    print(f"\n{name}")
    print("-"*40)
    print(f"Training Accuracy : {train_acc:.4f}")
    print(f"Testing Accuracy  : {test_acc:.4f}")

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_pipeline = pipeline
        best_model_name = name

# -------------------------------
# Comparison Table
# -------------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train Accuracy",
        "Test Accuracy"
    ]
)

print("\n")
print(results_df)

print("\nBest Model :", best_model_name)
print("Best Accuracy :", round(best_accuracy*100,2), "%")
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------
# Predictions
# -------------------------------------
y_pred = best_pipeline.predict(X_test)

print("\n")
print("="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(classification_report(y_test, y_pred))

# -------------------------------------
# Confusion Matrix
# -------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected","Approved"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "static/images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# -------------------------------------
# Correlation Heatmap
# -------------------------------------

numeric_df = df.copy()

numeric_df["Loan_Status"] = numeric_df["Loan_Status"].map({
    "N":0,
    "Y":1
})

numeric_df = numeric_df.select_dtypes(include=["number"])

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "static/images/heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# -------------------------------------
# Feature Importance
# -------------------------------------

if best_model_name in ["Random Forest","Decision Tree"]:

    clf = best_pipeline.named_steps["classifier"]

    X_processed = best_pipeline.named_steps["preprocessor"].transform(X_train)

    feature_names = best_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = clf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature":feature_names,
        "Importance":importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    plt.figure(figsize=(10,6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.title("Top 10 Important Features")

    plt.tight_layout()

    plt.savefig(
        "static/images/feature_importance.png",
        dpi=300
    )

    plt.close()
    joblib.dump(
    best_pipeline,
    "models/loan_model.pkl"
)

print("\nBest Model Saved Successfully")