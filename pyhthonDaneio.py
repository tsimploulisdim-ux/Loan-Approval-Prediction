import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Δημιουργία φακέλου εικόνων

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

# Αναπαραγωγιμότητα

np.random.seed(42)

# Φόρτωση δεδομένων

df = pd.read_csv(r"C:\Users\Gekyume\Documents\Loan-Approval-Prediction\data\train_u6lujuX_CVtuZ9i.csv")

# Feature Engineering

df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

df["DebtIncomeRatio"] = (
    df["LoanAmount"] /
    (df["TotalIncome"] + 1)
)

# Κωδικοποίηση μεταβλητής στόχου

df["Loan_Status_Encoded"] = (
    df["Loan_Status"]
    .map({"N": 0, "Y": 1})
)

# Features και Target

X = df.drop(
    columns=[
        "Loan_Status",
        "Loan_Status_Encoded",
        "Loan_ID"
    ]
)

y = df["Loan_Status_Encoded"]

# Αριθμητικές μεταβλητές

numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
    "DebtIncomeRatio"
]

# Κατηγορικές μεταβλητές

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

# Προεπεξεργασία

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numerical_cols
    ),
    (
        "cat",
        categorical_transformer,
        categorical_cols
    )
])

# Διαχωρισμός δεδομένων

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

# Μοντέλα

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            random_state=42
        )
}

# Εκπαίδευση και αξιολόγηση

results = []

plt.figure(figsize=(8, 6))

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        auc
    ])

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={auc:.3f}"
    )

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

# ROC Curve

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig(
    os.path.join(IMAGES_DIR, "roc_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Πίνακας αποτελεσμάτων

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ]
)

results_df = results_df.round(4)

print(results_df)

results_df.to_csv(
    os.path.join(BASE_DIR, "results.csv"),
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

# Σημαντικότητα χαρακτηριστικών

rf = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
    )
])

rf.fit(X_train, y_train)

feature_names = (
    numerical_cols +
    list(
        rf.named_steps[
            "preprocessor"
        ]
        .named_transformers_[
            "cat"
        ]
        .named_steps[
            "encoder"
        ]
        .get_feature_names_out(
            categorical_cols
        )
    )
)

importance = (
    rf.named_steps[
        "classifier"
    ]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")

plt.tight_layout()

plt.savefig(
    os.path.join(IMAGES_DIR, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Confusion Matrix

best_model = rf

y_pred = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(IMAGES_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nProject completed successfully.")
