"""Train and evaluate career recommendation models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "programming_skill",
    "math_skill",
    "communication_skill",
    "logic_score",
    "cgpa",
    "aptitude_score",
]
CATEGORICAL_FEATURES = ["interest_area"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "career"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Derive transformed feature names from fitted preprocessor."""
    names: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            cat_encoder: OneHotEncoder = trans
            cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
            names.extend(cat_names.tolist())
    return names


def get_feature_importances(model, feature_names: list[str]) -> list[tuple[str, float]]:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        return []

    pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return pairs


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "career_dataset.csv"
    model_dir = base_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y_raw = df[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
    }

    results = []
    trained_pipelines = {}

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for name, clf in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", clone(build_preprocessor())),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1 Score:  {metrics['F1 Score']:.4f}")
        print("Confusion Matrix:")
        print(metrics["Confusion Matrix"])

    comparison = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Accuracy": round(r["Accuracy"], 4),
                "Precision": round(r["Precision"], 4),
                "Recall": round(r["Recall"], 4),
                "F1 Score": round(r["F1 Score"], 4),
            }
            for r in results
        ]
    )
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(comparison.to_string(index=False))

    best_row = comparison.sort_values("F1 Score", ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_pipeline = trained_pipelines[best_name]
    best_preprocessor = best_pipeline.named_steps["preprocessor"]
    best_model = best_pipeline.named_steps["classifier"]
    feature_names = get_feature_names(best_preprocessor)

    print(f"\nBest model (by F1): {best_name}")

    joblib.dump(best_model, model_dir / "career_model.pkl")
    joblib.dump(best_preprocessor, model_dir / "preprocessor.pkl")
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")
    joblib.dump(FEATURE_COLUMNS, model_dir / "feature_names.pkl")

    importances = get_feature_importances(best_model, feature_names)
    if importances:
        print("\nFeature Importances (descending):")
        for feat, imp in importances:
            print(f"  {feat}: {imp:.4f}")

    y_test_labels = label_encoder.inverse_transform(y_test)
    best_pred_labels = label_encoder.inverse_transform(
        best_pipeline.predict(X_test)
    )
    print(f"\nClassification Report ({best_name}):")
    print(
        classification_report(
            y_test_labels,
            best_pred_labels,
            zero_division=0,
        )
    )

    print(f"\nSaved artifacts to {model_dir}/")
    print(f"  - career_model.pkl ({best_name})")
    print("  - preprocessor.pkl")
    print("  - label_encoder.pkl")
    print("  - feature_names.pkl")


if __name__ == "__main__":
    main()
