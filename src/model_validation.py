import mlflow
import pandas as pd

import config
from feature_engineering import prepare_features
from model_registry import load_model_artifacts

MODEL_URI = "models:/workspace.default.employee_attrition_model@qa"


def validate_qa_model(test_file_path):
    print("\n" + "=" * 80)
    print("QA MODEL VALIDATION")
    print("=" * 80)

    # 1. Load model from Unity Catalog using QA alias
    print("\nLoading model from Unity Catalog...")
    print(f"Model URI: {MODEL_URI}")

    model = mlflow.xgboost.load_model(MODEL_URI)

    print("Model loaded successfully")
    print(f"Model type: {type(model).__name__}")

    # 2. Load preprocessing artifacts
    print("\nLoading preprocessing artifacts...")

    artifacts = load_model_artifacts()

    encoders = artifacts["encoders"]
    feature_names = artifacts["feature_names"]

    print(f"Features expected by model: {len(feature_names)}")

    # 3. Prepare test data using saved encoders
    print("\nPreparing test data...")

    X_test, _, _, _, _, employee_ids = prepare_features(
        test_file_path,
        encoders=encoders,
        fit=False,
        select=False,
    )

    # 4. Align features with training features
    X_test = X_test[feature_names]

    print(f"Test records: {len(X_test):,}")
    print(f"Features: {len(X_test.columns)}")

    # 5. Generate predictions
    print("\nGenerating predictions...")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # 6. Validate predictions
    if len(predictions) == 0:
        raise ValueError(
            "QA validation failed: no predictions generated."
        )

    if len(predictions) != len(X_test):
        raise ValueError(
            "QA validation failed: prediction count does not match "
            "test record count."
        )

    if probabilities.shape[0] != len(X_test):
        raise ValueError(
            "QA validation failed: probability count does not match "
            "test record count."
        )

    # 7. Validate probability values
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(
            "QA validation failed: prediction probabilities are "
            "outside the range [0, 1]."
        )

    # 8. Validate Employee IDs
    if employee_ids is None:
        raise ValueError(
            "QA validation failed: Employee IDs were not available."
        )

    if len(employee_ids) != len(predictions):
        raise ValueError(
            "QA validation failed: Employee ID count does not match "
            "prediction count."
        )

    # 9. Prediction distribution
    prediction_distribution = pd.Series(predictions).value_counts()

    print("\nPrediction distribution:")
    print(prediction_distribution)

    print("\n" + "=" * 80)
    print("QA MODEL VALIDATION PASSED")
    print("=" * 80)

    print(f"Predictions generated: {len(predictions):,}")
    print(f"Features validated: {len(X_test.columns):,}")


if __name__ == "__main__":
    validate_qa_model(config.TEST_PATH)
