# =============================================================================
# ATTRITION ML PIPELINE - END-TO-END ORCHESTRATOR
# =============================================================================
"""
End-to-end employee attrition ML pipeline.

Pipeline:
1. Feature Engineering
2. Model Training
3. Model Registry / MLflow
4. Batch Scoring
"""

import config

from feature_engineering import prepare_features
from model_training import train_model_pipeline
from model_registry import register_model_pipeline
from scoring import score_pipeline


def main():

    print("\n" + "=" * 80)
    print("STARTING EMPLOYEE ATTRITION ML PIPELINE")
    print("=" * 80)

    # =========================================================================
    # 1. FEATURE ENGINEERING
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: FEATURE ENGINEERING")
    print("=" * 80)

    X_train, y_train, encoders, features, removed, ids = prepare_features(
        config.TRAIN_PATH,
        fit=True,
        select=True
    )

    print("\n✅ Feature Engineering completed")
    print(f"   Training data: {X_train.shape}")
    print(f"   Features: {len(features)}")

    # =========================================================================
    # 2. MODEL TRAINING
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)

    model, metrics, data_splits, best_params = train_model_pipeline(
        X_train,
        y_train
    )

    print("\n✅ Model Training completed")
    print(f"   Model: {type(model).__name__}")
    print(f"   Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")

    # =========================================================================
    # 3. MODEL REGISTRY / MLFLOW
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: MODEL REGISTRY / MLFLOW")
    print("=" * 80)

    run_id, artifacts = register_model_pipeline(
        model=model,
        X_train=data_splits["X_train"],
        y_train=data_splits["y_train"],
        metrics=metrics,
        best_params=best_params,
        encoders=encoders,
        feature_names=features,
        removed_features=removed
    )

    print("\n✅ Model Registry completed")
    print(f"   MLflow Run ID: {run_id}")

    # =========================================================================
    # 4. SCORING
    # =========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: BATCH SCORING")
    print("=" * 80)

    predictions, prediction_summary = score_pipeline(
        config.TEST_PATH,
        save_output=True,
        explain=False
    )

    print("\n✅ Scoring completed")

    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================

    print("\n" + "=" * 80)
    print("✅ EMPLOYEE ATTRITION ML PIPELINE COMPLETED")
    print("=" * 80)

    print("\nPipeline Summary:")
    print(f"   MLflow Run ID:     {run_id}")
    print(f"   Features:          {len(features)}")
    print(f"   Test ROC-AUC:      {metrics['test']['roc_auc']:.4f}")
    print(f"   Prediction Output: {config.OUTPUTS_DIR}/predictions.csv")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "best_params": best_params,
        "artifacts": artifacts,
        "predictions": predictions,
        "prediction_summary": prediction_summary
    }


if __name__ == "__main__":
    main()