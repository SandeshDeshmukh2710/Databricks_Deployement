# =============================================================================
# MODEL REGISTRY - MLflow logging and artifact management
# =============================================================================
"""
Model Registry Pipeline:
1. Log model and metrics to MLflow
2. Save model artifacts locally
3. Register model in Unity Catalog (optional)
4. Save preprocessing artifacts for inference
"""

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import json
import pickle
import os
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

import config

# =============================================================================
# MLFLOW LOGGING
# =============================================================================

def log_model_to_mlflow(model, X_train, y_train, metrics, best_params, 
                        feature_names, run_name="Optimized_XGBoost"):
    """
    Log trained model and metrics to MLflow
    
    Args:
        model: Trained XGBoost model
        X_train: Training features (for signature inference)
        y_train: Training target
        metrics (dict): Performance metrics
        best_params (dict): Best hyperparameters
        feature_names (list): List of feature names
        run_name (str): MLflow run name
    
    Returns:
        str: MLflow run ID
    """
    print("\n" + "="*80)
    print("MLFLOW MODEL LOGGING")
    print("="*80)

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Shared/Employee_Attrition_Prediction")

    with mlflow.start_run(run_name=run_name) as run:
        # Log hyperparameters
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "XGBoost_Optimized")
        mlflow.log_param("n_features", len(feature_names))
        
        # Log metrics for all datasets
        for split, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                mlflow.log_metric(f"{split}_{metric_name}", value)
        
        # Infer model signature
        signature = infer_signature(X_train, model.predict_proba(X_train))
        
        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3)
        )
        
        # Log feature names as artifact
        with open("feature_names.json", "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact("feature_names.json")
        os.remove("feature_names.json")
        
        run_id = run.info.run_id
        
        print(f"\n✅ Model logged to MLflow")
        print(f"   Run ID: {run_id}")
        print(f"   Run Name: {run_name}")
    
    return run_id


# =============================================================================
# SAVE MODEL ARTIFACTS
# =============================================================================

def save_model_artifacts(model, encoders, feature_names, removed_features, 
                         target_mapping=config.TARGET_MAPPING):
    """
    Save all model artifacts required for inference
    
    Args:
        model: Trained XGBoost model
        encoders (dict): Label encoders for categorical features
        feature_names (list): Final feature names used in model
        removed_features (list): Features removed during selection
        target_mapping (dict): Target encoding mapping
    
    Returns:
        dict: Paths to saved artifacts
    """
    print("\n" + "="*80)
    print("MODEL PERSISTENCE - SAVING ARTIFACTS")
    print("="*80)
    
    artifact_paths = {}
    
    # 1. Save XGBoost model
    model_path = f"{config.MODELS_DIR}/xgboost_attrition_model.json"
    model.save_model(model_path)
    artifact_paths['model'] = model_path
    print(f"✅ Model saved: {model_path}")
    
    # 2. Save target mapping
    target_path = f"{config.MODELS_DIR}/target_mapping.json"
    with open(target_path, 'w') as f:
        json.dump(target_mapping, f, indent=2)
    artifact_paths['target_mapping'] = target_path
    print(f"✅ Target mapping saved: {target_path}")
    
    # 3. Save label encoders
    encoders_path = f"{config.MODELS_DIR}/label_encoders.pkl"
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    artifact_paths['encoders'] = encoders_path
    print(f"✅ Label encoders saved: {encoders_path}")
    
    # 4. Save feature names
    features_path = f"{config.MODELS_DIR}/feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    artifact_paths['features'] = features_path
    print(f"✅ Feature names saved: {features_path}")
    
    # 5. Save removed features
    removed_path = f"{config.MODELS_DIR}/removed_features.json"
    with open(removed_path, 'w') as f:
        json.dump(removed_features, f, indent=2)
    artifact_paths['removed_features'] = removed_path
    print(f"✅ Removed features saved: {removed_path}")
    
    # 6. Save model metadata
    metadata = {
        'model_type': 'XGBoost Classifier',
        'task': 'Binary Classification',
        'target': 'Employee Attrition',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'removed_features': removed_features,
        'target_mapping': target_mapping,
        'artifacts': artifact_paths
    }
    metadata_path = f"{config.MODELS_DIR}/model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    artifact_paths['metadata'] = metadata_path
    print(f"✅ Model metadata saved: {metadata_path}")
    
    print(f"\n✅ All artifacts saved to: {config.MODELS_DIR}")
    
    return artifact_paths


# =============================================================================
# LOAD MODEL ARTIFACTS
# =============================================================================

def load_model_artifacts():
    """
    Load all saved model artifacts for inference
    
    Returns:
        dict: Dictionary containing all loaded artifacts
    """
    print("\n" + "="*80)
    print("LOADING MODEL ARTIFACTS")
    print("="*80)
    
    artifacts = {}
    
    # 1. Load XGBoost model
    model_path = f"{config.MODELS_DIR}/xgboost_attrition_model.json"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    artifacts['model'] = model
    print(f"✅ Model loaded: {model_path}")
    
    # 2. Load target mapping
    target_path = f"{config.MODELS_DIR}/target_mapping.json"
    with open(target_path, 'r') as f:
        artifacts['target_mapping'] = json.load(f)
    print(f"✅ Target mapping loaded: {target_path}")
    
    # 3. Load label encoders
    encoders_path = f"{config.MODELS_DIR}/label_encoders.pkl"
    with open(encoders_path, 'rb') as f:
        artifacts['encoders'] = pickle.load(f)
    print(f"✅ Label encoders loaded: {encoders_path}")
    
    # 4. Load feature names
    features_path = f"{config.MODELS_DIR}/feature_names.json"
    with open(features_path, 'r') as f:
        artifacts['feature_names'] = json.load(f)
    print(f"✅ Feature names loaded: {features_path}")
    
    # 5. Load model metadata
    metadata_path = f"{config.MODELS_DIR}/model_metadata.json"
    with open(metadata_path, 'r') as f:
        artifacts['metadata'] = json.load(f)
    print(f"✅ Model metadata loaded: {metadata_path}")
    
    print(f"\n✅ All artifacts loaded successfully")
    
    return artifacts


# =============================================================================
# COMPLETE REGISTRY PIPELINE
# =============================================================================

def register_model_pipeline(model, X_train, y_train, metrics, best_params, 
                           encoders, feature_names, removed_features):
    """
    Complete model registration pipeline
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        metrics: Performance metrics
        best_params: Hyperparameters
        encoders: Label encoders
        feature_names: Feature names
        removed_features: Removed features
    
    Returns:
        tuple: (mlflow_run_id, artifact_paths)
    """
    # Log to MLflow
    run_id = log_model_to_mlflow(
        model, X_train, y_train, metrics, best_params, feature_names
    )
    
    # Save artifacts
    artifact_paths = save_model_artifacts(
        model, encoders, feature_names, removed_features
    )
    
    print("\n" + "="*80)
    print("✅ MODEL REGISTRATION COMPLETE")
    print("="*80)
    print(f"   MLflow Run ID: {run_id}")
    print(f"   Artifacts saved to: {config.MODELS_DIR}")
    
    return run_id, artifact_paths


if __name__ == "__main__":
    # Test loading artifacts
    print("Testing artifact loading...")
    try:
        artifacts = load_model_artifacts()
        print("\n✅ Artifact loading test passed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Train a model first before loading artifacts.")