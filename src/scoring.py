# =============================================================================
# SCORING - Batch inference and explainability
# =============================================================================
"""
Scoring Pipeline:
1. Load saved model and artifacts
2. Batch prediction on new data
3. SHAP explainability
4. Risk categorization
5. Generate prediction outputs
"""

# import shap
import warnings

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

import config
from feature_engineering import prepare_features
from model_registry import load_model_artifacts

warnings.filterwarnings('ignore')

# =============================================================================
# LOAD MODEL FOR INFERENCE
# =============================================================================

def load_inference_model():
    """
    Load production model from Unity Catalog and
    load preprocessing artifacts required for inference.
    """

    print("\n" + "=" * 80)
    print("LOADING PRODUCTION INFERENCE MODEL")
    print("=" * 80)

    # Load preprocessing artifacts
    artifacts = load_model_artifacts()

    # Load production model from Unity Catalog
    model_uri = (
        f"models:/{config.MLFLOW_MODEL_NAME}@{config.PROD_MODEL_ALIAS}"
    )

    print("\nLoading model from Unity Catalog...")
    print(f"Model URI: {model_uri}")

    model = mlflow.xgboost.load_model(model_uri)

    # Replace locally stored model with UC production model
    artifacts["model"] = model

    print("\n✅ Production inference model ready")
    print(f"   Model: {config.MLFLOW_MODEL_NAME}")
    print(f"   Alias: @{config.PROD_MODEL_ALIAS}")
    print(f"   Features: {artifacts['metadata']['n_features']}")

    return artifacts


# =============================================================================
# BATCH PREDICTION
# =============================================================================

def predict_batch(model_artifacts, test_file_path):
    """
    Generate predictions for batch of employees
    
    Args:
        model_artifacts (dict): Loaded model and artifacts
        test_file_path (str): Path to test data
    
    Returns:
        pd.DataFrame: Predictions with employee IDs and risk scores
    """
    print("\n" + "="*80)
    print("BATCH PREDICTION")
    print("="*80)
    
    # Extract artifacts
    model = model_artifacts['model']
    encoders = model_artifacts['encoders']
    feature_names = model_artifacts['feature_names']
    
    # Prepare features (use existing encoders, no feature selection on test)
    X_test, _, _, _, _, employee_ids = prepare_features(
        test_file_path, 
        encoders=encoders, 
        fit=False, 
        select=False
    )
    
    # Align features with training features
    X_test = X_test[feature_names]
    
    print(f"\n⏳ Generating predictions for {len(X_test):,} employees...")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Get prediction labels
    prediction_labels = [config.REVERSE_TARGET_MAPPING[p] for p in y_pred]
    
    # Confidence scores
    confidence_scores = np.max(y_proba, axis=1)
    
    # Probability of leaving (Class 1)
    leave_probability = y_proba[:, 1]
    
    # Risk categorization
    risk_categories = [config.categorize_risk(prob) for prob in leave_probability]
    
    # Create results dataframe
    results = pd.DataFrame({
        'Employee_ID': employee_ids,
        'Prediction': prediction_labels,
        'Attrition_Probability': leave_probability,
        'Confidence': confidence_scores,
        'Risk_Category': risk_categories
    })
    
    print(f"\n✅ Predictions generated for {len(results):,} employees")
    
    # Prediction distribution
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    pred_dist = results['Prediction'].value_counts()
    for pred, count in pred_dist.items():
        pct = (count / len(results)) * 100
        print(f"  {pred}: {count:,} ({pct:.2f}%)")
    
    # Risk distribution
    print("\nRisk Distribution:")
    risk_dist = results['Risk_Category'].value_counts()
    for risk, count in risk_dist.items():
        pct = (count / len(results)) * 100
        print(f"  {risk}: {count:,} ({pct:.2f}%)")
    
    return results, X_test


# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================

def explain_predictions(model, X, feature_names, sample_size=1000):
    """
    Generate SHAP explanations for predictions
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features to explain
        feature_names (list): Feature names
        sample_size (int): Number of samples for SHAP background
    
    Returns:
        shap.Explainer: SHAP explainer object with computed values
    """
    import shap  # Import here to avoid issues if SHAP is not installed
    print("\n" + "="*80)
    print("SHAP EXPLAINABILITY")
    print("="*80)
    
    print(f"\n⏳ Computing SHAP values for {len(X)} predictions...")
    print(f"   (Using {min(sample_size, len(X))} samples for background)")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(
        model,
        X.sample(min(sample_size, len(X)), random_state=config.RANDOM_STATE)
    )
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    
    print("\n✅ SHAP values computed")
    
    return explainer, shap_values


def get_top_drivers(shap_values, feature_names, top_n=5):
    """
    Get top feature drivers for each prediction
    
    Args:
        shap_values (np.array): SHAP values for all predictions
        feature_names (list): Feature names
        top_n (int): Number of top drivers to return
    
    Returns:
        list: List of dicts with top drivers for each prediction
    """
    drivers_list = []
    
    for i in range(len(shap_values)):
        # Get absolute SHAP values for this prediction
        shap_abs = np.abs(shap_values[i])
        
        # Get top N indices
        top_indices = np.argsort(shap_abs)[-top_n:][::-1]
        
        # Create driver dictionary
        drivers = {
            f'driver_{j+1}': feature_names[idx]
            for j, idx in enumerate(top_indices)
        }
        
        drivers_list.append(drivers)
    
    return drivers_list


# =============================================================================
# SINGLE EMPLOYEE PREDICTION
# =============================================================================

def predict_single_employee(model_artifacts, employee_data):
    """
    Generate prediction for a single employee with detailed explanation
    
    Args:
        model_artifacts (dict): Loaded model and artifacts
        employee_data (dict): Dictionary with employee features
    
    Returns:
        dict: Prediction results with explanation
    """
    # Convert to DataFrame
    df = pd.DataFrame([employee_data])
    
    # Extract artifacts
    model = model_artifacts['model']
    encoders = model_artifacts['encoders']
    feature_names = model_artifacts['feature_names']
    
    # Prepare features
    from feature_engineering import create_features, encode_categoricals
    
    # Create features
    df_feat = create_features(df)
    
    # Encode categoricals
    df_encoded, _ = encode_categoricals(df_feat, encoders=encoders, fit=False)
    
    # Align features
    X = df_encoded[feature_names]
    
    # Predict
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]
    
    # Compute SHAP
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get top drivers
    shap_abs = np.abs(shap_values[0])
    top_indices = np.argsort(shap_abs)[-5:][::-1]
    top_drivers = [(feature_names[idx], shap_values[0][idx]) for idx in top_indices]
    
    result = {
        'prediction': config.REVERSE_TARGET_MAPPING[y_pred],
        'attrition_probability': y_proba[1],
        'confidence': np.max(y_proba),
        'risk_category': config.categorize_risk(y_proba[1]),
        'top_drivers': top_drivers
    }
    
    return result


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(results, output_path=None):
    """
    Save prediction results to CSV
    
    Args:
        results (pd.DataFrame): Prediction results
        output_path (str): Output file path
    
    Returns:
        str: Path to saved file
    """
    if output_path is None:
        output_path = f"{config.OUTPUTS_DIR}/predictions.csv"
    
    results.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")
    
    return output_path



def save_predictions_to_delta(results):
    """Upsert prediction results into the production Delta table."""

    from delta.tables import DeltaTable

    table_name = "workspace.default.employee_attrition_predictions"
    model_version = get_prod_model_version()

    output_df = results.copy()

    # Match the existing Delta table schema
    output_df["Employee_ID"] = output_df["Employee_ID"].astype(str)
    output_df["Prediction"] = output_df["Prediction"].astype(str)
    output_df["Attrition_Probability"] = (
        output_df["Attrition_Probability"].astype(float)
    )
    output_df["Confidence"] = output_df["Confidence"].astype(float)
    output_df["Risk_Category"] = output_df["Risk_Category"].astype(str)

    output_df["Prediction_Date"] = pd.Timestamp.now().date()
    output_df["Model_Version"] = model_version

    schema = StructType([
        StructField("Employee_ID", StringType(), True),
        StructField("Prediction", StringType(), True),
        StructField("Attrition_Probability", DoubleType(), True),
        StructField("Confidence", DoubleType(), True),
        StructField("Risk_Category", StringType(), True),
        StructField("Prediction_Date", DateType(), True),
        StructField("Model_Version", StringType(), True),
    ])

    spark = SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(output_df, schema=schema)

    # A MERGE source must have at most one row per employee.
    duplicate_ids = (
        spark_df.groupBy("Employee_ID")
        .count()
        .filter("count > 1")
        .limit(1)
        .count()
    )
    if duplicate_ids:
        raise ValueError(
            "Input predictions contain duplicate Employee_ID values."
        )

    print(f"Upserting {len(output_df):,} predictions into {table_name}")

    delta_table = DeltaTable.forName(spark, table_name)

    (
        delta_table.alias("target")
        .merge(
            spark_df.alias("source"),
            "target.Employee_ID = source.Employee_ID",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )

    print(f"Predictions upserted successfully: {table_name}")
    print(f"Model Version: {model_version}")
    print(f"Records processed: {len(output_df):,}")



# =============================================================================
# COMPLETE SCORING PIPELINE
# =============================================================================

def score_pipeline(test_file_path, save_output=True, explain=True):
    """
    Complete scoring pipeline
    
    Args:
        test_file_path (str): Path to test data
        save_output (bool): Whether to save predictions
        explain (bool): Whether to compute SHAP values
    
    Returns:
        tuple: (predictions_df, shap_values)
    """
    # Load model
    model_artifacts = load_inference_model()
    
    # Generate predictions
    predictions, X_test = predict_batch(model_artifacts, test_file_path)
    
    # Compute SHAP explanations
    shap_values = None
    if explain:
        _explainer, shap_values = explain_predictions(
            model_artifacts['model'],
            X_test,
            model_artifacts['feature_names']
        )
        
        # Add top drivers to predictions
        top_drivers = get_top_drivers(shap_values, model_artifacts['feature_names'])
        drivers_df = pd.DataFrame(top_drivers)
        predictions = pd.concat([predictions, drivers_df], axis=1)
    
    # Save predictions
    if save_output:
        save_predictions(predictions)
        save_predictions_to_delta(predictions)
    
    print("\n" + "="*80)
    print("✅ SCORING PIPELINE COMPLETE")
    print("="*80)
    
    return predictions, shap_values


def get_prod_model_version():
    """Get the model version currently assigned to the @prod alias."""

    client = mlflow.MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=config.MLFLOW_MODEL_NAME,
        alias=config.PROD_MODEL_ALIAS
    )

    return str(model_version.version)


if __name__ == "__main__":
    # Test scoring pipeline
    print("Testing Scoring Pipeline...")
    try:
        predictions, shap_values = score_pipeline(
            config.TEST_PATH,
            save_output=True,
            explain=False)
        print("\n✅ Scoring pipeline test complete!")
        print(f"   Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Train and save a model first.")