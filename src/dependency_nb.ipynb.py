# Databricks notebook source
# MAGIC %pip install --quiet xgboost optuna shap imbalanced-learn category-encoders

# COMMAND ----------

import sys

sys.path.append(
    "/Workspace/Users/sandesh.deshmukh@bizmetric.com/Attrition/DAB-MLOps-Attrition/src"
)

# COMMAND ----------

import config
import utils
import feature_engineering
import model_training
import model_registry
import scoring

print("All modules imported successfully")

# COMMAND ----------

from feature_engineering import prepare_features
import config

X_train, y_train, encoders, features, removed, ids = prepare_features(
    config.TRAIN_PATH,
    fit=True,
    select=True
)

print("Feature Engineering completed successfully")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Number of features:", len(features))

# COMMAND ----------

from model_training import train_model_pipeline

model, metrics, best_params = train_model_pipeline(
    X_train,
    y_train
)

print("\nModel Training completed successfully")
print("Model:", type(model).__name__)
print("Metrics:", metrics)
print("Best Parameters:", best_params)

# COMMAND ----------

import importlib
import model_training

importlib.reload(model_training)

print("✅ model_training.py reloaded")

# COMMAND ----------

from model_training import train_model_pipeline

print("✅ Updated train_model_pipeline imported")

# COMMAND ----------

model, metrics, data_splits, best_params = train_model_pipeline(
    X_train,
    y_train
)

print("\nModel Training completed successfully")
print("Model:", type(model).__name__)
print("Metrics:", metrics)
print("Best Parameters:", best_params)

# COMMAND ----------

print("===== MODEL TRAINING RESULT =====")

print("\nModel:")
print(type(model).__name__)

print("\nMetrics:")
for split, values in metrics.items():
    print(f"\n{split.upper()}:")
    for metric, value in values.items():
        print(f"  {metric:10s}: {value:.4f}")

print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"  {param:20s}: {value}")

print("\nData Splits:")
for name, data in data_splits.items():
    print(f"  {name:8s}: {data.shape}")

# COMMAND ----------

import inspect
from model_registry import register_model_pipeline

print(inspect.signature(register_model_pipeline))

# COMMAND ----------

from model_registry import register_model_pipeline

registry_result = register_model_pipeline(
    model=model,
    X_train=data_splits["X_train"],
    y_train=data_splits["y_train"],
    metrics=metrics,
    best_params=best_params,
    encoders=encoders,
    feature_names=features,
    removed_features=removed
)

print("\n✅ Model Registry pipeline completed")
print("Result:", registry_result)

# COMMAND ----------

import inspect
from scoring import score_pipeline

print(inspect.signature(score_pipeline))

# COMMAND ----------

from scoring import score_pipeline
import config

predictions = score_pipeline(
    test_file_path=config.TEST_PATH,
    save_output=True,
    explain=True
)

print("\n✅ Scoring pipeline completed")
print("Output type:", type(predictions))

# COMMAND ----------

import importlib
import model_registry

importlib.reload(model_registry)

print("✅ model_registry reloaded")

# COMMAND ----------

from scoring import score_pipeline
import config

predictions = score_pipeline(
    test_file_path=config.TEST_PATH,
    save_output=True,
    explain=True
)

print("\n✅ Scoring pipeline completed")
print("Output type:", type(predictions))

# COMMAND ----------

import sys
import importlib

bundle_src = "/Workspace/Users/sandesh.deshmukh@bizmetric.com/.bundle/Databricks_Deployement/dev/files/src"

if bundle_src not in sys.path:
    sys.path.insert(0, bundle_src)

import main
importlib.reload(main)

print("✅ Bundle main.py imported successfully")

# COMMAND ----------

import main

result = main.main()

print("\n" + "=" * 80)
print("FINAL PIPELINE RESULT")
print("=" * 80)

print("MLflow Run ID:", result["run_id"])
print("Test ROC-AUC:", result["metrics"]["test"]["roc_auc"])
print("Artifacts:", result["artifacts"])

# COMMAND ----------

import pandas as pd
import config

df = pd.read_csv(config.TRAIN_PATH)

print("Shape:", df.shape)
print("\nColumns:")
for col in df.columns:
    print(repr(col))