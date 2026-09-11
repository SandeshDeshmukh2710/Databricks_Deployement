# =============================================================================
# CONFIGURATION - All project settings and constants
# =============================================================================
"""
Centralized configuration for Employee Attrition MLOps Pipeline

Contains:
- Data paths
- Model hyperparameters
- Artifact directories
- Random seeds for reproducibility
- MLflow settings
"""

import os
import numpy as np

# =============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# DATA PATHS
# =============================================================================
DATA_PATH = "/Volumes/workspace/default/attrition/"
TRAIN_PATH = f"{DATA_PATH}train.csv"
TEST_PATH = f"{DATA_PATH}test.csv"

# =============================================================================
# ARTIFACT DIRECTORIES
# =============================================================================
ARTIFACTS_DIR = "/Workspace/Users/d83550cf-2251-43a8-911d-1403bdd563dc/Attrition/artifacts"
MODELS_DIR = f"{ARTIFACTS_DIR}/models"
PLOTS_DIR = f"{ARTIFACTS_DIR}/plots"
OUTPUTS_DIR = f"{ARTIFACTS_DIR}/outputs"

# Create directories if they don't exist
for directory in [ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# TARGET ENCODING
# =============================================================================
TARGET_MAPPING = {'Stayed': 0, 'Left': 1}
REVERSE_TARGET_MAPPING = {0: 'Stayed', 1: 'Left'}

# =============================================================================
# TRAIN-TEST SPLIT CONFIGURATION
# =============================================================================
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# =============================================================================
# XGBOOST DEFAULT PARAMETERS
# =============================================================================
XGBOOST_FIXED_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'enable_categorical': False
}

# =============================================================================
# OPTUNA HYPERPARAMETER TUNING SETTINGS
# =============================================================================
OPTUNA_N_TRIALS = 20
OPTUNA_TIMEOUT = None  # No timeout
OPTUNA_METRIC = 'roc_auc'  # Metric to optimize

# =============================================================================
# FEATURE SELECTION THRESHOLDS
# =============================================================================
CORRELATION_THRESHOLD = 0.95  # Remove features with correlation > 0.95

# =============================================================================
# MLFLOW SETTINGS
# =============================================================================
MLFLOW_EXPERIMENT_NAME = "Employee_Attrition_Prediction"

# =============================================================================
# RISK CATEGORIZATION THRESHOLDS
# =============================================================================
RISK_THRESHOLDS = {
    'low': 0.30,
    'medium': 0.50,
    'medium_high': 0.60
}

def categorize_risk(probability):
    """Convert attrition probability to business-friendly risk category
    
    Args:
        probability (float): Probability of employee leaving (0-1)
    
    Returns:
        str: Risk category ('Low Risk', 'Medium Risk', 'Medium-High Risk', 'High Risk')
    """
    if probability < RISK_THRESHOLDS['low']:
        return 'Low Risk'
    elif probability < RISK_THRESHOLDS['medium']:
        return 'Medium Risk'
    elif probability <= RISK_THRESHOLDS['medium_high']:
        return 'Medium-High Risk'
    else:
        return 'High Risk'

print(f"✅ Configuration loaded successfully")
print(f"   Random State: {RANDOM_STATE}")
print(f"   Train Path: {TRAIN_PATH}")
print(f"   Test Path: {TEST_PATH}")
print(f"   Artifacts Directory: {ARTIFACTS_DIR}")