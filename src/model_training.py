# =============================================================================
# MODEL TRAINING - Train-test split, hyperparameter tuning, model training
# =============================================================================
"""
Model Training Pipeline:
1. Split data into train/validation/test sets
2. Hyperparameter tuning with Optuna
3. Train final XGBoost model with optimal parameters
4. Evaluate model performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

import config

# =============================================================================
# TRAIN-VALIDATION-TEST SPLIT
# =============================================================================

def split_data(X, y, train_size=config.TRAIN_SIZE, val_size=config.VAL_SIZE, 
               test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Split data into train, validation, and test sets (stratified)
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        train_size (float): Training set proportion
        val_size (float): Validation set proportion
        test_size (float): Test set proportion
        random_state (int): Random seed
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*80)
    print("TRAIN-VALIDATION-TEST SPLIT")
    print("="*80)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"\n✅ Data split complete (stratified sampling):")
    print(f"   Training set:   {X_train.shape[0]:,} samples ({train_size*100:.0f}%)")
    print(f"   Validation set: {X_val.shape[0]:,} samples ({val_size*100:.0f}%)")
    print(f"   Test set:       {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    
    # Check class distribution
    print(f"\n✅ Class distribution preserved:")
    print(f"   Training - Left: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}%)")
    print(f"   Validation - Left: {y_val.sum()}/{len(y_val)} ({y_val.mean()*100:.1f}%)")
    print(f"   Test - Left: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# =============================================================================

def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=config.OPTUNA_N_TRIALS):
    """
    Tune XGBoost hyperparameters using Optuna (Bayesian optimization)
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        n_trials (int): Number of Optuna trials
    
    Returns:
        dict: Best hyperparameters found
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION - OPTUNA")
    print("="*80)
    print(f"\n📌 Tuning 4 most important parameters with {n_trials} trials")
    print(f"   Parameters: max_depth, learning_rate, n_estimators, subsample")
    
    def objective(trial):
        """
        Optuna objective function to maximize ROC-AUC
        """
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 1,
            **config.XGBOOST_FIXED_PARAMS
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Predict and evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        
        return score
    
    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=config.RANDOM_STATE)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Optimization complete!")
    print(f"   Best {config.OPTUNA_METRIC.upper()}: {study.best_value:.4f}")
    print(f"   Best parameters:")
    for param, value in study.best_params.items():
        print(f"      {param}: {value}")
    
    return study.best_params, study


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    """
    Train final XGBoost model with optimized hyperparameters
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        best_params (dict): Optimal hyperparameters from tuning
    
    Returns:
        tuple: (trained_model, metrics_dict)
    """
    print("\n" + "="*80)
    print("FINAL OPTIMIZED MODEL TRAINING")
    print("="*80)
    
    # Combine best parameters with fixed parameters
    final_params = {
        **config.XGBOOST_FIXED_PARAMS,
        **best_params
    }
    
    print("\n⏳ Training optimized XGBoost model...")
    
    # Train final model
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions on all sets
    y_train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)
    
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for all sets
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'val': {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred),
            'roc_auc': roc_auc_score(y_val, y_val_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
    }
    
    # Display results
    print("\n✅ Optimized model training complete")
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE")
    print("="*80)
    
    for split, split_metrics in metrics.items():
        print(f"\n{split.capitalize()} Set:")
        for metric, value in split_metrics.items():
            print(f"  {metric.upper():12s}: {value:.4f}")
    
    return final_model, metrics


# =============================================================================
# COMPLETE TRAINING PIPELINE
# =============================================================================

def train_model_pipeline(X, y, n_trials=config.OPTUNA_N_TRIALS):
    """
    Complete model training pipeline
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        n_trials (int): Number of Optuna trials
    
    Returns:
        tuple: (model, metrics, split_data, best_params)
    """
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Optimize hyperparameters
    best_params, study = optimize_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    
    # Train final model
    model, metrics = train_final_model(
        X_train, y_train, X_val, y_val, X_test, y_test, best_params
    )
    
    # Package split data
    data_splits = {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test
}
    
    print("\n" + "="*80)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*80)
    
    return model, metrics, data_splits, best_params


if __name__ == "__main__":
    # Test the pipeline
    print("This module should be imported, not run directly.")
    print("Use: from model_training import train_model_pipeline")