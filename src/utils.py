# =============================================================================
# UTILS - Helper functions for plotting and evaluation
# =============================================================================
"""
Utility functions:
1. Plotting (confusion matrix, ROC curves, feature importance)
2. Metric calculation and reporting
3. Data validation helpers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

import config

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# CONFUSION MATRIX PLOTTING
# =============================================================================

def plot_confusion_matrices(y_train, y_train_pred, y_val, y_val_pred, 
                           y_test, y_test_pred, save_path=None):
    """
    Plot confusion matrices for train, validation, and test sets
    
    Args:
        y_train, y_train_pred: Training true and predicted labels
        y_val, y_val_pred: Validation true and predicted labels
        y_test, y_test_pred: Test true and predicted labels
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        (y_train, y_train_pred, 'Training Set'),
        (y_val, y_val_pred, 'Validation Set'),
        (y_test, y_test_pred, 'Test Set')
    ]
    
    for idx, (y_true, y_pred, title) in enumerate(datasets):
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left'],
            ax=axes[idx], cbar=False
        )
        
        axes[idx].set_title(f'{title}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        # Add accuracy as subtitle
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        axes[idx].text(
            0.5, -0.15, f'Accuracy: {accuracy:.2%}',
            ha='center', transform=axes[idx].transAxes, fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrices saved: {save_path}")
    
    plt.show()


# =============================================================================
# ROC & PRECISION-RECALL CURVES
# =============================================================================

def plot_roc_pr_curves(y_train, y_train_proba, y_val, y_val_proba, 
                       y_test, y_test_proba, save_path=None):
    """
    Plot ROC and Precision-Recall curves
    
    Args:
        y_train, y_train_proba: Training labels and probabilities
        y_val, y_val_proba: Validation labels and probabilities
        y_test, y_test_proba: Test labels and probabilities
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    ax1 = axes[0]
    
    datasets = [
        (y_train, y_train_proba, 'Training', 'blue'),
        (y_val, y_val_proba, 'Validation', 'green'),
        (y_test, y_test_proba, 'Test', 'red')
    ]
    
    for y_true, y_proba, label, color in datasets:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
    ax2 = axes[1]
    
    for y_true, y_proba, label, color in datasets:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax2.plot(recall, precision, color=color, lw=2, label=f'{label} (AP = {ap:.3f})')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC & PR curves saved: {save_path}")
    
    plt.show()


# =============================================================================
# FEATURE IMPORTANCE PLOTTING
# =============================================================================

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Plot top N most important features
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): Feature names
        top_n (int): Number of top features to plot
        save_path (str): Path to save plot
    """
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=importance_df, x='Importance', y='Feature',
        palette='viridis'
    )
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved: {save_path}")
    
    plt.show()
    
    return importance_df


# =============================================================================
# METRICS REPORTING
# =============================================================================

def print_classification_metrics(y_true, y_pred, y_proba, dataset_name="Dataset"):
    """
    Print comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        dataset_name (str): Name of dataset for display
    """
    print("\n" + "="*80)
    print(f"{dataset_name.upper()} - CLASSIFICATION METRICS")
    print("="*80)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }
    
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Stayed', 'Left'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]:,}")
    print(f"  False Positives: {cm[0, 1]:,}")
    print(f"  False Negatives: {cm[1, 0]:,}")
    print(f"  True Positives:  {cm[1, 1]:,}")


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df, required_columns=None):
    """
    Validate input data
    
    Args:
        df (pd.DataFrame): Input dataframe
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check for empty dataframe
    if df.empty:
        validation['is_valid'] = False
        validation['errors'].append("DataFrame is empty")
        return validation
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        cols_with_missing = missing_counts[missing_counts > 0]
        validation['warnings'].append(
            f"Columns with missing values: {cols_with_missing.to_dict()}"
        )
    
    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        validation['warnings'].append(f"Found {n_duplicates} duplicate rows")
    
    return validation


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_model_summary(model, metrics, feature_names, params):
    """
    Generate a comprehensive model summary report
    
    Args:
        model: Trained model
        metrics (dict): Performance metrics
        feature_names (list): Feature names
        params (dict): Model parameters
    
    Returns:
        str: Formatted summary report
    """
    summary = []
    summary.append("\n" + "="*80)
    summary.append("MODEL SUMMARY REPORT")
    summary.append("="*80)
    
    summary.append("\nMODEL INFORMATION:")
    summary.append(f"  Model Type: {type(model).__name__}")
    summary.append(f"  Number of Features: {len(feature_names)}")
    summary.append(f"  Number of Trees: {params.get('n_estimators', 'N/A')}")
    
    summary.append("\nPERFORMANCE METRICS:")
    for split in ['train', 'val', 'test']:
        if split in metrics:
            summary.append(f"\n  {split.capitalize()} Set:")
            for metric, value in metrics[split].items():
                summary.append(f"    {metric.upper():12s}: {value:.4f}")
    
    summary.append("\nTOP 10 FEATURES:")
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    for idx, row in importance.iterrows():
        summary.append(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
    
    summary.append("\n" + "="*80)
    
    return "\n".join(summary)


if __name__ == "__main__":
    print("Utility functions loaded successfully")
    print("Available functions:")
    print("  - plot_confusion_matrices()")
    print("  - plot_roc_pr_curves()")
    print("  - plot_feature_importance()")
    print("  - print_classification_metrics()")
    print("  - validate_data()")
    print("  - generate_model_summary()")