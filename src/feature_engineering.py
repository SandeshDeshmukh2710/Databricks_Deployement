# =============================================================================
# FEATURE ENGINEERING - Data preparation and feature creation
# =============================================================================
"""
Feature Engineering Pipeline:
1. Load and clean data
2. Create business-meaningful features
3. Encode categorical variables
4. Select relevant features

All transformations are designed to be reproducible for inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import config

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Clean raw data (remove unnecessary columns)
    
    Args:
        df (pd.DataFrame): Raw dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Remove _rescued_data column if exists
    if '_rescued_data' in df_clean.columns:
        df_clean = df_clean.drop(columns=['_rescued_data'])
        print("✅ Removed '_rescued_data' column")
    
    return df_clean


# =============================================================================
# FEATURE CREATION
# =============================================================================

def create_features(df):
    """
    Create business-meaningful features from existing columns
    
    Features Created:
    - Tenure_Efficiency: Career progression speed
    - Years_Per_Promotion: Promotion frequency
    - Income_Per_Year: Salary growth rate
    - Income_Age_Ratio: Age-adjusted compensation
    - Overall_Satisfaction: Composite satisfaction score
    - Overtime_Flag: Binary overtime indicator
    - Distance_Burden: Commute stress metric
    - Promotion_Rate: Career advancement speed
    - Stagnation_Risk: Years without promotion
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    df_feat = df.copy()
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Convert ordinal categorical columns to numeric for feature engineering
    satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    balance_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    
    df_feat['Job Satisfaction'] = df_feat['Job Satisfaction'].map(satisfaction_map)
    df_feat['Work-Life Balance'] = df_feat['Work-Life Balance'].map(balance_map)
    
    # 1. Tenure-Based Features
    print("\n1. Creating Tenure-Based Features...")
    df_feat['Tenure_Efficiency'] = df_feat['Years at Company'] / (df_feat['Company Tenure'] + 1)
    print("   ✅ Tenure_Efficiency: Ratio of company tenure to total tenure")
    
    df_feat['Years_Per_Promotion'] = df_feat['Years at Company'] / (df_feat['Number of Promotions'] + 1)
    print("   ✅ Years_Per_Promotion: Years worked per promotion received")
    
    # 2. Compensation Features
    print("\n2. Creating Compensation Features...")
    df_feat['Income_Per_Year'] = df_feat['Monthly Income'] / (df_feat['Company Tenure'] + 1)
    print("   ✅ Income_Per_Year: Monthly income divided by tenure")
    
    df_feat['Income_Age_Ratio'] = df_feat['Monthly Income'] / (df_feat['Age'] + 1)
    print("   ✅ Income_Age_Ratio: Age-adjusted compensation")
    
    # 3. Satisfaction Composite
    print("\n3. Creating Satisfaction Composite...")
    df_feat['Overall_Satisfaction'] = (
        df_feat['Job Satisfaction'] + 
        df_feat['Work-Life Balance']
    ) / 2
    print("   ✅ Overall_Satisfaction: Average of job satisfaction and work-life balance")
    
    # 4. Work-Life Features
    print("\n4. Creating Work-Life Features...")
    df_feat['Overtime_Flag'] = df_feat['Overtime'].apply(lambda x: 1 if x == 'Yes' else 0)
    print("   ✅ Overtime_Flag: Binary indicator for overtime")
    
    df_feat['Distance_Burden'] = df_feat['Distance from Home'] * (df_feat['Overtime_Flag'] + 1)
    print("   ✅ Distance_Burden: Distance amplified by overtime status")
    
    # 5. Career Growth Features
    print("\n5. Creating Career Growth Features...")
    df_feat['Promotion_Rate'] = df_feat['Number of Promotions'] / (df_feat['Years at Company'] + 1)
    print("   ✅ Promotion_Rate: Career advancement speed (promotions per year)")
    
    df_feat['Stagnation_Risk'] = df_feat['Years at Company'] / (df_feat['Number of Promotions'] + 1)
    print("   ✅ Stagnation_Risk: Ratio indicating potential career stagnation")
    
    print(f"\n✅ Feature engineering complete: {len(df_feat.columns)} total features")
    
    return df_feat


# =============================================================================
# CATEGORICAL ENCODING
# =============================================================================

def encode_categoricals(df, encoders=None, fit=True):
    """
    Encode categorical variables using Label Encoding
    
    Args:
        df (pd.DataFrame): Input dataframe
        encoders (dict): Pre-fitted encoders (for inference)
        fit (bool): Whether to fit new encoders (True for training)
    
    Returns:
        tuple: (encoded_df, encoders_dict)
    """
    df_encoded = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column if present
    if 'Attrition' in categorical_cols:
        categorical_cols.remove('Attrition')
    
    # Remove Employee ID if present
    if 'Employee ID' in categorical_cols:
        categorical_cols.remove('Employee ID')
    
    print("\n" + "="*80)
    print("CATEGORICAL ENCODING - LABEL ENCODING")
    print("="*80)
    print(f"\nCategorical features to encode: {len(categorical_cols)}")
    print("\nEncoding Strategy: Label Encoding (one integer per category)\n")
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_cols:
        if fit:
            # Fit new encoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            
            # Get the mapping for display
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"✅ {col:30s} →  {len(le.classes_)} categories encoded (0 to {len(le.classes_)-1})")
            
            # Show full mapping for small categories, sample for large
            if len(mapping) <= 5:
                print(f"   Full mapping: {mapping}")
            else:
                sample_mapping = dict(list(mapping.items())[:3])
                print(f"   Sample mapping: {sample_mapping}...")
        else:
            # Use existing encoder
            if col in encoders:
                df_encoded[col] = encoders[col].transform(df_encoded[col])
            else:
                print(f"⚠️ Warning: No encoder found for {col}")
    
    print(f"\n✅ Categorical encoding complete")
    
    return df_encoded, encoders


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def select_features(df, threshold=config.CORRELATION_THRESHOLD):
    """
    Remove highly correlated features
    
    Args:
        df (pd.DataFrame): Input dataframe with all features
        threshold (float): Correlation threshold for feature removal
    
    Returns:
        tuple: (selected_df, removed_features)
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_remove = [column for column in upper_triangle.columns 
                 if any(upper_triangle[column] > threshold)]
    
    print(f"\nHighly correlated feature pairs (|corr| > {threshold}): {len(to_remove)}")
    
    # Remove highly correlated features
    df_selected = df.drop(columns=to_remove)
    
    print(f"\n✅ Feature selection complete")
    print(f"   Final features for modeling: {df_selected.shape[1]}")
    
    return df_selected, to_remove


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_features(file_path, encoders=None, fit=True, select=True):
    """
    Complete feature engineering pipeline
    
    Args:
        file_path (str): Path to data file
        encoders (dict): Pre-fitted encoders (for inference)
        fit (bool): Whether to fit new encoders
        select (bool): Whether to perform feature selection
    
    Returns:
        tuple: (X, y, encoders, feature_names, removed_features)
    """
    # Load and clean data
    df = load_data(file_path)
    df = clean_data(df)
    
    # Separate identifier, features, and target
    employee_ids = df['Employee ID'] if 'Employee ID' in df.columns else None
    
    # Check if target exists (training data)
    has_target = 'Attrition' in df.columns
    
    if has_target:
        X = df.drop(columns=['Employee ID', 'Attrition'])
        y = df['Attrition'].map(config.TARGET_MAPPING)
    else:
        X = df.drop(columns=['Employee ID'])
        y = None
    
    # Create features
    X = create_features(X)
    
    # Encode categoricals
    X, encoders = encode_categoricals(X, encoders=encoders, fit=fit)
    
    # Feature selection (only during training)
    removed_features = []
    if select and fit:
        X, removed_features = select_features(X)
    
    feature_names = X.columns.tolist()
    
    print("\n" + "="*80)
    print("FINAL DATASET SUMMARY")
    print("="*80)
    print(f"Features shape: {X.shape}")
    if has_target:
        print(f"Target shape: {y.shape}")
    print(f"Feature names: {len(feature_names)} features")
    
    return X, y, encoders, feature_names, removed_features, employee_ids


if __name__ == "__main__":
    # Test the pipeline
    print("Testing Feature Engineering Pipeline...")
    X_train, y_train, encoders, features, removed, ids = prepare_features(
        config.TRAIN_PATH, 
        fit=True, 
        select=True
    )
    print("\n✅ Feature engineering pipeline test complete!")