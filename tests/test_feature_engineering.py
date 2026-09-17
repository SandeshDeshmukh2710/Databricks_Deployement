import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_engineering import create_features


def test_create_features():
    df = pd.DataFrame({
        "Job Satisfaction": ["High", "Low"],
        "Work-Life Balance": ["Good", "Poor"],
        "Years at Company": [5, 10],
        "Company Tenure": [6, 12],
        "Number of Promotions": [2, 1],
        "Monthly Income": [50000, 60000],
        "Age": [30, 40],
        "Overtime": ["Yes", "No"],
        "Distance from Home": [10, 20],
    })

    result = create_features(df)

    expected_features = [
        "Tenure_Efficiency",
        "Years_Per_Promotion",
        "Income_Per_Year",
        "Income_Age_Ratio",
        "Overall_Satisfaction",
        "Overtime_Flag",
        "Distance_Burden",
        "Promotion_Rate",
        "Stagnation_Risk",
    ]

    for feature in expected_features:
        assert feature in result.columns

def test_overtime_flag():
    df = pd.DataFrame({
        "Job Satisfaction": ["High"],
        "Work-Life Balance": ["Good"],
        "Years at Company": [5],
        "Company Tenure": [6],
        "Number of Promotions": [2],
        "Monthly Income": [50000],
        "Age": [30],
        "Overtime": ["Yes"],
        "Distance from Home": [10],
    })

    result = create_features(df)

    assert result["Overtime_Flag"].iloc[0] == 1


def test_overall_satisfaction():
    df = pd.DataFrame({
        "Job Satisfaction": ["High"],
        "Work-Life Balance": ["Good"],
        "Years at Company": [5],
        "Company Tenure": [6],
        "Number of Promotions": [2],
        "Monthly Income": [50000],
        "Age": [30],
        "Overtime": ["Yes"],
        "Distance from Home": [10],
    })

    result = create_features(df)

    # High = 3, Good = 3
    # Overall Satisfaction = (3 + 3) / 2 = 3
    assert result["Overall_Satisfaction"].iloc[0] == 3


def test_invalid_satisfaction_value():
    df = pd.DataFrame({
        "Job Satisfaction": ["Unknown"],
        "Work-Life Balance": ["Good"],
        "Years at Company": [5],
        "Company Tenure": [6],
        "Number of Promotions": [2],
        "Monthly Income": [50000],
        "Age": [30],
        "Overtime": ["Yes"],
        "Distance from Home": [10],
    })

    result = create_features(df)

    assert result["Job Satisfaction"].isna().iloc[0]