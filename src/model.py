"""
src/model.py
------------
Trains and evaluates Logistic Regression and Random Forest classifiers
for at-risk SKU prediction.

Usage:
    from src.features import build_features
    from src.model import FEATURES, train_evaluate

    df = build_features('data/raw/supply_chain_inventory.csv')
    results, X_test, y_test = train_evaluate(df)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score,
    classification_report
)

# Feature list — uses engineered column names (all lowercase/snake_case)
# Note: raw spaced columns ('Stock Quantity' etc.) are not used directly here;
# build_features() produces these derived columns first.
FEATURES = [
    'stock_pressure',
    'expiry_urgency',
    'risk_score',
    'is_expired',
    'high_stock_flag',
    'near_expiry_flag',
    'price_tier_enc',
    'category_enc',
    'warranty_enc',
    'financial_exposure',
    'overhang_ratio',
    'pct_shelf_life_remaining',
    'shelf_life_days',
    'price_per_day_shelf',
    'inventory_value',
]

TARGET = 'is_at_risk'


def train_evaluate(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> tuple:
    """
    Train Logistic Regression and Random Forest, return metrics and test split.

    Parameters
    ----------
    df           : DataFrame output from build_features()
    target       : binary target column name
    test_size    : fraction held out for evaluation
    random_state : seed for reproducibility
    verbose      : print metrics during training

    Returns
    -------
    results  : dict  {model_name: {metric: value, 'model': fitted_model}}
    X_test   : pd.DataFrame
    y_test   : pd.Series
    """
    # Drop rows where any feature is NaN (e.g. price_per_day_shelf edge cases)
    df_model = df[FEATURES + [target]].dropna()
    if verbose:
        print(f"Modelling on {len(df_model):,} rows ({len(df) - len(df_model):,} dropped for NaN)")

    X = df_model[FEATURES]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")
        print(f"Target balance — train: {y_train.mean():.1%} at-risk | test: {y_test.mean():.1%} at-risk\n")

    models = [
        ('logistic_regression', LogisticRegression(max_iter=1000, random_state=random_state)),
        ('random_forest',       RandomForestClassifier(n_estimators=200, max_depth=10,
                                                        random_state=random_state, n_jobs=-1)),
    ]

    results = {}
    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'AUC-ROC':   round(roc_auc_score(y_test, prob), 4),
            'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, pred, zero_division=0), 4),
            'F1-Score':  round(f1_score(y_test, pred, zero_division=0), 4),
            'Accuracy':  round(accuracy_score(y_test, pred), 4),
            'model':     model,
        }
        results[name] = metrics

        if verbose:
            print(f"── {name} ──")
            for k, v in metrics.items():
                if k != 'model':
                    print(f"  {k}: {v}")
            print(classification_report(y_test, pred, target_names=['Safe', 'At-Risk']))

    return results, X_test, y_test


def get_feature_importance(results: dict, top_n: int = 15) -> pd.DataFrame:
    """
    Extract feature importance from the Random Forest model.

    Parameters
    ----------
    results : output dict from train_evaluate()
    top_n   : number of top features to return

    Returns
    -------
    pd.DataFrame sorted by importance descending
    """
    rf_model = results['random_forest']['model']
    importance_df = pd.DataFrame({
        'feature':    FEATURES,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)

    importance_df['pct'] = (
        importance_df['importance'] / importance_df['importance'].sum() * 100
    ).round(1)

    return importance_df


if __name__ == '__main__':
    from features import build_features
    df = build_features('data/raw/supply_chain_inventory.csv')
    results, X_test, y_test = train_evaluate(df)
    print("\nFeature importance:")
    print(get_feature_importance(results).to_string(index=False))
