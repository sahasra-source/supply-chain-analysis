"""
src/features.py
---------------
Builds all engineered features from raw inventory CSV.
Matches actual dataset column names (spaced, as-loaded from Kaggle).

Usage:
    from src.features import build_features
    df = pd.read_csv('../data/raw/supply_chain_inventory.csv')
"""

import pandas as pd
import numpy as np

# Fixed reference date — keep consistent across all notebooks and scripts
REFERENCE_DATE = pd.Timestamp('2024-01-01')

# Columns expected in raw CSV
RAW_COLS = {
    'id':           'Product ID',
    'name':         'Product Name',
    'category':     'Product Category',
    'price':        'Price',
    'stock':        'Stock Quantity',
    'warranty':     'Warranty Period',
    'mfg_date':     'Manufacturing Date',
    'exp_date':     'Expiration Date',
    'ratings':      'Product Ratings',
}


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Manufacturing Date and Expiration Date, derive Shelf_Life and days_to_expiry."""
    df['Manufacturing_Date'] = pd.to_datetime(df[RAW_COLS['mfg_date']], errors='coerce')
    df['Expiration_Date']    = pd.to_datetime(df[RAW_COLS['exp_date']], errors='coerce', dayfirst=True)

    df['days_to_expiry'] = (df['Expiration_Date'] - REFERENCE_DATE).dt.days
    df['Shelf_Life']     = (df['Expiration_Date'] - df['Manufacturing_Date']).dt.days.clip(lower=1)
    return df


def _assign_risk(row) -> str:
    """Rule-based risk segmentation using actual column names."""
    stock_high      = row['Stock Quantity'] > 75
    expiry_critical = row['days_to_expiry'] < 365   # expired or expiring within 12 months
    expiry_near     = row['days_to_expiry'] < 548   # expiring within 18 months

    if stock_high and expiry_critical:
        return 'Critical'
    elif stock_high and expiry_near:
        return 'High'
    elif stock_high or expiry_critical:
        return 'Medium'
    return 'Low'


def build_features(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV and build all 15 engineered features.

    Parameters
    ----------
    filepath : str
        Path to raw CSV, e.g. 'data/raw/supply_chain_inventory.csv'

    Returns
    -------
    pd.DataFrame with all original columns + engineered features
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # ── DATE FEATURES ──
    df = _parse_dates(df)

    # ── RISK SEGMENTATION ──
    df['Risk_Level'] = df.apply(_assign_risk, axis=1)
    df['is_at_risk'] = df['Risk_Level'].isin(['Critical', 'High']).astype(int)

    # ── FEATURE 1: Stock pressure ──
    df['stock_pressure'] = df['Stock Quantity'] / 100

    # ── FEATURE 2: Expiry urgency (handles negatives cleanly) ──
    df['expiry_urgency'] = (
        1 - (df['days_to_expiry'].clip(-731, 731) / 731)
    ).clip(0, 1)

    # ── FEATURE 3: Combined risk score ──
    df['risk_score'] = df['stock_pressure'] * df['expiry_urgency']

    # ── FEATURE 4: Is expired ──
    df['is_expired'] = (df['days_to_expiry'] <= 0).astype(int)

    # ── FEATURE 5: High stock flag ──
    df['high_stock_flag'] = (df['Stock Quantity'] > 75).astype(int)

    # ── FEATURE 6: Near expiry flag ──
    df['near_expiry_flag'] = (df['days_to_expiry'] < 180).astype(int)

    # ── FEATURE 7: Price tier ──
    df['price_tier']     = pd.qcut(df['Price'], q=3, labels=['low', 'mid', 'high'])
    df['price_tier_enc'] = df['price_tier'].map({'low': 0, 'mid': 1, 'high': 2})

    # ── FEATURE 8: Categorical encodings ──
    df['category_enc'] = df['Product Category'].astype('category').cat.codes
    df['warranty_enc'] = df['Warranty Period'].astype('category').cat.codes

    # ── FEATURE 9: Financial exposure ──
    df['financial_exposure'] = df['Stock Quantity'] * df['Price']

    # ── FEATURE 10: Overhang ratio ──
    df['overhang_ratio'] = (
        (df['Stock Quantity'] - 50).clip(0) / df['Stock Quantity'].clip(1)
    )

    # ── FEATURE 11: % shelf life remaining ──
    df['pct_shelf_life_remaining'] = (
        df['days_to_expiry'] / df['Shelf_Life'].clip(lower=1)
    ).clip(0, 1)

    # ── FEATURE 12: Shelf life days (clean alias) ──
    df['shelf_life_days'] = df['Shelf_Life']

    # ── FEATURE 13: Price per remaining shelf day ──
    df['price_per_day_shelf'] = df['Price'] / df['days_to_expiry'].clip(lower=1)

    # ── FEATURE 14: Inventory value ──
    df['inventory_value'] = df['Stock Quantity'] * df['Price']

    # ── FEATURE 15: Volume (placeholder — no dimension columns in dataset) ──
    if all(c in df.columns for c in ['Length', 'Width', 'Height']):
        df['volume_cm3'] = df['Length'] * df['Width'] * df['Height']
    else:
        df['volume_cm3'] = 0

    return df


if __name__ == '__main__':
    df = build_features('data/raw/supply_chain_inventory.csv')
    print(f"Shape: {df.shape}")
    print(f"At-risk SKUs: {df['is_at_risk'].sum():,} ({df['is_at_risk'].mean():.1%})")
    print(df[[
        'Stock Quantity', 'days_to_expiry', 'Risk_Level',
        'risk_score', 'inventory_value'
    ]].describe().round(3))
