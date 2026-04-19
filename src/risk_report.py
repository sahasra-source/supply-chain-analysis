"""
src/risk_report.py
------------------
Generates actionable risk flag reports from the processed inventory DataFrame.
Uses actual dataset column names ('Stock Quantity', 'days_to_expiry', etc.)

Usage:
    from src.features import build_features
    from src.risk_report import generate_flags, summary_by_category, export_report

    df    = build_features('data/raw/supply_chain_inventory.csv')
    flags = generate_flags(df)
    print(flags.head())
"""

import pandas as pd
from datetime import date

# Reference date — must match features.py and all notebooks
REFERENCE_DATE = '2024-01-01'


def generate_flags(
    df: pd.DataFrame,
    stock_threshold: int = 75,
    expiry_days: int = 365
) -> pd.DataFrame:
    """
    Flag SKUs that are overstocked, near/past expiry, or both (dual-risk).

    Parameters
    ----------
    df               : DataFrame output from build_features()
    stock_threshold  : Stock Quantity above which a SKU is considered overstocked
    expiry_days      : days_to_expiry below which a SKU is considered expiry-risk
                       (use negative values to flag only already-expired SKUs)

    Returns
    -------
    pd.DataFrame of flagged SKUs sorted by financial_exposure descending
    """
    overstock_mask   = df['Stock Quantity'] > stock_threshold
    expiry_mask      = df['days_to_expiry'] < expiry_days

    flags = df[overstock_mask | expiry_mask].copy()

    # Assign flag reason — DUAL_RISK takes priority
    flags['flag_reason'] = 'unknown'
    flags.loc[overstock_mask[flags.index], 'flag_reason']                         = 'OVERSTOCK'
    flags.loc[expiry_mask[flags.index],    'flag_reason']                         = 'EXPIRY_RISK'
    flags.loc[overstock_mask[flags.index] & expiry_mask[flags.index], 'flag_reason'] = 'DUAL_RISK'

    # Severity mapping
    severity_map = {'DUAL_RISK': 1, 'EXPIRY_RISK': 2, 'OVERSTOCK': 3}
    flags['severity_rank'] = flags['flag_reason'].map(severity_map)

    flags['report_date']       = REFERENCE_DATE
    flags['financial_exposure'] = flags['Stock Quantity'] * flags['Price']

    # Select and order output columns
    output_cols = [
        'Product ID', 'Product Name', 'Product Category',
        'Stock Quantity', 'Price', 'days_to_expiry',
        'Risk_Level', 'flag_reason', 'severity_rank',
        'financial_exposure', 'report_date'
    ]
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in flags.columns]

    return (
        flags[output_cols]
        .sort_values(['severity_rank', 'financial_exposure'], ascending=[True, False])
        .reset_index(drop=True)
    )


def summary_by_category(flags: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flag counts and financial exposure by product category.

    Parameters
    ----------
    flags : output DataFrame from generate_flags()

    Returns
    -------
    pd.DataFrame with one row per category, sorted by total_exposure descending
    """
    summary = flags.groupby('Product Category').agg(
        total_flagged      = ('Product ID',         'count'),
        dual_risk_count    = ('flag_reason',         lambda x: (x == 'DUAL_RISK').sum()),
        expiry_risk_count  = ('flag_reason',         lambda x: (x == 'EXPIRY_RISK').sum()),
        overstock_count    = ('flag_reason',         lambda x: (x == 'OVERSTOCK').sum()),
        total_exposure     = ('financial_exposure',  'sum'),
        avg_days_to_expiry = ('days_to_expiry',      'mean'),
        avg_stock_qty      = ('Stock Quantity',      'mean'),
    ).reset_index()

    summary['dual_risk_pct'] = (
        summary['dual_risk_count'] / summary['total_flagged'] * 100
    ).round(1)

    return summary.sort_values('total_exposure', ascending=False).reset_index(drop=True)


def export_report(
    flags: pd.DataFrame,
    output_path: str = 'outputs/risk_flag_report.csv'
) -> None:
    """
    Save the flagged SKU report to CSV.

    Parameters
    ----------
    flags       : output DataFrame from generate_flags()
    output_path : destination path for the CSV
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flags.to_csv(output_path, index=False)
    print(f"Report saved → {output_path}")
    print(f"  Total flagged:  {len(flags):,} SKUs")
    print(f"  DUAL_RISK:      {(flags['flag_reason'] == 'DUAL_RISK').sum():,}")
    print(f"  EXPIRY_RISK:    {(flags['flag_reason'] == 'EXPIRY_RISK').sum():,}")
    print(f"  OVERSTOCK:      {(flags['flag_reason'] == 'OVERSTOCK').sum():,}")
    print(f"  Total exposure: ${flags['financial_exposure'].sum():,.0f}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.features import build_features

    df    = build_features('data/raw/supply_chain_inventory.csv')
    flags = generate_flags(df, stock_threshold=75, expiry_days=365)

    print("=== RISK FLAG REPORT ===")
    print(f"Flagged SKUs: {len(flags):,} of {len(df):,}")
    print(flags['flag_reason'].value_counts().to_frame('count'))

    print("\n=== CATEGORY SUMMARY ===")
    print(summary_by_category(flags).to_string(index=False))

    export_report(flags)
