"""
PRR Calculator for ADR Pharmacovigilance
Calculates Proportional Reporting Ratio and related signal metrics
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_prr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PRR (Proportional Reporting Ratio) for every drug-ADR pair per quarter.
    
    PRR measures if a particular ADR is reported more frequently with a drug
    than would be expected compared to the background reporting rate.
    
    Formula:
    PRR = (a / (a + b)) / (c / (c + d))
    
    Where:
    a = reports of Drug X → ADR Y
    b = reports of Drug X → any other ADR  
    c = reports of any other drug → ADR Y
    d = reports of any other drug → any other ADR
    
    Args:
        df: DataFrame with required columns:
            - drug: drug name (string)
            - adr: adverse drug reaction name (string) 
            - quarter: reporting quarter (e.g. '2022Q1')
            - quarter_numeric: integer quarter index (1, 2, 3, ...)
            - report_count: number of reports for this drug-ADR pair this quarter
            - total_drug_reports: total reports for this drug this quarter
            - total_adr_reports: total reports for this ADR across all drugs this quarter
            - total_reports: total reports across all drugs and ADRs this quarter
    
    Returns:
        Same DataFrame with added columns:
            - PRR: proportional reporting ratio
            - PRR_signal: 1 if PRR >= 2.0 else 0 (FDA threshold)
            - ROR: reporting odds ratio (bonus signal)
            - log_PRR: natural log of PRR (clipped for numerical stability)
    
    Interpretation:
        PRR = 1.0 → no signal (same rate as background)
        PRR >= 2.0 → FDA standard threshold for signal detection
        PRR < 1.0 → lower than expected reporting rate
    """
    logger.info(f"Calculating PRR for {len(df)} drug-ADR-quarter observations")
    
    # Validate required columns
    required_cols = ['drug', 'adr', 'quarter', 'quarter_numeric', 'report_count', 
                   'total_drug_reports', 'total_adr_reports', 'total_reports']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate PRR components
    # a = reports of Drug X → ADR Y
    a = df['report_count']
    
    # b = reports of Drug X → any other ADR
    b = df['total_drug_reports'] - df['report_count']
    
    # c = reports of any other drug → ADR Y
    c = df['total_adr_reports'] - df['report_count']
    
    # d = reports of any other drug → any other ADR
    d = df['total_reports'] - df['total_drug_reports'] - df['total_adr_reports'] + df['report_count']
    
    # Calculate PRR with numerical stability
    # Handle division by zero cases
    denominator_a = a + b
    denominator_c = c + d
    
    # Avoid division by zero
    prr_numerator = np.where(denominator_a > 0, a / denominator_a, 0)
    prr_denominator = np.where(denominator_c > 0, c / denominator_c, 1)
    
    df['PRR'] = np.where(prr_denominator > 0, prr_numerator / prr_denominator, np.nan)
    
    # Calculate ROR (Reporting Odds Ratio)
    # ROR = (a * d) / (b * c)
    ror_numerator = a * d
    ror_denominator = b * c
    
    df['ROR'] = np.where(ror_denominator > 0, ror_numerator / ror_denominator, np.nan)
    
    # FDA signal threshold (PRR >= 2.0)
    df['PRR_signal'] = ((df['PRR'] >= 2.0) & (~df['PRR'].isna())).astype(int)
    
    # Log PRR for modeling (clipped for numerical stability)
    df['log_PRR'] = np.log(df['PRR'].clip(lower=0.01))
    
    # Add some additional useful metrics
    # Reporting ratio confidence interval (simplified)
    df['PRR_lower_ci'] = df['PRR'] * 0.8  # Simplified CI
    df['PRR_upper_ci'] = df['PRR'] * 1.2  # Simplified CI
    
    # Signal strength categorization
    df['signal_strength'] = pd.cut(
        df['PRR'], 
        bins=[0, 0.5, 1.0, 2.0, 4.0, np.inf],
        labels=['Very Low', 'Low', 'Background', 'Signal', 'Strong Signal']
    )
    
    # Calculate additional metrics for validation
    # Proportion of total reports
    df['drug_adr_proportion'] = df['report_count'] / df['total_reports']
    
    # Drug-specific reporting rate
    df['drug_reporting_rate'] = df['report_count'] / df['total_drug_reports']
    
    # ADR-specific reporting rate  
    df['adr_reporting_rate'] = df['report_count'] / df['total_adr_reports']
    
    # Log summary statistics
    total_pairs = len(df[['drug', 'adr']].drop_duplicates())
    signals_detected = df['PRR_signal'].sum()
    avg_prr = df['PRR'].mean()
    
    logger.info(f"PRR Calculation Summary:")
    logger.info(f"  Total drug-ADR pairs: {total_pairs}")
    logger.info(f"  Signals detected (PRR >= 2.0): {signals_detected}")
    logger.info(f"  Average PRR: {avg_prr:.3f}")
    logger.info(f"  PRR range: {df['PRR'].min():.3f} - {df['PRR'].max():.3f}")
    
    return df


def validate_prr_calculation(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate PRR calculation results
    
    Args:
        df: DataFrame with PRR calculations
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for NaN values in critical columns
    critical_cols = ['PRR', 'ROR', 'PRR_signal']
    nan_counts = df[critical_cols].isna().sum()
    
    if nan_counts.any():
        return False, f"NaN values found in critical columns: {nan_counts[nan_counts > 0].to_dict()}"
    
    # Check PRR signal consistency
    inconsistent_signals = df[
        (df['PRR'] >= 2.0) & (df['PRR_signal'] == 0)
    ].shape[0]
    
    if inconsistent_signals > 0:
        return False, f"{inconsistent_signals} records have PRR >= 2.0 but signal = 0"
    
    # Check for impossible values
    impossible_prr = df[df['PRR'] < 0].shape[0]
    if impossible_prr > 0:
        return False, f"{impossible_prr} records have negative PRR values"
    
    return True, None


def get_prr_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for PRR calculations
    
    Args:
        df: DataFrame with PRR calculations
        
    Returns:
        DataFrame with summary statistics
    """
    summary_stats = []
    
    # Overall statistics
    total_observations = len(df)
    unique_pairs = len(df[['drug', 'adr']].drop_duplicates())
    total_signals = df['PRR_signal'].sum()
    signal_rate = total_signals / total_observations
    
    summary_stats.append({
        'metric': 'Total Observations',
        'value': total_observations
    })
    
    summary_stats.append({
        'metric': 'Unique Drug-ADR Pairs',
        'value': unique_pairs
    })
    
    summary_stats.append({
        'metric': 'Total Signals (PRR >= 2.0)',
        'value': total_signals
    })
    
    summary_stats.append({
        'metric': 'Signal Rate',
        'value': signal_rate
    })
    
    # PRR distribution statistics
    prr_stats = df['PRR'].describe()
    for stat_name, stat_value in prr_stats.items():
        summary_stats.append({
            'metric': f'PRR {stat_name}',
            'value': stat_value
        })
    
    # Signal strength distribution
    signal_dist = df['signal_strength'].value_counts()
    for strength, count in signal_dist.items():
        summary_stats.append({
            'metric': f'Signal Strength: {strength}',
            'value': count
        })
    
    return pd.DataFrame(summary_stats)


def calculate_prr_by_time_window(df: pd.DataFrame, window_size: int = 4) -> pd.DataFrame:
    """
    Calculate rolling PRR statistics over time windows
    
    Args:
        df: DataFrame with PRR calculations
        window_size: Number of quarters to include in each window
        
    Returns:
        DataFrame with rolling PRR statistics
    """
    logger.info(f"Calculating rolling PRR statistics with window size {window_size}")
    
    # Sort by drug, adr, quarter_numeric for proper rolling calculation
    df_sorted = df.sort_values(['drug', 'adr', 'quarter_numeric'])
    
    # Calculate rolling statistics for each drug-ADR pair
    rolling_stats = []
    
    for (drug, adr), group in df_sorted.groupby(['drug', 'adr']):
        group = group.copy().sort_values('quarter_numeric')
        
        # Rolling PRR statistics
        group['PRR_rolling_mean'] = group['PRR'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        group['PRR_rolling_std'] = group['PRR'].rolling(
            window=window_size, min_periods=1
        ).std()
        
        group['PRR_rolling_max'] = group['PRR'].rolling(
            window=window_size, min_periods=1
        ).max()
        
        # Rolling signal count
        group['PRR_signals_rolling'] = group['PRR_signal'].rolling(
            window=window_size, min_periods=1
        ).sum()
        
        rolling_stats.append(group)
    
    return pd.concat(rolling_stats, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    sample_data = {
        'drug': ['DrugA', 'DrugA', 'DrugB', 'DrugB'],
        'adr': ['ADR1', 'ADR1', 'ADR2', 'ADR2'],
        'quarter': ['2022Q1', '2022Q2', '2022Q1', '2022Q2'],
        'quarter_numeric': [1, 2, 1, 2],
        'report_count': [10, 15, 5, 8],
        'total_drug_reports': [100, 120, 80, 90],
        'total_adr_reports': [50, 60, 30, 35],
        'total_reports': [1000, 1100, 800, 900],
        'will_signal': [1, 1, 0, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Calculate PRR
    df_with_prr = calculate_prr(df)
    
    # Validate results
    is_valid, error_msg = validate_prr_calculation(df_with_prr)
    if is_valid:
        print("✅ PRR calculation validation passed")
    else:
        print(f"❌ PRR calculation validation failed: {error_msg}")
    
    # Show results
    print("\nPRR Calculation Results:")
    print(df_with_prr[['drug', 'adr', 'quarter', 'PRR', 'PRR_signal', 'ROR']].round(3))
    
    # Summary statistics
    summary = get_prr_summary_stats(df_with_prr)
    print("\nSummary Statistics:")
    print(summary)
