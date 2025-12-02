"""
Aggregation module for system evaluation.

This module handles aggregating metrics across runs and computing
hypothesis comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_confidence_interval(data: pd.Series, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for a series of values.
    
    Args:
        data: Series of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    
    if len(data) == 1:
        mean_val = data.iloc[0]
        return (mean_val, mean_val, mean_val)
    
    mean_val = data.mean()
    std_val = data.std()
    n = len(data)
    
    # Use t-distribution for small samples, normal for large
    if n < 30:
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_critical * std_val / np.sqrt(n)
    else:
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin = z_critical * std_val / np.sqrt(n)
    
    lower = mean_val - margin
    upper = mean_val + margin
    
    return (mean_val, lower, upper)


def aggregate_by_group(metrics_df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """
    Aggregate metrics by grouping keys, computing mean ± std and 95% CI.
    
    Args:
        metrics_df: DataFrame with per-run metrics
        group_keys: List of column names to group by (e.g., ['ablation_name', 'sequence_id'])
        
    Returns:
        DataFrame with aggregated metrics (one row per group)
    """
    # Identify numeric columns (exclude grouping keys and identifiers)
    exclude_cols = group_keys + ['run_id', 'repeat_index']
    numeric_cols = [
        col for col in metrics_df.columns 
        if col not in exclude_cols and 
        pd.api.types.is_numeric_dtype(metrics_df[col])
    ]
    
    aggregated_rows = []
    
    for group_vals, group_df in metrics_df.groupby(group_keys):
        # Handle both single and multiple grouping keys
        if isinstance(group_vals, tuple):
            group_dict = dict(zip(group_keys, group_vals))
        else:
            group_dict = {group_keys[0]: group_vals}
        
        # Compute statistics for each numeric column
        for col in numeric_cols:
            col_data = group_df[col].dropna()
            
            if len(col_data) > 0:
                mean_val, ci_lower, ci_upper = compute_confidence_interval(col_data)
                std_val = col_data.std()
                
                group_dict[f'{col}_mean'] = mean_val
                group_dict[f'{col}_std'] = std_val
                group_dict[f'{col}_ci_lower'] = ci_lower
                group_dict[f'{col}_ci_upper'] = ci_upper
                group_dict[f'{col}_count'] = len(col_data)
            else:
                group_dict[f'{col}_mean'] = np.nan
                group_dict[f'{col}_std'] = np.nan
                group_dict[f'{col}_ci_lower'] = np.nan
                group_dict[f'{col}_ci_upper'] = np.nan
                group_dict[f'{col}_count'] = 0
        
        aggregated_rows.append(group_dict)
    
    agg_df = pd.DataFrame(aggregated_rows)
    logger.info(f"Aggregated {len(metrics_df)} runs into {len(agg_df)} groups")
    
    return agg_df


def compute_hypothesis_comparisons(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly compute deltas for hypothesis testing.
    
    Hypotheses:
    - H1 (Adaptive Tiering & Offload): config 1 vs 4 - latency reduction, F1 maintained
    - H2 (Sensor Fusion Stability): config 2 vs 4 - oscillation reduction, stability improvement
    - H3 (Multi-Model Consensus): config 4 vs 4b, config 2 vs 2b - accuracy improvement vs energy cost
    - Fast-motion safety: config 4 vs 5 - latency during fast motion
    - Always-offload baseline: config 4 vs 6 - energy efficiency comparison
    - Graceful degradation: config 4 vs 3 - local-only vs offload performance
    
    Args:
        metrics_df: DataFrame with per-run metrics
        
    Returns:
        DataFrame with hypothesis comparisons (one row per hypothesis)
    """
    comparisons = []
    
    # Helper function to get aggregated metric for a config
    def get_metric(config_name: str, metric_name: str, sequence_id: Optional[str] = None) -> pd.Series:
        """Get metric values for a specific config."""
        mask = metrics_df['ablation_name'] == config_name
        if sequence_id is not None:
            mask = mask & (metrics_df['sequence_id'] == sequence_id)
        return metrics_df[mask][metric_name].dropna()
    
    # Helper function to compute comparison
    def compute_delta(config_a: str, config_b: str, metric_name: str, 
                     sequence_id: Optional[str] = None) -> Dict:
        """Compute delta between two configs for a metric."""
        values_a = get_metric(config_a, metric_name, sequence_id)
        values_b = get_metric(config_b, metric_name, sequence_id)
        
        if len(values_a) == 0 or len(values_b) == 0:
            return {
                'delta_mean': np.nan,
                'delta_std': np.nan,
                'delta_pct': np.nan,
                'p_value': np.nan,
                'n_a': len(values_a),
                'n_b': len(values_b)
            }
        
        mean_a = values_a.mean()
        mean_b = values_b.mean()
        delta = mean_b - mean_a  # B - A (positive means B is higher)
        delta_pct = (delta / mean_a * 100) if mean_a != 0 else np.nan
        
        # Statistical test (t-test if both have >1 sample, else just report means)
        if len(values_a) > 1 and len(values_b) > 1:
            try:
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
            except:
                p_value = np.nan
        else:
            p_value = np.nan
        
        return {
            'delta_mean': delta,
            'delta_std': np.sqrt(values_a.var() + values_b.var()) if len(values_a) > 1 and len(values_b) > 1 else np.nan,
            'delta_pct': delta_pct,
            'p_value': p_value,
            'n_a': len(values_a),
            'n_b': len(values_b),
            'mean_a': mean_a,
            'mean_b': mean_b
        }
    
    # H1: Adaptive Tiering & Offload (config 1 vs 4)
    # Metrics: latency reduction, F1 maintained
    h1_latency = compute_delta('1', '4', 'latency_mean')
    h1_f1 = compute_delta('1', '4', 'macro_f1')
    
    comparisons.append({
        'hypothesis': 'H1',
        'description': 'Adaptive Tiering & Offload: config 1 vs 4',
        'metric': 'latency_mean',
        **h1_latency
    })
    comparisons.append({
        'hypothesis': 'H1',
        'description': 'Adaptive Tiering & Offload: config 1 vs 4',
        'metric': 'macro_f1',
        **h1_f1
    })
    
    # H2: Sensor Fusion Stability (config 2 vs 4)
    # Metrics: oscillation reduction, stability improvement
    h2_oscillation = compute_delta('2', '4', 'oscillation_count')
    h2_f1 = compute_delta('2', '4', 'macro_f1')
    
    comparisons.append({
        'hypothesis': 'H2',
        'description': 'Sensor Fusion Stability: config 2 vs 4',
        'metric': 'oscillation_count',
        **h2_oscillation
    })
    comparisons.append({
        'hypothesis': 'H2',
        'description': 'Sensor Fusion Stability: config 2 vs 4',
        'metric': 'macro_f1',
        **h2_f1
    })
    
    # H3: Multi-Model Consensus (config 4 vs 4b, config 2 vs 2b)
    # Metrics: accuracy improvement vs energy cost
    h3_4_f1 = compute_delta('4', '4b', 'macro_f1')
    h3_4_energy = compute_delta('4', '4b', 'energy_total_run')
    
    comparisons.append({
        'hypothesis': 'H3',
        'description': 'Multi-Model Consensus: config 4 vs 4b',
        'metric': 'macro_f1',
        **h3_4_f1
    })
    comparisons.append({
        'hypothesis': 'H3',
        'description': 'Multi-Model Consensus: config 4 vs 4b',
        'metric': 'energy_total_run',
        **h3_4_energy
    })
    
    h3_2_f1 = compute_delta('2', '2b', 'macro_f1')
    h3_2_energy = compute_delta('2', '2b', 'energy_total_run')
    
    comparisons.append({
        'hypothesis': 'H3',
        'description': 'Multi-Model Consensus: config 2 vs 2b',
        'metric': 'macro_f1',
        **h3_2_f1
    })
    comparisons.append({
        'hypothesis': 'H3',
        'description': 'Multi-Model Consensus: config 2 vs 2b',
        'metric': 'energy_total_run',
        **h3_2_energy
    })
    
    # Fast-motion safety (config 4 vs 5)
    # Metrics: latency during fast motion
    h4_latency_fast = compute_delta('4', '5', 'latency_mean_fast')
    h4_latency = compute_delta('4', '5', 'latency_mean')
    
    comparisons.append({
        'hypothesis': 'H4',
        'description': 'Fast-motion safety: config 4 vs 5',
        'metric': 'latency_mean_fast',
        **h4_latency_fast
    })
    comparisons.append({
        'hypothesis': 'H4',
        'description': 'Fast-motion safety: config 4 vs 5',
        'metric': 'latency_mean',
        **h4_latency
    })
    
    # Always-offload baseline (config 4 vs 6)
    # Metrics: energy efficiency comparison
    h5_energy = compute_delta('4', '6', 'energy_total_run')
    h5_f1 = compute_delta('4', '6', 'macro_f1')
    
    comparisons.append({
        'hypothesis': 'H5',
        'description': 'Always-offload baseline: config 4 vs 6',
        'metric': 'energy_total_run',
        **h5_energy
    })
    comparisons.append({
        'hypothesis': 'H5',
        'description': 'Always-offload baseline: config 4 vs 6',
        'metric': 'macro_f1',
        **h5_f1
    })
    
    # Graceful degradation (config 4 vs 3)
    # Metrics: local-only vs offload performance
    h6_f1 = compute_delta('4', '3', 'macro_f1')
    h6_latency = compute_delta('4', '3', 'latency_mean')
    
    comparisons.append({
        'hypothesis': 'H6',
        'description': 'Graceful degradation: config 4 vs 3',
        'metric': 'macro_f1',
        **h6_f1
    })
    comparisons.append({
        'hypothesis': 'H6',
        'description': 'Graceful degradation: config 4 vs 3',
        'metric': 'latency_mean',
        **h6_latency
    })
    
    hyp_df = pd.DataFrame(comparisons)
    logger.info(f"Computed {len(hyp_df)} hypothesis comparisons")
    
    return hyp_df


def aggregate_all_results(metrics_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Aggregate results by different grouping strategies.
    
    Args:
        metrics_df: DataFrame with per-run metrics
        
    Returns:
        Dictionary with different aggregated DataFrames:
        - by_config_sequence: Grouped by (ablation_name, sequence_id)
        - by_config_sequence_sensor: Grouped by (ablation_name, sequence_id, sensor_prediction)
        - hypothesis_comparisons: Hypothesis comparison deltas
    """
    results = {}
    
    # Aggregate by (ablation_name, sequence_id)
    results['by_config_sequence'] = aggregate_by_group(
        metrics_df, 
        ['ablation_name', 'sequence_id']
    )
    
    # Aggregate by (ablation_name, sequence_id, sensor_prediction)
    if 'sensor_prediction' in metrics_df.columns:
        results['by_config_sequence_sensor'] = aggregate_by_group(
            metrics_df,
            ['ablation_name', 'sequence_id', 'sensor_prediction']
        )
    
    # Hypothesis comparisons
    results['hypothesis_comparisons'] = compute_hypothesis_comparisons(metrics_df)
    
    return results

