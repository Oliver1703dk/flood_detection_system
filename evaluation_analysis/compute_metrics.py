"""
Metrics computation module for system evaluation.

This module computes per-run metrics including accuracy, latency,
energy, stability, coverage, and sensor metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_accuracy_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute accuracy metrics from frame-level predictions.
    
    Includes:
    - Confusion matrix (3x3 for classes 0, 1, 2)
    - Per-class precision, recall, F1
    - Macro F1 score
    - For FSM configs: F1 on ambiguous frames (combined_score in [0.3, 0.8])
    
    Args:
        frame_df: DataFrame with gt_label and pred_label columns
        
    Returns:
        Dictionary with accuracy metrics
    """
    metrics = {}
    
    # Filter out frames without ground truth or prediction
    valid_frames = frame_df[
        frame_df['gt_label'].notna() & 
        frame_df['pred_label'].notna()
    ].copy()
    
    if len(valid_frames) == 0:
        logger.warning("No valid frames for accuracy computation")
        return {
            'num_frames': 0,
            'confusion_matrix': None,
            'precision_0': np.nan, 'recall_0': np.nan, 'f1_0': np.nan,
            'precision_1': np.nan, 'recall_1': np.nan, 'f1_1': np.nan,
            'precision_2': np.nan, 'recall_2': np.nan, 'f1_2': np.nan,
            'macro_f1': np.nan,
            'f1_ambiguous_frames': np.nan
        }
    
    metrics['num_frames'] = len(valid_frames)
    
    # Build confusion matrix
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    
    y_true = valid_frames['gt_label'].astype(int)
    y_pred = valid_frames['pred_label'].astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    for i, label in enumerate([0, 1, 2]):
        metrics[f'precision_{label}'] = precision[i] if not np.isnan(precision[i]) else 0.0
        metrics[f'recall_{label}'] = recall[i] if not np.isnan(recall[i]) else 0.0
        metrics[f'f1_{label}'] = f1[i] if not np.isnan(f1[i]) else 0.0
        metrics[f'support_{label}'] = int(support[i])
    
    # Macro F1 (average of per-class F1 scores)
    metrics['macro_f1'] = np.mean([metrics['f1_0'], metrics['f1_1'], metrics['f1_2']])
    
    # F1 on ambiguous frames (FSM configs only)
    # Ambiguous frames: combined_score in [0.3, 0.8]
    ambiguous_frames = valid_frames[
        valid_frames['combined_score'].notna() &
        (valid_frames['combined_score'] >= 0.3) &
        (valid_frames['combined_score'] <= 0.8)
    ]
    
    if len(ambiguous_frames) > 0:
        amb_y_true = ambiguous_frames['gt_label'].astype(int)
        amb_y_pred = ambiguous_frames['pred_label'].astype(int)
        _, _, amb_f1, _ = precision_recall_fscore_support(
            amb_y_true, amb_y_pred, labels=[0, 1, 2], zero_division=0, average='macro'
        )
        metrics['f1_ambiguous_frames'] = amb_f1 if not np.isnan(amb_f1) else np.nan
        metrics['num_ambiguous_frames'] = len(ambiguous_frames)
    else:
        metrics['f1_ambiguous_frames'] = np.nan
        metrics['num_ambiguous_frames'] = 0
    
    return metrics


def compute_latency_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute latency metrics using total_pipeline_latency_s.
    
    Includes:
    - p50, p90, p99, max, mean, std
    - For FSM configs: stratify by motion and tier_used
    
    Args:
        frame_df: DataFrame with total_pipeline_latency_s column
        
    Returns:
        Dictionary with latency metrics
    """
    metrics = {}
    
    # Filter valid latency measurements
    valid_latencies = frame_df[frame_df['total_pipeline_latency_s'].notna()]['total_pipeline_latency_s']
    
    if len(valid_latencies) == 0:
        logger.warning("No valid latency measurements")
        return {
            'latency_p50': np.nan,
            'latency_p90': np.nan,
            'latency_p99': np.nan,
            'latency_max': np.nan,
            'latency_mean': np.nan,
            'latency_std': np.nan,
            'latency_count': 0
        }
    
    metrics['latency_count'] = len(valid_latencies)
    metrics['latency_p50'] = valid_latencies.quantile(0.50)
    metrics['latency_p90'] = valid_latencies.quantile(0.90)
    metrics['latency_p99'] = valid_latencies.quantile(0.99)
    metrics['latency_max'] = valid_latencies.max()
    metrics['latency_mean'] = valid_latencies.mean()
    metrics['latency_std'] = valid_latencies.std()
    
    # FSM-specific: stratify by motion
    if 'motion' in frame_df.columns and frame_df['motion'].notna().any():
        for motion_type in ['slow', 'fast']:
            motion_frames = frame_df[
                (frame_df['motion'] == motion_type) &
                frame_df['total_pipeline_latency_s'].notna()
            ]
            if len(motion_frames) > 0:
                motion_latencies = motion_frames['total_pipeline_latency_s']
                metrics[f'latency_p50_{motion_type}'] = motion_latencies.quantile(0.50)
                metrics[f'latency_p90_{motion_type}'] = motion_latencies.quantile(0.90)
                metrics[f'latency_mean_{motion_type}'] = motion_latencies.mean()
            else:
                metrics[f'latency_p50_{motion_type}'] = np.nan
                metrics[f'latency_p90_{motion_type}'] = np.nan
                metrics[f'latency_mean_{motion_type}'] = np.nan
    
    # FSM-specific: stratify by tier_used
    if 'tier_used' in frame_df.columns and frame_df['tier_used'].notna().any():
        for tier in ['nano', 'small', 'medium', 'large']:
            tier_frames = frame_df[
                (frame_df['tier_used'] == tier) &
                frame_df['total_pipeline_latency_s'].notna()
            ]
            if len(tier_frames) > 0:
                tier_latencies = tier_frames['total_pipeline_latency_s']
                metrics[f'latency_p50_tier_{tier}'] = tier_latencies.quantile(0.50)
                metrics[f'latency_p90_tier_{tier}'] = tier_latencies.quantile(0.90)
                metrics[f'latency_mean_tier_{tier}'] = tier_latencies.mean()
            else:
                metrics[f'latency_p50_tier_{tier}'] = np.nan
                metrics[f'latency_p90_tier_{tier}'] = np.nan
                metrics[f'latency_mean_tier_{tier}'] = np.nan
    
    return metrics


def compute_energy_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute energy metrics (when energy data available).
    
    Includes:
    - Total run energy (from last measurement)
    - Per-frame energy (mean/p50/p90/std)
    - Mean/max power during frames (mean/p50/p90)
    - Energy per decision-bearing frame
    
    Args:
        frame_df: DataFrame with energy columns
        
    Returns:
        Dictionary with energy metrics
    """
    metrics = {}
    
    # Check if energy data is available
    if 'energy_total_run' not in frame_df.columns:
        return {
            'energy_total_run': np.nan,
            'energy_per_frame_mean': np.nan,
            'energy_per_frame_p50': np.nan,
            'energy_per_frame_p90': np.nan,
            'energy_per_frame_std': np.nan,
            'power_mean_mean': np.nan,
            'power_mean_p50': np.nan,
            'power_mean_p90': np.nan,
            'power_max_mean': np.nan,
            'power_max_p50': np.nan,
            'power_max_p90': np.nan,
            'energy_per_decision_frame': np.nan
        }
    
    # Total run energy (same for all frames, take first non-nan value)
    run_energy = frame_df['energy_total_run'].dropna()
    if len(run_energy) > 0:
        metrics['energy_total_run'] = run_energy.iloc[0]
    else:
        metrics['energy_total_run'] = np.nan
    
    # Per-frame energy statistics
    per_frame_energy = frame_df['energy_total_per_frame'].dropna()
    if len(per_frame_energy) > 0:
        metrics['energy_per_frame_mean'] = per_frame_energy.mean()
        metrics['energy_per_frame_p50'] = per_frame_energy.quantile(0.50)
        metrics['energy_per_frame_p90'] = per_frame_energy.quantile(0.90)
        metrics['energy_per_frame_std'] = per_frame_energy.std()
    else:
        metrics['energy_per_frame_mean'] = np.nan
        metrics['energy_per_frame_p50'] = np.nan
        metrics['energy_per_frame_p90'] = np.nan
        metrics['energy_per_frame_std'] = np.nan
    
    # Power statistics
    power_mean = frame_df['power_total_mean'].dropna()
    if len(power_mean) > 0:
        metrics['power_mean_mean'] = power_mean.mean()
        metrics['power_mean_p50'] = power_mean.quantile(0.50)
        metrics['power_mean_p90'] = power_mean.quantile(0.90)
    else:
        metrics['power_mean_mean'] = np.nan
        metrics['power_mean_p50'] = np.nan
        metrics['power_mean_p90'] = np.nan
    
    power_max = frame_df['power_total_max'].dropna()
    if len(power_max) > 0:
        metrics['power_max_mean'] = power_max.mean()
        metrics['power_max_p50'] = power_max.quantile(0.50)
        metrics['power_max_p90'] = power_max.quantile(0.90)
    else:
        metrics['power_max_mean'] = np.nan
        metrics['power_max_p50'] = np.nan
        metrics['power_max_p90'] = np.nan
    
    # Energy per decision-bearing frame (frames where prediction was made)
    decision_frames = frame_df[
        frame_df['pred_label'].notna() &
        frame_df['energy_total_per_frame'].notna()
    ]
    if len(decision_frames) > 0:
        total_decision_energy = decision_frames['energy_total_per_frame'].sum()
        num_decision_frames = len(decision_frames)
        metrics['energy_per_decision_frame'] = total_decision_energy / num_decision_frames if num_decision_frames > 0 else np.nan
    else:
        metrics['energy_per_decision_frame'] = np.nan
    
    return metrics


def compute_stability_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute stability metrics (FSM configs only).
    
    Includes:
    - Label oscillation count (consecutive prediction changes)
    - FSM state usage fractions (time in S0, S1, S2, S3, S5)
    - Tier usage fractions (nano, small, medium, large) by motion regime
    - S2/S3 episode durations
    - S5 entry count
    
    Args:
        frame_df: DataFrame with FSM-specific columns
        
    Returns:
        Dictionary with stability metrics
    """
    metrics = {}
    
    # Check if this is an FSM config
    if 'fsm_state' not in frame_df.columns or frame_df['fsm_state'].isna().all():
        # Non-FSM config, return empty metrics
        return {
            'oscillation_count': np.nan,
            'state_s0_fraction': np.nan,
            'state_s1_fraction': np.nan,
            'state_s2_fraction': np.nan,
            'state_s3_fraction': np.nan,
            'state_s5_fraction': np.nan,
            'tier_nano_fraction': np.nan,
            'tier_small_fraction': np.nan,
            'tier_medium_fraction': np.nan,
            'tier_large_fraction': np.nan,
            's2_episode_count': np.nan,
            's2_episode_mean_duration': np.nan,
            's3_episode_count': np.nan,
            's3_episode_mean_duration': np.nan,
            's5_entry_count': np.nan
        }
    
    # Label oscillation count (consecutive prediction changes)
    valid_predictions = frame_df[frame_df['pred_label'].notna()].copy()
    if len(valid_predictions) > 1:
        valid_predictions = valid_predictions.sort_values('video_timestamp_sec')
        prediction_changes = (valid_predictions['pred_label'].diff() != 0).sum()
        metrics['oscillation_count'] = int(prediction_changes)
    else:
        metrics['oscillation_count'] = 0
    
    # FSM state usage fractions
    valid_states = frame_df[frame_df['fsm_state'].notna()]
    if len(valid_states) > 0:
        state_counts = valid_states['fsm_state'].value_counts()
        total_states = len(valid_states)
        
        for state in ['S0', 'S1', 'S2', 'S3', 'S5']:
            count = state_counts.get(state, 0)
            metrics[f'state_{state.lower()}_fraction'] = count / total_states if total_states > 0 else 0.0
    else:
        for state in ['S0', 'S1', 'S2', 'S3', 'S5']:
            metrics[f'state_{state.lower()}_fraction'] = np.nan
    
    # Tier usage fractions
    valid_tiers = frame_df[frame_df['tier_used'].notna()]
    if len(valid_tiers) > 0:
        tier_counts = valid_tiers['tier_used'].value_counts()
        total_tiers = len(valid_tiers)
        
        for tier in ['nano', 'small', 'medium', 'large']:
            count = tier_counts.get(tier, 0)
            metrics[f'tier_{tier}_fraction'] = count / total_tiers if total_tiers > 0 else 0.0
    else:
        for tier in ['nano', 'small', 'medium', 'large']:
            metrics[f'tier_{tier}_fraction'] = np.nan
    
    # Tier usage by motion regime
    if 'motion' in frame_df.columns:
        for motion_type in ['slow', 'fast']:
            motion_tiers = frame_df[
                (frame_df['motion'] == motion_type) &
                frame_df['tier_used'].notna()
            ]
            if len(motion_tiers) > 0:
                motion_tier_counts = motion_tiers['tier_used'].value_counts()
                motion_total = len(motion_tiers)
                
                for tier in ['nano', 'small', 'medium', 'large']:
                    count = motion_tier_counts.get(tier, 0)
                    metrics[f'tier_{tier}_fraction_{motion_type}'] = count / motion_total if motion_total > 0 else 0.0
            else:
                for tier in ['nano', 'small', 'medium', 'large']:
                    metrics[f'tier_{tier}_fraction_{motion_type}'] = np.nan
    
    # S2/S3 episode durations
    # An episode is a consecutive sequence of frames in the same state
    valid_states_sorted = frame_df[frame_df['fsm_state'].notna()].sort_values('video_timestamp_sec')
    
    if len(valid_states_sorted) > 0:
        # Find S2 episodes
        s2_episodes = []
        in_s2 = False
        episode_start = None
        
        for idx, row in valid_states_sorted.iterrows():
            state = row['fsm_state']
            if state == 'S2':
                if not in_s2:
                    in_s2 = True
                    episode_start = row['video_timestamp_sec']
            else:
                if in_s2:
                    in_s2 = False
                    if episode_start is not None:
                        episode_duration = row['video_timestamp_sec'] - episode_start
                        s2_episodes.append(episode_duration)
        
        # Check if still in S2 at end
        if in_s2 and episode_start is not None:
            last_timestamp = valid_states_sorted.iloc[-1]['video_timestamp_sec']
            episode_duration = last_timestamp - episode_start
            s2_episodes.append(episode_duration)
        
        metrics['s2_episode_count'] = len(s2_episodes)
        metrics['s2_episode_mean_duration'] = np.mean(s2_episodes) if s2_episodes else np.nan
        
        # Find S3 episodes (same logic)
        s3_episodes = []
        in_s3 = False
        episode_start = None
        
        for idx, row in valid_states_sorted.iterrows():
            state = row['fsm_state']
            if state == 'S3':
                if not in_s3:
                    in_s3 = True
                    episode_start = row['video_timestamp_sec']
            else:
                if in_s3:
                    in_s3 = False
                    if episode_start is not None:
                        episode_duration = row['video_timestamp_sec'] - episode_start
                        s3_episodes.append(episode_duration)
        
        if in_s3 and episode_start is not None:
            last_timestamp = valid_states_sorted.iloc[-1]['video_timestamp_sec']
            episode_duration = last_timestamp - episode_start
            s3_episodes.append(episode_duration)
        
        metrics['s3_episode_count'] = len(s3_episodes)
        metrics['s3_episode_mean_duration'] = np.mean(s3_episodes) if s3_episodes else np.nan
    else:
        metrics['s2_episode_count'] = 0
        metrics['s2_episode_mean_duration'] = np.nan
        metrics['s3_episode_count'] = 0
        metrics['s3_episode_mean_duration'] = np.nan
    
    # S5 entry count (count transitions into S5)
    if len(valid_states_sorted) > 1:
        state_changes = valid_states_sorted['fsm_state'].diff()
        s5_entries = (state_changes == 'S5').sum()
        metrics['s5_entry_count'] = int(s5_entries)
    else:
        metrics['s5_entry_count'] = 0
    
    return metrics


def compute_coverage_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute coverage metrics.
    
    Includes:
    - Decision coverage by GT level (0/1/2)
    - Critical-frame coverage (GT ∈ {1, 2})
    - Correct coverage on critical frames
    
    Args:
        frame_df: DataFrame with gt_label and pred_label columns
        
    Returns:
        Dictionary with coverage metrics
    """
    metrics = {}
    
    # Filter frames with ground truth
    valid_frames = frame_df[frame_df['gt_label'].notna()].copy()
    
    if len(valid_frames) == 0:
        return {
            'coverage_gt_0': np.nan,
            'coverage_gt_1': np.nan,
            'coverage_gt_2': np.nan,
            'coverage_critical': np.nan,
            'coverage_critical_correct': np.nan
        }
    
    # Decision coverage by GT level
    for gt_level in [0, 1, 2]:
        gt_frames = valid_frames[valid_frames['gt_label'] == gt_level]
        if len(gt_frames) > 0:
            # Coverage = fraction of frames with predictions
            coverage = gt_frames['pred_label'].notna().sum() / len(gt_frames)
            metrics[f'coverage_gt_{gt_level}'] = coverage
        else:
            metrics[f'coverage_gt_{gt_level}'] = np.nan
    
    # Critical-frame coverage (GT ∈ {1, 2})
    critical_frames = valid_frames[valid_frames['gt_label'].isin([1, 2])]
    if len(critical_frames) > 0:
        coverage = critical_frames['pred_label'].notna().sum() / len(critical_frames)
        metrics['coverage_critical'] = coverage
        
        # Correct coverage on critical frames
        correct_critical = critical_frames[
            (critical_frames['pred_label'].notna()) &
            (critical_frames['pred_label'] == critical_frames['gt_label'])
        ]
        if len(critical_frames) > 0:
            correct_coverage = len(correct_critical) / len(critical_frames)
            metrics['coverage_critical_correct'] = correct_coverage
        else:
            metrics['coverage_critical_correct'] = np.nan
    else:
        metrics['coverage_critical'] = np.nan
        metrics['coverage_critical_correct'] = np.nan
    
    return metrics


def compute_sensor_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute sensor metrics (FSM configs only).
    
    Includes:
    - Mean/std sensor_boost
    - Correlation image_score vs sensor_boost
    - sensor_prediction distribution (wet/neutral/dry counts and percentages)
    
    Args:
        frame_df: DataFrame with sensor-related columns
        
    Returns:
        Dictionary with sensor metrics
    """
    metrics = {}
    
    # Check if sensor data is available
    if 'sensor_boost' not in frame_df.columns:
        return {
            'sensor_boost_mean': np.nan,
            'sensor_boost_std': np.nan,
            'sensor_boost_image_correlation': np.nan,
            'sensor_prediction_wet_count': np.nan,
            'sensor_prediction_wet_pct': np.nan,
            'sensor_prediction_neutral_count': np.nan,
            'sensor_prediction_neutral_pct': np.nan,
            'sensor_prediction_dry_count': np.nan,
            'sensor_prediction_dry_pct': np.nan
        }
    
    # Sensor boost statistics
    valid_boost = frame_df[frame_df['sensor_boost'].notna()]['sensor_boost']
    if len(valid_boost) > 0:
        metrics['sensor_boost_mean'] = valid_boost.mean()
        metrics['sensor_boost_std'] = valid_boost.std()
    else:
        metrics['sensor_boost_mean'] = np.nan
        metrics['sensor_boost_std'] = np.nan
    
    # Correlation image_score vs sensor_boost
    valid_both = frame_df[
        frame_df['image_score'].notna() &
        frame_df['sensor_boost'].notna()
    ]
    if len(valid_both) > 1:
        correlation = valid_both['image_score'].corr(valid_both['sensor_boost'])
        metrics['sensor_boost_image_correlation'] = correlation if not np.isnan(correlation) else np.nan
    else:
        metrics['sensor_boost_image_correlation'] = np.nan
    
    # Sensor prediction distribution
    if 'sensor_prediction' in frame_df.columns:
        valid_predictions = frame_df[frame_df['sensor_prediction'].notna()]
        if len(valid_predictions) > 0:
            pred_counts = valid_predictions['sensor_prediction'].value_counts()
            total = len(valid_predictions)
            
            for pred_type in ['wet', 'neutral', 'dry']:
                count = pred_counts.get(pred_type, 0)
                metrics[f'sensor_prediction_{pred_type}_count'] = int(count)
                metrics[f'sensor_prediction_{pred_type}_pct'] = count / total if total > 0 else 0.0
        else:
            for pred_type in ['wet', 'neutral', 'dry']:
                metrics[f'sensor_prediction_{pred_type}_count'] = 0
                metrics[f'sensor_prediction_{pred_type}_pct'] = np.nan
    else:
        for pred_type in ['wet', 'neutral', 'dry']:
            metrics[f'sensor_prediction_{pred_type}_count'] = np.nan
            metrics[f'sensor_prediction_{pred_type}_pct'] = np.nan
    
    return metrics


def compute_all_metrics(frame_df: pd.DataFrame) -> Dict:
    """
    Compute all metrics for a run.
    
    Args:
        frame_df: DataFrame with frame-level data for a single run
        
    Returns:
        Dictionary with all computed metrics
    """
    all_metrics = {}
    
    # Add run identifiers
    if 'run_id' in frame_df.columns and len(frame_df) > 0:
        all_metrics['run_id'] = frame_df['run_id'].iloc[0]
    if 'ablation_name' in frame_df.columns and len(frame_df) > 0:
        all_metrics['ablation_name'] = frame_df['ablation_name'].iloc[0]
    if 'sequence_id' in frame_df.columns and len(frame_df) > 0:
        all_metrics['sequence_id'] = frame_df['sequence_id'].iloc[0]
    if 'sensor_prediction' in frame_df.columns and len(frame_df) > 0:
        all_metrics['sensor_prediction'] = frame_df['sensor_prediction'].iloc[0]
    if 'repeat_index' in frame_df.columns and len(frame_df) > 0:
        all_metrics['repeat_index'] = frame_df['repeat_index'].iloc[0]
    if 'config_type' in frame_df.columns and len(frame_df) > 0:
        all_metrics['config_type'] = frame_df['config_type'].iloc[0]
    
    # Compute all metric types
    all_metrics.update(compute_accuracy_metrics(frame_df))
    all_metrics.update(compute_latency_metrics(frame_df))
    all_metrics.update(compute_energy_metrics(frame_df))
    all_metrics.update(compute_stability_metrics(frame_df))
    all_metrics.update(compute_coverage_metrics(frame_df))
    all_metrics.update(compute_sensor_metrics(frame_df))
    
    return all_metrics


def compute_per_run_metrics(frame_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics for each run in the frame DataFrame.
    
    Args:
        frame_df: DataFrame with all frame-level data
        
    Returns:
        DataFrame with one row per run_id, all metrics as columns
    """
    run_metrics = []
    
    for run_id, run_frames in frame_df.groupby('run_id'):
        logger.debug(f"Computing metrics for run {run_id}")
        metrics = compute_all_metrics(run_frames)
        run_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(run_metrics)
    logger.info(f"Computed metrics for {len(metrics_df)} runs")
    
    return metrics_df

