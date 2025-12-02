"""
Data export module for system evaluation.

This module handles exporting all DataFrames to CSV files with proper formatting.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_frame_level_data(frame_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export full per-frame DataFrame to CSV.
    
    Args:
        frame_df: DataFrame with all frame-level data
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / 'frames_all.csv'
    frame_df.to_csv(file_path, index=False)
    logger.info(f"Exported {len(frame_df)} frames to {file_path}")


def export_run_level_data(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export per-run metrics to CSV.
    
    Args:
        metrics_df: DataFrame with per-run metrics
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / 'runs_metrics.csv'
    metrics_df.to_csv(file_path, index=False)
    logger.info(f"Exported {len(metrics_df)} run metrics to {file_path}")


def export_aggregated_data(agg_df: pd.DataFrame, output_dir: Path, 
                          filename: str = 'aggregated_by_config_sequence.csv') -> None:
    """
    Export aggregated metrics to CSV.
    
    Args:
        agg_df: DataFrame with aggregated metrics
        output_dir: Output directory path
        filename: Output filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / filename
    agg_df.to_csv(file_path, index=False)
    logger.info(f"Exported aggregated data to {file_path}")


def export_hypothesis_data(hyp_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export hypothesis comparisons to CSV.
    
    Args:
        hyp_df: DataFrame with hypothesis comparisons
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / 'hypothesis_comparisons.csv'
    hyp_df.to_csv(file_path, index=False)
    logger.info(f"Exported {len(hyp_df)} hypothesis comparisons to {file_path}")


def export_motion_stratified_data(frame_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export motion-stratified summaries to CSV.
    
    Includes tier usage, oscillations, latency by motion regime.
    
    Args:
        frame_df: DataFrame with frame-level data
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summaries = []
    
    # Group by ablation_name and motion
    if 'motion' in frame_df.columns and 'ablation_name' in frame_df.columns:
        for (ablation, motion), group_df in frame_df.groupby(['ablation_name', 'motion']):
            summary = {
                'ablation_name': ablation,
                'motion': motion,
                'num_frames': len(group_df)
            }
            
            # Tier usage by motion
            if 'tier_used' in group_df.columns:
                tier_counts = group_df['tier_used'].value_counts()
                total_tiers = group_df['tier_used'].notna().sum()
                if total_tiers > 0:
                    for tier in ['nano', 'small', 'medium', 'large']:
                        count = tier_counts.get(tier, 0)
                        summary[f'tier_{tier}_count'] = count
                        summary[f'tier_{tier}_fraction'] = count / total_tiers
            
            # Oscillations (for FSM configs)
            if 'pred_label' in group_df.columns:
                valid_preds = group_df[group_df['pred_label'].notna()].sort_values('video_timestamp_sec')
                if len(valid_preds) > 1:
                    oscillations = (valid_preds['pred_label'].diff() != 0).sum()
                    summary['oscillation_count'] = int(oscillations)
                else:
                    summary['oscillation_count'] = 0
            
            # Latency statistics by motion
            if 'total_pipeline_latency_s' in group_df.columns:
                latencies = group_df['total_pipeline_latency_s'].dropna()
                if len(latencies) > 0:
                    summary['latency_mean'] = latencies.mean()
                    summary['latency_p50'] = latencies.quantile(0.50)
                    summary['latency_p90'] = latencies.quantile(0.90)
                    summary['latency_std'] = latencies.std()
            
            summaries.append(summary)
    
    if summaries:
        summary_df = pd.DataFrame(summaries)
        file_path = output_path / 'motion_stratified_summary.csv'
        summary_df.to_csv(file_path, index=False)
        logger.info(f"Exported motion-stratified summary to {file_path}")
    else:
        logger.warning("No motion-stratified data to export")


def export_sensor_prediction_data(frame_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export sensor prediction distribution data to CSV.
    
    Args:
        frame_df: DataFrame with frame-level data
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summaries = []
    
    # Group by ablation_name and sensor_prediction
    if 'sensor_prediction' in frame_df.columns and 'ablation_name' in frame_df.columns:
        for (ablation, sensor_pred), group_df in frame_df.groupby(['ablation_name', 'sensor_prediction']):
            summary = {
                'ablation_name': ablation,
                'sensor_prediction': sensor_pred,
                'num_frames': len(group_df)
            }
            
            # Sensor prediction distribution
            if 'sensor_prediction' in group_df.columns:
                pred_counts = group_df['sensor_prediction'].value_counts()
                total = len(group_df)
                for pred_type in ['wet', 'neutral', 'dry']:
                    count = pred_counts.get(pred_type, 0)
                    summary[f'{pred_type}_count'] = count
                    summary[f'{pred_type}_fraction'] = count / total if total > 0 else 0.0
            
            # Sensor boost statistics
            if 'sensor_boost' in group_df.columns:
                boosts = group_df['sensor_boost'].dropna()
                if len(boosts) > 0:
                    summary['sensor_boost_mean'] = boosts.mean()
                    summary['sensor_boost_std'] = boosts.std()
            
            summaries.append(summary)
    
    if summaries:
        summary_df = pd.DataFrame(summaries)
        file_path = output_path / 'sensor_prediction_summary.csv'
        summary_df.to_csv(file_path, index=False)
        logger.info(f"Exported sensor prediction summary to {file_path}")
    else:
        logger.warning("No sensor prediction data to export")


def export_all_data(frame_df: pd.DataFrame, metrics_df: pd.DataFrame,
                   aggregated_results: Dict[str, pd.DataFrame],
                   output_dir: Path) -> None:
    """
    Export all data files to CSV.
    
    Args:
        frame_df: DataFrame with frame-level data
        metrics_df: DataFrame with per-run metrics
        aggregated_results: Dictionary with aggregated DataFrames
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting all data to {output_path}")
    
    # Export frame-level data
    export_frame_level_data(frame_df, output_path)
    
    # Export run-level metrics
    export_run_level_data(metrics_df, output_path)
    
    # Export aggregated data
    if 'by_config_sequence' in aggregated_results:
        export_aggregated_data(
            aggregated_results['by_config_sequence'],
            output_path,
            'aggregated_by_config_sequence.csv'
        )
    
    if 'by_config_sequence_sensor' in aggregated_results:
        export_aggregated_data(
            aggregated_results['by_config_sequence_sensor'],
            output_path,
            'aggregated_by_config_sequence_sensor.csv'
        )
    
    # Export hypothesis comparisons
    if 'hypothesis_comparisons' in aggregated_results:
        export_hypothesis_data(
            aggregated_results['hypothesis_comparisons'],
            output_path
        )
    
    # Export motion-stratified data
    export_motion_stratified_data(frame_df, output_path)
    
    # Export sensor prediction data
    export_sensor_prediction_data(frame_df, output_path)
    
    logger.info("All data export complete")

