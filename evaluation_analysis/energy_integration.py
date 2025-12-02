"""
Energy data integration module for system evaluation.

This module handles loading and integrating energy measurement data
with frame-level data. Energy data is optional.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_energy_file(energy_dir: Path, run_id: str) -> Optional[Path]:
    """
    Find energy file for a run by matching run_id.
    
    Search pattern: files containing run_id in filename or within JSON run_id field.
    
    Args:
        energy_dir: Path to storage/video_energy_results/ directory
        run_id: Run ID to match (e.g., "20251202-194508-743316")
        
    Returns:
        Path to energy file or None if not found
    """
    energy_path = Path(energy_dir)
    
    if not energy_path.exists():
        return None
    
    # First, try matching by filename
    for energy_file in energy_path.glob('*.jsonl'):
        if run_id in energy_file.name:
            return energy_file
    
    # If not found by filename, try matching run_id field in JSON
    for energy_file in energy_path.glob('*.jsonl'):
        try:
            with open(energy_file, 'r') as f:
                data = json.load(f)
                if data.get('run_id') == run_id:
                    return energy_file
        except Exception as e:
            logger.debug(f"Could not read {energy_file.name} for run_id matching: {e}")
            continue
    
    return None


def load_energy_data(energy_file: Path) -> Dict:
    """
    Load energy file as single JSON object.
    
    Expected format:
    {
        "run_id": "20251202-194508-743316",
        "video_file": "flood_video_20251005_145739.mp4",
        "measurements": [
            {"t": 0.0, "energy_total_j": 0.0, "power_total_w": 12.5},
            {"t": 0.1, "energy_total_j": 1.25, "power_total_w": 12.5},
            ...
        ]
    }
    
    Args:
        energy_file: Path to energy JSONL file
        
    Returns:
        Dictionary with run_id, video_file, and measurements array
    """
    try:
        with open(energy_file, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if 'measurements' not in data:
            logger.warning(f"Energy file {energy_file.name} missing 'measurements' field")
            return {}
        
        return data
    except Exception as e:
        logger.warning(f"Failed to load energy file {energy_file.name}: {e}")
        return {}


def match_energy_to_frames(energy_data: Dict, frame_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match energy measurements to frames by timestamp.
    
    For each frame, find energy and power values at frame timestamp.
    Extract energy at frame timestamp from cumulative energy values.
    Extract power samples within frame interval for aggregation.
    
    Args:
        energy_data: Energy data dictionary with measurements array
        frame_df: DataFrame with frame data (must have video_timestamp_sec column)
        
    Returns:
        DataFrame with energy columns added
    """
    if not energy_data or 'measurements' not in energy_data:
        logger.warning("No energy measurements to match")
        return frame_df.copy()
    
    measurements = energy_data['measurements']
    if not measurements:
        logger.warning("Empty measurements array")
        return frame_df.copy()
    
    # Convert measurements to DataFrame for easier processing
    energy_df = pd.DataFrame(measurements)
    
    # Ensure required columns exist
    required_cols = ['t', 'energy_total_j', 'power_total_w']
    missing_cols = [col for col in required_cols if col not in energy_df.columns]
    if missing_cols:
        logger.warning(f"Energy data missing columns: {missing_cols}")
        return frame_df.copy()
    
    # Sort by timestamp
    energy_df = energy_df.sort_values('t').reset_index(drop=True)
    
    # Create output DataFrame
    result_df = frame_df.copy()
    
    # Initialize energy columns
    result_df['energy_total_j'] = np.nan
    result_df['energy_total_per_frame'] = np.nan
    result_df['power_total_mean'] = np.nan
    result_df['power_total_max'] = np.nan
    
    # For each frame, find matching energy measurements
    for idx, row in result_df.iterrows():
        frame_timestamp = row.get('video_timestamp_sec')
        
        if pd.isna(frame_timestamp):
            continue
        
        # Find energy measurement at or just before frame timestamp
        # Use interpolation to get exact energy value
        matching_measurements = energy_df[energy_df['t'] <= frame_timestamp]
        
        if len(matching_measurements) == 0:
            # Frame is before first measurement, use first measurement
            if len(energy_df) > 0:
                result_df.at[idx, 'energy_total_j'] = energy_df.iloc[0]['energy_total_j']
        elif len(matching_measurements) == len(energy_df):
            # Frame is after last measurement, use last measurement
            result_df.at[idx, 'energy_total_j'] = energy_df.iloc[-1]['energy_total_j']
        else:
            # Interpolate between measurements
            prev_idx = matching_measurements.index[-1]
            next_idx = prev_idx + 1
            
            if next_idx < len(energy_df):
                prev_t = energy_df.iloc[prev_idx]['t']
                next_t = energy_df.iloc[next_idx]['t']
                prev_energy = energy_df.iloc[prev_idx]['energy_total_j']
                next_energy = energy_df.iloc[next_idx]['energy_total_j']
                
                # Linear interpolation
                if next_t > prev_t:
                    alpha = (frame_timestamp - prev_t) / (next_t - prev_t)
                    interpolated_energy = prev_energy + alpha * (next_energy - prev_energy)
                    result_df.at[idx, 'energy_total_j'] = interpolated_energy
                else:
                    result_df.at[idx, 'energy_total_j'] = prev_energy
            else:
                result_df.at[idx, 'energy_total_j'] = energy_df.iloc[prev_idx]['energy_total_j']
        
        # Compute per-frame energy (energy consumed during this frame)
        # Find previous frame's timestamp to compute delta
        if idx > 0:
            prev_frame_timestamp = result_df.iloc[idx - 1].get('video_timestamp_sec')
            if not pd.isna(prev_frame_timestamp) and not pd.isna(frame_timestamp):
                # Find energy at previous frame
                prev_matching = energy_df[energy_df['t'] <= prev_frame_timestamp]
                if len(prev_matching) > 0:
                    prev_energy_idx = prev_matching.index[-1]
                    if prev_energy_idx < len(energy_df):
                        prev_energy = energy_df.iloc[prev_energy_idx]['energy_total_j']
                        curr_energy = result_df.at[idx, 'energy_total_j']
                        if not pd.isna(curr_energy) and not pd.isna(prev_energy):
                            result_df.at[idx, 'energy_total_per_frame'] = curr_energy - prev_energy
        
        # Compute power statistics within frame interval
        if idx > 0:
            prev_frame_timestamp = result_df.iloc[idx - 1].get('video_timestamp_sec')
            if not pd.isna(prev_frame_timestamp) and not pd.isna(frame_timestamp):
                # Find power samples within frame interval
                frame_start = prev_frame_timestamp
                frame_end = frame_timestamp
                
                power_samples = energy_df[
                    (energy_df['t'] >= frame_start) & 
                    (energy_df['t'] <= frame_end)
                ]['power_total_w']
                
                if len(power_samples) > 0:
                    result_df.at[idx, 'power_total_mean'] = power_samples.mean()
                    result_df.at[idx, 'power_total_max'] = power_samples.max()
    
    # Add total run energy (same for all frames in run)
    if len(energy_df) > 0:
        total_run_energy = energy_df.iloc[-1]['energy_total_j']
        result_df['energy_total_run'] = total_run_energy
    else:
        result_df['energy_total_run'] = np.nan
    
    return result_df


def compute_per_frame_energy(energy_data: Dict, frame_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute energy consumed per frame and power statistics.
    
    This is a convenience wrapper around match_energy_to_frames.
    Energy consumed per frame = energy_at_frame_end - energy_at_frame_start.
    Mean/max power during frame interval from power samples.
    
    Args:
        energy_data: Energy data dictionary
        frame_df: DataFrame with frame data
        
    Returns:
        DataFrame with energy columns added
    """
    return match_energy_to_frames(energy_data, frame_df)


def integrate_energy_data(frame_df: pd.DataFrame, energy_dir: Optional[Path]) -> pd.DataFrame:
    """
    Integrate energy data for all runs in the frame DataFrame.
    
    Args:
        frame_df: DataFrame with frame data
        energy_dir: Path to storage/video_energy_results/ directory (optional)
        
    Returns:
        DataFrame with energy columns added (NaN if energy data not available)
    """
    if energy_dir is None:
        logger.info("No energy directory provided, skipping energy integration")
        # Add empty energy columns
        frame_df = frame_df.copy()
        frame_df['energy_total_j'] = np.nan
        frame_df['energy_total_per_frame'] = np.nan
        frame_df['power_total_mean'] = np.nan
        frame_df['power_total_max'] = np.nan
        frame_df['energy_total_run'] = np.nan
        return frame_df
    
    energy_path = Path(energy_dir)
    if not energy_path.exists():
        logger.warning(f"Energy directory does not exist: {energy_dir}")
        # Add empty energy columns
        frame_df = frame_df.copy()
        frame_df['energy_total_j'] = np.nan
        frame_df['energy_total_per_frame'] = np.nan
        frame_df['power_total_mean'] = np.nan
        frame_df['power_total_max'] = np.nan
        frame_df['energy_total_run'] = np.nan
        return frame_df
    
    # Group by run_id and process each run
    result_frames = []
    runs_processed = 0
    runs_with_energy = 0
    
    for run_id, run_frames in frame_df.groupby('run_id'):
        runs_processed += 1
        
        # Find energy file for this run
        energy_file = discover_energy_file(energy_path, run_id)
        
        if energy_file is None:
            logger.debug(f"No energy file found for run {run_id}")
            # Add NaN energy columns
            run_frames = run_frames.copy()
            run_frames['energy_total_j'] = np.nan
            run_frames['energy_total_per_frame'] = np.nan
            run_frames['power_total_mean'] = np.nan
            run_frames['power_total_max'] = np.nan
            run_frames['energy_total_run'] = np.nan
        else:
            # Load and integrate energy data
            energy_data = load_energy_data(energy_file)
            if energy_data:
                run_frames = match_energy_to_frames(energy_data, run_frames)
                runs_with_energy += 1
                logger.debug(f"Integrated energy data for run {run_id}")
            else:
                # Add NaN energy columns
                run_frames = run_frames.copy()
                run_frames['energy_total_j'] = np.nan
                run_frames['energy_total_per_frame'] = np.nan
                run_frames['power_total_mean'] = np.nan
                run_frames['power_total_max'] = np.nan
                run_frames['energy_total_run'] = np.nan
        
        result_frames.append(run_frames)
    
    # Combine all runs
    result_df = pd.concat(result_frames, ignore_index=True)
    
    logger.info(f"Energy integration complete: {runs_with_energy}/{runs_processed} runs have energy data")
    
    return result_df

