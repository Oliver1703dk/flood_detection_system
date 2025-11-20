"""
Utility functions for evaluation analysis scripts.
"""

import csv
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """
    Load ground truth annotations from CSV file.
    
    Expected format:
    video_file, frame_number, timestamp_sec, label, notes
    
    Returns:
        DataFrame with columns: video_file, frame_number, timestamp_sec, label
    """
    df = pd.read_csv(csv_path)
    required_cols = ['video_file', 'frame_number', 'timestamp_sec', 'label']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in ground truth CSV")
    
    return df[required_cols]


def load_json_results(results_dir: str, pattern: str = "**/*.json") -> List[Dict]:
    """
    Load all JSON result files from the specified directory.
    
    Args:
        results_dir: Path to storage/data_results/ directory
        pattern: Glob pattern to match JSON files
        
    Returns:
        List of dictionaries containing parsed JSON results
    """
    results = []
    results_path = Path(results_dir)
    
    for json_file in results_path.glob(pattern):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Add file path for reference
                data['_file_path'] = str(json_file)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def extract_prediction(result: Dict) -> Optional[int]:
    """
    Extract prediction from a result dictionary.
    Handles both FSM (FrameDecision) and yolo_sensor formats.
    
    Returns:
        Prediction: 0 (no-flood), 1 (watch), 2 (flood), or None if not found
    """
    # FSM mode: check classification_result dict
    if 'classification_result' in result:
        cls_result = result['classification_result']
        if isinstance(cls_result, dict) and 'prediction' in cls_result:
            return cls_result['prediction']
    
    # Direct prediction field
    if 'prediction' in result:
        return result['prediction']
    
    # yolo_sensor mode: check final_prediction
    if 'final_prediction' in result:
        return result['final_prediction']
    
    # Check classification_result string format
    if 'classification_result' in result:
        cls_str = str(result['classification_result']).lower()
        if 'no flood' in cls_str or 'no-flood' in cls_str:
            return 0
        elif 'watch' in cls_str or 'some water' in cls_str:
            return 1
        elif 'flood' in cls_str:
            return 2
    
    return None


def extract_frame_identifier(result: Dict) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    """
    Extract frame identification information from result.
    
    Returns:
        Tuple of (video_file, timestamp_sec, frame_number)
    """
    metadata = result.get('metadata', {})
    
    video_file = metadata.get('video_file')
    timestamp_sec = metadata.get('video_timestamp_sec')
    frame_number = metadata.get('frame_number')
    
    # Try alternative locations
    if video_file is None:
        video_file = result.get('video_file')
    if timestamp_sec is None:
        timestamp_sec = result.get('video_timestamp_sec')
    if frame_number is None:
        frame_number = result.get('frame_number')
    
    return video_file, timestamp_sec, frame_number


def match_result_to_ground_truth(result: Dict, ground_truth: pd.DataFrame, 
                                  tolerance_sec: float = 0.1) -> Optional[int]:
    """
    Match a result to ground truth annotation.
    
    Args:
        result: Result dictionary
        ground_truth: Ground truth DataFrame
        tolerance_sec: Timestamp matching tolerance in seconds
        
    Returns:
        Ground truth label or None if no match found
    """
    video_file, timestamp_sec, frame_number = extract_frame_identifier(result)
    
    if video_file is None:
        return None
    
    # Filter by video file
    video_gt = ground_truth[ground_truth['video_file'] == video_file]
    
    if video_gt.empty:
        return None
    
    # Try matching by frame number first (most reliable)
    if frame_number is not None:
        frame_match = video_gt[video_gt['frame_number'] == frame_number]
        if not frame_match.empty:
            return int(frame_match.iloc[0]['label'])
    
    # Fall back to timestamp matching
    if timestamp_sec is not None:
        video_gt['timestamp_diff'] = abs(video_gt['timestamp_sec'] - timestamp_sec)
        closest = video_gt.loc[video_gt['timestamp_diff'].idxmin()]
        
        if closest['timestamp_diff'] <= tolerance_sec:
            return int(closest['label'])
    
    return None


def load_power_logs(power_log_path: str) -> pd.DataFrame:
    """
    Load HMC power analyzer logs from CSV.
    
    Expected format: timestamp, voltage_V, current_A, power_W
    
    Returns:
        DataFrame with power measurements
    """
    df = pd.read_csv(power_log_path)
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in df.columns:
        raise ValueError("Power log must have 'timestamp' column")
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def integrate_power(power_df: pd.DataFrame, start_time: datetime, 
                    end_time: datetime) -> Dict[str, float]:
    """
    Integrate power measurements over time to compute energy.
    
    Args:
        power_df: DataFrame with power measurements
        start_time: Start of measurement window
        end_time: End of measurement window
        
    Returns:
        Dictionary with energy_j, avg_power_w, peak_power_w
    """
    # Filter to time window
    mask = (power_df['timestamp'] >= start_time) & (power_df['timestamp'] <= end_time)
    window_df = power_df[mask].copy()
    
    if window_df.empty:
        return {'energy_j': 0.0, 'avg_power_w': 0.0, 'peak_power_w': 0.0}
    
    # Compute time deltas between samples
    window_df = window_df.sort_values('timestamp')
    window_df['time_delta_s'] = window_df['timestamp'].diff().dt.total_seconds()
    
    # For first sample, use half the delta to next sample
    if len(window_df) > 1:
        window_df.loc[window_df.index[0], 'time_delta_s'] = \
            window_df.loc[window_df.index[1], 'time_delta_s'] / 2
    
    # Integrate: Energy = Sum(Power * time_delta)
    window_df['energy_j'] = window_df['power_W'] * window_df['time_delta_s']
    total_energy_j = window_df['energy_j'].sum()
    
    # Compute statistics
    avg_power_w = window_df['power_W'].mean()
    peak_power_w = window_df['power_W'].max()
    
    return {
        'energy_j': total_energy_j,
        'avg_power_w': avg_power_w,
        'peak_power_w': peak_power_w,
        'duration_s': (end_time - start_time).total_seconds(),
        'sample_count': len(window_df)
    }


def parse_timestamp(ts_value) -> Optional[datetime]:
    """
    Parse various timestamp formats to datetime.
    
    Args:
        ts_value: String, float (unix timestamp), or datetime
        
    Returns:
        datetime object or None
    """
    if ts_value is None:
        return None
    
    if isinstance(ts_value, datetime):
        return ts_value
    
    if isinstance(ts_value, (int, float)):
        try:
            return datetime.fromtimestamp(ts_value)
        except:
            return None
    
    if isinstance(ts_value, str):
        try:
            # Try ISO format
            return datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
        except:
            try:
                # Try unix timestamp as string
                return datetime.fromtimestamp(float(ts_value))
            except:
                return None
    
    return None

