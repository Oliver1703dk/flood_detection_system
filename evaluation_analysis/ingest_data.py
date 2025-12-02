"""
Data ingestion module for system evaluation.

This module handles discovering runs, loading ground truth labels,
parsing frame JSON files, and matching frames to ground truth.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RunInfo:
    """Information about a single run."""
    def __init__(self, ablation_name: str, run_path: Path, run_index: int):
        self.ablation_name = ablation_name
        self.run_path = run_path
        self.run_index = run_index


def discover_runs(results_dir: Path) -> List[RunInfo]:
    """
    Walk storage/video_results/ and discover all ablation folders and run directories.
    
    For each ablation folder, discover all run_<timestamp> subdirectories
    and sort them chronologically.
    
    Args:
        results_dir: Path to storage/video_results/ directory
        
    Returns:
        List of RunInfo objects, sorted by ablation name and run timestamp
    """
    runs = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return runs
    
    # Find all ablation folders (ablation1, ablation1b, ablation2, etc.)
    for ablation_dir in sorted(results_path.iterdir()):
        if not ablation_dir.is_dir():
            continue
        
        ablation_name = ablation_dir.name
        if not ablation_name.startswith('ablation'):
            continue
        
        # Extract ablation number/variant (e.g., "1", "1b", "2", "2b")
        match = re.match(r'ablation(\d+[a-z]?)', ablation_name)
        if not match:
            logger.warning(f"Could not parse ablation name: {ablation_name}")
            continue
        
        # Find all run_<timestamp> subdirectories
        run_dirs = []
        for item in ablation_dir.iterdir():
            if item.is_dir() and item.name.startswith('run_'):
                # Extract timestamp from directory name
                timestamp_str = item.name.replace('run_', '')
                try:
                    # Parse timestamp: YYYYMMDD-HHMMSS-XXXXXX
                    parts = timestamp_str.split('-')
                    if len(parts) >= 2:
                        date_part = parts[0]  # YYYYMMDD
                        time_part = parts[1]  # HHMMSS
                        # Create sortable key
                        sort_key = f"{date_part}{time_part}"
                        run_dirs.append((sort_key, item))
                except Exception as e:
                    logger.warning(f"Could not parse run timestamp from {item.name}: {e}")
                    continue
        
        # Sort runs chronologically
        run_dirs.sort(key=lambda x: x[0])
        
        # Create RunInfo objects with 0-based index
        for run_index, (_, run_path) in enumerate(run_dirs):
            runs.append(RunInfo(ablation_name, run_path, run_index))
    
    logger.info(f"Discovered {len(runs)} runs across {len(set(r.ablation_name for r in runs))} ablation configs")
    return runs


def extract_run_metadata(run_path: Path, sample_json: Dict, run_index_in_ablation: int) -> Dict:
    """
    Extract run-level metadata from a sample JSON file.
    
    Args:
        run_path: Path to the run directory
        sample_json: A sample JSON file from the run (for metadata extraction)
        run_index_in_ablation: 0-based index of run within its ablation folder
        
    Returns:
        Dictionary with run metadata:
        - ablation_name: Ablation number/variant (e.g., "1", "1b", "2")
        - sequence_id: 6-digit timestamp from video filename (e.g., "145739")
        - sensor_prediction: Sensor prediction from classification_result (if FSM)
        - repeat_index: 0-based index of this run within its ablation
    """
    metadata = {}
    
    # Extract ablation name (preserve variant suffix)
    ablation_folder = run_path.parent.name
    match = re.match(r'ablation(\d+[a-z]?)', ablation_folder)
    if match:
        metadata['ablation_name'] = match.group(1)  # e.g., "1", "1b", "2"
    else:
        metadata['ablation_name'] = ablation_folder.replace('ablation', '')
    
    # Extract sequence_id from video_file in metadata
    video_file = sample_json.get('metadata', {}).get('video_file', '')
    match = re.search(r'flood_video_\d{8}_(\d{6})\.mp4', video_file)
    if match:
        metadata['sequence_id'] = match.group(1)  # e.g., "145739"
    else:
        metadata['sequence_id'] = None
        logger.warning(f"Could not extract sequence_id from video_file: {video_file}")
    
    # Extract sensor_prediction from classification_result
    classification_result = sample_json.get('classification_result', {})
    if isinstance(classification_result, dict):
        scores = classification_result.get('scores', {})
        sensor_pred = scores.get('sensor_prediction', 'neutral')
        metadata['sensor_prediction'] = sensor_pred
    else:
        # Non-FSM configs don't have sensor_prediction
        metadata['sensor_prediction'] = 'neutral'
    
    # Set repeat_index
    metadata['repeat_index'] = run_index_in_ablation
    
    return metadata


def load_ground_truth(gt_dir: Path) -> Dict[str, Dict[int, int]]:
    """
    Load all ground truth label files from test_videos/labels/.
    
    Each file is a single JSON object (not line-by-line JSONL) with format:
    {"video": "flood_video_20251005_145739.mp4", "fps_truth": 1, "labels": [{"t": 0, "y": "flood"}, ...]}
    
    Label values: "flood" → 2, "sus" → 1, "no_flood" → 0
    
    Args:
        gt_dir: Path to test_videos/labels/ directory
        
    Returns:
        Dictionary mapping video_filename -> {timestamp_int: label_int}
    """
    gt_dict = {}
    gt_path = Path(gt_dir)
    
    if not gt_path.exists():
        logger.warning(f"Ground truth directory does not exist: {gt_dir}")
        return gt_dict
    
    label_mapping = {
        "flood": 2,
        "sus": 1,
        "no_flood": 0
    }
    
    for gt_file in sorted(gt_path.glob('*.jsonl')):
        try:
            with open(gt_file, 'r') as f:
                # Read entire file as single JSON object
                data = json.load(f)
                
                video_filename = data.get('video', '')
                labels = data.get('labels', [])
                
                if not video_filename:
                    logger.warning(f"No video field in {gt_file.name}")
                    continue
                
                # Build timestamp -> label mapping
                video_gt = {}
                for label_entry in labels:
                    timestamp = label_entry.get('t')
                    label_str = label_entry.get('y', '').lower()
                    
                    if timestamp is not None and label_str in label_mapping:
                        video_gt[int(timestamp)] = label_mapping[label_str]
                
                gt_dict[video_filename] = video_gt
                logger.debug(f"Loaded {len(video_gt)} labels for {video_filename}")
                
        except Exception as e:
            logger.warning(f"Failed to load ground truth file {gt_file.name}: {e}")
    
    logger.info(f"Loaded ground truth for {len(gt_dict)} videos")
    return gt_dict


def load_frame_data(json_path: Path) -> Dict:
    """
    Parse a single JSON file and extract all fields.
    
    Handles both formats for classification_result:
    - String format (non-FSM): "No Flood", "Some Water", "Flooded"
    - Dict format (FSM): Detailed dict with scores, counters, etc.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary with all extracted fields
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load JSON file {json_path}: {e}")
        return {}
    
    return data


def normalize_classification_result(classification_result) -> Dict:
    """
    Normalize classification_result to a consistent format.
    
    For string format (non-FSM): Convert to dict with prediction mapped to int.
    For dict format (FSM): Return as-is.
    
    Args:
        classification_result: Either a string or dict
        
    Returns:
        Normalized dict with all fields
    """
    if isinstance(classification_result, str):
        # Non-FSM format: convert string to normalized dict
        string_to_int = {
            "No Flood": 0,
            "Some Water": 1,
            "Flooded": 2
        }
        prediction_str = classification_result.strip()
        prediction_int = string_to_int.get(prediction_str, None)
        
        return {
            'prediction': prediction_int,
            'state': None,
            'model_tier': None,
            'scores': {
                'combined_score': np.nan,
                'image_score': np.nan,
                'sensor_boost': np.nan,
                'sensor_prediction': 'neutral'
            },
            'counters': {
                'high': None,
                'low': None,
                'ambiguous': None,
                'conflict': None,
                'mid': None
            },
            'llm_used': False,
            'llm_prediction': None,
            'conflict': False,
            'drift': False,
            'flapping': False,
            'skipped': False,
            'model_switched': False,
            'tier_requested': None,
            'backend': None,
            'backend_info': {},
            's2_llm_confirmed': False
        }
    elif isinstance(classification_result, dict):
        # FSM format: return as-is, but ensure all fields exist
        normalized = classification_result.copy()
        
        # Ensure scores dict exists
        if 'scores' not in normalized:
            normalized['scores'] = {}
        
        # Ensure counters dict exists
        if 'counters' not in normalized:
            normalized['counters'] = {}
        
        # Fill missing fields with defaults
        defaults = {
            'state': None,
            'model_tier': None,
            'prediction': None,
            'llm_used': False,
            'llm_prediction': None,
            'conflict': False,
            'drift': False,
            'flapping': False,
            'skipped': False,
            'model_switched': False,
            'tier_requested': None,
            'backend': None,
            'backend_info': {},
            's2_llm_confirmed': False
        }
        
        for key, default_value in defaults.items():
            if key not in normalized:
                normalized[key] = default_value
        
        return normalized
    else:
        logger.warning(f"Unexpected classification_result type: {type(classification_result)}")
        return {}


def match_frame_to_ground_truth(frame_data: Dict, gt: Dict[str, Dict[int, int]]) -> Optional[int]:
    """
    Match a frame to ground truth by video_file + rounded video_timestamp_sec.
    
    Args:
        frame_data: Frame data dictionary
        gt: Ground truth dictionary {video_filename: {timestamp_int: label_int}}
        
    Returns:
        Ground truth label (0, 1, or 2) or None if no match
    """
    metadata = frame_data.get('metadata', {})
    video_file = metadata.get('video_file')
    video_timestamp_sec = metadata.get('video_timestamp_sec')
    
    if video_file is None or video_timestamp_sec is None:
        return None
    
    # Round timestamp to nearest integer second
    timestamp_rounded = int(round(video_timestamp_sec))
    
    # Lookup in GT dict
    if video_file in gt:
        video_gt = gt[video_file]
        return video_gt.get(timestamp_rounded)
    
    return None


def build_frame_dataframe(runs: List[RunInfo], gt_dict: Dict[str, Dict[int, int]]) -> pd.DataFrame:
    """
    Build a comprehensive DataFrame with all frames from all runs.
    
    Args:
        runs: List of RunInfo objects
        gt_dict: Ground truth dictionary
        
    Returns:
        DataFrame with all frame-level data
    """
    all_frames = []
    
    for run_info in runs:
        logger.info(f"Processing run {run_info.ablation_name}/{run_info.run_path.name} ({run_info.run_index + 1}/{len(runs)})")
        
        # Find all JSON files in run directory
        json_files = sorted(run_info.run_path.glob('*.json'))
        
        if not json_files:
            logger.warning(f"No JSON files found in {run_info.run_path}")
            continue
        
        # Load a sample JSON to extract run metadata
        sample_data = load_frame_data(json_files[0])
        if not sample_data:
            logger.warning(f"Could not load sample JSON from {run_info.run_path}")
            continue
        
        run_metadata = extract_run_metadata(run_info.run_path, sample_data, run_info.run_index)
        
        # Process each frame
        for json_file in json_files:
            frame_data = load_frame_data(json_file)
            if not frame_data:
                continue
            
            # Normalize classification_result
            classification_result = frame_data.get('classification_result', {})
            normalized_result = normalize_classification_result(classification_result)
            
            # Determine config type
            original_result = frame_data.get('classification_result', {})
            config_type = "fsm" if isinstance(original_result, dict) else "baseline"
            
            # Extract metadata
            metadata = frame_data.get('metadata', {})
            sensor_data = frame_data.get('sensor_data', {})
            timing = frame_data.get('timing', {})
            
            # Match to ground truth
            gt_label = match_frame_to_ground_truth(frame_data, gt_dict)
            
            # Build frame record
            frame_record = {
                # Identifiers
                'run_id': metadata.get('run_id'),
                'ablation_name': run_metadata['ablation_name'],
                'sequence_id': run_metadata['sequence_id'],
                'sensor_prediction': run_metadata['sensor_prediction'],
                'repeat_index': run_metadata['repeat_index'],
                'config_type': config_type,
                
                # Frame info
                'frame_index': normalized_result.get('frame_index'),
                'video_file': metadata.get('video_file'),
                'video_timestamp_sec': metadata.get('video_timestamp_sec'),
                'video_timestamp_sec_rounded': int(round(metadata.get('video_timestamp_sec', 0))) if metadata.get('video_timestamp_sec') is not None else None,
                
                # Labels
                'gt_label': gt_label,
                'pred_label': normalized_result.get('prediction'),
                
                # Scores (FSM only, NaN for non-FSM)
                'image_score': normalized_result.get('scores', {}).get('image_score'),
                'sensor_boost': normalized_result.get('scores', {}).get('sensor_boost'),
                'combined_score': normalized_result.get('scores', {}).get('combined_score'),
                
                # FSM state (FSM only)
                'fsm_state': normalized_result.get('state'),
                'tier_used': normalized_result.get('model_tier'),
                'tier_requested': normalized_result.get('tier_requested'),
                
                # Metadata
                'motion': metadata.get('motion'),
                'resource_constrained': metadata.get('resource_constrained'),
                'backend': normalized_result.get('backend'),
                
                # Flags (FSM only)
                'skipped': normalized_result.get('skipped'),
                'conflict': normalized_result.get('conflict'),
                'drift': normalized_result.get('drift'),
                'flapping': normalized_result.get('flapping'),
                'model_switched': normalized_result.get('model_switched'),
                'llm_used': normalized_result.get('llm_used'),
                's2_llm_confirmed': normalized_result.get('s2_llm_confirmed'),
                
                # Counters (FSM only)
                'counter_high': normalized_result.get('counters', {}).get('high'),
                'counter_low': normalized_result.get('counters', {}).get('low'),
                'counter_ambiguous': normalized_result.get('counters', {}).get('ambiguous'),
                'counter_conflict': normalized_result.get('counters', {}).get('conflict'),
                'counter_mid': normalized_result.get('counters', {}).get('mid'),
                
                # Timing
                'total_pipeline_latency_s': timing.get('total_pipeline_latency_s'),
                
                # Sensor data
                'temperature': sensor_data.get('temperature'),
                'humidity': sensor_data.get('humidity'),
                'pressure': sensor_data.get('pressure'),
                
                # Sensor baselines (from metadata if present)
                'temperature_baseline': metadata.get('sensor_baseline', {}).get('temperature_baseline') if isinstance(metadata.get('sensor_baseline'), dict) else None,
                'humidity_baseline': metadata.get('sensor_baseline', {}).get('humidity_baseline') if isinstance(metadata.get('sensor_baseline'), dict) else None,
                'pressure_baseline': metadata.get('sensor_baseline', {}).get('pressure_baseline') if isinstance(metadata.get('sensor_baseline'), dict) else None,
                
                # Sensor anomalies (if present)
                'delta_temperature': metadata.get('sensor_anomalies', {}).get('delta_temperature') if isinstance(metadata.get('sensor_anomalies'), dict) else None,
                'delta_humidity': metadata.get('sensor_anomalies', {}).get('delta_humidity') if isinstance(metadata.get('sensor_anomalies'), dict) else None,
                'delta_pressure': metadata.get('sensor_anomalies', {}).get('delta_pressure') if isinstance(metadata.get('sensor_anomalies'), dict) else None,
            }
            
            all_frames.append(frame_record)
        
        logger.info(f"  Loaded {len(json_files)} frames from run {run_info.run_path.name}")
    
    # Create DataFrame
    df = pd.DataFrame(all_frames)
    
    # Data validation
    if len(df) > 0:
        # Check for duplicate frames
        duplicates = df.duplicated(subset=['run_id', 'frame_index'], keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate frames (same run_id + frame_index)")
        
        # Report frames without ground truth
        missing_gt = df['gt_label'].isna().sum()
        if missing_gt > 0:
            logger.warning(f"{missing_gt} frames ({missing_gt/len(df)*100:.1f}%) have no ground truth match")
        
        # Check for incomplete runs
        run_frame_counts = df.groupby('run_id').size()
        min_frames = run_frame_counts.min()
        max_frames = run_frame_counts.max()
        logger.info(f"Frames per run: min={min_frames}, max={max_frames}, mean={run_frame_counts.mean():.1f}")
    
    logger.info(f"Built DataFrame with {len(df)} frames from {len(runs)} runs")
    return df

