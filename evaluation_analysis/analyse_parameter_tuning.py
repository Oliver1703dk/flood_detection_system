"""
Comprehensive parameter tuning analysis for flood detection system.
Analyzes results across all ablations to identify optimal thresholds and parameters.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Map ground truth labels to numeric values
GT_LABEL_MAP = {
    "no_flood": 0,
    "sus": 1,  # suspicious/watch
    "flood": 2
}

# Map predictions to numeric values
PRED_MAP = {
    "No Flood": 0,
    "no_flood": 0,
    "Watch": 1,
    "Some Water": 1,
    "some_water": 1,
    "Flooded": 2,
    "flood": 2
}


def load_ground_truth_jsonl(jsonl_path: str) -> Dict[str, Dict[int, int]]:
    """Load ground truth from JSONL file. Returns dict: {video_file: {t: label}}"""
    gt = defaultdict(dict)
    
    with open(jsonl_path, 'r') as f:
        data = json.load(f)
        video_file = data['video']
        for label_entry in data['labels']:
            t = int(label_entry['t'])
            label_str = label_entry['y']
            label_num = GT_LABEL_MAP.get(label_str, 1)  # Default to sus if unknown
            gt[video_file][t] = label_num
    
    return dict(gt)


def extract_prediction_from_result(result: Dict) -> Optional[int]:
    """Extract numeric prediction from result dict."""
    cls_result = result.get('classification_result')
    
    if isinstance(cls_result, dict):
        # FSM format with detailed classification_result
        pred = cls_result.get('prediction')
        if pred is not None:
            return int(pred)
    
    if isinstance(cls_result, str):
        # String format like "Flooded", "No Flood", etc.
        cls_lower = cls_result.lower()
        if 'no flood' in cls_lower or 'no-flood' in cls_lower:
            return 0
        elif 'watch' in cls_lower or 'some water' in cls_lower:
            return 1
        elif 'flood' in cls_lower:
            return 2
    
    return None


def extract_scores_from_result(result: Dict) -> Dict[str, float]:
    """Extract scores from result dict."""
    scores = {}
    
    cls_result = result.get('classification_result')
    if isinstance(cls_result, dict) and 'scores' in cls_result:
        scores_dict = cls_result['scores']
        scores['combined_score'] = scores_dict.get('combined_score', 0.0)
        scores['image_score'] = scores_dict.get('image_score', 0.0)
        scores['sensor_boost'] = scores_dict.get('sensor_boost', 0.0)
    
    return scores


def load_ablation_results(results_dir: Path, ablation_name: str) -> List[Dict]:
    """Load all JSON results for a specific ablation."""
    ablation_dir = results_dir / ablation_name
    if not ablation_dir.exists():
        return []
    
    results = []
    for video_dir in ablation_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        for json_file in video_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['_ablation'] = ablation_name
                    data['_file_path'] = str(json_file)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    return results


def match_result_to_ground_truth(result: Dict, gt: Dict[str, Dict[int, int]], 
                                  tolerance_sec: float = 0.5) -> Optional[int]:
    """Match result to ground truth by video_timestamp_sec."""
    metadata = result.get('metadata', {})
    video_file = metadata.get('video_file')
    timestamp_sec = metadata.get('video_timestamp_sec')
    
    if video_file is None or timestamp_sec is None:
        return None
    
    if video_file not in gt:
        return None
    
    # Round to nearest second
    t = int(round(timestamp_sec))
    
    # Check exact match first
    if t in gt[video_file]:
        return gt[video_file][t]
    
    # Check within tolerance
    for gt_t, label in gt[video_file].items():
        if abs(gt_t - timestamp_sec) <= tolerance_sec:
            return label
    
    return None


def analyze_ablation(results: List[Dict], gt: Dict[str, Dict[int, int]], 
                     ablation_name: str) -> pd.DataFrame:
    """Analyze results for a single ablation."""
    matched_data = []
    
    for result in results:
        pred = extract_prediction_from_result(result)
        if pred is None:
            continue
        
        gt_label = match_result_to_ground_truth(result, gt)
        if gt_label is None:
            continue
        
        scores = extract_scores_from_result(result)
        metadata = result.get('metadata', {})
        
        matched_data.append({
            'ablation': ablation_name,
            'video_file': metadata.get('video_file', 'unknown'),
            'video_timestamp_sec': metadata.get('video_timestamp_sec', 0),
            'prediction': pred,
            'ground_truth': gt_label,
            'combined_score': scores.get('combined_score', None),
            'image_score': scores.get('image_score', None),
            'sensor_boost': scores.get('sensor_boost', None),
            'correct': pred == gt_label,
        })
    
    return pd.DataFrame(matched_data)


def compute_metrics(df: pd.DataFrame) -> Dict:
    """Compute accuracy metrics."""
    if len(df) == 0:
        return {}
    
    correct = df['correct'].sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-class metrics
    metrics = {'accuracy': accuracy, 'total': total, 'correct': correct}
    
    for label in [0, 1, 2]:
        label_df = df[df['ground_truth'] == label]
        if len(label_df) > 0:
            label_correct = label_df['correct'].sum()
            metrics[f'class_{label}_accuracy'] = label_correct / len(label_df)
            metrics[f'class_{label}_support'] = len(label_df)
    
    return metrics


def analyze_threshold_performance(df: pd.DataFrame, threshold_low: float, 
                                   threshold_high: float, verbose: bool = False,
                                   use_balanced_accuracy: bool = True,
                                   class_weights: Optional[Dict[int, float]] = None) -> Dict:
    """Analyze performance with specific thresholds.
    
    Args:
        class_weights: Optional dict mapping class label to weight.
                      If None, uses equal weights (balanced accuracy).
                      Example: {0: 1.0, 1: 0.0, 2: 1.0} to ignore class 1.
    """
    if 'combined_score' not in df.columns or df['combined_score'].isna().all():
        return {}
    
    # Apply thresholds
    df_copy = df.copy()
    df_copy['pred_thresh'] = 0
    df_copy.loc[df_copy['combined_score'] >= threshold_low, 'pred_thresh'] = 1
    df_copy.loc[df_copy['combined_score'] >= threshold_high, 'pred_thresh'] = 2
    
    # Compute accuracy
    correct = (df_copy['pred_thresh'] == df_copy['ground_truth']).sum()
    accuracy = correct / len(df_copy) if len(df_copy) > 0 else 0.0
    
    # Compute per-class metrics
    metrics = {
        'threshold_low': threshold_low,
        'threshold_high': threshold_high,
        'accuracy': accuracy,
        'correct': correct,
        'total': len(df_copy)
    }
    
    # Per-class accuracy
    class_accuracies = []
    class_weights_list = []
    for label in [0, 1, 2]:
        label_df = df_copy[df_copy['ground_truth'] == label]
        if len(label_df) > 0:
            label_correct = (label_df['pred_thresh'] == label_df['ground_truth']).sum()
            class_acc = label_correct / len(label_df)
            metrics[f'class_{label}_accuracy'] = class_acc
            metrics[f'class_{label}_support'] = len(label_df)
            class_accuracies.append(class_acc)
            
            # Get weight for this class
            if class_weights is not None:
                weight = class_weights.get(label, 0.0)
            else:
                weight = 1.0  # Equal weight for balanced accuracy
            class_weights_list.append(weight)
            
            # Confusion matrix entries
            pred_counts = label_df['pred_thresh'].value_counts().to_dict()
            metrics[f'class_{label}_pred_0'] = pred_counts.get(0, 0)
            metrics[f'class_{label}_pred_1'] = pred_counts.get(1, 0)
            metrics[f'class_{label}_pred_2'] = pred_counts.get(2, 0)
        else:
            # If class doesn't exist in data, set accuracy to 0
            metrics[f'class_{label}_accuracy'] = 0.0
            metrics[f'class_{label}_support'] = 0
            class_accuracies.append(0.0)
            if class_weights is not None:
                class_weights_list.append(class_weights.get(label, 0.0))
            else:
                class_weights_list.append(1.0)
    
    # Compute weighted accuracy (prioritizes classes 0 and 2)
    if class_weights is not None and len(class_accuracies) > 0:
        # Weighted average: sum(accuracy * weight) / sum(weights) for classes with support
        weighted_sum = sum(acc * weight for acc, weight in zip(class_accuracies, class_weights_list))
        total_weight = sum(class_weights_list)
        weighted_acc = weighted_sum / total_weight if total_weight > 0 else 0.0
        metrics['weighted_accuracy'] = weighted_acc
        metrics['score'] = weighted_acc
    elif use_balanced_accuracy and len(class_accuracies) > 0:
        # Original balanced accuracy (equal weights)
        balanced_acc = np.mean(class_accuracies)
        metrics['balanced_accuracy'] = balanced_acc
        metrics['score'] = balanced_acc
    else:
        metrics['balanced_accuracy'] = accuracy
        metrics['score'] = accuracy
    
    # Also compute accuracy on only classes 0 and 2 (ignoring class 1)
    if len(class_accuracies) >= 3:
        class_0_2_accuracies = [class_accuracies[0], class_accuracies[2]]
        metrics['class_0_2_accuracy'] = np.mean(class_0_2_accuracies)
    
    if verbose:
        print(f"  Thresholds: low={threshold_low:.3f}, high={threshold_high:.3f}")
        print(f"  Overall accuracy: {accuracy:.4f} ({correct}/{len(df_copy)})")
        if 'weighted_accuracy' in metrics:
            print(f"  Weighted accuracy: {metrics['weighted_accuracy']:.4f}")
        if 'balanced_accuracy' in metrics:
            print(f"  Balanced accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        if 'class_0_2_accuracy' in metrics:
            print(f"  Classes 0+2 accuracy (ignoring class 1): {metrics['class_0_2_accuracy']:.4f}")
        for label in [0, 1, 2]:
            if f'class_{label}_support' in metrics:
                print(f"  Class {label}: accuracy={metrics[f'class_{label}_accuracy']:.4f}, "
                      f"support={metrics[f'class_{label}_support']}, "
                      f"predictions: 0={metrics.get(f'class_{label}_pred_0', 0)}, "
                      f"1={metrics.get(f'class_{label}_pred_1', 0)}, "
                      f"2={metrics.get(f'class_{label}_pred_2', 0)}")
    
    return metrics


def find_optimal_thresholds(df: pd.DataFrame, step: float = 0.05, 
                            use_distribution_aware: bool = True,
                            debug: bool = False,
                            min_threshold_gap: float = 0.2,
                            use_balanced_accuracy: bool = True,
                            class_weights: Optional[Dict[int, float]] = None) -> Dict:
    """
    Grid search for optimal thresholds with distribution-aware initialization.
    
    Args:
        df: DataFrame with 'combined_score' and 'ground_truth' columns
        step: Grid search step size (default 0.05 for coarse, 0.01 for fine)
        use_distribution_aware: If True, use score distributions to guide search
        debug: If True, print detailed debugging information
        min_threshold_gap: Minimum gap required between low and high thresholds
        use_balanced_accuracy: If True, use balanced accuracy (ignored if class_weights provided)
        class_weights: Optional dict mapping class label to weight.
                      If provided, uses weighted accuracy instead of balanced.
                      Example: {0: 1.0, 1: 0.0, 2: 1.0} to optimize for classes 0 and 2 only.
    """
    if 'combined_score' not in df.columns or df['combined_score'].isna().all():
        return {}
    
    valid_df = df[df['combined_score'].notna()].copy()
    if len(valid_df) == 0:
        return {}
    
    # Analyze score distributions
    distributions = analyze_score_distributions(valid_df)
    
    if debug:
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION DEBUG")
        print("="*60)
        print(f"Total samples: {len(valid_df)}")
        print(f"Score range: [{valid_df['combined_score'].min():.3f}, {valid_df['combined_score'].max():.3f}]")
        print("\nScore distributions by class:")
        for label in [0, 1, 2]:
            label_df = valid_df[valid_df['ground_truth'] == label]
            if len(label_df) > 0:
                print(f"  Class {label}: count={len(label_df)}, "
                      f"median={distributions.get(f'class_{label}_median', 0):.3f}, "
                      f"q25={distributions.get(f'class_{label}_q25', 0):.3f}, "
                      f"q75={distributions.get(f'class_{label}_q75', 0):.3f}")
    
    min_score = valid_df['combined_score'].min()
    max_score = valid_df['combined_score'].max()
    
    # Distribution-aware initial search bounds
    if use_distribution_aware and distributions:
        # Low threshold should be above class 0's q75 (to avoid false positives)
        class_0_q75 = distributions.get('class_0_q75', 0.11)
        class_1_q25 = distributions.get('class_1_q25', 0.11)
        class_1_q75 = distributions.get('class_1_q75', 0.5)
        class_2_q25 = distributions.get('class_2_q25', 1.0)
        
        # Account for sensor boost baseline (typically 0.11)
        sensor_boost_baseline = 0.11
        suggested_low_min = max(class_0_q75 + 0.05, sensor_boost_baseline + 0.05)
        suggested_low_max = min(class_1_q75, 0.5)  # Don't go too high
        
        suggested_high_min = max(class_1_q75, class_2_q25 * 0.5)  # Between class 1 and 2
        suggested_high_max = min(class_2_q25 * 1.5, max_score)
        
        if debug:
            print(f"\nDistribution-aware search bounds:")
            print(f"  Suggested low threshold range: [{suggested_low_min:.3f}, {suggested_low_max:.3f}]")
            print(f"  Suggested high threshold range: [{suggested_high_min:.3f}, {suggested_high_max:.3f}]")
        
        # Use distribution-aware bounds, but don't restrict too much
        search_low_min = max(0.1, suggested_low_min - 0.1)
        search_low_max = min(max_score, suggested_low_max + 0.2)
        # Ensure high threshold is at least min_threshold_gap above low threshold
        search_high_min = max(0.2, suggested_high_min - 0.2, search_low_min + min_threshold_gap)
        search_high_max = min(max_score, suggested_high_max + 0.5)
    else:
        search_low_min = 0.1
        search_low_max = min(max_score, 1.0)  # Cap at 1.0 for low threshold
        # Ensure high threshold is at least min_threshold_gap above low threshold
        search_high_min = max(0.2, search_low_min + min_threshold_gap)
        search_high_max = max_score
    
    if debug:
        print(f"\nGrid search bounds:")
        print(f"  Low threshold: [{search_low_min:.3f}, {search_low_max:.3f}]")
        print(f"  High threshold: [{search_high_min:.3f}, {search_high_max:.3f}]")
        print(f"  Step size: {step}")
        print(f"  Minimum threshold gap: {min_threshold_gap:.3f}")
        if class_weights is not None:
            print(f"  Using weighted accuracy with weights: {class_weights}")
        else:
            print(f"  Using balanced accuracy: {use_balanced_accuracy}")
    
    best_accuracy = 0.0
    best_thresholds = None
    best_metrics = None
    iterations = 0
    
    # Coarse grid search
    if debug:
        print(f"\nStarting coarse grid search...")
    
    for thresh_low in np.arange(search_low_min, search_low_max, step):
        # Ensure high threshold is at least min_threshold_gap above low threshold
        high_min = max(thresh_low + min_threshold_gap, search_high_min)
        for thresh_high in np.arange(high_min, 
                                     min(search_high_max, max_score + step), step):
            iterations += 1
            metrics = analyze_threshold_performance(valid_df, thresh_low, thresh_high,
                                                    use_balanced_accuracy=use_balanced_accuracy,
                                                    class_weights=class_weights)
            # Use weighted accuracy, balanced accuracy, or overall accuracy based on flags
            score = metrics.get('score', metrics.get('accuracy', 0))
            
            if score > best_accuracy:
                best_accuracy = score
                best_thresholds = (thresh_low, thresh_high)
                best_metrics = metrics.copy()
                if debug and iterations % 50 == 0:
                    if class_weights is not None:
                        score_type = "weighted_acc"
                    elif use_balanced_accuracy:
                        score_type = "balanced_acc"
                    else:
                        score_type = "accuracy"
                    print(f"  New best: low={thresh_low:.3f}, high={thresh_high:.3f}, "
                          f"{score_type}={score:.4f}")
    
    if debug:
        print(f"\nCoarse search complete: {iterations} iterations")
        if best_thresholds:
            if class_weights is not None:
                score_type = "weighted_acc"
            elif use_balanced_accuracy:
                score_type = "balanced_acc"
            else:
                score_type = "accuracy"
            print(f"Best thresholds found: low={best_thresholds[0]:.3f}, high={best_thresholds[1]:.3f}")
            print(f"Best {score_type}: {best_accuracy:.4f}")
            if best_metrics:
                analyze_threshold_performance(valid_df, best_thresholds[0], 
                                            best_thresholds[1], verbose=True,
                                            use_balanced_accuracy=use_balanced_accuracy,
                                            class_weights=class_weights)
    
    # Fine-tune around best result if we found one
    if best_thresholds and step >= 0.05:
        if debug:
            print(f"\nStarting fine-tuning search (step=0.01)...")
        
        fine_step = 0.01
        fine_low_min = max(0.05, best_thresholds[0] - 0.1)
        fine_low_max = min(max_score, best_thresholds[0] + 0.1)
        fine_high_min = max(0.1, best_thresholds[1] - 0.2)
        fine_high_max = min(max_score, best_thresholds[1] + 0.2)
        
        for thresh_low in np.arange(fine_low_min, fine_low_max, fine_step):
            # Ensure high threshold is at least min_threshold_gap above low threshold
            fine_high_min_actual = max(thresh_low + min_threshold_gap, fine_high_min)
            for thresh_high in np.arange(fine_high_min_actual,
                                         min(fine_high_max, max_score + fine_step), fine_step):
                metrics = analyze_threshold_performance(valid_df, thresh_low, thresh_high,
                                                        use_balanced_accuracy=use_balanced_accuracy,
                                                        class_weights=class_weights)
                score = metrics.get('score', metrics.get('accuracy', 0))
                
                if score > best_accuracy:
                    best_accuracy = score
                    best_thresholds = (thresh_low, thresh_high)
                    best_metrics = metrics.copy()
        
        if debug:
            print(f"Fine-tuning complete")
            if best_thresholds:
                if class_weights is not None:
                    score_type = "weighted_acc"
                elif use_balanced_accuracy:
                    score_type = "balanced_acc"
                else:
                    score_type = "accuracy"
                print(f"Final thresholds: low={best_thresholds[0]:.3f}, high={best_thresholds[1]:.3f}")
                print(f"Final {score_type}: {best_accuracy:.4f}")
    
    if best_thresholds:
        # Get current performance for comparison
        current_metrics = analyze_threshold_performance(valid_df, 0.35, 0.65, 
                                                       use_balanced_accuracy=use_balanced_accuracy,
                                                       class_weights=class_weights)
        
        result = {
            'optimal_threshold_low': best_thresholds[0],
            'optimal_threshold_high': best_thresholds[1],
            # Store both overall and balanced/weighted accuracy
            'optimal_accuracy': best_metrics.get('accuracy', best_accuracy) if best_metrics else best_accuracy,
            'current_threshold_low': 0.35,  # From config
            'current_threshold_high': 0.65,  # From config
            'current_accuracy': current_metrics.get('accuracy', 0),
        }
        
        # Store the optimization metric (weighted or balanced)
        if class_weights is not None:
            result['optimal_weighted_accuracy'] = best_metrics.get('weighted_accuracy', best_accuracy) if best_metrics else best_accuracy
            result['current_weighted_accuracy'] = current_metrics.get('weighted_accuracy', 0)
            if 'class_0_2_accuracy' in best_metrics:
                result['optimal_class_0_2_accuracy'] = best_metrics['class_0_2_accuracy']
            if 'class_0_2_accuracy' in current_metrics:
                result['current_class_0_2_accuracy'] = current_metrics['class_0_2_accuracy']
        else:
            result['optimal_balanced_accuracy'] = best_metrics.get('balanced_accuracy', best_accuracy) if best_metrics else best_accuracy
            result['current_balanced_accuracy'] = current_metrics.get('balanced_accuracy', 0)
        
        # Add detailed metrics if available
        if best_metrics:
            for key in ['class_0_accuracy', 'class_1_accuracy', 'class_2_accuracy',
                       'class_0_support', 'class_1_support', 'class_2_support',
                       'class_0_2_accuracy']:
                if key in best_metrics:
                    result[key] = best_metrics[key]
        
        return result
    
    return {}


def analyze_score_distributions(df: pd.DataFrame) -> Dict:
    """Analyze score distributions by ground truth label."""
    if 'combined_score' not in df.columns:
        return {}
    
    valid_df = df[df['combined_score'].notna()].copy()
    if len(valid_df) == 0:
        return {}
    
    distributions = {}
    for label in [0, 1, 2]:
        label_df = valid_df[valid_df['ground_truth'] == label]
        if len(label_df) > 0:
            scores = label_df['combined_score'].values
            distributions[f'class_{label}_mean'] = float(np.mean(scores))
            distributions[f'class_{label}_std'] = float(np.std(scores))
            distributions[f'class_{label}_min'] = float(np.min(scores))
            distributions[f'class_{label}_max'] = float(np.max(scores))
            distributions[f'class_{label}_median'] = float(np.median(scores))
            distributions[f'class_{label}_q25'] = float(np.percentile(scores, 25))
            distributions[f'class_{label}_q75'] = float(np.percentile(scores, 75))
    
    return distributions


def analyze_false_positives_negatives(df: pd.DataFrame) -> Dict:
    """Analyze false positives and false negatives."""
    if len(df) == 0:
        return {}
    
    # False positives: predicted flood (2) but ground truth is no_flood (0)
    fp = df[(df['prediction'] == 2) & (df['ground_truth'] == 0)]
    
    # False negatives: predicted no_flood (0) but ground truth is flood (2)
    fn = df[(df['prediction'] == 0) & (df['ground_truth'] == 2)]
    
    # Analyze scores for these cases
    analysis = {
        'false_positives_count': len(fp),
        'false_negatives_count': len(fn),
        'false_positive_mean_score': float(fp['combined_score'].mean()) if len(fp) > 0 and 'combined_score' in fp.columns else None,
        'false_negative_mean_score': float(fn['combined_score'].mean()) if len(fn) > 0 and 'combined_score' in fn.columns else None,
    }
    
    return analysis


def main():
    # Paths
    results_dir = Path("storage/video_results")
    gt_dir = Path("test_videos/labels")
    
    # Load ground truth - automatically discover all JSONL files
    print("Loading ground truth...")
    all_gt = {}
    gt_files = sorted(gt_dir.glob("*.jsonl"))
    if len(gt_files) == 0:
        print(f"⚠️  Warning: No ground truth files found in {gt_dir}")
    else:
        for gt_file in gt_files:
            print(f"  Loading {gt_file.name}...")
            try:
                gt_data = load_ground_truth_jsonl(str(gt_file))
                all_gt.update(gt_data)
            except Exception as e:
                print(f"    ⚠️  Warning: Failed to load {gt_file.name}: {e}")
    print(f"Loaded ground truth for {len(all_gt)} videos")
    
    # Automatically discover all ablation folders
    print("\nDiscovering ablation folders...")
    ablations = []
    if results_dir.exists():
        for item in sorted(results_dir.iterdir()):
            if item.is_dir() and item.name.startswith("ablation"):
                ablations.append(item.name)
    else:
        print(f"⚠️  Warning: Results directory {results_dir} does not exist")
    
    if len(ablations) == 0:
        print("⚠️  Warning: No ablation folders found")
    else:
        print(f"Found {len(ablations)} ablations: {ablations}")
    
    all_results = []
    for ablation in ablations:
        print(f"\nLoading {ablation}...")
        results = load_ablation_results(results_dir, ablation)
        print(f"  Loaded {len(results)} results")
        all_results.extend(results)
    
    print(f"\nTotal results loaded: {len(all_results)}")
    
    # Analyze each ablation
    print("\n" + "="*80)
    print("ANALYZING EACH ABLATION")
    print("="*80)
    
    ablation_dfs = {}
    for ablation in ablations:
        ablation_results = [r for r in all_results if r.get('_ablation') == ablation]
        if len(ablation_results) == 0:
            continue
        
        df = analyze_ablation(ablation_results, all_gt, ablation)
        if len(df) > 0:
            ablation_dfs[ablation] = df
            metrics = compute_metrics(df)
            print(f"\n{ablation}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f} ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
    
    # Combine all results for threshold analysis
    print("\n" + "="*80)
    print("THRESHOLD TUNING ANALYSIS")
    print("="*80)
    
    # Optimize thresholds using ONLY ablation 4 (full production system)
    ablation4_key = "ablation4"
    optimal = None
    ablation4_optimal = None  # Save ablation4 thresholds separately to avoid overwriting
    all_df = pd.concat(ablation_dfs.values(), ignore_index=True) if ablation_dfs else pd.DataFrame()
    
    if ablation4_key in ablation_dfs:
        ablation4_df = ablation_dfs[ablation4_key]
        if 'combined_score' in ablation4_df.columns:
            ablation4_valid = ablation4_df[ablation4_df['combined_score'].notna()].copy()
            if len(ablation4_valid) > 0:
                print(f"\nOptimizing thresholds using {ablation4_key} (full production system)...")
                print(f"Using {len(ablation4_valid)} samples from {ablation4_key}")
                
                # Optimize for classes 0 and 2 only (ignore class 1)
                class_weights = {0: 1.0, 1: 0.0, 2: 1.0}
                print(f"Using weighted accuracy: class 0 weight={class_weights[0]}, "
                      f"class 1 weight={class_weights[1]}, class 2 weight={class_weights[2]}")
                
                # Use improved optimization with debugging enabled
                # Use weighted accuracy and enforce minimum threshold gap
                optimal = find_optimal_thresholds(ablation4_valid, step=0.05, 
                                                  use_distribution_aware=True, debug=True,
                                                  min_threshold_gap=0.2,
                                                  use_balanced_accuracy=False,  # Use weighted instead
                                                  class_weights=class_weights)
                ablation4_optimal = optimal  # Save before it gets overwritten in the per-ablation loop
                if optimal:
                    print(f"\n" + "="*60)
                    print(f"OPTIMAL THRESHOLDS (optimized on {ablation4_key}):")
                    print(f"="*60)
                    print(f"  Thresholds:")
                    print(f"    Low:  {optimal['optimal_threshold_low']:.3f} (current: {optimal['current_threshold_low']:.3f})")
                    print(f"    High: {optimal['optimal_threshold_high']:.3f} (current: {optimal['current_threshold_high']:.3f})")
                    
                    print(f"\n  Overall Accuracy:")
                    print(f"    Current: {optimal['current_accuracy']:.4f}")
                    print(f"    Optimal: {optimal['optimal_accuracy']:.4f}")
                    print(f"    Improvement: {optimal['optimal_accuracy'] - optimal['current_accuracy']:.4f}")
                    
                    # Print weighted accuracy (what's being optimized)
                    if 'optimal_weighted_accuracy' in optimal:
                        print(f"\n  Weighted Accuracy (optimization target, classes 0+2 only):")
                        print(f"    Current: {optimal.get('current_weighted_accuracy', 0):.4f}")
                        print(f"    Optimal: {optimal['optimal_weighted_accuracy']:.4f}")
                        print(f"    Improvement: {optimal['optimal_weighted_accuracy'] - optimal.get('current_weighted_accuracy', 0):.4f}")
                    
                    # Print classes 0+2 accuracy if available
                    if 'optimal_class_0_2_accuracy' in optimal:
                        print(f"\n  Classes 0+2 Accuracy (ignoring class 1):")
                        print(f"    Current: {optimal.get('current_class_0_2_accuracy', 0):.4f}")
                        print(f"    Optimal: {optimal['optimal_class_0_2_accuracy']:.4f}")
                        print(f"    Improvement: {optimal['optimal_class_0_2_accuracy'] - optimal.get('current_class_0_2_accuracy', 0):.4f}")
                    
                    # Print per-class accuracies if available
                    if 'class_0_accuracy' in optimal:
                        print(f"\n  Per-Class Accuracies (Optimal thresholds):")
                        for label in [0, 1, 2]:
                            acc_key = f'class_{label}_accuracy'
                            supp_key = f'class_{label}_support'
                            if acc_key in optimal and supp_key in optimal:
                                support = optimal[supp_key]
                                accuracy = optimal[acc_key]
                                print(f"    Class {label}: {accuracy:.4f} (support: {support})")
    else:
        print(f"\n⚠️  Warning: {ablation4_key} not found. Cannot optimize thresholds.")
    
    if len(all_df) > 0 and 'combined_score' in all_df.columns:
        valid_df = all_df[all_df['combined_score'].notna()].copy()
        print(f"\nAnalyzing {len(valid_df)} results with combined_score...")
        
        # Score distributions
        print("\nScore Distributions by Ground Truth Label:")
        distributions = analyze_score_distributions(valid_df)
        for key, value in distributions.items():
            print(f"  {key}: {value:.4f}")
        
        # False positives/negatives
        print("\nFalse Positives and Negatives:")
        fp_fn = analyze_false_positives_negatives(valid_df)
        print(f"  False Positives: {fp_fn['false_positives_count']}")
        if fp_fn['false_positive_mean_score'] is not None:
            print(f"    Mean score: {fp_fn['false_positive_mean_score']:.4f}")
        print(f"  False Negatives: {fp_fn['false_negatives_count']}")
        if fp_fn['false_negative_mean_score'] is not None:
            print(f"    Mean score: {fp_fn['false_negative_mean_score']:.4f}")
    
    # Per-ablation detailed analysis (keep existing)
    print("\n" + "="*80)
    print("PER-ABLATION DETAILED ANALYSIS")
    print("="*80)
    
    for ablation, df in ablation_dfs.items():
        print(f"\n{ablation}:")
        if 'combined_score' in df.columns:
            valid_df = df[df['combined_score'].notna()].copy()
            if len(valid_df) > 0:
                # Use improved optimization without debug for per-ablation analysis
                optimal = find_optimal_thresholds(valid_df, step=0.05, 
                                                  use_distribution_aware=True, debug=False,
                                                  min_threshold_gap=0.2,
                                                  use_balanced_accuracy=True)
                if optimal:
                    print(f"  Optimal thresholds (for this ablation only):")
                    print(f"    low={optimal.get('optimal_threshold_low', 'N/A'):.3f}, high={optimal.get('optimal_threshold_high', 'N/A'):.3f}")
                    print(f"  Performance metrics:")
                    print(f"    Overall accuracy: {optimal.get('optimal_accuracy', 0):.4f}")
                    if 'optimal_balanced_accuracy' in optimal:
                        print(f"    Balanced accuracy (optimized): {optimal.get('optimal_balanced_accuracy', 0):.4f}")
                    else:
                        print(f"    Balanced accuracy (optimized): {optimal.get('optimal_accuracy', 0):.4f}")
                    
                    # Show per-class accuracies if available
                    if any(f'class_{i}_accuracy' in optimal for i in [0, 1, 2]):
                        print(f"  Per-class accuracies:")
                        for label in [0, 1, 2]:
                            acc_key = f'class_{label}_accuracy'
                            supp_key = f'class_{label}_support'
                            if acc_key in optimal and supp_key in optimal:
                                support = optimal.get(supp_key, 0)
                                accuracy = optimal.get(acc_key, 0)
                                print(f"    Class {label}: {accuracy:.4f} (support: {support})")
    
    # Save detailed results
    output_dir = Path("evaluation_analysis/parameter_tuning_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(all_df) > 0:
        all_df.to_csv(output_dir / "all_results.csv", index=False)
        print(f"\n✅ Saved detailed results to {output_dir / 'all_results.csv'}")
    
    # Save summary
    summary = {
        'optimization_ablation': ablation4_key if ablation4_key in ablation_dfs else None,
        'optimal_thresholds': ablation4_optimal if ablation4_optimal else {},
        'ablations_analyzed': list(ablation_dfs.keys()),
        'total_results': len(all_df),
        'score_distributions': distributions if 'distributions' in locals() else {},
    }
    
    with open(output_dir / "parameter_tuning_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved summary to {output_dir / 'parameter_tuning_summary.json'}")


if __name__ == '__main__':
    main()