"""
Compute accuracy metrics from evaluation results and ground truth annotations.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score
import json

from utils import (
    load_ground_truth,
    load_json_results,
    extract_prediction,
    match_result_to_ground_truth
)


def compute_accuracy_metrics(results_dir: str, ground_truth_csv: str, 
                             output_file: str = None, tolerance_sec: float = 0.1):
    """
    Compute accuracy metrics by comparing predictions to ground truth.
    
    Args:
        results_dir: Path to results directory (e.g., storage/data_results/)
        ground_truth_csv: Path to ground truth annotations CSV
        output_file: Optional output file for detailed results
        tolerance_sec: Timestamp matching tolerance
    """
    print("=" * 60)
    print("ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Load ground truth
    print(f"\nLoading ground truth from: {ground_truth_csv}")
    ground_truth = load_ground_truth(ground_truth_csv)
    print(f"Loaded {len(ground_truth)} ground truth annotations")
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    results = load_json_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    # Match results to ground truth
    matched_data = []
    unmatched_count = 0
    
    for result in results:
        prediction = extract_prediction(result)
        if prediction is None:
            continue
        
        gt_label = match_result_to_ground_truth(result, ground_truth, tolerance_sec)
        
        if gt_label is not None:
            matched_data.append({
                'prediction': prediction,
                'ground_truth': gt_label,
                'video_file': result.get('metadata', {}).get('video_file', 'unknown'),
                'frame_number': result.get('metadata', {}).get('frame_number', -1),
                'file_path': result.get('_file_path', '')
            })
        else:
            unmatched_count += 1
    
    print(f"\nMatched {len(matched_data)} predictions to ground truth")
    print(f"Unmatched: {unmatched_count}")
    
    if len(matched_data) == 0:
        print("\n❌ No matched data available for accuracy computation!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(matched_data)
    
    # Extract predictions and ground truth
    y_true = df['ground_truth'].values
    y_pred = df['prediction'].values
    
    # Compute metrics
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    
    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    class_names = ['No Flood', 'Watch/Some Water', 'Flood']
    
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 68)
    for i, name in enumerate(class_names):
        print(f"{name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}")
    
    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print("\nMacro-Averaged Metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    
    # Weighted-averaged metrics
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print("\nWeighted-Averaged Metrics:")
    print(f"  Precision: {weighted_precision:.4f}")
    print(f"  Recall:    {weighted_recall:.4f}")
    print(f"  F1-Score:  {weighted_f1:.4f}")
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy:  {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("\n" + " " * 20 + "Predicted")
    print(" " * 15 + "  ".join([f"{name:<15}" for name in class_names]))
    print(" " * 10 + "-" * 50)
    for i, name in enumerate(class_names):
        print(f"Actual {name:<12} {' '.join([f'{cm[i,j]:<16}' for j in range(3)])}")
    
    # Save detailed results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV with predictions vs ground truth
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"\n✅ Detailed results saved to: {output_path.with_suffix('.csv')}")
        
        # Save summary JSON
        summary = {
            'total_samples': len(matched_data),
            'unmatched_samples': unmatched_count,
            'per_class': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(3)
            },
            'macro_avg': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1_score': float(macro_f1)
            },
            'weighted_avg': {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1_score': float(weighted_f1)
            },
            'balanced_accuracy': float(balanced_acc),
            'overall_accuracy': float(accuracy),
            'confusion_matrix': cm.tolist()
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved to: {output_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute accuracy metrics from evaluation results'
    )
    parser.add_argument(
        'results_dir',
        help='Path to results directory (e.g., storage/data_results/2025-11-19)'
    )
    parser.add_argument(
        'ground_truth_csv',
        help='Path to ground truth annotations CSV'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (without extension)',
        default=None
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=0.1,
        help='Timestamp matching tolerance in seconds (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    compute_accuracy_metrics(
        args.results_dir,
        args.ground_truth_csv,
        args.output,
        args.tolerance
    )


if __name__ == '__main__':
    main()

