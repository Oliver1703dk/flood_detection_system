"""
Compute latency metrics from evaluation results.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from utils import load_json_results


def extract_latency_metrics(result: Dict) -> Dict:
    """
    Extract all latency metrics from a result.
    
    Returns:
        Dictionary with timing fields
    """
    timing = result.get('timing', {})
    
    # Extract primary metrics
    metrics = {
        'pipeline_latency_s': timing.get('pipeline_latency_s'),
        'inference_s': timing.get('inference_s'),
        'classification_s': timing.get('classification_s'),
        'queue_wait_s': timing.get('queue_wait_s'),
        'backend_latency_s': timing.get('backend_latency_s'),
        'backend_roundtrip_s': timing.get('backend_roundtrip_s'),
        'llm_latency_s': timing.get('llm_latency_s'),
        'fsm_total_s': timing.get('fsm_total_s'),
        'fsm_classification_core_s': timing.get('fsm_classification_core_s'),
        'preprocess_s': timing.get('preprocess_s'),
        'result_format_s': timing.get('result_format_s'),
        'validation_storage_s': timing.get('validation_storage_s'),
        'baseline_update_s': timing.get('baseline_update_s'),
        'total_pipeline_latency_s': timing.get('total_pipeline_latency_s'),
    }
    
    # Extract FSM state if available
    cls_result = result.get('classification_result', {})
    if isinstance(cls_result, dict):
        metrics['state'] = cls_result.get('state')
        metrics['backend'] = cls_result.get('backend', 'local')
        metrics['model_tier'] = cls_result.get('model_tier')
    
    return metrics


def compute_latency_statistics(results_dir: str, output_file: str = None, 
                               plot: bool = False):
    """
    Compute latency statistics from evaluation results.
    
    Args:
        results_dir: Path to results directory
        output_file: Optional output file path
        plot: Whether to generate distribution plots
    """
    print("=" * 60)
    print("LATENCY ANALYSIS")
    print("=" * 60)
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    results = load_json_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    # Extract latency metrics
    latency_data = []
    for result in results:
        metrics = extract_latency_metrics(result)
        if metrics['pipeline_latency_s'] is not None:
            latency_data.append(metrics)
    
    print(f"Extracted latency data from {len(latency_data)} results")
    
    if len(latency_data) == 0:
        print("\n❌ No latency data available!")
        return
    
    df = pd.DataFrame(latency_data)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL LATENCY STATISTICS")
    print("=" * 60)
    
    metrics_to_analyze = [
        ('pipeline_latency_s', 'Pipeline Latency'),
        ('total_pipeline_latency_s', 'Total Pipeline Latency (inc. baseline update)'),
        ('inference_s', 'Inference'),
        ('classification_s', 'Classification'),
        ('queue_wait_s', 'Queue Wait'),
        ('backend_latency_s', 'Backend (Remote)'),
        ('fsm_total_s', 'FSM Total'),
        ('fsm_classification_core_s', 'FSM Classification Core'),
    ]
    
    for col, name in metrics_to_analyze:
        if col not in df.columns or df[col].isna().all():
            continue
        
        values = df[col].dropna()
        if len(values) == 0:
            continue
        
        print(f"\n{name}:")
        print(f"  Count:  {len(values)}")
        print(f"  Mean:   {values.mean():.4f}s")
        print(f"  Std:    {values.std():.4f}s")
        print(f"  Min:    {values.min():.4f}s")
        print(f"  p50:    {values.quantile(0.50):.4f}s")
        print(f"  p90:    {values.quantile(0.90):.4f}s")
        print(f"  p99:    {values.quantile(0.99):.4f}s")
        print(f"  Max:    {values.max():.4f}s")
    
    # Break down by state (FSM only)
    if 'state' in df.columns and df['state'].notna().any():
        print("\n" + "=" * 60)
        print("LATENCY BY FSM STATE")
        print("=" * 60)
        
        for state in sorted(df['state'].dropna().unique()):
            state_df = df[df['state'] == state]
            latencies = state_df['pipeline_latency_s'].dropna()
            
            if len(latencies) == 0:
                continue
            
            print(f"\nState {state}:")
            print(f"  Count:  {len(latencies)}")
            print(f"  Mean:   {latencies.mean():.4f}s")
            print(f"  p50:    {latencies.quantile(0.50):.4f}s")
            print(f"  p90:    {latencies.quantile(0.90):.4f}s")
            print(f"  p99:    {latencies.quantile(0.99):.4f}s")
    
    # Break down by backend (local vs remote)
    if 'backend' in df.columns and df['backend'].notna().any():
        print("\n" + "=" * 60)
        print("LATENCY BY BACKEND (Local vs Remote)")
        print("=" * 60)
        
        for backend in ['local', 'remote']:
            backend_df = df[df['backend'] == backend]
            latencies = backend_df['pipeline_latency_s'].dropna()
            
            if len(latencies) == 0:
                continue
            
            print(f"\n{backend.capitalize()} Backend:")
            print(f"  Count:  {len(latencies)}")
            print(f"  Mean:   {latencies.mean():.4f}s")
            print(f"  p50:    {latencies.quantile(0.50):.4f}s")
            print(f"  p90:    {latencies.quantile(0.90):.4f}s")
            print(f"  p99:    {latencies.quantile(0.99):.4f}s")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"\n✅ Detailed latency data saved to: {output_path.with_suffix('.csv')}")
        
        # Save summary JSON
        summary = {
            'total_samples': len(df),
            'overall': {}
        }
        
        for col, name in metrics_to_analyze:
            if col not in df.columns or df[col].isna().all():
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            summary['overall'][col] = {
                'count': len(values),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'p50': float(values.quantile(0.50)),
                'p90': float(values.quantile(0.90)),
                'p99': float(values.quantile(0.99)),
                'max': float(values.max())
            }
        
        # Add per-state breakdown if available
        if 'state' in df.columns and df['state'].notna().any():
            summary['by_state'] = {}
            for state in sorted(df['state'].dropna().unique()):
                state_df = df[df['state'] == state]
                latencies = state_df['pipeline_latency_s'].dropna()
                
                if len(latencies) > 0:
                    summary['by_state'][state] = {
                        'count': len(latencies),
                        'mean': float(latencies.mean()),
                        'p50': float(latencies.quantile(0.50)),
                        'p90': float(latencies.quantile(0.90)),
                        'p99': float(latencies.quantile(0.99))
                    }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved to: {output_path.with_suffix('.json')}")
    
    # Generate plots
    if plot and output_file:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Latency Distribution Analysis')
        
        # Pipeline latency histogram
        axes[0, 0].hist(df['pipeline_latency_s'].dropna(), bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('Pipeline Latency (s)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Pipeline Latency Distribution')
        
        # CDF
        latencies_sorted = np.sort(df['pipeline_latency_s'].dropna())
        cdf = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)
        axes[0, 1].plot(latencies_sorted, cdf)
        axes[0, 1].set_xlabel('Pipeline Latency (s)')
        axes[0, 1].set_ylabel('CDF')
        axes[0, 1].set_title('Cumulative Distribution Function')
        axes[0, 1].grid(True)
        
        # By state (if available)
        if 'state' in df.columns and df['state'].notna().any():
            states = sorted(df['state'].dropna().unique())
            state_data = [df[df['state'] == s]['pipeline_latency_s'].dropna() for s in states]
            axes[1, 0].boxplot(state_data, labels=states)
            axes[1, 0].set_xlabel('FSM State')
            axes[1, 0].set_ylabel('Pipeline Latency (s)')
            axes[1, 0].set_title('Latency by FSM State')
        
        # By backend (if available)
        if 'backend' in df.columns and df['backend'].notna().any():
            backends = ['local', 'remote']
            backend_data = [df[df['backend'] == b]['pipeline_latency_s'].dropna() 
                          for b in backends if (df['backend'] == b).any()]
            if len(backend_data) > 0:
                axes[1, 1].boxplot(backend_data, labels=[b for b in backends if (df['backend'] == b).any()])
                axes[1, 1].set_xlabel('Backend')
                axes[1, 1].set_ylabel('Pipeline Latency (s)')
                axes[1, 1].set_title('Latency by Backend')
        
        plt.tight_layout()
        plot_path = output_path.with_name(output_path.stem + '_plots.png')
        plt.savefig(plot_path, dpi=150)
        print(f"✅ Plots saved to: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute latency statistics from evaluation results'
    )
    parser.add_argument(
        'results_dir',
        help='Path to results directory'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (without extension)',
        default=None
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate distribution plots'
    )
    
    args = parser.parse_args()
    
    compute_latency_statistics(args.results_dir, args.output, args.plot)


if __name__ == '__main__':
    main()

