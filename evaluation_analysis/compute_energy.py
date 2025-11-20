"""
Compute energy metrics from HMC power logs and evaluation results.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

from utils import load_json_results, load_power_logs, integrate_power, parse_timestamp


def compute_energy_metrics(results_dir: str, power_pi_csv: str, 
                           power_jetson_csv: str = None, 
                           idle_baseline_w: float = 0.0,
                           output_file: str = None):
    """
    Compute energy consumption metrics.
    
    Args:
        results_dir: Path to results directory
        power_pi_csv: Path to Pi power log CSV
        power_jetson_csv: Optional path to Jetson power log CSV
        idle_baseline_w: Idle baseline power to subtract (watts)
        output_file: Optional output file path
    """
    print("=" * 60)
    print("ENERGY ANALYSIS")
    print("=" * 60)
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    results = load_json_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    # Load power logs
    print(f"\nLoading Pi power log: {power_pi_csv}")
    power_pi = load_power_logs(power_pi_csv)
    print(f"Loaded {len(power_pi)} Pi power samples")
    
    power_jetson = None
    if power_jetson_csv:
        print(f"Loading Jetson power log: {power_jetson_csv}")
        power_jetson = load_power_logs(power_jetson_csv)
        print(f"Loaded {len(power_jetson)} Jetson power samples")
    
    # Compute per-frame energy
    energy_data = []
    
    for result in results:
        timing = result.get('timing', {})
        
        # Extract timestamps
        start_ts = parse_timestamp(timing.get('process_start_ts'))
        pipeline_latency = timing.get('pipeline_latency_s') or timing.get('total_pipeline_latency_s')
        
        if start_ts is None or pipeline_latency is None:
            continue
        
        end_ts = start_ts + timedelta(seconds=pipeline_latency)
        
        # Integrate Pi power
        pi_energy = integrate_power(power_pi, start_ts, end_ts)
        
        # Subtract idle baseline
        pi_net_energy_j = pi_energy['energy_j'] - (idle_baseline_w * pipeline_latency)
        pi_net_energy_j = max(0, pi_net_energy_j)  # Can't be negative
        
        # Integrate Jetson power if available
        jetson_energy_j = 0.0
        if power_jetson is not None:
            jetson_energy = integrate_power(power_jetson, start_ts, end_ts)
            jetson_energy_j = jetson_energy['energy_j'] - (idle_baseline_w * pipeline_latency)
            jetson_energy_j = max(0, jetson_energy_j)
        
        # Total energy
        total_energy_j = pi_net_energy_j + jetson_energy_j
        
        # Extract backend info
        cls_result = result.get('classification_result', {})
        backend = 'local'
        state = None
        model_tier = None
        
        if isinstance(cls_result, dict):
            backend = cls_result.get('backend', 'local')
            state = cls_result.get('state')
            model_tier = cls_result.get('model_tier')
        
        energy_data.append({
            'pi_energy_j': pi_net_energy_j,
            'jetson_energy_j': jetson_energy_j,
            'total_energy_j': total_energy_j,
            'pipeline_latency_s': pipeline_latency,
            'pi_avg_power_w': pi_energy['avg_power_w'],
            'backend': backend,
            'state': state,
            'model_tier': model_tier,
            'file_path': result.get('_file_path', '')
        })
    
    print(f"\nComputed energy for {len(energy_data)} frames")
    
    if len(energy_data) == 0:
        print("\n❌ No energy data available!")
        return
    
    df = pd.DataFrame(energy_data)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL ENERGY STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal Frames: {len(df)}")
    print(f"Total Energy (Pi):     {df['pi_energy_j'].sum():.2f} J")
    if power_jetson is not None:
        print(f"Total Energy (Jetson): {df['jetson_energy_j'].sum():.2f} J")
    print(f"Total Energy (System): {df['total_energy_j'].sum():.2f} J")
    
    print(f"\nEnergy per Frame:")
    print(f"  Mean:   {df['total_energy_j'].mean():.4f} J/frame")
    print(f"  Std:    {df['total_energy_j'].std():.4f} J/frame")
    print(f"  Min:    {df['total_energy_j'].min():.4f} J/frame")
    print(f"  p50:    {df['total_energy_j'].quantile(0.50):.4f} J/frame")
    print(f"  p90:    {df['total_energy_j'].quantile(0.90):.4f} J/frame")
    print(f"  p99:    {df['total_energy_j'].quantile(0.99):.4f} J/frame")
    print(f"  Max:    {df['total_energy_j'].max():.4f} J/frame")
    
    print(f"\nAverage Power (Pi):")
    print(f"  Mean:   {df['pi_avg_power_w'].mean():.2f} W")
    print(f"  p50:    {df['pi_avg_power_w'].quantile(0.50):.2f} W")
    print(f"  p90:    {df['pi_avg_power_w'].quantile(0.90):.2f} W")
    
    # Break down by state (FSM only)
    if 'state' in df.columns and df['state'].notna().any():
        print("\n" + "=" * 60)
        print("ENERGY BY FSM STATE")
        print("=" * 60)
        
        for state in sorted(df['state'].dropna().unique()):
            state_df = df[df['state'] == state]
            
            print(f"\nState {state}:")
            print(f"  Count:           {len(state_df)}")
            print(f"  Mean Energy:     {state_df['total_energy_j'].mean():.4f} J/frame")
            print(f"  p50 Energy:      {state_df['total_energy_j'].quantile(0.50):.4f} J/frame")
            print(f"  Total Energy:    {state_df['total_energy_j'].sum():.2f} J")
    
    # Break down by backend
    if 'backend' in df.columns and df['backend'].notna().any():
        print("\n" + "=" * 60)
        print("ENERGY BY BACKEND (Local vs Remote)")
        print("=" * 60)
        
        for backend in ['local', 'remote']:
            backend_df = df[df['backend'] == backend]
            
            if len(backend_df) == 0:
                continue
            
            print(f"\n{backend.capitalize()} Backend:")
            print(f"  Count:           {len(backend_df)}")
            print(f"  Mean Energy:     {backend_df['total_energy_j'].mean():.4f} J/frame")
            print(f"  p50 Energy:      {backend_df['total_energy_j'].quantile(0.50):.4f} J/frame")
            print(f"  Total Energy:    {backend_df['total_energy_j'].sum():.2f} J")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"\n✅ Detailed energy data saved to: {output_path.with_suffix('.csv')}")
        
        # Save summary JSON
        summary = {
            'total_frames': len(df),
            'total_energy_j': float(df['total_energy_j'].sum()),
            'total_pi_energy_j': float(df['pi_energy_j'].sum()),
            'idle_baseline_w': idle_baseline_w,
            'overall': {
                'mean_energy_per_frame_j': float(df['total_energy_j'].mean()),
                'std_energy_per_frame_j': float(df['total_energy_j'].std()),
                'p50_energy_per_frame_j': float(df['total_energy_j'].quantile(0.50)),
                'p90_energy_per_frame_j': float(df['total_energy_j'].quantile(0.90)),
                'p99_energy_per_frame_j': float(df['total_energy_j'].quantile(0.99)),
                'mean_pi_power_w': float(df['pi_avg_power_w'].mean())
            }
        }
        
        if power_jetson is not None:
            summary['total_jetson_energy_j'] = float(df['jetson_energy_j'].sum())
        
        # Add per-state breakdown
        if 'state' in df.columns and df['state'].notna().any():
            summary['by_state'] = {}
            for state in sorted(df['state'].dropna().unique()):
                state_df = df[df['state'] == state]
                summary['by_state'][state] = {
                    'count': len(state_df),
                    'mean_energy_j': float(state_df['total_energy_j'].mean()),
                    'total_energy_j': float(state_df['total_energy_j'].sum())
                }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved to: {output_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute energy metrics from HMC power logs'
    )
    parser.add_argument(
        'results_dir',
        help='Path to results directory'
    )
    parser.add_argument(
        'power_pi_csv',
        help='Path to Pi power log CSV'
    )
    parser.add_argument(
        '--power-jetson',
        help='Path to Jetson power log CSV (for remote configs)',
        default=None
    )
    parser.add_argument(
        '--idle-baseline',
        type=float,
        default=0.0,
        help='Idle baseline power in watts (to subtract from measurements)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (without extension)',
        default=None
    )
    
    args = parser.parse_args()
    
    compute_energy_metrics(
        args.results_dir,
        args.power_pi_csv,
        args.power_jetson,
        args.idle_baseline,
        args.output
    )


if __name__ == '__main__':
    main()

