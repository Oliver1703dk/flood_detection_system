"""
Analyze FSM-specific metrics: state transitions, tier usage, label oscillations.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter

from utils import load_json_results, extract_prediction


def analyze_fsm_behavior(results_dir: str, output_file: str = None):
    """
    Analyze FSM behavior from evaluation results.
    
    Args:
        results_dir: Path to results directory
        output_file: Optional output file path
    """
    print("=" * 60)
    print("FSM BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    # Load results
    print(f"\nLoading results from: {results_dir}")
    results = load_json_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    # Extract FSM data
    fsm_data = []
    
    for result in results:
        cls_result = result.get('classification_result', {})
        
        if not isinstance(cls_result, dict):
            continue
        
        # Extract FSM fields
        state = cls_result.get('state')
        model_tier = cls_result.get('model_tier')
        prediction = extract_prediction(result)
        
        if state is None:
            continue
        
        fsm_data.append({
            'state': state,
            'model_tier': model_tier,
            'prediction': prediction,
            'tier_requested': cls_result.get('tier_requested'),
            'model_switched': cls_result.get('model_switched', False),
            'backend': cls_result.get('backend', 'local'),
            'skipped': cls_result.get('skipped', False),
            'conflict': cls_result.get('conflict', False),
            'flapping': cls_result.get('flapping', False),
            'frame_index': cls_result.get('frame_index', -1),
            'file_path': result.get('_file_path', '')
        })
    
    print(f"Extracted FSM data from {len(fsm_data)} results")
    
    if len(fsm_data) == 0:
        print("\n❌ No FSM data available!")
        return
    
    df = pd.DataFrame(fsm_data).sort_values('frame_index')
    
    # State distribution
    print("\n" + "=" * 60)
    print("STATE DISTRIBUTION")
    print("=" * 60)
    
    state_counts = df['state'].value_counts().sort_index()
    print("\nFrames per State:")
    for state, count in state_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {state}: {count:>6} ({percentage:>5.1f}%)")
    
    # Tier usage
    print("\n" + "=" * 60)
    print("MODEL TIER USAGE")
    print("=" * 60)
    
    tier_counts = df['model_tier'].value_counts()
    print("\nFrames per Tier:")
    for tier, count in tier_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {tier}: {count:>6} ({percentage:>5.1f}%)")
    
    # Backend usage
    backend_counts = df['backend'].value_counts()
    print("\nBackend Usage:")
    for backend, count in backend_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {backend.capitalize()}: {count:>6} ({percentage:>5.1f}%)")
    
    # State transitions
    print("\n" + "=" * 60)
    print("STATE TRANSITIONS")
    print("=" * 60)
    
    transitions = []
    prev_state = None
    
    for _, row in df.iterrows():
        if prev_state is not None and prev_state != row['state']:
            transitions.append((prev_state, row['state']))
        prev_state = row['state']
    
    print(f"\nTotal Transitions: {len(transitions)}")
    
    if len(transitions) > 0:
        transition_counts = Counter(transitions)
        print("\nMost Common Transitions:")
        for (from_state, to_state), count in transition_counts.most_common(10):
            print(f"  {from_state} → {to_state}: {count:>4}")
        
        # Compute average dwell time per state
        print("\n" + "=" * 60)
        print("AVERAGE DWELL TIME PER STATE")
        print("=" * 60)
        
        dwell_times = {state: [] for state in df['state'].unique()}
        current_state = None
        dwell_count = 0
        
        for _, row in df.iterrows():
            if current_state != row['state']:
                if current_state is not None and dwell_count > 0:
                    dwell_times[current_state].append(dwell_count)
                current_state = row['state']
                dwell_count = 1
            else:
                dwell_count += 1
        
        # Add final dwell
        if current_state is not None and dwell_count > 0:
            dwell_times[current_state].append(dwell_count)
        
        print("\nFrames per State Visit:")
        for state in sorted(dwell_times.keys()):
            if len(dwell_times[state]) > 0:
                avg_dwell = np.mean(dwell_times[state])
                max_dwell = np.max(dwell_times[state])
                visits = len(dwell_times[state])
                print(f"  {state}: {avg_dwell:>6.1f} frames/visit (max: {max_dwell}, visits: {visits})")
    
    # Label oscillations
    print("\n" + "=" * 60)
    print("LABEL OSCILLATIONS")
    print("=" * 60)
    
    predictions = df['prediction'].dropna().values
    oscillations = 0
    
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i-1]:
            oscillations += 1
    
    oscillation_rate = (oscillations / len(predictions)) * 100 if len(predictions) > 0 else 0
    
    print(f"\nTotal Prediction Changes: {oscillations}")
    print(f"Oscillation Rate: {oscillation_rate:.2f}%")
    
    # Flapping detection
    flapping_count = df['flapping'].sum()
    flapping_rate = (flapping_count / len(df)) * 100
    
    print(f"\nFlapping Detected: {flapping_count} frames ({flapping_rate:.2f}%)")
    
    # Conflict detection
    conflict_count = df['conflict'].sum()
    conflict_rate = (conflict_count / len(df)) * 100
    
    print(f"Conflicts Detected: {conflict_count} frames ({conflict_rate:.2f}%)")
    
    # Model switches
    switch_count = df['model_switched'].sum()
    switch_rate = (switch_count / len(df)) * 100
    
    print(f"Model Switches: {switch_count} ({switch_rate:.2f}%)")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"\n✅ Detailed FSM data saved to: {output_path.with_suffix('.csv')}")
        
        # Save summary JSON
        summary = {
            'total_frames': len(df),
            'state_distribution': {
                state: int(count) for state, count in state_counts.items()
            },
            'tier_usage': {
                tier: int(count) for tier, count in tier_counts.items()
            },
            'backend_usage': {
                backend: int(count) for backend, count in backend_counts.items()
            },
            'transitions': {
                'total_count': len(transitions),
                'most_common': [
                    {'from': from_state, 'to': to_state, 'count': count}
                    for (from_state, to_state), count in transition_counts.most_common(10)
                ] if len(transitions) > 0 else []
            },
            'dwell_times': {
                state: {
                    'avg_frames': float(np.mean(times)),
                    'max_frames': int(np.max(times)),
                    'visits': len(times)
                }
                for state, times in dwell_times.items() if len(times) > 0
            } if len(transitions) > 0 else {},
            'oscillations': {
                'count': int(oscillations),
                'rate_percent': float(oscillation_rate)
            },
            'flapping_count': int(flapping_count),
            'flapping_rate_percent': float(flapping_rate),
            'conflict_count': int(conflict_count),
            'conflict_rate_percent': float(conflict_rate),
            'model_switch_count': int(switch_count),
            'model_switch_rate_percent': float(switch_rate)
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary saved to: {output_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FSM behavior from evaluation results'
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
    
    args = parser.parse_args()
    
    analyze_fsm_behavior(args.results_dir, args.output)


if __name__ == '__main__':
    main()

