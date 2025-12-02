"""
Main orchestration script for system evaluation.

This script coordinates all evaluation modules to process ablation runs
and generate comprehensive structured data for analysis.
"""

import argparse
import sys
from pathlib import Path
import logging

# Import evaluation modules
from ingest_data import discover_runs, load_ground_truth, build_frame_dataframe
from energy_integration import integrate_energy_data
from compute_metrics import compute_per_run_metrics
from aggregate_results import aggregate_all_results
from export_data import export_all_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate flood detection system ablation runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation without energy data
  python evaluate_system.py \\
      --results-dir storage/video_results \\
      --gt-dir test_videos/labels \\
      --output-dir evaluation_analysis/evaluation_results

  # Evaluation with energy data
  python evaluate_system.py \\
      --results-dir storage/video_results \\
      --gt-dir test_videos/labels \\
      --energy-dir storage/video_energy_results \\
      --output-dir evaluation_analysis/evaluation_results
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to storage/video_results/ directory'
    )
    
    parser.add_argument(
        '--gt-dir',
        type=str,
        required=True,
        help='Path to test_videos/labels/ directory'
    )
    
    parser.add_argument(
        '--energy-dir',
        type=str,
        default=None,
        help='Path to storage/video_energy_results/ directory (optional)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_analysis/evaluation_results',
        help='Output directory for evaluation results (default: evaluation_analysis/evaluation_results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directories
    results_dir = Path(args.results_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    energy_dir = Path(args.energy_dir) if args.energy_dir else None
    
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    if not gt_dir.exists():
        logger.error(f"Ground truth directory does not exist: {gt_dir}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("System Evaluation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Ground truth directory: {gt_dir}")
    logger.info(f"Energy directory: {energy_dir if energy_dir else 'None (skipping energy integration)'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Stage 1: Discover runs
    logger.info("Stage 1: Discovering runs...")
    runs = discover_runs(results_dir)
    if not runs:
        logger.error("No runs found. Exiting.")
        sys.exit(1)
    logger.info(f"Found {len(runs)} runs across {len(set(r.ablation_name for r in runs))} ablation configs")
    logger.info("")
    
    # Stage 2: Load ground truth
    logger.info("Stage 2: Loading ground truth labels...")
    gt_dict = load_ground_truth(gt_dir)
    if not gt_dict:
        logger.warning("No ground truth labels loaded. Continuing without GT matching.")
    else:
        logger.info(f"Loaded ground truth for {len(gt_dict)} videos")
    logger.info("")
    
    # Stage 3: Load frame data and match to GT
    logger.info("Stage 3: Loading frame data and matching to ground truth...")
    frame_df = build_frame_dataframe(runs, gt_dict)
    if len(frame_df) == 0:
        logger.error("No frame data loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(frame_df)} frames from {frame_df['run_id'].nunique()} runs")
    logger.info("")
    
    # Stage 4: Integrate energy data (optional)
    if energy_dir:
        logger.info("Stage 4: Integrating energy data...")
        frame_df = integrate_energy_data(frame_df, energy_dir)
        logger.info("")
    else:
        logger.info("Stage 4: Skipping energy integration (no energy directory provided)")
        logger.info("")
    
    # Stage 5: Compute per-run metrics
    logger.info("Stage 5: Computing per-run metrics...")
    metrics_df = compute_per_run_metrics(frame_df)
    if len(metrics_df) == 0:
        logger.error("No metrics computed. Exiting.")
        sys.exit(1)
    logger.info(f"Computed metrics for {len(metrics_df)} runs")
    logger.info("")
    
    # Stage 6: Aggregate results
    logger.info("Stage 6: Aggregating results...")
    aggregated_results = aggregate_all_results(metrics_df)
    logger.info(f"Generated {len(aggregated_results)} aggregated result sets")
    logger.info("")
    
    # Stage 7: Export all data
    logger.info("Stage 7: Exporting data...")
    export_all_data(frame_df, metrics_df, aggregated_results, output_dir)
    logger.info("")
    
    # Summary statistics
    logger.info("=" * 80)
    logger.info("Evaluation Complete")
    logger.info("=" * 80)
    logger.info(f"Total frames processed: {len(frame_df)}")
    logger.info(f"Total runs processed: {len(metrics_df)}")
    logger.info(f"Ablation configs: {sorted(frame_df['ablation_name'].unique())}")
    logger.info(f"Sequences: {sorted(frame_df['sequence_id'].dropna().unique())}")
    logger.info(f"Output files written to: {output_dir}")
    logger.info("")
    logger.info("Output files:")
    for file_path in sorted(output_dir.glob('*.csv')):
        logger.info(f"  - {file_path.name}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

