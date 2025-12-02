<!-- 7934eedc-8421-4a4e-b0e9-fb8b690fea9a f7f91d13-1867-4b15-b88e-1bdf0ae099ff -->
# System Evaluation Script Plan

## Overview

Build a data extraction and metric computation pipeline that processes all ablation runs and outputs comprehensive structured data for analysis. The script focuses on data extraction rather than report generation.

## Input Data Structure

### Per-Frame JSON Files

- Location: `storage/video_results/ablationX/run_<timestamp>/*.json`
- Contains:
  - `sensor_data`: temperature, humidity, pressure
  - `metadata`: video_file, video_timestamp_sec, run_id, motion, resource_constrained, timing fields
  - `classification_result`: Can be either:
    - **String format**: Simple string like "No Flood", "Some Water", "Flooded" (legacy format)
    - **Dict format**: Detailed dict with frame_index, state, model_tier, prediction, scores (combined_score, image_score, sensor_boost), counters, backend, timing breakdown, sensor_prediction
  - `timing`: per-stage latency metrics

### Ground Truth Labels

- Location: `test_videos/labels/*.jsonl`
- Format: `{"video": "filename.mp4", "labels": [{"t": seconds, "y": "flood|sus|no_flood"}]}`

### Energy Data (flexible format)

- JSON files with entries every 100-500ms
- Expected fields: timestamp, power/energy measurements (format to be determined)

## Implementation Stages

### Stage 1: Data Ingestion Module (`evaluation_analysis/ingest_data.py`)

**Functions:**

- `discover_runs(results_dir: Path) -> List[RunInfo]`: Walk `storage/video_results/` and discover all ablation folders and run directories. For each ablation folder, discover all `run_<timestamp>` subdirectories and sort them chronologically.
- `extract_run_metadata(run_path: Path, sample_json: Dict, run_index_in_ablation: int) -> Dict`: Extract:
  - `ablation_name`: From folder name (e.g., "ablation1" → "1", "ablation2" → "2")
  - `sequence_id`: Extract from `video_file` in metadata (e.g., "flood_video_20251005_145739.mp4" → extract sequence identifier)
  - `sensor_variant`: Extract from metadata field `sensor_variant` if present, otherwise infer from sensor data patterns or default to "Neutral"
  - `repeat_index`: Use `run_index_in_ablation` (0-based index of run within its ablation folder, sorted by timestamp)
- `load_ground_truth(gt_dir: Path) -> Dict[str, Dict[int, int]]`: Load all JSONL files, return `{video_file: {timestamp_sec: label_int}}`
- `load_frame_data(json_path: Path) -> Dict`: Parse single JSON file, extract all fields. Handle both string and dict formats for `classification_result`:
  - If string: Convert to dict format with `prediction` mapped from string ("No Flood"→0, "Some Water"→1, "Flooded"→2)
  - If dict: Extract all fields as-is
- `match_frame_to_ground_truth(frame_data: Dict, gt: Dict, tolerance_sec: float = 0.5) -> Optional[int]`: Match frame to GT by video_file + video_timestamp_sec

**Output:** Single long DataFrame with all frames from all runs, columns:

- `run_id`, `ablation_name`, `sequence_id`, `sensor_variant`, `repeat_index`
- `frame_index`, `video_file`, `video_timestamp_sec`
- `gt_label`, `pred_label`
- `image_score`, `sensor_boost`, `combined_score`, `sensor_prediction` (wet/neutral/dry - wet when sensor_boost > 0, dry when sensor_boost < 0, neutral when sensor_boost == 0)
- `fsm_state`, `tier_used`, `tier_requested`, `motion`
- `backend`, `skipped`, `conflict`, `drift`, `flapping`
- All timing fields (end-to-end, per-stage)
- `counters` (high, low, ambiguous, conflict, mid)

### Stage 2: Energy Data Integration Module (`evaluation_analysis/energy_integration.py`)

**Functions:**

- `discover_energy_files(energy_dir: Path, run_id: str) -> List[Path]`: Find energy JSON files for a run
- `load_energy_data(energy_file: Path) -> pd.DataFrame`: Load energy JSON, handle flexible formats (timestamp-based or interval-based)
- `match_energy_to_frames(energy_df: pd.DataFrame, frame_timestamps: pd.Series, run_start_ts: float) -> pd.DataFrame`: Align energy samples to frames (interpolate or aggregate)
- `compute_per_frame_energy(energy_df: pd.DataFrame, frame_intervals: List[Tuple[float, float]]) -> pd.Series`: Compute energy per frame from power samples

**Output:** Extended frame DataFrame with energy columns:

- `energy_pi_total`, `energy_jetson_total`, `energy_total`
- `energy_pi_per_frame`, `energy_jetson_per_frame`, `energy_total_per_frame`
- `power_pi_mean`, `power_jetson_mean` (if available)

### Stage 3: Per-Run Metrics Computation (`evaluation_analysis/compute_metrics.py`)

**Functions:**

- `compute_accuracy_metrics(frame_df: pd.DataFrame) -> Dict`: Confusion matrix, per-class precision/recall/F1, macro F1, F1 on ambiguous frames (combined_score in [0.3, 0.8])
- `compute_latency_metrics(frame_df: pd.DataFrame) -> Dict`: p50/p90/p99/max end-to-end latency, stratified by motion and tier
- `compute_energy_metrics(frame_df: pd.DataFrame) -> Dict`: Total energy, per-frame energy (mean/p50/p90), normalized by decision-bearing frames
- `compute_stability_metrics(frame_df: pd.DataFrame) -> Dict`: Label oscillation count, FSM state usage fractions, tier usage fractions (by motion regime), S2/S3 episode durations, S5 entries
- `compute_coverage_metrics(frame_df: pd.DataFrame) -> Dict`: Decision coverage by GT level (0/1/2), critical-frame coverage (GT ∈ {1,2}), correct coverage on critical frames
- `compute_sensor_metrics(frame_df: pd.DataFrame) -> Dict`: Mean/std sensor_boost, correlation image_score vs sensor_boost, sensor-variant comparisons, sensor_prediction distribution (wet/neutral/dry counts and percentages), dry condition detection rate

**Output:** Per-run metrics DataFrame with one row per run_id, all computed metrics as columns.

### Stage 4: Aggregation Module (`evaluation_analysis/aggregate_results.py`)

**Functions:**

- `aggregate_by_group(metrics_df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame`: Group by (config, sequence, sensor_variant) or (config, sequence), compute mean ± std, 95% CI
- `compute_hypothesis_comparisons(metrics_df: pd.DataFrame) -> pd.DataFrame`: Explicitly compute deltas for H1-H4 and ablation claims:
  - H1: config 1 vs 4 (slow_creeping, Neutral)
  - H2: Motion-aware metrics (config 4, Neutral, all sequences)
  - H3: Consensus (4 vs 4b, 2 vs 2b, slow_creeping, Neutral)
  - H4: Sensor variants (config 4, slow_creeping, all variants)
  - Fast-motion safety: config 4 vs 5 (fast_passing, Neutral)
  - Always-offload: config 4 vs 6 (slow_creeping, Neutral)
  - Graceful degradation: config 4 vs 3 (slow_creeping, Neutral)

**Output:**

- Aggregated metrics CSV: mean ± CI per (config, sequence, sensor_variant)
- Hypothesis comparisons CSV: explicit deltas for each hypothesis

### Stage 5: Data Export Module (`evaluation_analysis/export_data.py`)

**Functions:**

- `export_frame_level_data(frame_df: pd.DataFrame, output_dir: Path)`: Export full per-frame DataFrame to CSV
- `export_run_level_data(metrics_df: pd.DataFrame, output_dir: Path)`: Export per-run metrics to CSV
- `export_aggregated_data(agg_df: pd.DataFrame, output_dir: Path)`: Export aggregated metrics to CSV
- `export_hypothesis_data(hyp_df: pd.DataFrame, output_dir: Path)`: Export hypothesis comparisons to CSV
- `export_motion_stratified_data(frame_df: pd.DataFrame, output_dir: Path)`: Export motion-stratified summaries (tier usage, oscillations, latency) to CSV
- `export_sensor_variant_data(frame_df: pd.DataFrame, output_dir: Path)`: Export sensor-variant comparison data to CSV

**Output:** All data files in `evaluation_analysis/evaluation_results/`:

- `frames_all.csv`: Complete per-frame data
- `runs_metrics.csv`: Per-run aggregated metrics
- `aggregated_by_config_sequence_sensor.csv`: Grouped means ± CI
- `hypothesis_comparisons.csv`: Explicit hypothesis deltas
- `motion_stratified_summary.csv`: Motion-regime breakdowns
- `sensor_variant_comparison.csv`: Sensor variant analysis

### Stage 6: Main Orchestration Script (`evaluation_analysis/evaluate_system.py`)

**Main function:**

1. Discover all runs from `storage/video_results/`
2. Load ground truth from `test_videos/labels/`
3. Load all frame data and match to GT
4. (Optional) Load and integrate energy data if energy_dir provided
5. Compute per-run metrics
6. Aggregate across repeats
7. Compute hypothesis comparisons
8. Export all data files

**Command-line interface:**

```bash
python evaluate_system.py \
    --results-dir storage/video_results \
    --gt-dir test_videos/labels \
    --energy-dir storage/energy_data \  # optional
    --output-dir evaluation_analysis/evaluation_results
```

## Key Design Decisions

1. **Flexible Energy Integration**: Energy module handles multiple formats, matches by timestamp or run_id
2. **Flexible Metadata Extraction**: Supports folder-based (ablationX) and metadata-field extraction, with fallback logic
3. **Comprehensive Frame-Level Data**: Export full frame DataFrame for custom analysis
4. **Explicit Hypothesis Computations**: Script explicitly computes all hypothesis deltas as specified
5. **Data-Only Output**: No plots or reports, only structured data files (CSV/JSON)
6. **Motion Stratification**: All metrics computed per motion regime where relevant
7. **Coverage Metrics**: Explicit coverage calculations for system-level hazard prioritization claim
8. **String Classification Result Handling**: Data ingestion handles both legacy string format ("No Flood") and modern dict format, normalizing to consistent structure
9. **Multiple Runs Per Ablation**: Repeat indexing based on chronological ordering of run directories within each ablation folder
10. **Sensor Prediction Tracking**: Sensor prediction (wet/neutral/dry) extracted from classification_result when available, computed from sensor_boost if missing. Wet when sensor_boost > 0, dry when sensor_boost < 0 (heatwave/dry spell), neutral when sensor_boost == 0

## File Structure

```
evaluation_analysis/
├── ingest_data.py          # Stage 1: Data loading
├── energy_integration.py    # Stage 2: Energy data handling
├── compute_metrics.py      # Stage 3: Metric computation
├── aggregate_results.py    # Stage 4: Aggregation & comparisons
├── export_data.py          # Stage 5: Data export
├── evaluate_system.py       # Stage 6: Main orchestration
└── evaluation_results/     # Output directory
    ├── frames_all.csv
    ├── runs_metrics.csv
    ├── aggregated_by_config_sequence_sensor.csv
    ├── hypothesis_comparisons.csv
    ├── motion_stratified_summary.csv
    └── sensor_variant_comparison.csv
```

## Dependencies

- pandas, numpy for data manipulation
- scipy for confidence intervals
- pathlib for file handling
- json for parsing

## Testing Strategy

- Unit tests for each metric computation function
- Integration test with sample data
- Validation that all expected columns are present in outputs

### To-dos

- [ ] Implement data ingestion module: discover runs, load ground truth, parse frame JSONs, match to GT
- [ ] Implement energy integration module: flexible format handling, timestamp matching, per-frame energy computation
- [ ] Implement metrics computation module: accuracy, latency, energy, stability, coverage, sensor metrics
- [ ] Implement aggregation module: group by config/sequence/sensor, compute means ± CI, explicit hypothesis comparisons
- [ ] Implement data export module: export all DataFrames to CSV with proper column naming
- [ ] Implement main orchestration script with CLI interface, wire all modules together