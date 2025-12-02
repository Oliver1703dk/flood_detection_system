<!-- 7934eedc-8421-4a4e-b0e9-fb8b690fea9a f7f91d13-1867-4b15-b88e-1bdf0ae099ff -->
# System Evaluation Script Plan

## Overview

Build a data extraction and metric computation pipeline that processes all ablation runs and outputs comprehensive structured data for analysis. The script focuses on data extraction rather than report generation.

## Input Data Structure

### Per-Frame JSON Files

- Location: `storage/video_results/ablationX/run_<timestamp>/*.json`
- Contains:
  - `sensor_data`: temperature, humidity, pressure
  - `metadata`: video_file, video_timestamp_sec, run_id, motion, resource_constrained, sensor_baseline, sensor_anomalies
  - `classification_result`: Can be either:
    - **String format** (non-FSM configs 1, 1b): Simple string like "No Flood", "Some Water", "Flooded"
    - **Dict format** (FSM configs 2-6): Detailed dict with frame_index, state, model_tier, prediction, scores (combined_score, image_score, sensor_boost, sensor_prediction), counters, backend, backend_info, timing breakdown, llm_used, llm_prediction, s2_llm_confirmed, conflict, drift, flapping, skipped, model_switched
  - `timing`: per-stage latency metrics including `total_pipeline_latency_s`

### Ground Truth Labels

- Location: `test_videos/labels/*.jsonl`
- Format: Single JSON object per file (not line-by-line JSONL):
  ```json
  {"video": "flood_video_20251005_145739.mp4", "fps_truth": 1, "labels": [{"t": 0, "y": "flood"}, {"t": 1, "y": "sus"}, ...]}
  ```

- Label values: `"flood"` → 2, `"sus"` → 1, `"no_flood"` → 0
- Timestamps `t` are integer seconds

### Energy Data (optional)

- Location: `storage/video_energy_results/*.jsonl`
- Format: Single JSON object per file (same structure as ground truth labels):
  ```json
  {
    "run_id": "20251202-194508-743316",
    "video_file": "flood_video_20251005_145739.mp4",
    "measurements": [
      {"t": 0.0, "energy_total_j": 0.0, "power_total_w": 12.5},
      {"t": 0.1, "energy_total_j": 1.25, "power_total_w": 12.5},
      {"t": 0.2, "energy_total_j": 2.5, "power_total_w": 12.5},
      ...
    ]
  }
  ```

- Timestamps `t` are relative to run start (in seconds, float)
- `energy_total_j`: Cumulative total system energy (all devices combined) in Joules
- `power_total_w`: Instantaneous total system power in Watts
- Samples typically at 100-500ms intervals

## Implementation Stages

### Stage 1: Data Ingestion Module (`evaluation_analysis/ingest_data.py`)

**Functions:**

- `discover_runs(results_dir: Path) -> List[RunInfo]`: Walk `storage/video_results/` and discover all ablation folders and run directories. For each ablation folder, discover all `run_<timestamp>` subdirectories and sort them chronologically.

- `extract_run_metadata(run_path: Path, sample_json: Dict, run_index_in_ablation: int) -> Dict`: Extract:
  - `ablation_name`: From folder name, preserve variant suffix (e.g., "ablation1" → "1", "ablation1b" → "1b", "ablation2" → "2")
  - `sequence_id`: Extract timestamp portion from `video_file` in metadata using regex pattern `flood_video_\d{8}_(\d{6})\.mp4` (e.g., "flood_video_20251005_145739.mp4" → "145739"). This matches GT file naming `video_145739.jsonl`.
  - `sensor_prediction`: Extract from `classification_result.scores.sensor_prediction` if present (values: "wet", "neutral", "dry"), otherwise default to "neutral"
  - `repeat_index`: Use `run_index_in_ablation` (0-based index of run within its ablation folder, sorted by timestamp)

- `load_ground_truth(gt_dir: Path) -> Dict[str, Dict[int, int]]`: 
  - Load all `.jsonl` files from `test_videos/labels/`
  - Parse each file as a single JSON object (not line-by-line)
  - Extract `video` field (full filename) and `labels` array
  - Convert labels: `"flood"` → 2, `"sus"` → 1, `"no_flood"` → 0
  - Return `{video_filename: {timestamp_int: label_int}}`

- `load_frame_data(json_path: Path) -> Dict`: Parse single JSON file, extract all fields. Handle both formats for `classification_result`:
  - **If string** (non-FSM): Convert to normalized format with `prediction` mapped from string ("No Flood"→0, "Some Water"→1, "Flooded"→2). Set FSM-specific fields to None/NaN.
  - **If dict** (FSM): Extract all fields as-is including scores, counters, backend_info, timing

- `match_frame_to_ground_truth(frame_data: Dict, gt: Dict) -> Optional[int]`: 
  - Match frame to GT by `video_file` + `round(video_timestamp_sec)` to nearest integer second
  - Lookup in GT dict using rounded timestamp
  - Return GT label or None if no match

**Config Type Detection:**

- Configs 1, 1b: Non-FSM mode (classification_result is string)
- Configs 2, 2b, 3, 3b, 4, 4b, 5, 6: FSM mode (classification_result is dict)
- Add `config_type` column: "baseline" for non-FSM, "fsm" for FSM configs

**Output:** Single long DataFrame with all frames from all runs, columns:

- **Identifiers:** `run_id`, `ablation_name`, `sequence_id`, `sensor_prediction`, `repeat_index`, `config_type`
- **Frame info:** `frame_index`, `video_file`, `video_timestamp_sec`, `video_timestamp_sec_rounded`
- **Labels:** `gt_label`, `pred_label`
- **Scores (FSM only, NaN for non-FSM):** `image_score`, `sensor_boost`, `combined_score`
- **FSM state (FSM only):** `fsm_state`, `tier_used`, `tier_requested`
- **Metadata:** `motion`, `resource_constrained`, `backend`
- **Flags (FSM only):** `skipped`, `conflict`, `drift`, `flapping`, `model_switched`, `llm_used`, `s2_llm_confirmed`
- **Counters (FSM only):** `counter_high`, `counter_low`, `counter_ambiguous`, `counter_conflict`, `counter_mid`
- **Timing:** `total_pipeline_latency_s` (primary latency metric for all configs)
- **Sensor data:** `temperature`, `humidity`, `pressure`, `temperature_baseline`, `humidity_baseline`, `pressure_baseline`
- **Sensor anomalies (if present):** `delta_temperature`, `delta_humidity`, `delta_pressure`

### Stage 2: Energy Data Integration Module (`evaluation_analysis/energy_integration.py`)

**Note:** This module is optional and will be implemented when energy data is available.

**Functions:**

- `discover_energy_file(energy_dir: Path, run_id: str) -> Optional[Path]`: 
  - Find energy file for a run by matching `run_id` in files from `storage/video_energy_results/`
  - Search pattern: files containing `run_id` in filename or within JSON `run_id` field
  - Return single file path or None if not found

- `load_energy_data(energy_file: Path) -> Dict`: 
  - Load energy file as single JSON object (same structure as label files)
  - Parse `measurements` array with `t` (timestamp), `energy_total_j` (cumulative energy), and `power_total_w` (instantaneous power)
  - Return dict with `run_id`, `video_file`, and `measurements` array

- `match_energy_to_frames(energy_data: Dict, frame_timestamps: pd.Series) -> pd.DataFrame`: 
  - Match energy measurements to frames by timestamp
  - For each frame, find energy and power values at frame start and frame end
  - Extract energy at frame timestamp from cumulative energy values
  - Extract power samples within frame interval for aggregation

- `compute_per_frame_energy(energy_data: Dict, frame_df: pd.DataFrame) -> pd.DataFrame`: 
  - Compute energy consumed per frame: `energy_at_frame_end - energy_at_frame_start`
  - Compute mean/max power during frame interval from power samples
  - Return DataFrame with `energy_total_per_frame`, `power_total_mean`, `power_total_max` for each frame

**Output:** Extended frame DataFrame with energy columns (when available):

- `energy_total_j`: Cumulative total system energy at frame timestamp (Joules)
- `energy_total_per_frame`: Energy consumed during this frame (Joules)
- `power_total_mean`: Mean power during frame interval (Watts)
- `power_total_max`: Maximum power during frame interval (Watts)
- `energy_total_run`: Total energy for entire run (Joules) - same value for all frames in run

### Stage 3: Per-Run Metrics Computation (`evaluation_analysis/compute_metrics.py`)

**Functions:**

- `compute_accuracy_metrics(frame_df: pd.DataFrame) -> Dict`: 
  - Confusion matrix (3x3 for classes 0, 1, 2)
  - Per-class precision, recall, F1
  - Macro F1 score
  - For FSM configs: F1 on ambiguous frames (combined_score in [0.3, 0.8])

- `compute_latency_metrics(frame_df: pd.DataFrame) -> Dict`: 
  - Use `total_pipeline_latency_s` as the primary latency metric
  - Compute p50, p90, p99, max, mean, std
  - For FSM configs: stratify by motion and tier_used

- `compute_energy_metrics(frame_df: pd.DataFrame) -> Dict`: (when energy data available)
  - Total run energy (from last measurement)
  - Per-frame energy (mean/p50/p90/std)
  - Mean/max power during frames (mean/p50/p90)
  - Energy per decision-bearing frame (frames where prediction was made)

- `compute_stability_metrics(frame_df: pd.DataFrame) -> Dict`: (FSM configs only)
  - Label oscillation count (consecutive prediction changes)
  - FSM state usage fractions (time in S0, S1, S2, S3, S5)
  - Tier usage fractions (nano, small, medium, large) by motion regime
  - S2/S3 episode durations
  - S5 entry count

- `compute_coverage_metrics(frame_df: pd.DataFrame) -> Dict`: 
  - Decision coverage by GT level (0/1/2)
  - Critical-frame coverage (GT ∈ {1, 2})
  - Correct coverage on critical frames

- `compute_sensor_metrics(frame_df: pd.DataFrame) -> Dict`: (FSM configs only)
  - Mean/std sensor_boost
  - Correlation image_score vs sensor_boost
  - sensor_prediction distribution (wet/neutral/dry counts and percentages)

**Config-Aware Metrics:**

- For non-FSM configs (1, 1b): Skip FSM-specific metrics (state usage, tier usage, counters, stability)
- For FSM configs (2-6): Compute full metric suite

**Output:** Per-run metrics DataFrame with one row per run_id, all computed metrics as columns.

### Stage 4: Aggregation Module (`evaluation_analysis/aggregate_results.py`)

**Functions:**

- `aggregate_by_group(metrics_df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame`: 
  - Group by (ablation_name, sequence_id) or (ablation_name, sequence_id, sensor_prediction)
  - Compute mean ± std, 95% CI for each metric

- `compute_hypothesis_comparisons(metrics_df: pd.DataFrame) -> pd.DataFrame`: Explicitly compute deltas for hypothesis testing:
  - **H1 (Adaptive Tiering & Offload):** config 1 vs 4 - latency reduction, F1 maintained
  - **H2 (Sensor Fusion Stability):** config 2 vs 4 - oscillation reduction, stability improvement
  - **H3 (Multi-Model Consensus):** config 4 vs 4b, config 2 vs 2b - accuracy improvement vs energy cost
  - **Fast-motion safety:** config 4 vs 5 - latency during fast motion
  - **Always-offload baseline:** config 4 vs 6 - energy efficiency comparison
  - **Graceful degradation:** config 4 vs 3 - local-only vs offload performance

**Output:**

- Aggregated metrics CSV: mean ± CI per (ablation_name, sequence_id)
- Hypothesis comparisons CSV: explicit deltas for each hypothesis with statistical significance

### Stage 5: Data Export Module (`evaluation_analysis/export_data.py`)

**Functions:**

- `export_frame_level_data(frame_df: pd.DataFrame, output_dir: Path)`: Export full per-frame DataFrame to CSV
- `export_run_level_data(metrics_df: pd.DataFrame, output_dir: Path)`: Export per-run metrics to CSV
- `export_aggregated_data(agg_df: pd.DataFrame, output_dir: Path)`: Export aggregated metrics to CSV
- `export_hypothesis_data(hyp_df: pd.DataFrame, output_dir: Path)`: Export hypothesis comparisons to CSV
- `export_motion_stratified_data(frame_df: pd.DataFrame, output_dir: Path)`: Export motion-stratified summaries (tier usage, oscillations, latency) to CSV
- `export_sensor_prediction_data(frame_df: pd.DataFrame, output_dir: Path)`: Export sensor prediction distribution data to CSV

**Output:** All data files in `evaluation_analysis/evaluation_results/`:

- `frames_all.csv`: Complete per-frame data
- `runs_metrics.csv`: Per-run aggregated metrics
- `aggregated_by_config_sequence.csv`: Grouped means ± CI
- `hypothesis_comparisons.csv`: Explicit hypothesis deltas
- `motion_stratified_summary.csv`: Motion-regime breakdowns
- `sensor_prediction_summary.csv`: Sensor prediction analysis

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
    --energy-dir storage/video_energy_results \  # optional
    --output-dir evaluation_analysis/evaluation_results
```

**Progress Reporting:**

- Log: "Processing run X/Y", "Loaded N frames", "Computed metrics for M runs"
- Summary statistics at completion

## Key Design Decisions

1. **Ground Truth Matching:** Match frames to GT by rounding `video_timestamp_sec` to nearest integer second, then lookup in GT dict by video filename + rounded timestamp
2. **Sequence ID Extraction:** Extract 6-digit timestamp from video filename pattern `flood_video_YYYYMMDD_HHMMSS.mp4` → `HHMMSS` to match GT file naming `video_HHMMSS.jsonl`
3. **Ablation Name Handling:** Preserve variant suffix (1, 1b, 2, 2b, etc.) from folder names
4. **Sensor Prediction:** Extract directly from `classification_result.scores.sensor_prediction` field (values: "wet", "neutral", "dry")
5. **Primary Latency Metric:** Use `total_pipeline_latency_s` from timing for all configs (available in both FSM and non-FSM results)
6. **Config-Aware Processing:** Detect FSM vs non-FSM configs based on classification_result type; skip FSM-specific metrics for non-FSM configs
7. **String Classification Handling:** Convert string predictions ("No Flood", "Some Water", "Flooded") to integers (0, 1, 2) for non-FSM configs
8. **Multiple Runs Per Ablation:** Repeat indexing based on chronological ordering of run directories within each ablation folder
9. **Data-Only Output:** No plots or reports, only structured data files (CSV) for downstream analysis
10. **Optional Energy Integration:** Energy module designed to be optional; analysis proceeds without energy data
11. **Energy Data Format:** Total system energy format with cumulative energy (`energy_total_j`) and instantaneous power (`power_total_w`), matched by `run_id`, timestamps relative to run start

## Ablation Configuration Reference

| Ablation | Config Type | Models | FSM | Sensor Fusion | Remote Offload | Notes |

|----------|-------------|--------|-----|---------------|----------------|-------|

| 1        | baseline    | 1x medium | ❌ | ❌ | ❌ | Single-model baseline |

| 1b       | baseline    | 3x medium | ❌ | ❌ | ❌ | Multi-model baseline |

| 2        | fsm         | 3x nano | ✅ | ❌ | ✅ | Vision-only FSM |

| 2b       | fsm         | 1x nano | ✅ | ❌ | ✅ | Single-model FSM |

| 3        | fsm         | 3x nano | ✅ | ✅ | ❌ | Full system local |

| 3b       | fsm         | 1x nano | ✅ | ✅ | ❌ | Single-model full local |

| 4        | fsm         | 3x nano | ✅ | ✅ | ✅ | **Production system** |

| 4b       | fsm         | 1x nano | ✅ | ✅ | ✅ | Single-model production |

| 5        | fsm         | 3x nano | ✅ | ✅ | ✅ | Fast motion force Jetson |

| 6        | fsm         | 1x medium | ✅* | ✅ | ✅ | Always-offload baseline |

*Config 6 uses FSM mode but with FIXED_TIER override

## File Structure

```
evaluation_analysis/
├── ingest_data.py          # Stage 1: Data loading
├── energy_integration.py    # Stage 2: Energy data handling (optional)
├── compute_metrics.py      # Stage 3: Metric computation
├── aggregate_results.py    # Stage 4: Aggregation & comparisons
├── export_data.py          # Stage 5: Data export
├── evaluate_system.py       # Stage 6: Main orchestration
└── evaluation_results/     # Output directory
    ├── frames_all.csv
    ├── runs_metrics.csv
    ├── aggregated_by_config_sequence.csv
    ├── hypothesis_comparisons.csv
    ├── motion_stratified_summary.csv
    └── sensor_prediction_summary.csv
```

## Dependencies

- pandas, numpy for data manipulation
- scipy for confidence intervals and statistical tests
- pathlib for file handling
- json, re for parsing

## Data Validation & Error Handling

- Skip malformed JSON files with warning logs
- Handle missing fields gracefully (fill with NaN/None)
- Report frames without ground truth matches (count per run)
- Detect and warn about duplicate frames (same run_id + frame_index)
- Check for incomplete runs (runs with very few frames)
- Validate timestamp consistency

## Testing Strategy

- Unit tests for each metric computation function
- Integration test with sample data from each ablation type
- Validation that all expected columns are present in outputs
- Cross-check GT matching accuracy

## To-dos

- [ ] Implement data ingestion module: discover runs, load ground truth (JSONL format), parse frame JSONs, match to GT by rounded timestamp
- [ ] Implement energy integration module: flexible format handling, timestamp matching, per-frame energy computation (optional)
- [ ] Implement metrics computation module: accuracy, latency (using total_pipeline_latency_s), stability, coverage, sensor metrics with config-aware logic
- [ ] Implement aggregation module: group by config/sequence, compute means ± CI, explicit hypothesis comparisons
- [ ] Implement data export module: export all DataFrames to CSV with proper column naming
- [ ] Implement main orchestration script with CLI interface, wire all modules together