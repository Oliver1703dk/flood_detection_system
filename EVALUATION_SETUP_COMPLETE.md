# Evaluation System Setup - Complete ✅

## Summary

The flood detection system has been successfully prepared for rigorous evaluation following the protocol in the evaluation guide. All 4 ablation configurations are ready to run, and comprehensive post-processing analysis scripts have been created.

## What Was Implemented

### Phase 1: Sensor Fusion Toggle ✅

**Modified Files:**
1. `flood_classifier/inference/classifier_both.py`
   - Added `disable_sensor_fusion` parameter to `__init__`
   - Wrapped sensor fusion logic in conditional check
   - When disabled, skips baseline lookups and sensor anomaly calculations

2. `flood_classifier/classification/strategies/yolo_sensor.py`
   - Reads `DISABLE_SENSOR_FUSION` from config
   - Passes flag to ClassifierBoth constructor

3. `flood_classifier/classification/strategies/fsm.py`
   - Added `vision_only` parameter to FSMStrategy
   - Modified `_get_default_fsm()` to create ClassifierBoth with sensor fusion disabled when needed
   - Imported ClassifierBoth for explicit instantiation

### Phase 2: Configuration Files ✅

**Created Files:**

**Core Configurations (3x models):**
1. `config_ablation1.py` - Static Single Medium-YOLO (Baseline)
   - **Single** medium model (no consensus)
   - No FSM (uses yolo_sensor mode)
   - Sensor fusion disabled
   - No remote offload

1b. `config_ablation1b.py` - Multi-Model Medium-YOLO Baseline
   - **3x** medium models with consensus
   - No FSM (uses yolo_sensor mode)
   - Sensor fusion disabled
   - No remote offload

2. `config_ablation2.py` - Vision-Only + FSM + Multi-Model + Offload
   - FSM enabled
   - **3x** nano models (multi-model consensus)
   - Sensor fusion disabled
   - Remote offload enabled for S1/S2/S3

3. `config_ablation3.py` - Full System Local-Only
   - FSM enabled
   - **3x** nano models
   - Sensor fusion enabled
   - All inference forced local

4. `config_ablation4.py` - Full System Remote-Enabled (Production)
   - FSM enabled
   - **3x** nano models
   - Sensor fusion enabled
   - Remote offload enabled for S1/S2/S3

**Single-Model Variants (1x model):**
2b. `config_ablation2b.py` - Single-Model Vision-Only + FSM + Offload
   - FSM enabled
   - **1x** nano model (tests FSM without consensus)
   - Sensor fusion disabled
   - Remote offload enabled for S1/S2/S3

3b. `config_ablation3b.py` - Single-Model Full System Local-Only
   - FSM enabled
   - **1x** nano model
   - Sensor fusion enabled
   - All inference forced local

4b. `config_ablation4b.py` - Single-Model Full System Remote-Enabled
   - FSM enabled
   - **1x** nano model
   - Sensor fusion enabled
   - Remote offload enabled for S1/S2/S3

**Key Settings Per Config:**

| Feature | 1 | 1b | 2 | 2b | 3 | 3b | 4 | 4b |
|---------|---|----|----|----|----|----|----|-----|
| CLASSIFICATION_MODE | yolo_sensor | yolo_sensor | fsm | fsm | fsm | fsm | fsm | fsm |
| model_size | medium | medium | nano | nano | nano | nano | nano | nano |
| model_number | 1 | **3** | 3 | **1** | 3 | **1** | 3 | **1** |
| DISABLE_SENSOR_FUSION | True | True | True | True | False | False | False | False |
| INFERENCE_ROUTING | all local | all local | S1/S2/S3 remote | S1/S2/S3 remote | all local | all local | S1/S2/S3 remote | S1/S2/S3 remote |
| ENABLE_FSM_ENERGY_TRACKING | False | False | False | False | False | False | False | False |

### Phase 3: Entry Point Scripts ✅

**Created Files:**

**Core Configurations:**
1. `main_ablation1.py` - Imports config_ablation1 (1x medium, no FSM)
1b. `main_ablation1b.py` - Imports config_ablation1b (3x medium, no FSM)
2. `main_ablation2.py` - Imports config_ablation2 (3x nano, FSM, no fusion)
3. `main_ablation3.py` - Imports config_ablation3 (3x nano, FSM, with fusion, local)
4. `main_ablation4.py` - Imports config_ablation4 (3x nano, FSM, with fusion, remote)

**Single-Model Variants:**
2b. `main_ablation2b.py` - Imports config_ablation2b (1x nano, FSM, no fusion)
3b. `main_ablation3b.py` - Imports config_ablation3b (1x nano, FSM, with fusion, local)
4b. `main_ablation4b.py` - Imports config_ablation4b (1x nano, FSM, with fusion, remote)

Each file:
- Overrides `sys.modules['config']` before other imports
- Runs the full MQTT processing pipeline from main_final.py
- Displays ablation-specific header on startup with configuration details

### Phase 4: Post-Processing Analysis Scripts ✅

**Created Directory:** `evaluation_analysis/`

**Scripts:**

1. **`utils.py`** - Utility functions
   - `load_ground_truth()` - Parse annotation CSV
   - `load_json_results()` - Load result JSON files
   - `extract_prediction()` - Extract predictions from various formats
   - `match_result_to_ground_truth()` - Match results to annotations
   - `load_power_logs()` - Parse HMC power logs
   - `integrate_power()` - Compute energy from power measurements
   - `parse_timestamp()` - Handle various timestamp formats

2. **`compute_accuracy.py`** - Accuracy analysis
   - Compares predictions to manual ground truth annotations
   - Computes precision, recall, F1 per class
   - Generates confusion matrix
   - Outputs: CSV with detailed matches, JSON with summary metrics

3. **`compute_latency.py`** - Latency analysis
   - Extracts timing metrics from result JSONs
   - Computes mean, std, p50, p90, p99 percentiles
   - Breaks down by FSM state and backend (local vs remote)
   - Optional distribution plots
   - Outputs: CSV with per-frame latencies, JSON with statistics, PNG plots

4. **`compute_energy.py`** - Energy analysis
   - Integrates HMC power logs over frame processing time
   - Handles Pi + Jetson measurements separately
   - Subtracts idle baseline
   - Computes energy per frame
   - Outputs: CSV with per-frame energy, JSON with statistics

5. **`analyze_fsm.py`** - FSM behavior analysis
   - State distribution and transitions
   - Tier usage distribution
   - Average dwell time per state
   - Label oscillation detection
   - Flapping and conflict detection
   - Outputs: CSV with FSM data, JSON with summary

6. **`requirements.txt`** - Analysis dependencies
   - pandas, numpy, matplotlib, scikit-learn

7. **`README.md`** - Complete usage documentation
   - Script usage examples
   - Ground truth format specification
   - Power log format specification
   - Hypothesis validation guide

8. **`__init__.py`** - Package initialization

## How to Use

### 1. Run Evaluations

**Configs 1, 1b, 3, 3b (No Jetson needed - all local):**
```bash
python main_ablation1.py   # 1x medium, no FSM
python main_ablation1b.py  # 3x medium, no FSM
python main_ablation3.py   # 3x nano, FSM, fusion, local
python main_ablation3b.py  # 1x nano, FSM, fusion, local
```

**Configs 2, 2b, 4, 4b (Jetson required - remote offload enabled):**
```bash
# Terminal 1 (Jetson):
python jetson_worker/worker.py

# Terminal 2 (Processing Pi):
python main_ablation2.py   # 3x nano, FSM, no fusion
python main_ablation2b.py  # 1x nano, FSM, no fusion
python main_ablation4.py   # 3x nano, FSM, fusion (production)
python main_ablation4b.py  # 1x nano, FSM, fusion
```

### 2. Collect Data

- Results saved to: `storage/data_results/YYYY-MM-DD/`
- Each run should use:
  - Same annotated test video
  - HMC power measurement running
  - Fixed duration or frame count

### 3. Analyze Results

```bash
cd evaluation_analysis

# Accuracy
python compute_accuracy.py \
    ../storage/data_results/2025-11-19 \
    ground_truth_annotations.csv \
    -o results/ablation1/accuracy

# Latency
python compute_latency.py \
    ../storage/data_results/2025-11-19 \
    -o results/ablation1/latency \
    --plot

# Energy
python compute_energy.py \
    ../storage/data_results/2025-11-19 \
    power_ablation1_pi.csv \
    --idle-baseline 2.5 \
    -o results/ablation1/energy

# FSM (for configs 2, 3, 4)
python analyze_fsm.py \
    ../storage/data_results/2025-11-19 \
    -o results/ablation2/fsm
```

## Experimental Design Rationale

The expanded ablation suite systematically isolates the effects of:

1. **Multi-Model Consensus** (1 vs 3b, 2 vs 2b, 3 vs 3b, 4 vs 4b)
   - Tests whether multi-model voting improves accuracy
   - Measures energy/latency overhead of consensus

2. **Model Size** (1/1b vs 2/3/4)
   - Medium vs nano models
   - Baseline comparison for tiering

3. **FSM Logic** (1/1b vs 2/2b)
   - Tests adaptive state machine benefits
   - Isolates FSM from sensor fusion

4. **Sensor Fusion** (2/2b vs 3/3b)
   - Tests diurnal baseline integration
   - Measures impact on accuracy

5. **Remote Offload** (3/3b vs 4/4b)
   - Tests distributed inference benefits
   - Measures network overhead

## Hypothesis Validation

The analysis scripts provide all metrics needed to validate the paper's hypotheses:

### H1: Adaptive Tiering & Offload (Efficiency)
- **Compare**: Config 1b (3x medium, no FSM) vs Config 4 (3x nano, FSM, fusion, offload)
- **Metrics**: mean latency, energy per frame, F1-score
- **Target**: ≥20% reduction in latency & energy, ≤5% F1 degradation
- **Isolates**: Full system optimization (FSM + adaptive tiering + offload)

### H2: Motion-Aware Escalation & Hysteresis (Stability)
- **Compare**: Config 2 (FSM, no fusion) vs Config 4 (FSM, with fusion)
- **Metrics**: oscillation rate, p99 latency, dwell times
- **Target**: ≥15% reduction in oscillations, tighter p99
- **Isolates**: Sensor fusion stabilization effect

### H3: Diurnal Sensor Fusion (Environmental Robustness)
- **Compare**: Config 2 (vision-only) vs Config 3 (with fusion, both local)
- **Metrics**: false positives, false negatives from confusion matrix
- **Target**: ≥15% reduction in FP and FN, no latency increase
- **Isolates**: Pure sensor fusion benefit without offload confound

### H4 (New): Multi-Model Consensus Value
- **Compare**: Config 1 (1x medium) vs Config 1b (3x medium)
- **Compare**: Config 4b (1x nano) vs Config 4 (3x nano)
- **Metrics**: accuracy improvement, energy overhead, latency overhead
- **Target**: Justify multi-model consensus cost
- **Isolates**: Pure consensus effect

## Files Modified

**Phase 1 (3 files):**
- `flood_classifier/inference/classifier_both.py`
- `flood_classifier/classification/strategies/yolo_sensor.py`
- `flood_classifier/classification/strategies/fsm.py`

**Phase 2-4 (25 new files):**
- 8 config files: `config_ablation1.py` through `config_ablation4b.py`
- 8 main files: `main_ablation1.py` through `main_ablation4b.py`
- 8 analysis files in `evaluation_analysis/`
- 1 summary document (this file)

## Next Steps

1. **Prepare Ground Truth:**
   - Manually annotate test video frames
   - Create `ground_truth_annotations.csv`

2. **Setup HMC Power Analyzers:**
   - Connect to Pi power supply
   - Connect to Jetson power supply (if applicable)
   - Configure sampling rate ≥10 Hz
   - Synchronize clocks via NTP

3. **Run Evaluations:**
   - Start with Config 1 (simplest)
   - Run 3-5 times per configuration
   - Use consistent test video and settings

4. **Analyze Results:**
   - Run all analysis scripts per configuration
   - Compare metrics across configurations
   - Validate hypotheses

5. **Generate Report:**
   - Compile results into comparison tables
   - Create visualizations
   - Document findings for paper

## Status: ✅ READY FOR EVALUATION

All code modifications and analysis tools are complete and tested.
The system is ready for rigorous evaluation following the protocol.

