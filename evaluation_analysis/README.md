# Evaluation Analysis Tools

This directory contains post-processing scripts for analyzing flood detection system evaluation results.

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. `compute_accuracy.py`

Compute accuracy metrics by comparing predictions to manually annotated ground truth.

**Usage:**
```bash
python compute_accuracy.py <results_dir> <ground_truth_csv> [-o output_file] [-t tolerance]
```

**Example:**
```bash
python compute_accuracy.py \
    ../storage/data_results/2025-11-19 \
    ground_truth_annotations.csv \
    -o ablation1_accuracy \
    -t 0.1
```

**Outputs:**
- `ablation1_accuracy.csv`: Detailed predictions vs ground truth
- `ablation1_accuracy.json`: Summary metrics (precision, recall, F1, confusion matrix)

---

### 2. `compute_latency.py`

Compute latency statistics from timing data in result JSON files.

**Usage:**
```bash
python compute_latency.py <results_dir> [-o output_file] [--plot]
```

**Example:**
```bash
python compute_latency.py \
    ../storage/data_results/2025-11-19 \
    -o ablation1_latency \
    --plot
```

**Outputs:**
- `ablation1_latency.csv`: Per-frame latency data
- `ablation1_latency.json`: Summary statistics (mean, p50, p90, p99)
- `ablation1_latency_plots.png`: Distribution plots (if --plot flag used)

---

### 3. `compute_energy.py`

Compute energy consumption from HMC power analyzer logs.

**Usage:**
```bash
python compute_energy.py <results_dir> <power_pi_csv> \
    [--power-jetson jetson_csv] [--idle-baseline watts] [-o output_file]
```

**Example:**
```bash
# Config 1 (Pi only)
python compute_energy.py \
    ../storage/data_results/2025-11-19 \
    power_ablation1_run1_pi.csv \
    --idle-baseline 2.5 \
    -o ablation1_energy

# Config 4 (Pi + Jetson)
python compute_energy.py \
    ../storage/data_results/2025-11-19 \
    power_ablation4_run1_pi.csv \
    --power-jetson power_ablation4_run1_jetson.csv \
    --idle-baseline 2.5 \
    -o ablation4_energy
```

**Outputs:**
- `ablation1_energy.csv`: Per-frame energy consumption
- `ablation1_energy.json`: Summary statistics (total energy, energy per frame)

---

### 4. `analyze_fsm.py`

Analyze FSM-specific behavior (state transitions, tier usage, oscillations).

**Usage:**
```bash
python analyze_fsm.py <results_dir> [-o output_file]
```

**Example:**
```bash
python analyze_fsm.py \
    ../storage/data_results/2025-11-19 \
    -o ablation2_fsm
```

**Outputs:**
- `ablation2_fsm.csv`: Per-frame FSM data
- `ablation2_fsm.json`: Summary (state distribution, transitions, oscillations)

---

## Ground Truth Format

The `ground_truth_annotations.csv` file must have the following columns:

```csv
video_file,frame_number,timestamp_sec,label,notes
flood_video_001.mp4,0,0.0,0,Clear road
flood_video_001.mp4,30,1.0,0,Still clear
flood_video_001.mp4,60,2.0,1,Water appearing
flood_video_001.mp4,90,3.0,2,Flooded
```

Where:
- `video_file`: Name of the video file
- `frame_number`: Frame number (0-indexed)
- `timestamp_sec`: Timestamp in seconds from video start
- `label`: 0 = no-flood, 1 = watch/some-water, 2 = flood
- `notes`: Optional annotation notes

---

## Power Log Format

HMC power analyzer logs must be CSV files with the following columns:

```csv
timestamp,voltage_V,current_A,power_W
2025-11-19 10:00:00.000,12.0,0.42,5.04
2025-11-19 10:00:00.100,12.0,0.43,5.16
```

Where:
- `timestamp`: ISO format datetime
- `voltage_V`: Voltage in volts
- `current_A`: Current in amperes
- `power_W`: Power in watts

---

## Hypothesis Validation

Use the scripts to validate the paper hypotheses:

### H1: Adaptive Tiering & Offload (Efficiency)

Compare Config 1 vs Config 4:

```bash
# Config 1 metrics
python compute_latency.py ../storage/data_results/config1/ -o config1_latency
python compute_energy.py ../storage/data_results/config1/ power_config1_pi.csv -o config1_energy
python compute_accuracy.py ../storage/data_results/config1/ ground_truth.csv -o config1_accuracy

# Config 4 metrics
python compute_latency.py ../storage/data_results/config4/ -o config4_latency
python compute_energy.py ../storage/data_results/config4/ power_config4_pi.csv --power-jetson power_config4_jetson.csv -o config4_energy
python compute_accuracy.py ../storage/data_results/config4/ ground_truth.csv -o config4_accuracy

# Compare:
# - Mean latency reduced ≥20%?
# - Energy per frame reduced ≥20%?
# - F1-score degradation ≤5%?
```

### H2: Motion-Aware Escalation & Hysteresis (Stability)

Compare Config 2 vs Config 4:

```bash
# Analyze oscillations
python analyze_fsm.py ../storage/data_results/config2/ -o config2_fsm
python analyze_fsm.py ../storage/data_results/config4/ -o config4_fsm

# Compare:
# - Oscillation rate reduced ≥15%?
# - P99 latency lower?
```

### H3: Diurnal Sensor Fusion (Environmental Robustness)

Compare Config 2 vs Config 3:

```bash
# Config 2 (vision-only)
python compute_accuracy.py ../storage/data_results/config2/ ground_truth.csv -o config2_accuracy

# Config 3 (with fusion)
python compute_accuracy.py ../storage/data_results/config3/ ground_truth.csv -o config3_accuracy

# Compare confusion matrices:
# - False positives reduced ≥15%?
# - Missed detections (false negatives) reduced ≥15%?
```

---

## Output Directory Structure

Recommended structure for analysis outputs:

```
evaluation_analysis/
├── results/
│   ├── ablation1/
│   │   ├── accuracy.json
│   │   ├── accuracy.csv
│   │   ├── latency.json
│   │   ├── latency.csv
│   │   ├── latency_plots.png
│   │   ├── energy.json
│   │   └── energy.csv
│   ├── ablation2/
│   │   ├── accuracy.json
│   │   ├── latency.json
│   │   ├── energy.json
│   │   ├── fsm.json
│   │   └── fsm.csv
│   ├── ablation3/
│   │   └── ...
│   └── ablation4/
│       └── ...
└── comparison_report.md  # Manual report comparing all configs
```

