# Comprehensive Ablation Study Guide

## Overview

This document describes the expanded 10-configuration ablation study designed to systematically evaluate each component of the flood detection system.

## Configuration Matrix

| Config | Models | Size | FSM | Sensor Fusion | Remote Offload | Jetson Needed | Purpose |
|--------|--------|------|-----|---------------|----------------|---------------|---------|
| **1**  | 1x | medium | ❌ | ❌ | ❌ | No | Single-model baseline |
| **1b** | 3x | medium | ❌ | ❌ | ❌ | No | Multi-model baseline |
| **2**  | 3x | nano | ✅ | ❌ | ✅ | **Yes** | Vision-only FSM |
| **2b** | 1x | nano | ✅ | ❌ | ✅ | **Yes** | Single-model FSM |
| **3**  | 3x | nano | ✅ | ✅ | ❌ | No | Full system local |
| **3b** | 1x | nano | ✅ | ✅ | ❌ | No | Single-model full local |
| **4**  | 3x | nano | ✅ | ✅ | ✅ | **Yes** | **Production system (original)** |
| **4b** | 1x | nano | ✅ | ✅ | ✅ | **Yes** | Single-model production |
| **5**  | 3x | nano | ✅ | ✅ | ✅ | **Yes** | **Production with fast motion force Jetson** |
| **6**  | 1x | medium | ✅* | ✅ | ✅ | **Yes** | **Naive always-offload baseline** |

*Config 6 uses FSM mode but with FIXED_TIER override, effectively disabling adaptive logic

## Experimental Design Dimensions

### 1. Multi-Model Consensus (1 vs 3 models)

**Question**: Does multi-model voting improve accuracy enough to justify the overhead?

**Comparisons**:
- Config 1 (1x medium) vs Config 1b (3x medium)
- Config 2b (1x nano) vs Config 2 (3x nano)
- Config 3b (1x nano) vs Config 3 (3x nano)
- Config 4b (1x nano) vs Config 4 (3x nano)

**Metrics**:
- Accuracy: F1-score, precision, recall, confusion matrix
- Energy: Energy per frame (should be ~3x higher for 3 models)
- Latency: Inference time (may be similar if parallel, or ~3x if sequential)

**Expected Outcome**: 3-model consensus should show improved accuracy, especially in ambiguous cases, at the cost of higher energy.

---

### 2. Model Size (Medium vs Nano)

**Question**: Can nano models + FSM match medium model performance?

**Comparisons**:
- Config 1b (3x medium, no FSM) vs Config 2 (3x nano, with FSM)

**Metrics**:
- Accuracy: F1-score comparison
- Energy: Should be lower for nano
- Latency: Should be lower for nano

**Expected Outcome**: FSM with adaptive tiering should compensate for nano's lower base accuracy.

---

### 3. FSM Adaptive Logic

**Question**: Does the FSM provide value over static classification?

**Comparisons**:
- Config 1b (3x medium, no FSM) vs Config 2 (3x nano, FSM, no fusion)

**Metrics**:
- Stability: Oscillation rate, flapping detection
- Latency: P50, P90, P99 percentiles
- State distribution: Time spent in each state
- Accuracy: May be similar or better with hysteresis

**Expected Outcome**: FSM should show reduced oscillations and more stable predictions.

---

### 4. Sensor Fusion

**Question**: Does diurnal sensor fusion improve robustness?

**Comparisons**:
- Config 2 (FSM, vision-only) vs Config 3 (FSM, with fusion)
- Config 2b (single, vision-only) vs Config 3b (single, with fusion)

**Metrics**:
- False Positives: Should decrease with fusion
- False Negatives: Should decrease with fusion
- Environmental robustness: Performance across different weather/lighting
- Latency: Should be minimal overhead

**Expected Outcome**: Sensor fusion should reduce false alarms and improve detection in challenging conditions.

---

### 5. Remote Offload

**Question**: Does distributed inference provide efficiency gains?

**Comparisons**:
- Config 3 (all local) vs Config 4 (with offload)
- Config 3b (single, all local) vs Config 4b (single, with offload)

**Metrics**:
- Latency: Should be lower with Jetson for higher tiers
- Energy: Pi energy should be lower, but total (Pi + Jetson) may vary
- Tier usage: Should see more medium/large tier usage with offload
- Network: Round-trip times, queue superseding

**Expected Outcome**: Offload should reduce Pi workload and improve responsiveness for complex scenes.

---

## Hypothesis Testing

### H1: Adaptive Tiering & Offload (Efficiency)

**Hypothesis**: The full system (FSM + adaptive tiering + offload) reduces latency and energy by ≥20% while maintaining accuracy (≤5% F1 degradation).

**Test**: Config 1b vs Config 4
- **1b**: 3x medium, no FSM, all local (strong baseline)
- **4**: 3x nano, FSM, fusion, offload (production)

**Why this comparison?**
- Both use 3-model consensus (fair comparison)
- Isolates the combined effect of all optimizations
- Shows if nano+FSM+offload can beat static medium

---

### H2: Motion-Aware Escalation & Hysteresis (Stability)

**Hypothesis**: Sensor fusion reduces label oscillations by ≥15% and provides tighter p99 latency.

**Test**: Config 2 vs Config 4
- **2**: FSM, vision-only
- **4**: FSM, with fusion

**Why this comparison?**
- Both use FSM with offload
- Isolates pure sensor fusion effect
- Tests stability claims

---

### H3: Diurnal Sensor Fusion (Environmental Robustness)

**Hypothesis**: Sensor fusion reduces false positives and missed detections by ≥15% without latency increase.

**Test**: Config 2 vs Config 3
- **2**: FSM, vision-only, remote
- **3**: FSM, with fusion, local

**Why this comparison?**
- Isolates sensor fusion
- Both use FSM with same model setup
- Local vs remote is orthogonal concern

---

### H4: Multi-Model Consensus Value

**Hypothesis**: Multi-model consensus improves accuracy, justifying the energy overhead.

**Test**: Config 1 vs Config 1b, Config 4b vs Config 4
- **1** vs **1b**: 1x vs 3x medium (baseline)
- **4b** vs **4**: 1x vs 3x nano (production)

**Why these comparisons?**
- Isolates pure consensus effect
- Tests at different model sizes
- Validates design choice

---

## Execution Strategy

### Phase 1: Baseline Establishment (Configs 1, 1b)
```bash
python main_ablation1.py   # 1x medium baseline
python main_ablation1b.py  # 3x medium baseline
```
- No Jetson needed
- Establishes performance ceiling
- Tests consensus value on medium models

### Phase 2: Vision-Only FSM (Configs 2, 2b)
```bash
# Terminal 1 (Jetson):
python jetson_worker/worker.py

# Terminal 2 (Pi):
python main_ablation2.py   # 3x nano, FSM, vision-only
python main_ablation2b.py  # 1x nano, FSM, vision-only
```
- Jetson required
- Tests FSM without sensor fusion confound
- Tests consensus value with FSM

### Phase 3: Local Full System (Configs 3, 3b)
```bash
python main_ablation3.py   # 3x nano, FSM, fusion, local
python main_ablation3b.py  # 1x nano, FSM, fusion, local
```
- No Jetson needed
- Tests full system on Pi only
- Baseline for offload comparison

### Phase 4: Production System (Configs 4, 4b)
```bash
# Terminal 1 (Jetson):
python jetson_worker/worker.py

# Terminal 2 (Pi):
python main_ablation4.py   # 3x nano, FSM, fusion, offload (PRODUCTION ORIGINAL)
python main_ablation4b.py  # 1x nano, FSM, fusion, offload
```
- Jetson required
- Full production configuration (original motion-aware policy)
- Final comparison point

### Phase 5: Motion-Aware Policies (Configs 5, 6)
```bash
# Terminal 1 (Jetson):
python jetson_worker/worker.py

# Terminal 2 (Pi):
python main_ablation5.py   # 3x nano, FSM, fusion, offload, FAST_MOTION_TIER=medium
python main_ablation6.py  # 1x medium, FIXED_TIER, always offload (naive baseline)
```
- Jetson required
- Config 5: Tests new real-time policy for fast motion (safety > energy)
- Config 6: Naive "always use Jetson" baseline showing why blind offloading is terrible for energy

---

## Data Collection Requirements

For each configuration, collect:

1. **Run 3-5 repetitions** with same test video
2. **Ground truth annotations** (same for all runs)
3. **HMC power logs** (Pi + Jetson where applicable)
4. **Result JSONs** (automatically saved)
5. **Detection images** (automatically saved)

---

## Analysis Workflow

### 1. Accuracy Analysis
```bash
cd evaluation_analysis

# For each config
python compute_accuracy.py ../storage/data_results/config1/ ground_truth.csv -o results/config1/accuracy
python compute_accuracy.py ../storage/data_results/config1b/ ground_truth.csv -o results/config1b/accuracy
# ... repeat for all configs
```

### 2. Latency Analysis
```bash
python compute_latency.py ../storage/data_results/config1/ -o results/config1/latency --plot
python compute_latency.py ../storage/data_results/config1b/ -o results/config1b/latency --plot
# ... repeat for all configs
```

### 3. Energy Analysis
```bash
# Pi-only configs (1, 1b, 3, 3b)
python compute_energy.py ../storage/data_results/config1/ power_config1_pi.csv --idle-baseline 2.5 -o results/config1/energy

# Pi + Jetson configs (2, 2b, 4, 4b, 5, 6)
python compute_energy.py ../storage/data_results/config2/ power_config2_pi.csv --power-jetson power_config2_jetson.csv --idle-baseline 2.5 -o results/config2/energy
python compute_energy.py ../storage/data_results/ablation5/ power_config5_pi.csv --power-jetson power_config5_jetson.csv --idle-baseline 2.5 -o results/ablation5/energy
python compute_energy.py ../storage/data_results/ablation6/ power_config6_pi.csv --power-jetson power_config6_jetson.csv --idle-baseline 2.5 -o results/ablation6/energy
```

### 4. FSM Analysis (Configs 2-6, 2b-4b only)
```bash
python analyze_fsm.py ../storage/data_results/config2/ -o results/config2/fsm
python analyze_fsm.py ../storage/data_results/config2b/ -o results/config2b/fsm
python analyze_fsm.py ../storage/data_results/ablation5/ -o results/ablation5/fsm
# ... repeat for FSM configs (note: config 6 uses FSM but with fixed tier)
```

---

## Expected Results Summary

| Comparison | Metric | Expected Change | Validates |
|------------|--------|-----------------|-----------|
| 1 vs 1b | F1 score | +5-10% | Consensus value |
| 1 vs 1b | Energy/frame | +200% | Consensus cost |
| 1b vs 2 | Latency | -20-30% | Nano efficiency |
| 1b vs 4 | Latency | -20%+ | H1: Full system |
| 1b vs 4 | Energy | -20%+ | H1: Full system |
| 1b vs 4 | F1 score | -0 to -5% | H1: Accuracy maintained |
| 2 vs 4 | Oscillations | -15%+ | H2: Stability |
| 2 vs 3 | False Positives | -15%+ | H3: Sensor fusion |
| 2 vs 3 | False Negatives | -15%+ | H3: Sensor fusion |
| 3 vs 4 | Latency (p50) | -10-20% | Offload benefit |
| 4b vs 4 | F1 score | +5-10% | Consensus value |
| 4b vs 4 | Energy | +150-200% | Consensus cost |
| 4 vs 5 | Energy (fast motion) | +10-20% | Fast motion policy cost |
| 4 vs 5 | Latency (fast motion) | -5-10% | Fast motion policy benefit |
| 4 vs 6 | Energy | +200-300% | Naive offload cost |
| 4 vs 6 | Latency | -5-10% | Naive offload benefit (marginal) |

---

## Quick Reference Commands

```bash
# Run all local configs (no Jetson)
for config in 1 1b 3 3b; do
    echo "Running config ${config}..."
    python main_ablation${config}.py
done

# Run all remote configs (requires Jetson)
# Terminal 1 (Jetson):
python jetson_worker/worker.py

# Terminal 2 (Pi), for each:
for config in 2 2b 4 4b 5 6; do
    echo "Running config ${config}..."
    python main_ablation${config}.py
done
```

---

## Additional Experimental Dimensions

### 6. Fast Motion Policy (Config 5)

**Question**: Does forcing medium tier and offload on fast motion improve real-time responsiveness?

**Comparisons**:
- Config 4 (original policy) vs Config 5 (fast motion force Jetson)

**Metrics**:
- Latency during fast motion: Should be lower with forced medium tier
- Energy during fast motion: Should be higher due to always offloading
- Overall accuracy: Should be similar or better with higher tier

**Expected Outcome**: Config 5 should show improved latency for fast motion scenes at the cost of higher energy, validating the safety > energy tradeoff.

---

### 7. Naive Offload Baseline (Config 6)

**Question**: Why is adaptive offloading better than always offloading?

**Comparisons**:
- Config 4 (adaptive) vs Config 6 (always offload)
- Config 1b (local medium) vs Config 6 (remote medium)

**Metrics**:
- Energy: Should be significantly higher for Config 6 (always offload)
- Latency: May be slightly better but not enough to justify energy cost
- Network usage: Much higher for Config 6

**Expected Outcome**: Config 6 demonstrates why blind offloading is inefficient, validating the need for adaptive policies.

---

## Files Created

- `config_ablation1.py` through `config_ablation6.py` (10 files)
- `main_ablation1.py` through `main_ablation6.py` (10 files)
- This guide: `ABLATION_STUDY_GUIDE.md`

Total: **21 new files** for comprehensive ablation study.

