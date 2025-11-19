# **📋 Updated Evaluation Guide for Full MQTT Pipeline**

**(Fully Revised Protocol with External HMC Power Measurement, Manual Ground Truth Annotation, and All Required Adjustments)**

## **🏗️ System Architecture** 

Your evaluation uses the **3-node distributed architecture**:

1. **Gathering/Collector Pi**: Captures camera \+ Netatmo sensor data → publishes to MQTT `sensor/data`  
2. **Processing Pi**: Subscribes to `sensor/data` → validates → runs classification → saves results  
3. **Jetson Worker** (some configs): Handles offloaded heavy YOLO tiers \+ optional LLM

---

## **📝 Preparation Steps** 

### **1\. Create Configuration Variants**

- Duplicate `config.py` → `config_ablation1.py` … `config_ablation4.py`

### **2\. Create Entry Point Scripts**

- Duplicate `main_final.py` → `main_ablation1.py` … `main_ablation4.py`  
- Each imports its corresponding config

### **3\. Add Sensor Fusion Toggle** 

- Modify `ClassifierBoth` to accept `disable_sensor_fusion` flag

### **4\. Update FSM Strategy Initialization (unchanged)**

- Allow `vision_only=True` parameter

### **5\. Disable Internal Energy Tracking**

In **every** `config_ablationN.py` add:

ENABLE\_FSM\_ENERGY\_TRACKING \= False   \# Disable software energy tracking to eliminate CPU overhead during external HMC measurement

---

## **📹 Ground Truth Annotation** 

### **Before Running Any Evaluations:**

1. **Select test video(s)**  
     
   - Representative flood/no-flood transitions  
   - Varied lighting, weather, time of day  
   - Minimum 300–500 frames for statistical significance

   

2. **Manual frame-by-frame annotation**  
     
   - Labels: `0` \= no-flood, `1` \= watch/some water, `2` \= flood  
   - Record exact frame number or timestamp

   

3. **Output format** → `ground_truth_annotations.csv`:

video\_file,frame\_number,timestamp\_sec,label,notes

flood\_video\_001.mp4,0,0.0,0,Clear road

flood\_video\_001.mp4,30,1.0,0,Still clear

flood\_video\_001.mp4,60,2.0,1,Water appearing

flood\_video\_001.mp4,90,3.0,2,Fully flooded

...

4. **Replay consistency**  
   - Collector must publish the exact same video at fixed FPS (e.g., 1 fps)  
   - Include `video_file` and `video_timestamp_sec` in MQTT metadata

---

## **⚡ HMC Power Analyzer Setup** 

### **Hardware Configuration**

- **Processing Pi**: HMC between PSU and Pi (≥10 Hz sampling)  
- **Jetson** (Configs 2 & 4): Separate HMC on Jetson PSU  
- Log format: `timestamp, voltage_V, current_A, power_W`  
- Synchronize clocks on Pi, Jetson, and HMC via NTP

### **Baseline Measurement**

- Record 30 s idle power before each run  
- Subtract baseline from active measurements

### **Synchronization Strategy**

- Note exact wall-clock start/end time when launching `main_ablationN.py`  
- Each result JSON contains `process_start_ts` (wall-clock timestamp)  
- Align power samples using absolute timestamps

---

## **🎯 Configuration 1: Static Medium-YOLO Only (Baseline – unchanged logic, updated settings)**

**What to Change in `config_ablation1.py`:**

CLASSIFICATION\_MODE \= "yolo\_sensor"

MODEL\_SIZE \= "medium"

MODEL\_NUMBER \= 1

USE\_LLM\_CONFIRMATION \= False

ENABLE\_FSM\_ENERGY\_TRACKING \= False

\# Vision-only (disable sensor fusion)

\# Pass disable\_sensor\_fusion=True in YoloSensorStrategy

No Jetson needed.

---

## **🎯 Configuration 2: Vision-Only \+ FSM \+ Multi-Model \+ Offload**

**What to Change in `config_ablation2.py`:**

CLASSIFICATION\_MODE \= "fsm"

MODEL\_SIZE \= "nano"

MODEL\_NUMBER \= 3

LOCAL\_YOLO\_TIER \= "small"          \# medium/large → Jetson

ENABLE\_FSM\_ENERGY\_TRACKING \= False

\# Pass vision\_only=True → ClassifierBoth(disable\_sensor\_fusion=True)

Jetson required.

---

## **🎯 Configuration 3: Full System Local-Only (No Offload, With Sensor Fusion)**

**What to Change in `config_ablation3.py`:**

CLASSIFICATION\_MODE \= "fsm"

MODEL\_SIZE \= "nano"

MODEL\_NUMBER \= 3

\# Force everything local

INFERENCE\_ROUTING \= {"S0": "local", "S1": "local", "S2": "local", "S3": "local", "S5": "local", "default": "local"}

USE\_DEFAULT\_BASELINE \= True

ENABLE\_FSM\_ENERGY\_TRACKING \= False

\# Sensor fusion ENABLED (default)

No Jetson needed.

---

## **🎯 Configuration 4: Full System Remote-Enabled (Production configuration)**

**What to Change in `config_ablation4.py`:**

CLASSIFICATION\_MODE \= "fsm"

MODEL\_SIZE \= "nano"

MODEL\_NUMBER \= 3

LOCAL\_YOLO\_TIER \= "small"

USE\_DEFAULT\_BASELINE \= True

ENABLE\_FSM\_ENERGY\_TRACKING \= False

\# Default routing: medium/large → Jetson

\# Sensor fusion ENABLED

Jetson required.

---

## **📊 Execution Matrix (unchanged)**

| Config | Nodes Running | Jetson | Sensor Fusion | Key Focus |
| :---- | :---- | :---: | :---: | :---- |
| 1 | Collector \+ Processor | ❌ | ❌ | Static baseline |
| 2 | Collector \+ Processor \+ Jetson | ✅ | ❌ | FSM \+ offload (vision-only) |
| 3 | Collector \+ Processor | ❌ | ✅ | Full system local-only |
| 4 | Collector \+ Processor \+ Jetson | ✅ | ✅ | Full production system |

---

## **🔬 Data Collection Strategy** 

### **For Each Configuration (run 3–5 times):**

1. **Pre-run**  
     
   - Clear or create dated result folder  
   - Start HMC recording (both Pi and Jetson if applicable)  
   - Record idle baseline (30 s)  
   - Note exact start wall-clock time

   

2. **During run**  
     
   - Replay the **exact same annotated video** at fixed FPS  
   - Process fixed number of frames (e.g., 500\) or fixed duration (e.g., 5 min)  
   - Let system save results to `storage/data_results/YYYY-MM-DD/<timestamp>.json`

   

3. **Immediately after run**  
     
   - Note exact end wall-clock time  
   - Stop HMC recording  
   - Save power logs as `power_ablationN_runX_pi.csv` and `power_ablationN_runX_jetson.csv`

   

4. **Post-run analysis** (automated scripts recommended)

| Metric | Source | Extraction Method | Notes |
| :---- | :---- | :---- | :---- |
| **Accuracy (F1, Precision, Recall)** | JSON predictions vs `ground_truth_annotations.csv` | Match by `video_file` \+ `video_timestamp_sec` or frame number → compare `final_prediction`/`decision.prediction` | Per-config, per-state (FSM) |
| **Latency (p50/p90/p99)** | JSON `timing.pipeline_latency_s` | Collect all frames → numpy percentile | Breakdown: inference, classification, queue, backend |
| **Energy per frame (J)** | HMC power logs | Integrate power over exact run window → subtract idle baseline → total\_J / frame\_count | Sum Pi \+ Jetson for distributed configs |
| **State transitions** | JSON `decision.state` (FSM only) | Count transitions, compute dwell time per state | FSM configs only |
| **Tier usage** | JSON `decision.model_tier` (FSM only) | Histogram nano/small/medium/large | FSM configs only |
| **Queue superseding** | Console "superseded" messages | Grep/count logs | Remote configs only |

---

## **🔬 Updated Hypothesis Validation**

### **H1: Adaptive Tiering & Offload (Efficiency)**

**Config 1 → Config 4**

- Mean latency reduction ≥20%?  
- Energy per frame reduction ≥20%?  
- F1-score degradation ≤5%?

### **H2: Motion-Aware Escalation & Hysteresis (Stability)**

**Config 2 → Config 4**

- Label oscillations (consecutive flips) reduced ≥15%?  
- P99 latency lower/tighter?  
- Longer average state dwell time?

### **H3: Diurnal Sensor Fusion (Environmental Robustness)**

**Config 2 (vision-only) → Config 3 (with fusion)**

- False positives reduced ≥15%?  
- Missed floods reduced ≥15%?  
- Mean latency increase ≈0?

---

## **✅ Final Checklist (Before Starting Evaluations)**

- [ ] Ground truth CSV completed and verified  
- [ ] HMC analyzers connected, calibrated, ≥10 Hz  
- [ ] NTP clock sync on Pi, Jetson, and HMC PC  
- [ ] `ENABLE_FSM_ENERGY_TRACKING = False` in all 4 config files  
- [ ] Test video replays identically (same FPS, same metadata)  
- [ ] Post-processing scripts ready (JSON parsing, power integration, accuracy comparison)  
- [ ] Storage directories clean or dated  
- [ ] Idle baseline power measured