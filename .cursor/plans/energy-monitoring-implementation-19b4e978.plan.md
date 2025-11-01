<!-- 19b4e978-06b0-42cf-a1c6-07e0677a4156 5efd35b6-e908-4f40-9ae5-8fa4b7ef1ecf -->
# Energy Monitoring Implementation Plan

## Overview

Add energy profiling to both the Processing Pi and Jetson worker to track power consumption during flood detection operations. Energy data will be stored in result JSON files alongside existing timing metrics.

## Architecture Changes

### Processing Pi (main_final.py)

- Use **codecarbon** to track CPU-based energy for expensive operations
- Measure energy for: FSM execution and YOLO-nano inference runs
- Store energy metrics in the `timing` payload of result JSON files

### Jetson (jetson_worker/worker.py)

- Use **jetson-stats (jtop)** to sample real-time power readings
- Sample at 100ms intervals during YOLO and LLM inference tasks
- Integrate power measurements over time to calculate energy (Joules)
- Include average power, CPU/GPU utilization in response metadata

## Implementation Steps

### 1. Dependencies

**File: requirements.txt**

- Add `codecarbon` for Pi energy estimation
- Add `jetson-stats` for Jetson power monitoring

### 2. Processing Pi Energy Tracker

**New file: flood_classifier/utils/energy_tracker.py**

- Create `EnergyTracker` class wrapping codecarbon's `EmissionsTracker`
- Methods: `start_measurement(operation_name)`, `stop_measurement()` → returns energy dict
- Handle codecarbon initialization for Raspberry Pi hardware
- Configure for offline mode (no cloud emissions API calls)

### 3. Integrate Energy Tracking in FSM

**File: flood_classifier/fsm/flood_fsm.py**

- In `FloodFSM.next_state()`: wrap the non-YOLO portions of the FSM (classification, counters, LLM calls) with the energy tracker so YOLO measurements stay separate
- Track logic-only energy and add metrics to the `timing_info` dict returned in `FrameDecision`
- Store: `fsm_energy_j` (Joules), `fsm_power_w` (average Watts), `fsm_cpu_util_%`

### 4. Integrate Energy Tracking for YOLO on Pi

**File: yolov8_processor/inference/multi_model_inference.py**

- In `run_all_inference()`: wrap YOLO execution with energy tracker
- Measure energy for all model inference calls
- Return energy metrics alongside inference results (via metadata)

**File: flood_classifier/fsm/flood_fsm.py (ModelManager)**

- In `ModelManager.infer()`: capture energy from YOLO inference
- Only populate YOLO energy metrics when `backend_meta["backend"] == "local"`; set them to `None` for remote runs so downstream aggregation does not double count
- Add YOLO energy to `backend_meta["metadata"]["energy"]`
- Propagate energy data to FSM's `backend_info`

### 5. Store Energy in Results

**File: main_final.py (process_message function)**

- After FSM classification, extract energy from `final_result.timing` and `backend_info`
- Add to `timing_payload`: `fsm_energy_j`, `yolo_energy_j`, `total_energy_j`
- Compute `total_energy_j` as `fsm_energy_j + (yolo_energy_j or 0) + (backend_energy_j or 0) + (llm_energy_j or 0)` while guarding against `None` values and only including Jetson metrics when the backend was remote
- Include power and utilization metrics: `fsm_power_w`, `cpu_util_%`

### 6. Jetson Power Monitor

**New file: jetson_worker/power_monitor.py**

- Create `JetsonPowerMonitor` class using `jtop` library
- Start background thread that samples power every 100ms
- Methods: `start_sampling(job_id)`, `stop_sampling(job_id)` → returns power stats
- Track: total energy (Joules), average power (Watts), peak power, CPU/GPU utilization
- Handle graceful fallback if jtop unavailable (return None metrics)

### 7. Integrate Power Monitoring in Jetson Worker

**File: jetson_worker/worker.py**

- Initialize `JetsonPowerMonitor` singleton in module scope
- In `run_yolo_inference()`: wrap inference with power monitor start/stop
- In `run_llm_inference()`: wrap LLM call with power monitor start/stop
- Add power metrics directly into the in-flight `_timing` dict before `export_fields` is built so the response `timing` dict includes `backend_energy_j`, `backend_avg_power_w`, `backend_peak_power_w`, `backend_cpu_util_%`, `backend_gpu_util_%`

**File: jetson_worker/worker.py (_process_payload)**

- Capture power metrics from inference functions
- Ensure the prefixed power metrics remain in `_timing`, include them in `export_fields`, and therefore in `response["timing"]`
- Ensure power data flows back to Pi in MQTT response

### 8. Propagate Jetson Energy to Pi Results

**File: main_final.py**

- In `process_message()`: extract Jetson energy from `backend_info["timing"]` and `llm_backend_info["timing"]`
- Add to `timing_payload`: `backend_energy_j`, `llm_energy_j` with power/utilization metrics
- Only include Jetson metrics when the backend metadata indicates a remote run, then calculate total system energy as Pi energy plus any remote Jetson contributions

## Key Files Modified

- `requirements.txt` - Add codecarbon, jetson-stats
- `flood_classifier/utils/energy_tracker.py` (NEW) - Codecarbon wrapper
- `jetson_worker/power_monitor.py` (NEW) - Jtop power monitor
- `flood_classifier/fsm/flood_fsm.py` - FSM energy tracking
- `yolov8_processor/inference/multi_model_inference.py` - YOLO energy tracking
- `jetson_worker/worker.py` - Jetson power monitoring integration
- `main_final.py` - Aggregate and store all energy metrics

## Output Schema

Energy metrics added to result JSON `timing` section:

```json
{
  "timing": {
    "pipeline_latency_s": 1.234,
    "fsm_energy_j": 2.5,
    "fsm_power_w": 3.2,
    "cpu_util_%": 75.3,
    "yolo_energy_j": 1.8,
    "backend_energy_j": 15.2,
    "backend_avg_power_w": 12.5,
    "backend_cpu_util_%": 45.0,
    "backend_gpu_util_%": 85.0,
    "llm_energy_j": 25.0,
    "total_energy_j": 44.5
  }
}
```

### To-dos

- [x] Add codecarbon and jetson-stats to requirements.txt
- [x] Create EnergyTracker wrapper class for codecarbon on Pi
- [x] Integrate energy tracking in FloodFSM.next_state()
- [x] Add energy tracking to YOLO inference on Pi
- [x] Create JetsonPowerMonitor class using jtop
- [x] Integrate power monitoring in Jetson worker inference functions
- [x] Aggregate and store all energy metrics in main_final.py result JSON without double counting and confirm Jetson timing payload includes power stats
- [ ] Test energy monitoring on both Pi and Jetson with sample workloads