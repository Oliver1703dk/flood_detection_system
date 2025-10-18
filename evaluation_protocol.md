**Evaluation Protocol**

This protocol defines how the distributed flood detection system will be evaluated in real-world conditions across three key metrics: **latency**, **accuracy**, and **energy efficiency**. The evaluation ensures reproducibility, transparency, and alignment with typical applied AI research standards.

---

### 1. Preparation and Baseline Setup

1. **Dataset synchronization**

   * Copy the full test video set and corresponding hand-labelled ground-truth JSONL files onto the **Processing Pi**.
   * Verify that each entry matches on `camera_id`, `video_file`, and `video_timestamp_sec`.

2. **Instrumentation check**

   * Confirm correct operation of `main_final.py` on the Processing Pi, including timing logs for pipeline and queue stages.
   * Ensure the Jetson worker is running with inference timing logs enabled.
   * Attach external power monitors to both the Pi and Jetson for energy recording.

3. **Environment snapshot**

   * Record active software versions, configuration file (`config.py`), selected model tiers, and current network conditions.
   * Save this snapshot in `storage/env_snapshots/` for reproducibility.

---

### 2. Latency Evaluation (Processing Pi + Jetson)

1. **System execution**

   * Run `python main_final.py` on the Processing Pi and replay the collector dataset.
   * Replays can be performed either through live MQTT publishing or scripted re-send of stored data.

2. **Data collection**

   * Extract per-frame timing data from the stored decision files, including:

     * `pipeline_latency_s`
     * Queue waiting time
     * FSM and backend metrics
     * `backend_*_ts` timestamps
   * Export all timing data to CSV for post-processing.

3. **Analysis**

   * Compute **p50**, **p90**, and **p99** latency percentiles for the entire pipeline.
   * Split latency statistics by **backend** (local vs remote) and **model_tier** (small, medium, large).
   * Break down component-level timings: preprocessing, inference, LLM, and baseline updates.
   * Validate MQTT round-trip latency by comparing `backend_roundtrip_s` against internal inference times and flag outliers.

---

### 3. Accuracy Evaluation (Prediction vs Ground Truth)

1. **Comparison script**

   * Develop a Python script to join system outputs with ground-truth JSONL labels using `video_file` and `video_timestamp_sec` (or frame index).
   * Extract for each frame: predicted flood state (`prediction`, `FrameDecision.state`) and the expected label.

2. **Metrics computation**

   * Compute confusion matrix (TP, FP, TN, FN).
   * Derive **accuracy**, **precision**, **recall**, **specificity**, **F1 score**, and **balanced accuracy**.
   * Compute **per-state accuracy** (S0–S3) to analyze FSM escalation effectiveness.

3. **Configuration experiments**

   * Repeat evaluation under varied configurations:

     * Local-only inference
     * Remote inference enabled
     * Different model tiers
   * Quantify performance trade-offs among these settings.

4. **Error inspection**

   * Log and analyze misclassified frames.
   * Correlate misclassifications with latency values and sensor readings to identify root causes.

---

### 4. Energy Evaluation (Processing Pi + Jetson)

1. **Measurement setup**

   * Use smart plugs, INA monitors, or Jetson’s `tegrastats` to record power usage.
   * Synchronize timestamps between power logs and pipeline events.

2. **Measurement phases**

   * Record power during:

     * **Idle phase** (system on, no data).
     * **Baseline phase** (system running, no active replay).
     * **Active phase** (during replay).

3. **Computation**

   * Calculate:

     * Mean and peak power consumption (W).
     * Energy per frame (J/frame).
   * Correlate energy data with backend type and model tier.
   * Build an **energy–latency–accuracy trade-off table** summarizing efficiency across configurations.

---

### 5. Reporting and Validation

1. **Result summary**

   * Aggregate and log all metrics, including dataset version, configurations, latency percentiles, accuracy statistics, and energy totals.

2. **Visualization**

   * Plot latency and energy distributions.
   * Include FSM state transitions and annotate outliers.

3. **Regression tracking**

   * Re-run this protocol after any major software or configuration update.
   * Store all runs in a shared repository for longitudinal performance comparison.

4. **Automation (optional)**

   * Implement a script to replay datasets, extract metrics, and generate Markdown summaries automatically for repeatable evaluations.

---

### 6. Recommended Extensions for Publication

* Add **environmental robustness tests** (lighting, weather, camera angle).
* Include **confidence intervals** or variance across multiple runs.
* Compare with a **baseline single-model detector** to quantify FSM and multi-tier gains.

This finalized evaluation plan is complete for research reporting and aligns with common practices in embedded AI and edge-computing system papers.
