Short list of defensible novelties for this system:

1. Adaptive FSM orchestrator for edge AI
   A resource-aware FSM (S0/S1/S2/S3/S5) that jointly optimizes latency, energy, and accuracy using weighted objectives, explicit hysteresis counters (M/N/K), and an S5 “anchor-state” recovery. Show fewer oscillations and bounded tail latency vs naive thresholds.

2. Diurnal baseline + gating for sensor fusion
   Exponential moving baselines per time-of-day window feed anomaly deltas and cooldown gates. Prove reduced false positives in stable weather and faster detection after regime shifts.

3. Multi-model YOLO consensus with provenance
   Tiered checkpoints per frame, IoU≥0.5 grouping, confidence-weighted xywh fusion, and explicit model vote counts (M=·). Use consensus both in the score and the UI overlays. Show F1 and calibration gains vs single-model or naive NMS.

4. On-demand local VLM confirmation at the edge
   Quantized VLM on Jetson used only on first S2 entry and in ambiguous S1/S3 frames. Integrates its verdict into the score and state transitions. Report ablation showing fewer false alarms at small latency/energy cost.

5. Bounded-backlog offload pattern over MQTT
   “Latest-payload buffer” on the Pi plus a single-job queue on Jetson. Retained heartbeats and timeouts yield predictable p99 and graceful fallback to local tiers. Provide tail-latency and drop-rate evidence under bursty input.

6. Unified scoring that fuses image, sensors, and consensus
   Explicit combination of image_score, sensor_boost, and consensus bonus with suppression gates. Publish a reproducible JSON schema and per-frame counters enabling post-hoc audits.

7. Tiered local/remote routing policy with health checks
   Policy chooses local small vs remote medium/large per state, device health, and SLA. Demonstrate a Pareto shift (accuracy↑ for same energy, or energy↓ for same latency).

8. Reproducible telemetry and storage contract
   Stable topic set and file layout that logs tiers, backend, counters, states, and p50/p90/p99 timings. Enables deterministic replays and fair cross-setting comparisons.

