"""
Main entry point for Ablation 4b: Single-Model Full System Remote-Enabled
- FSM enabled, single nano model, sensor fusion enabled, remote offload enabled
- Tests full system with single model vs Ablation 4 (3x nano)
"""

# Override config module BEFORE any other imports
import config_ablation4b as config
import sys
sys.modules['config'] = config

# Now import the rest (they will use config_ablation4b)
from pathlib import Path
from datetime import datetime
import main_final
from main_final import *

def initialize_strategy():
    """Initialize the classification strategy at startup to trigger pre-warming."""
    # Update the _strategy in main_final module (since process_message uses that)
    if main_final._strategy is not None:
        return main_final._strategy
    
    print("\n🚀 Initializing classification strategy...")
    strategy_map = {
        "yolo_sensor": YoloSensorStrategy,
        "llm_only": LLMOnlyStrategy,
        "fsm": FSMStrategy,
    }
    classification_mode = getattr(config, 'CLASSIFICATION_MODE', 'fsm')
    strategy_cls = strategy_map.get(classification_mode)
    if strategy_cls is None:
        raise ValueError(f"Invalid classification mode: {classification_mode}")
    
    main_final._strategy = strategy_cls()
    print("✅ Strategy initialized (models pre-warmed if applicable)\n")
    return main_final._strategy

def main():
    """
    Main method for Ablation 4b.
    Initializes the MQTT receiver and directs each incoming message payload
    to the process_message callback for processing.
    """
    print("=" * 60)
    print("ABLATION 4b: Single-Model Full System Remote-Enabled")
    print("=" * 60)
    
    # Extract ablation name from config file name (e.g., "config_ablation4b.py" -> "ablation4b")
    config_file = getattr(config, '__file__', '')
    if 'ablation' in config_file:
        ablation_name = config_file.split('config_')[1].split('.py')[0]
    else:
        # Fallback: extract from RESULTS_STORAGE_DIR if available
        results_dir = getattr(config, 'RESULTS_STORAGE_DIR', 'storage/data_results/ablation4b')
        if '/ablation' in results_dir:
            ablation_name = results_dir.split('/ablation')[1].split('/')[0]
            ablation_name = f"ablation{ablation_name}"
        else:
            ablation_name = "ablation4b"  # default fallback
    
    # Generate a single run_id for this entire run
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    config.RUN_ID = run_id  # Store in config for access throughout the run
    print(f"📁 [PI] Run ID: {run_id}")
    print(f"📁 [PI] Ablation: {ablation_name}")
    
    # Create run-specific storage directories
    # Structure: storage/image-detections/ablation4b/run_<timestamp>/
    run_image_detections = Path(f"storage/image-detections/{ablation_name}/run_{run_id}")
    run_video_results = Path(f"storage/video_results/{ablation_name}/run_{run_id}")
    
    # Update config paths
    config.DETECTION_OUTPUT_DIR = run_image_detections
    config.RESULTS_VIDEO_STORAGE_DIR = str(run_video_results)
    
    # Create directories
    run_image_detections.mkdir(parents=True, exist_ok=True)
    run_video_results.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 [PI] Storage directories created:")
    print(f"   - Image detections: {run_image_detections}")
    print(f"   - Video results: {run_video_results}")
    
    print("Configuration:")
    print("  - Mode: FSM")
    print("  - Models: 1x nano (single model)")
    print("  - Sensor Fusion: Enabled")
    print("  - Remote Offload: Enabled (S1/S2/S3)")
    print("  ⚠️  Jetson worker required!")
    
    # Debug: Print config file being used
    print(f"📋 [PI] Using config file: {config_file}")
    print(f"📋 [PI] Config settings:")
    print(f"   - CLASSIFICATION_MODE: {getattr(config, 'CLASSIFICATION_MODE', 'unknown')}")
    print(f"   - model_size: {getattr(config, 'model_size', 'unknown')}, model_number: {getattr(config, 'model_number', 'unknown')}")
    print(f"   - DISABLE_SENSOR_FUSION: {getattr(config, 'DISABLE_SENSOR_FUSION', 'unknown')}")
    print(f"   - INFERENCE_ROUTING: {getattr(config, 'INFERENCE_ROUTING', {})}")
    print(f"   - MQTT_BROKER_URL: {getattr(config, 'MQTT_BROKER_URL', 'unknown')}")
    print(f"   - ENABLE_BASELINE_UPDATES: {getattr(config, 'ENABLE_BASELINE_UPDATES', True)}")
    print("=" * 60)
    
    # Initialize strategy at startup (triggers pre-warming for FSM mode)
    initialize_strategy()
    
    # Instantiate your MQTTReceiver with the broker configuration.
    receiver = MQTTReceiver(
        broker_url=config.MQTT_BROKER_URL,
        broker_port=config.MQTT_BROKER_PORT,
        topic=config.MQTT_TOPIC,
    )

    config.IMAGE_MODE = "MQTT_Final"

    latest_messages = LatestPayloadBuffer()
    queue_monitor = QueueMonitor()
    stop_event = threading.Event()

    def custom_on_message(_client, _userdata, msg):
        print(f"Message received on topic: {msg.topic}")
        entry_ts = queue_monitor.enter()
        superseded = latest_messages.offer(msg.topic, msg.payload)
        if superseded is not None:
            print("Superseded older collector payload; keeping newest frame only.")
        
        # Check for queue alerts
        alert = queue_monitor.check_alert_threshold(threshold_depth=5, threshold_wait_s=0.5)
        if alert:
            print(alert)

    receiver.on_message = custom_on_message

    def worker_loop() -> None:
        while not stop_event.is_set():
            item = latest_messages.take()
            if item is None:
                continue
            topic, payload, offered_perf, offered_ts = item
            queue_wait_s = queue_monitor.exit(offered_perf) if offered_perf is not None else 0.0
            
            # Log queue stats periodically
            stats = queue_monitor.get_stats()
            if stats["total_entries"] % 10 == 0:  # Every 10 messages
                print(f"📊 Queue stats: depth={stats['current_depth']:.0f}, "
                      f"avg_wait={stats['avg_wait_s']*1000:.1f}ms, "
                      f"max_wait={stats['max_wait_s']*1000:.1f}ms")
                
                # Memory monitoring - check memory usage periodically
                # Monitor BOTH process and system-wide memory to detect OOM conditions
                try:
                    import psutil
                    import os
                    import gc
                    process = psutil.Process(os.getpid())
                    
                    # Check BOTH process and system memory
                    process_mem_mb = process.memory_info().rss / 1024 / 1024
                    system_mem = psutil.virtual_memory()
                    system_mem_mb = system_mem.total / 1024 / 1024
                    system_mem_available_mb = system_mem.available / 1024 / 1024
                    system_mem_percent = system_mem.percent
                    
                    # Alert if system memory is low (more critical than process memory)
                    # This detects OOM conditions that cause system freezes
                    if system_mem_percent > 85 or system_mem_available_mb < 200:
                        print(f"🚨 CRITICAL SYSTEM MEMORY: {system_mem_percent:.1f}% used, {system_mem_available_mb:.0f}MB available - forcing aggressive cleanup")
                        gc.collect()
                    elif process_mem_mb > 800:  # Lower threshold for Pi (was 1000MB)
                        print(f"⚠️  HIGH PROCESS MEMORY: {process_mem_mb:.0f}MB - forcing garbage collection")
                        gc.collect()
                    elif stats["total_entries"] % 20 == 0:  # More frequent logging (was every 50)
                        print(f"💾 Memory: process={process_mem_mb:.0f}MB, system={system_mem_percent:.1f}% ({system_mem_available_mb:.0f}MB free)")
                except ImportError:
                    # psutil not available, skip memory monitoring
                    pass
                except Exception as e:
                    # Don't fail if memory monitoring fails
                    print(f"Memory monitoring error: {e}")
                    pass
            try:
                config.IMAGE_NAME = (
                    "MQTT_Image"
                    + str(Timestamp.now().date())
                    + str(Timestamp.now().time())
                )
            except Exception:
                config.IMAGE_NAME = "MQTT_Image"
            # Clean up payload reference after processing
            try:
                process_message(
                    payload,
                    image_name=config.IMAGE_NAME,
                    queue_wait_s=queue_wait_s,
                    queue_enter_ts=offered_ts,
                )
            except Exception as exc:
                print(f"Error processing buffered message: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up payload reference
                del payload
                # Garbage collection only when memory is high (not every 5 messages)
                try:
                    import psutil
                    import os
                    import gc
                    process = psutil.Process(os.getpid())
                    process_mem_mb = process.memory_info().rss / 1024 / 1024
                    if process_mem_mb > 600:  # Only when memory is actually high
                        gc.collect()
                except (ImportError, Exception):
                    # If psutil not available or error, skip GC check
                    pass

    worker_thread = threading.Thread(target=worker_loop, name="processor-worker", daemon=True)
    worker_thread.start()

    print("Starting MQTT receiver. Waiting for messages...")
    try:
        receiver.start()
    except KeyboardInterrupt:
        print("Stopping MQTT receiver (KeyboardInterrupt)")
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

