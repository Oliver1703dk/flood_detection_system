"""
Main entry point for Ablation 1b: Multi-Model Medium-YOLO Baseline (3x medium)
- Three medium models with consensus, no FSM, no sensor fusion, no offload
- Tests multi-model consensus effect vs Ablation 1 (1x medium)
"""

# Override config module BEFORE any other imports
import config_ablation1b as config
import sys
sys.modules['config'] = config

# Now import the rest (they will use config_ablation1b)
from main_final import *

def main():
    """
    Main method for Ablation 1b.
    Initializes the MQTT receiver and directs each incoming message payload
    to the process_message callback for processing.
    """
    print("=" * 60)
    print("ABLATION 1b: Multi-Model Medium-YOLO Baseline (3x medium)")
    print("=" * 60)
    print("Configuration:")
    print("  - Mode: yolo_sensor (no FSM)")
    print("  - Models: 3x medium with consensus")
    print("  - Sensor Fusion: Disabled")
    print("  - Remote Offload: Disabled")
    print("=" * 60)
    
    # Debug: Print config file being used
    config_file = getattr(config, '__file__', 'unknown')
    print(f"📋 [PI] Using config file: {config_file}")
    print(f"📋 [PI] Config settings:")
    print(f"   - CLASSIFICATION_MODE: {getattr(config, 'CLASSIFICATION_MODE', 'unknown')}")
    print(f"   - model_size: {getattr(config, 'model_size', 'unknown')}, model_number: {getattr(config, 'model_number', 'unknown')}")
    print(f"   - DISABLE_SENSOR_FUSION: {getattr(config, 'DISABLE_SENSOR_FUSION', 'unknown')}")
    print(f"   - MQTT_BROKER_URL: {getattr(config, 'MQTT_BROKER_URL', 'unknown')}")
    print(f"   - ENABLE_BASELINE_UPDATES: {getattr(config, 'ENABLE_BASELINE_UPDATES', True)}")
    print("=" * 60)
    
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
            try:
                config.IMAGE_NAME = (
                    "MQTT_Image"
                    + str(Timestamp.now().date())
                    + str(Timestamp.now().time())
                )
            except Exception:
                config.IMAGE_NAME = "MQTT_Image"
            try:
                process_message(
                    payload,
                    image_name=config.IMAGE_NAME,
                    queue_wait_s=queue_wait_s,
                    queue_enter_ts=offered_ts,
                )
            except Exception as exc:
                print(f"Error processing buffered message: {exc}")

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

