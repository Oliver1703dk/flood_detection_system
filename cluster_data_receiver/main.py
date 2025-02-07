from storage.storage_manager import StorageManager
from validation.data_validator import DataValidator
from receiver.mqtt_receiver import MQTTReceiver

if __name__ == "__main__":
    # MQTT Broker configuration
    broker_url = "localhost"  # Replace with your broker's address
    broker_port = 1883        # Default MQTT port
    topic = "sensor/data"  # Topic to subscribe to

    # Initialize components
    validator = DataValidator()
    storage_manager = StorageManager(storage_dir="storage")

    # Create and start the MQTT receiver
    mqtt_receiver = MQTTReceiver(
        broker_url=broker_url,
        broker_port=broker_port,
        topic=topic,
        validator=validator,
        storage_manager=storage_manager
    )

    try:
        print("Starting MQTT receiver...")
        mqtt_receiver.start()
    except KeyboardInterrupt:
        print("\nMQTT receiver stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")