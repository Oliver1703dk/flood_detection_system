import paho.mqtt.client as mqtt
import json
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager

class MQTTReceiver:
    def __init__(self, broker_url, broker_port, topic, validator=None, storage_manager=None):
        self.broker_url = broker_url
        self.broker_port = broker_port
        self.topic = topic
        self.validator = validator or DataValidator()
        self.storage_manager = storage_manager or StorageManager()
        
        # CRITICAL: Use fixed client_id and clean_session=True
        # This ensures old session queues are cleared on reconnect
        self.client = mqtt.Client(
            client_id="flood-detection-processor",
            clean_session=True  # Clear any old session queues
        )
        
        # Limit paho-mqtt's internal message buffer (prevents memory buildup)
        # Only keep a small buffer - we only want the newest message anyway
        self.client.max_queued_messages_set(1)  # Only buffer 1 message max
        
        # Limit inflight messages (only matters for QoS > 0, but good practice)
        self.client.max_inflight_messages_set(1)

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        # CRITICAL: Use QoS 0 - broker won't queue messages
        # If client is busy, broker will drop older undelivered messages
        # This is exactly what you want - only newest message matters
        client.subscribe(self.topic, qos=0)
        print(f"Subscribed to {self.topic} with QoS 0 (no broker queuing - only newest message)")

    def on_message(self, client, userdata, msg):
        try:
            print(f"Message received on topic {msg.topic}")
            data = json.loads(msg.payload.decode("utf-8"))
            if self.validator.validate(data):
                self.storage_manager.store(data)
                print("Data validated and stored successfully.")
            else:
                print("Validation failed for received data.")
        except Exception as e:
            print(f"Error processing message: {e}")

    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker_url, self.broker_port, 60)
        print("Starting MQTT receiver...")
        print("⚠️  Configured for 'newest message only' - older messages will be dropped")
        self.client.loop_forever()
