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
        self.client = mqtt.Client()

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT broker with result code {rc}")
        client.subscribe(self.topic)

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
        self.client.loop_forever()
