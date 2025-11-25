import paho.mqtt.client as mqtt
import json
import time
import threading
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager

class MQTTReceiver:
    def __init__(self, broker_url, broker_port, topic, validator=None, storage_manager=None):
        self.broker_url = broker_url
        self.broker_port = broker_port
        self.topic = topic
        self.validator = validator or DataValidator()
        self.storage_manager = storage_manager or StorageManager()
        
        # Connection state tracking
        self._connected = threading.Event()
        self._last_connect_time = None
        self._reconnect_count = 0
        self._last_message_time = None
        self._connection_monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # CRITICAL: Use fixed client_id and clean_session=True
        # This ensures old session queues are cleared on reconnect
        self.client = mqtt.Client(
            client_id="flood-detection-processor",
            clean_session=True  # Clear any old session queues
        )
        
        # Enable automatic reconnection
        # paho-mqtt will automatically reconnect with exponential backoff
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)
        
        # Limit paho-mqtt's internal message buffer (prevents memory buildup)
        # Only keep a small buffer - we only want the newest message anyway
        self.client.max_queued_messages_set(1)  # Only buffer 1 message max
        
        # Limit inflight messages (only matters for QoS > 0, but good practice)
        self.client.max_inflight_messages_set(1)

    def on_connect(self, client, userdata, flags, rc):
        """Called when the broker responds to our connection request."""
        if rc == 0:
            self._connected.set()
            self._last_connect_time = time.time()
            if self._reconnect_count > 0:
                print(f"✅ [MQTT] Reconnected to broker (attempt {self._reconnect_count})")
                self._reconnect_count = 0
            else:
                print(f"✅ [MQTT] Connected to broker with result code {rc}")
            
            # CRITICAL: Use QoS 0 - broker won't queue messages
            # If client is busy, broker will drop older undelivered messages
            # This is exactly what you want - only newest message matters
            client.subscribe(self.topic, qos=0)
            print(f"📡 [MQTT] Subscribed to {self.topic} with QoS 0 (no broker queuing - only newest message)")
        else:
            print(f"❌ [MQTT] Connection failed with result code {rc}")
            self._connected.clear()

    def on_disconnect(self, client, userdata, rc):
        """Called when the client disconnects from the broker."""
        self._connected.clear()
        
        if rc == 0:
            print("ℹ️  [MQTT] Disconnected normally")
        else:
            self._reconnect_count += 1
            print(f"⚠️  [MQTT] Unexpected disconnection (rc={rc}). Reconnect attempt {self._reconnect_count}...")
            print(f"⚠️  [MQTT] paho-mqtt will automatically attempt to reconnect")

    def on_message(self, client, userdata, msg):
        """Called when a message has been received on a topic that the client subscribes to."""
        try:
            self._last_message_time = time.time()
            print(f"📨 [MQTT] Message received on topic {msg.topic}")
            data = json.loads(msg.payload.decode("utf-8"))
            if self.validator.validate(data):
                self.storage_manager.store(data)
                print("✅ [MQTT] Data validated and stored successfully.")
            else:
                print("❌ [MQTT] Validation failed for received data.")
        except Exception as e:
            print(f"❌ [MQTT] Error processing message: {e}")

    def _connection_monitor(self):
        """Monitor connection health and log warnings if messages stop arriving."""
        MESSAGE_TIMEOUT = 10.0  # seconds - warn if no messages for this long
        CHECK_INTERVAL = 5.0  # seconds - check every 5 seconds
        
        while not self._stop_monitoring.is_set():
            time.sleep(CHECK_INTERVAL)
            
            if not self._connected.is_set():
                continue  # Already disconnected, on_disconnect will handle it
            
            # Check if we're connected but not receiving messages
            if self._last_message_time is not None:
                time_since_last_message = time.time() - self._last_message_time
                if time_since_last_message > MESSAGE_TIMEOUT:
                    print(f"⚠️  [MQTT] WARNING: No messages received for {time_since_last_message:.1f}s")
                    print(f"⚠️  [MQTT] Connection status: {'connected' if self.client.is_connected() else 'disconnected'}")
                    # Reset to avoid spam
                    self._last_message_time = time.time()

    def start(self):
        """Start the MQTT receiver with automatic reconnection."""
        # Set up callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        # Connect to broker
        print(f"🔌 [MQTT] Connecting to broker at {self.broker_url}:{self.broker_port}...")
        try:
            self.client.connect(self.broker_url, self.broker_port, keepalive=60)
        except Exception as e:
            print(f"❌ [MQTT] Failed to connect: {e}")
            raise
        
        print("🚀 [MQTT] Starting MQTT receiver...")
        print("⚠️  [MQTT] Configured for 'newest message only' - older messages will be dropped")
        print("🔄 [MQTT] Automatic reconnection enabled (will retry with exponential backoff)")
        
        # Start connection monitoring thread
        self._stop_monitoring.clear()
        self._connection_monitor_thread = threading.Thread(
            target=self._connection_monitor,
            name="mqtt-connection-monitor",
            daemon=True
        )
        self._connection_monitor_thread.start()
        
        # Start the MQTT loop (blocks forever, with automatic reconnection)
        try:
            self.client.loop_forever(retry_first_connection=True)
        except KeyboardInterrupt:
            print("\n🛑 [MQTT] Stopping MQTT receiver (KeyboardInterrupt)")
        finally:
            self._stop_monitoring.set()
            self.client.loop_stop()
            self.client.disconnect()
            print("✅ [MQTT] MQTT receiver stopped")
