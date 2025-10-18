"""MQTT utilities for synchronous request/response style inference offloading.

This module provides :class:`MQTTInferenceClient` which allows the Pi to publish
an inference job and wait for the Jetson's reply using correlation IDs.  It
assumes the Jetson publishes heartbeat messages so we can quickly detect
connectivity problems and fall back to local inference.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import config

try:  # pragma: no cover - optional dependency for remote inference
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - allow unit tests without paho-mqtt
    mqtt = None  # type: ignore


@dataclass
class MQTTResponse:
    """Container storing a Jetson response payload and metadata."""

    data: Dict[str, object]
    latency_s: float
    sent_at: float
    received_at: float


class MQTTInferenceClient:
    """Publish inference requests to the Jetson and wait for replies."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        request_topic: str = config.MQTT_INFERENCE_REQUEST_TOPIC,
        response_topic: str = config.MQTT_INFERENCE_RESPONSE_TOPIC,
        heartbeat_topic: str = config.MQTT_INFERENCE_HEARTBEAT_TOPIC,
        qos: int = config.MQTT_INFERENCE_QOS,
        timeout_s: float = config.MQTT_INFERENCE_TIMEOUT,
    ) -> None:
        if mqtt is None:
            raise RuntimeError(
                "paho-mqtt is not installed; cannot use MQTTInferenceClient"
            )

        self.request_topic = request_topic.rstrip("/")
        self.response_topic = response_topic.rstrip("/")
        self.heartbeat_topic = heartbeat_topic
        self.qos = qos
        self.timeout_s = timeout_s

        cid = client_id or f"pi-inference-{uuid.uuid4().hex[:8]}"
        self._client = mqtt.Client(client_id=cid)
        self._client.enable_logger()

        self._pending: Dict[str, threading.Event] = {}
        self._payloads: Dict[str, MQTTResponse] = {}
        self._sent_at: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._last_heartbeat = 0.0

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        self._client.connect(config.MQTT_BROKER_URL, config.MQTT_BROKER_PORT)
        self._client.loop_start()

        if not self._connected.wait(timeout=5.0):
            raise RuntimeError("MQTTInferenceClient failed to connect to broker")

        self._client.subscribe(f"{self.response_topic}/#", qos=self.qos)
        self._client.subscribe(self.heartbeat_topic, qos=0)

    # ------------------------------------------------------------------
    def _on_connect(self, _client, _userdata, _flags, rc):
        if rc == 0:
            self._connected.set()

    # ------------------------------------------------------------------
    def _on_disconnect(self, _client, _userdata, _rc):
        self._connected.clear()

    # ------------------------------------------------------------------
    def _on_message(self, _client, _userdata, msg):
        topic = msg.topic
        if topic == self.heartbeat_topic:
            self._last_heartbeat = time.time()
            return

        if not topic.startswith(self.response_topic):
            return

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        correlation_id = payload.get("id")
        if not correlation_id:
            return

        with self._lock:
            event = self._pending.get(correlation_id)
            if event is None:
                return
            received_at = time.time()
            sent_at = self._sent_at.get(correlation_id, received_at)
            latency = received_at - sent_at
            self._payloads[correlation_id] = MQTTResponse(payload, latency, sent_at, received_at)
            event.set()

    # ------------------------------------------------------------------
    def publish_request(self, message: Dict[str, object]) -> MQTTResponse:
        """Publish a request and block until the response arrives or times out."""

        correlation_id = message.setdefault("id", uuid.uuid4().hex)
        message["ts"] = time.time()

        event = threading.Event()
        with self._lock:
            self._pending[correlation_id] = event

        try:
            payload = json.dumps(message).encode("utf-8")
            if len(payload) > config.MQTT_INFERENCE_MAX_PAYLOAD:
                raise ValueError("Inference payload exceeds MQTT_INFERENCE_MAX_PAYLOAD")

            result = self._client.publish(self.request_topic, payload, qos=self.qos)
            result.wait_for_publish(timeout=self.timeout_s)

            with self._lock:
                self._sent_at[correlation_id] = time.time()

            if not event.wait(timeout=self.timeout_s):
                raise TimeoutError("Timed out waiting for Jetson inference response")

            with self._lock:
                resp = self._payloads.pop(correlation_id)
                self._pending.pop(correlation_id, None)
                self._sent_at.pop(correlation_id, None)
                return resp

        finally:
            with self._lock:
                self._pending.pop(correlation_id, None)
                self._payloads.pop(correlation_id, None)
                self._sent_at.pop(correlation_id, None)

    # ------------------------------------------------------------------
    def is_healthy(self, heartbeat_timeout: float = 10.0) -> bool:
        if not self._connected.is_set():
            return False
        if self._last_heartbeat == 0.0:
            return False
        return (time.time() - self._last_heartbeat) <= heartbeat_timeout

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


def encode_image_for_transport(image_bytes: bytes) -> str:
    """Return a compact base64 representation suitable for MQTT."""

    return base64.b64encode(image_bytes).decode("utf-8")
