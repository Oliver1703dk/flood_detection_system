"""Local vision-language model inference using VILA or similar models.

This module provides a drop-in replacement for the OpenAI-based LLMImageClassifier
that runs entirely on the Jetson using models like VILA1.5-3B.
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from PIL import Image


class LocalVLMClassifier:
    """Classify flood severity using a local vision-language model.
    
    This class is designed to be a drop-in replacement for LLMImageClassifier
    but runs entirely on local GPU hardware.
    
    Parameters
    ----------
    model_name:
        The model to use. Options:
        - "Efficient-Large-Model/VILA1.5-3b" (recommended for Jetson)
        - "Efficient-Large-Model/VILA-7b" (if you have more memory)
        - Any other HuggingFace vision-language model
    device:
        Device to run on ('cuda', 'cpu', or None for auto-detect)
    default_label:
        Label returned when inference fails (0 = no-flood)
    raise_exceptions:
        If True, raise exceptions instead of returning default_label
    use_4bit:
        Use 4-bit quantization to reduce memory usage (recommended for Jetson)
    """

    def __init__(
        self,
        model_name: str = "Efficient-Large-Model/VILA1.5-3b",
        device: Optional[str] = None,
        default_label: int = 0,
        raise_exceptions: bool = False,
        use_4bit: bool = True,
    ) -> None:
        requested_path = Path(model_name).expanduser()
        module_dir = Path(__file__).resolve().parent
        local_candidate = module_dir / "models" / Path(model_name)

        if requested_path.exists():
            resolved_model = requested_path
        elif local_candidate.exists():
            resolved_model = local_candidate
        else:
            resolved_model = None

        self.model_name = str(resolved_model) if resolved_model else model_name
        self.default_label = default_label
        self.raise_exceptions = raise_exceptions
        self.use_4bit = use_4bit
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self._model = None
        self._processor = None
        self._initialized = False
        
        print(f"LocalVLMClassifier initialized with model: {self.model_name}")
        print(f"Device: {self.device}, 4-bit quantization: {use_4bit}")

    def _initialize_model(self) -> None:
        """Lazy initialization of the model to avoid loading during import."""
        if self._initialized:
            return
            
        print(f"Loading vision-language model: {self.model_name}")
        
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            import torch
            
            # Load processor
            print("Loading processor...")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with optional quantization
            print("Loading model...")
            if self.use_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self._model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                self._model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                self._model.to(self.device)
            
            self._model.eval()
            self._initialized = True
            print(f"✓ Model loaded successfully on {self.device}")
            
            # Print memory usage
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
        except Exception as exc:
            print(f"Error loading model: {exc}")
            if self.raise_exceptions:
                raise
            self._initialized = False

    def classify_flood(
        self,
        image_bytes: bytes,
        sensor_data: Optional[Dict[str, Any]] = None,
        sensor_baseline: Optional[Dict[str, Any]] = None,
        sensor_anomalies: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Classify flood level in an image using local VLM.
        
        Parameters
        ----------
        image_bytes:
            Raw bytes of the image to classify.
        sensor_data:
            Optional sensor readings to include in the prompt
        sensor_baseline:
            Optional baseline sensor values
        sensor_anomalies:
            Optional sensor anomaly information
            
        Returns
        -------
        int
            0 for no flood, 1 for little-flood, 2 for flood.
        """
        try:
            # Initialize model if needed
            if not self._initialized:
                self._initialize_model()
                
            if not self._initialized:
                print("Model not initialized, returning default label")
                return self.default_label
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Build prompt with sensor context
            prompt = self._build_prompt(sensor_data, sensor_baseline, sensor_anomalies)
            
            print(f"Running local VLM inference...")
            
            # Prepare inputs
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                )
            
            # Decode response
            generated_text = self._processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract the answer (remove the prompt)
            if prompt in generated_text:
                answer = generated_text.replace(prompt, "").strip()
            else:
                answer = generated_text.strip()
            
            print(f"VLM response: {answer}")
            
            # Parse the response
            result = self._parse_response(answer)
            return result
            
        except Exception as exc:
            print(f"Error during local VLM classification: {exc}")
            if self.raise_exceptions:
                raise
            return self.default_label

    def _build_prompt(
        self,
        sensor_data: Optional[Dict[str, Any]],
        sensor_baseline: Optional[Dict[str, Any]],
        sensor_anomalies: Optional[Dict[str, Any]],
    ) -> str:
        """Build the prompt for the vision-language model."""
        
        base_prompt = (
            "USER: <image>\n"
            "Analyze this image for flood conditions. "
        )
        
        # Add sensor context if available
        sensor_context = self._build_sensor_context(
            sensor_data, sensor_baseline, sensor_anomalies
        )
        if sensor_context:
            base_prompt += f"\n\n{sensor_context}\n\n"
        
        base_prompt += (
            "Classify the flood severity. Respond with ONLY ONE of these exact words:\n"
            "- 'no-flood' if there is no flooding\n"
            "- 'little-flood' if there is minor water accumulation\n"
            "- 'flood' if there is significant flooding\n\n"
            "ASSISTANT: "
        )
        
        return base_prompt

    @staticmethod
    def _build_sensor_context(
        sensor_data: Optional[Dict[str, Any]],
        sensor_baseline: Optional[Dict[str, Any]],
        sensor_anomalies: Optional[Dict[str, Any]],
    ) -> str:
        """Build sensor context string (same as original implementation)."""
        if not sensor_data:
            return ""

        lines: List[str] = []
        for key, value in sensor_data.items():
            if value is None:
                continue
            baseline_key = f"{key}_baseline"
            baseline_val = None
            if sensor_baseline:
                baseline_val = sensor_baseline.get(baseline_key, sensor_baseline.get(key))

            anomaly_key = f"delta_{key}"
            delta_val = None
            if sensor_anomalies:
                delta_val = sensor_anomalies.get(anomaly_key, sensor_anomalies.get(key))
            if delta_val is None and baseline_val is not None:
                try:
                    delta_val = value - baseline_val
                except TypeError:
                    delta_val = None

            summary = f"{key.title()}: current {value}"
            if baseline_val is not None:
                summary += f", baseline {baseline_val}"
            if isinstance(delta_val, (int, float)):
                summary += f", delta {delta_val:+.2f}"
            lines.append(summary)

        if not lines:
            return ""

        context = (
            "Sensor readings:\n"
            + "\n".join(lines)
        )
        return context

    def _parse_response(self, response: str) -> int:
        """Parse the model's response into a flood classification."""
        response_lower = response.lower().strip()
        
        # Define mapping with various possible responses
        mapping = {
            "no-flood": 0,
            "no flood": 0,
            "noflood": 0,
            "none": 0,
            "no": 0,
            "little-flood": 1,
            "little flood": 1,
            "littleflood": 1,
            "minor": 1,
            "some": 1,
            "flood": 2,
            "flooded": 2,
            "severe": 2,
            "major": 2,
        }
        
        # Try exact match first
        if response_lower in mapping:
            return mapping[response_lower]
        
        # Try substring matching
        for key, value in mapping.items():
            if key in response_lower:
                return value
        
        print(f"Warning: Could not parse response '{response}', returning default")
        return self.default_label


class LocalLLMImageDetector(LocalVLMClassifier):
    """Thin wrapper for compatibility with existing code."""
    
    def detect_flood(self, image_bytes: bytes) -> int:
        """Detect flood presence in an image.
        
        Parameters
        ----------
        image_bytes:
            Raw image bytes.
            
        Returns
        -------
        int
            0 for no flood, 1 for little-flood, 2 for flood.
        """
        return self.classify_flood(image_bytes)


# Backward compatibility alias
LLMImageClassifier = LocalVLMClassifier
LLMImageDetector = LocalLLMImageDetector