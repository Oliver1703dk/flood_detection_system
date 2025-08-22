"""Inference module using a vision capable LLM.

This module provides :class:`LLMImageClassifier` which sends an image to a
vision enabled Large Language Model (LLM) such as GPT‑4.1-mini and interprets the
model's response to determine flood severity.

The classifier expects the LLM to reply with one of the following labels:
``"flood"``, ``"little-flood"`` or ``"no-flood"``.  The return values map to
integers ``2``, ``1`` and ``0`` respectively.
"""
from __future__ import annotations

import base64
import os
from typing import Optional
from dotenv import load_dotenv  # Import python-dotenv

# Load .env file
load_dotenv()

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI might not be installed
    OpenAI = None  # type: ignore[misc]


class LLMImageClassifier:
    """Classify flood severity in an image using a vision-capable LLM.

    Parameters
    ----------
    model:
        The name of the LLM model to use (defaults to ``"gpt-4.1-mini"``).
    api_key:
        Optional API key.  If not provided, ``OPENAI_API_KEY`` environment
        variable will be used.
    default_label:
        Label returned when the request fails or the response cannot be
        interpreted.  Defaults to ``0`` (no-flood).
    raise_exceptions:
        If ``True`` any request/response errors will be raised instead of
        returning ``default_label``.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        default_label: int = 0,
        raise_exceptions: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.default_label = default_label
        self.raise_exceptions = raise_exceptions
        self._client: Optional["OpenAI"] = None

    # ------------------------------------------------------------------
    def _get_client(self) -> "OpenAI":
        """Return an OpenAI client, instantiating it if required."""
        if OpenAI is None:
            raise ImportError(
                "openai package is required for LLMImageClassifier"
            )
        if self._client is None:
            print("Creating OpenAI client...")
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key)
        return self._client

    # ------------------------------------------------------------------
    def classify_flood(self, image_bytes: bytes) -> int:
        """Classify flood level in an image.

        The method base64 encodes the image, sends it to the configured LLM and
        interprets the response.  When an error occurs, ``default_label`` is
        returned unless ``raise_exceptions`` is set.

        Parameters
        ----------
        image_bytes:
            Raw bytes of the image to classify.

        Returns
        -------
        int
            ``0`` for no flood, ``1`` for little-flood and ``2`` for flood.
        """

        try:
            client = self._get_client()
            print("Encoding image to base64...")
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt_text = (
                "Classify the flood severity in this image. Respond with one of "
                "'flood', 'little-flood', or 'no-flood' only."
            )

            print("Sending request to LLM...")

            print("Model:", self.model)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            print("LLM response received: " + response.choices[0].message.content)

            # Extract the textual content from the first choice
            output = ""
            if response and getattr(response, "choices", None):
                message = response.choices[0].message
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    text_parts = [
                        part.get("text", "") for part in content if part.get("type") == "text"
                    ]
                    output = "".join(text_parts)
                elif isinstance(content, str):
                    output = content
            output = output.strip().lower()
            mapping = {
                "no-flood": 0,
                "no flood": 0,
                "little-flood": 1,
                "little flood": 1,
                "flood": 2,
            }
            return mapping.get(output, self.default_label)

        except Exception:  # pragma: no cover - network errors are hard to unit test
            print("Error during LLM classification:", str(Exception))
            if self.raise_exceptions:
                raise
            return self.default_label


class LLMImageDetector(LLMImageClassifier):
    """Thin wrapper around :class:`LLMImageClassifier` used for confirmation.

    It exposes :meth:`detect_flood` which simply delegates to
    :meth:`LLMImageClassifier.classify_flood`.
    """

    def detect_flood(self, image_bytes: bytes) -> int:
        """Detect flood presence in an image.

        Parameters
        ----------
        image_bytes:
            Raw image bytes.

        Returns
        -------
        int
            ``0`` for no flood, ``1`` for little-flood and ``2`` for flood.
        """

        return super().classify_flood(image_bytes)

