"""Zero-shot visual evidence for JevPilot. Leaves stay sampled trajectories.

CLIP is the default encoder (small, cached). SigLIP/MobileCLIP load if the
env var SEMIF_VISION_MODEL points at them. Tests use synthetic labels, not weights.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, Optional

VISION_FIELDS = ("backend", "signal", "red", "green", "pedestrian", "vehicle", "construction")

_PROMPTS = (
    ("red", "a red traffic light facing the camera"),
    ("green", "a green traffic light facing the camera"),
    ("pedestrian", "a pedestrian on the road in front of the car"),
    ("vehicle", "the rear of a car on the road ahead"),
    ("construction", "orange traffic cones and a construction barrier on the road"),
    ("clear", "an empty asphalt road with no people or cars"),
)


def compact_vision(vision: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(vision, dict):
        return None
    packed: Dict[str, Any] = {}
    for key in VISION_FIELDS:
        if key in vision and vision[key] is not None:
            packed[key] = vision[key]
    return packed or None


def synthetic_vision(**labels: Any) -> Dict[str, Any]:
    """Fixture evidence. Not a substitute for GPU CLIP scores."""
    red = float(labels.get("red", 0.0))
    green = float(labels.get("green", 0.0))
    signal = "unknown"
    if red >= 0.35 and red >= green:
        signal = "red"
    elif green >= 0.35:
        signal = "green"
    return {
        "backend": "synthetic",
        "signal": signal,
        "red": round(red, 3),
        "green": round(green, 3),
        "pedestrian": round(float(labels.get("pedestrian", 0.0)), 3),
        "vehicle": round(float(labels.get("vehicle", 0.0)), 3),
        "construction": round(float(labels.get("construction", 0.0)), 3),
    }


def vision_from_scenario(scenario: str) -> Dict[str, Any]:
    """Map a closed-loop scenario name to synthetic visual evidence."""
    table: Dict[str, Dict[str, float]] = {
        "traffic_light_red": {"red": 0.82, "green": 0.04},
        "speed_zone_city": {"green": 0.4},
        "pedestrian_jaywalking": {"pedestrian": 0.88},
        "roadside_parked_hazard": {"vehicle": 0.7},
        "cut_in_vehicle": {"vehicle": 0.8},
        "construction_detour": {"construction": 0.85},
        "emergency_vehicle": {"vehicle": 0.55},
        "ambiguous_priority": {"vehicle": 0.45},
        "sharp_curve": {},
        "sensor_anomaly": {},
    }
    return synthetic_vision(**table.get(scenario, {}))


def decode_image_bytes(image_b64: str) -> Any:
    raw = image_b64.split(",", 1)[-1]
    blob = base64.b64decode(raw)
    from PIL import Image

    return Image.open(io.BytesIO(blob)).convert("RGB")


class VisionEncoder:
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id or os.environ.get("SEMIF_VISION_MODEL", "openai/clip-vit-base-patch32")
        self.device = device
        self.backend = "stub"
        self._model = None
        self._processor = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()
            self.backend = self.model_id
            self._torch = torch
        except Exception:
            self.backend = "stub"
            self._model = None

    def infer_pil(self, image: Any) -> Dict[str, Any]:
        if self._model is None or self._processor is None:
            return synthetic_vision()
        torch = self._torch
        texts = [text for _key, text in _PROMPTS]
        inputs = self._processor(text=texts, images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            probs = out.logits_per_image.softmax(dim=-1)[0].tolist()
        scores = {key: float(prob) for (key, _prompt), prob in zip(_PROMPTS, probs)}
        signal = "unknown"
        if scores["red"] >= 0.28 and scores["red"] >= scores["green"]:
            signal = "red"
        elif scores["green"] >= 0.28:
            signal = "green"
        return {
            "backend": self.backend,
            "signal": signal,
            "red": round(scores["red"], 3),
            "green": round(scores["green"], 3),
            "pedestrian": round(scores["pedestrian"], 3),
            "vehicle": round(scores["vehicle"], 3),
            "construction": round(scores["construction"], 3),
        }

    def infer_b64(self, image_b64: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        image = decode_image_bytes(image_b64)
        evidence = self.infer_pil(image)
        evidence["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return evidence


_encoder: Optional[VisionEncoder] = None


def get_vision_encoder() -> VisionEncoder:
    global _encoder
    if _encoder is None:
        device = os.environ.get("SEMIF_VISION_DEVICE", "cpu")
        _encoder = VisionEncoder(device=device)
    return _encoder
