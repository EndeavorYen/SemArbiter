"""Visual evidence for JevPilot. Leaves stay sampled trajectories.

Default encoder is SigLIP (#48). Patch tokens (32–64) are a visual prefix for
the sliced head; compact scores are HUD-only. Tests use synthetic labels.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any, Dict, Optional

VISION_FIELDS = (
    "backend",
    "signal",
    "red",
    "green",
    "pedestrian",
    "vehicle",
    "construction",
    "prefix_tokens",
)

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
        "prefix_tokens": 0,
    }


def render_scenario_frame(scenario: str, env: Any = None) -> Any:
    """Draw a crude forward view so CLIP has pixels. Not a camera from 3D."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224, 224), (120, 170, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 120, 224, 224), fill=(70, 75, 78))
    draw.polygon([(112, 120), (40, 224), (184, 224)], fill=(50, 52, 54))
    dist = 30.0
    if env is not None:
        dist = max(4.0, 55.0 - float(getattr(env, "z", 0.0)))
    scale = max(8, int(80 * (20.0 / dist)))
    cx = 112
    if scenario == "traffic_light_red":
        draw.rectangle((cx - 10, 20, cx + 10, 90), fill=(30, 30, 30))
        draw.ellipse((cx - 14, 24, cx + 14, 52), fill=(220, 30, 30))
    elif scenario == "speed_zone_city":
        draw.rectangle((cx - 10, 20, cx + 10, 90), fill=(30, 30, 30))
        draw.ellipse((cx - 14, 54, cx + 14, 82), fill=(30, 180, 50))
    elif scenario == "pedestrian_jaywalking":
        draw.rectangle((cx - 8, 130, cx + 8, 130 + scale), fill=(20, 20, 20))
        draw.ellipse((cx - 10, 118, cx + 10, 138), fill=(40, 30, 25))
    elif scenario in ("cut_in_vehicle", "roadside_parked_hazard", "emergency_vehicle", "ambiguous_priority"):
        w = scale
        draw.rectangle((cx - w, 150, cx + w, 150 + int(scale * 0.8)), fill=(180, 40, 40) if scenario == "emergency_vehicle" else (90, 90, 95))
    elif scenario == "construction_detour":
        for x in (70, 100, 130):
            draw.polygon([(x, 200), (x + 16, 140), (x + 32, 200)], fill=(230, 120, 20))
    return img


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


def prefix_count() -> int:
    raw = os.environ.get("SEMIF_VISION_PATCHES", "32")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 32
    return max(32, min(64, n))


class VisionEncoder:
    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id or os.environ.get(
            "SEMIF_VISION_MODEL",
            "google/siglip-base-patch16-224",
        )
        self.device = device
        self.backend = "stub"
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._image_proc = None
        self.last_patches = None
        self._null_patches = None
        self._load()

    def _load(self) -> None:
        try:
            import torch

            self._torch = torch
            if "clip-vit" in self.model_id and "siglip" not in self.model_id.lower():
                self._load_clip()
            else:
                try:
                    self._load_auto()
                except Exception:
                    if "siglip" in self.model_id.lower():
                        raise
                    self._load_clip()
        except Exception:
            self.backend = "stub"
            self._model = None
            self._processor = None
            self._tokenizer = None
            self._image_proc = None
            self.last_patches = None

    def _load_auto(self) -> None:
        from transformers import AutoModel, SiglipImageProcessor

        # AutoProcessor needs SentencePiece; AutoImageProcessor needs torchvision.
        # Image processor + AutoModel is enough for patch tokens (#48).
        self._image_proc = SiglipImageProcessor.from_pretrained(self.model_id)
        self._processor = None
        try:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception:
            self._processor = None
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self.backend = self.model_id

    def _load_clip(self) -> None:
        from tokenizers import processors as tok_processors
        from transformers.models.clip import tokenization_clip as clip_tok

        orig = tok_processors.RobertaProcessing

        def _roberta(sep, cls=None, cls_token=None, trim_offsets=True, add_prefix_space=True, **_kw):
            token = cls_token if cls_token is not None else cls
            return orig(sep, token, trim_offsets=trim_offsets, add_prefix_space=add_prefix_space)

        tok_processors.RobertaProcessing = _roberta
        clip_tok.processors.RobertaProcessing = _roberta

        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self._image_proc = CLIPImageProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        self.backend = self.model_id

    def encode_patches(self, image: Any):
        """Return [prefix_count, dim] patch tokens. None if the encoder is a stub."""
        if self._model is None:
            return None
        from semif_phase1.visual_prefix import pool_patches

        torch = self._torch
        pixel = self._pixel_values(image)
        vision = getattr(self._model, "vision_model", None)
        if vision is None:
            return None
        with torch.no_grad():
            out = vision(pixel_values=pixel)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]
        if hidden.shape[1] > 1:
            # Drop class token when the sequence is patches+1.
            maybe_cls = hidden[:, 1:, :]
            if maybe_cls.shape[1] in {49, 196, 256, 729} or maybe_cls.shape[1] % 7 == 0:
                hidden = maybe_cls
        pooled = pool_patches(hidden, prefix_count())
        return pooled[0].detach()

    def _pixel_values(self, image: Any):
        torch = self._torch
        if self._processor is not None:
            packed = self._processor(images=image, return_tensors="pt")
            pixel = packed["pixel_values"]
        else:
            packed = self._image_proc(images=image, return_tensors="pt")
            pixel = packed["pixel_values"]
        return pixel.to(self.device)

    def null_patches(self):
        if self._null_patches is not None:
            return self._null_patches
        from PIL import Image

        blank = Image.new("RGB", (224, 224), (127, 127, 127))
        self._null_patches = self.encode_patches(blank)
        return self._null_patches

    def infer_pil(self, image: Any) -> Dict[str, Any]:
        patches = self.encode_patches(image)
        self.last_patches = patches
        n_prefix = int(patches.shape[0]) if patches is not None else 0
        if self._model is None:
            ev = synthetic_vision()
            ev["prefix_tokens"] = 0
            return ev
        torch = self._torch
        texts = [text for _key, text in _PROMPTS]
        try:
            if self._processor is not None:
                inputs = self._processor(text=texts, images=image, padding=True, return_tensors="pt")
            else:
                text_inputs = self._tokenizer(texts, padding=True, return_tensors="pt")
                image_inputs = self._image_proc(images=image, return_tensors="pt")
                inputs = {**text_inputs, **image_inputs}
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
                logits = out.logits_per_image
                probs = logits.softmax(dim=-1)[0].tolist()
            scores = {key: float(prob) for (key, _prompt), prob in zip(_PROMPTS, probs)}
        except Exception:
            scores = {key: 0.0 for key, _prompt in _PROMPTS}
            scores["clear"] = 1.0
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
            "prefix_tokens": n_prefix,
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
