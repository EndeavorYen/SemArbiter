"""Keep the default Jev letter-slot prompt inside the 512 CUDA Graph bucket (#86)."""

from semif_phase1.core import direct_messages
from jevpilot_vision.trajectory_sampler import VECTOR_INSTRUCTIONS, compact_jev_state, vector_option_tag

GRAPH_BUCKET = 512
# Qwen English JSON is typically ~3.5–4 characters per token. Stay under 4× bucket.
MAX_CHARS = GRAPH_BUCKET * 4


def _fat_row():
    options = []
    for i in range(16):
        options.append(
            {
                "id": f"t{i:02d}",
                "description": vector_option_tag(
                    [16.2, -0.14 if i % 2 else 0.22, 0.3, 0.0, i == 3, i == 0],
                    style="words",
                    ego_x=0.4,
                    signal="red",
                ),
            }
        )
    state = compact_jev_state(
        {
            "speed_mps": 16.0,
            "speed_ceiling_mps": 18.0,
            "on_road": True,
            "intersection": {"control": "signal", "distance_to_line_m": 22.0, "signal": "red"},
            "vision": {
                "signal": "red",
                "event": "RED signal ahead, mandatory stop; caution: vehicle cutting toward frame center, growing in the camera",
            },
            "directive": "RED_LIGHT_STOP",
            "lateral_offset_m": 0.4,
        }
    )
    return {
        "id": "budget",
        "state": state,
        "question": f"{VECTOR_INSTRUCTIONS} Intent: RED_LIGHT_STOP.",
        "options": options,
    }


def test_default_jev_prompt_fits_512_char_budget():
    msgs = direct_messages(_fat_row())
    blob = msgs[0]["content"] + msgs[1]["content"]
    assert len(blob) <= MAX_CHARS, f"{len(blob)} chars would miss the 512 bucket"


def test_prompt_tokens_fit_512_when_tokenizer_available():
    pytest = __import__("pytest")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers missing")
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not cached")
    msgs = direct_messages(_fat_row())
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    n = len(tok.encode(prompt, add_special_tokens=False))
    assert n <= GRAPH_BUCKET, f"{n} tokens miss the 512 CUDA Graph bucket"


def test_compact_truncates_long_vision_event():
    packed = compact_jev_state(
        {"vision": {"event": "x" * 200, "signal": "red", "backend": "clip"}}
    )
    assert len(packed["vision"]["event"]) <= 96
    assert "backend" not in packed["vision"]
