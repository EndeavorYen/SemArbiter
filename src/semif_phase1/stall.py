"""Ego-centric stall detector for #107. No NPC world coordinates."""

from __future__ import annotations

from typing import Any, Optional

STALL_SPEED_MPS = 0.2
STALL_PROGRESS_M = 0.5
STALL_HOLD_S = 3.0
STALL_COOLDOWN_S = 8.0


def stall_update(
    *,
    autopilot: bool,
    speed_mps: float,
    route_s: Optional[float],
    dt: float,
    held_s: float,
    s0: Optional[float],
    cooldown_s: float = 0.0,
) -> dict[str, Any]:
    """Advance stall timers. Trigger when slow and path progress < 0.5 m for 3 s."""
    if cooldown_s > 0:
        cooldown_s = max(0.0, cooldown_s - dt)
    if not autopilot:
        return {"trigger": False, "held_s": 0.0, "s0": route_s, "cooldown_s": cooldown_s}
    if abs(float(speed_mps)) >= STALL_SPEED_MPS:
        return {"trigger": False, "held_s": 0.0, "s0": route_s, "cooldown_s": cooldown_s}
    if s0 is None:
        s0 = route_s
    held_s = float(held_s) + float(dt)
    progress = 0.0
    if route_s is not None and s0 is not None:
        progress = abs(float(route_s) - float(s0))
    trigger = held_s >= STALL_HOLD_S and progress < STALL_PROGRESS_M and cooldown_s <= 0
    if trigger:
        cooldown_s = STALL_COOLDOWN_S
        held_s = 0.0
        s0 = route_s
    return {"trigger": trigger, "held_s": held_s, "s0": s0, "cooldown_s": cooldown_s}
