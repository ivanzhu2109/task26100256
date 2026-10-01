"""Track model and evaluation."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from .segments import Segment


@dataclass(frozen=True)
class Track:
    """A scalar animation track covering [0, T)."""

    segments: tuple[Segment, ...]
    T: float


def evaluate(track: Track, t: float) -> float:
    """Return the track value at time t (0 <= t < T)."""
    if not 0.0 <= t < track.T:
        raise ValueError(f"time {t} outside track domain [0, {track.T})")
    starts = [seg.t0 for seg in track.segments]
    idx = bisect_right(starts, t) - 1
    return track.segments[idx].value_at(t)
