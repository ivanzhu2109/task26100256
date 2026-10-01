"""Clipboard operations: copy_range and paste."""

from __future__ import annotations

from dataclasses import dataclass

from .segments import Bezier, Hold, Linear, Segment, split_bezier
from .track import Track


@dataclass(frozen=True)
class Clip:
    """An independent clipboard fragment covering local time [0, length)."""

    segments: tuple[Segment, ...]
    length: float


def copy_range(track: Track, a: float, b: float) -> Clip:
    """Return a clip of length b-a; clip at s equals track at a+s."""
    if not 0.0 <= a < b <= track.T:
        raise ValueError(f"range [{a}, {b}) outside track domain [0, {track.T})")
    segs: list[Segment] = []
    for seg in track.segments:
        lo = max(a, seg.t0)
        hi = min(b, seg.t1)
        if lo < hi:
            segs.append(_restrict(seg, lo, hi).shifted(-a))
    return Clip(tuple(segs), b - a)


def paste(track: Track, d: float, e: float, clip: Clip, reverse: bool = False) -> Track:
    """Return a new track with the clip scaled into [d, e).

    Values outside [d, e) are unchanged. With reverse=True the clip is
    played backwards using left limits at jump points. Inputs are never
    modified.
    """
    if not 0.0 <= d < e <= track.T:
        raise ValueError(f"range [{d}, {e}) outside track domain [0, {track.T})")
    left = [
        _restrict(seg, seg.t0, min(seg.t1, d))
        for seg in track.segments
        if seg.t0 < d
    ]
    right = [
        _restrict(seg, max(seg.t0, e), seg.t1)
        for seg in track.segments
        if seg.t1 > e
    ]
    source = reversed(clip.segments) if reverse else clip.segments
    middle = [_map_segment(seg, d, e, clip.length, reverse) for seg in source]
    return Track(tuple(left + middle + right), track.T)


def _restrict(seg: Segment, lo: float, hi: float) -> Segment:
    """Restrict a segment to the sub-interval [lo, hi)."""
    if isinstance(seg, Hold):
        return Hold(lo, hi, seg.value)
    if isinstance(seg, Linear):
        return Linear(lo, hi, seg.value_at(lo), seg.value_at(hi))
    if isinstance(seg, Bezier):
        result = seg
        if hi < seg.t1:
            result = split_bezier(result, hi)[0]
        if lo > seg.t0:
            result = split_bezier(result, lo)[1]
        return result
    raise TypeError(f"unknown segment type: {type(seg)!r}")


def _map_segment(seg: Segment, d: float, e: float, length: float, reverse: bool) -> Segment:
    """Map a clip segment into the target interval [d, e).

    Forward: local time u maps to d + u*k with k = (e-d)/length.
    Reverse: local time u maps to d + (length-u)*k, so each segment is
    mirrored and its value at the left edge of its target interval is
    the left limit of the original segment at its right edge.
    """
    k = (e - d) / length
    if not reverse:
        t0 = d + seg.t0 * k
        t1 = d + seg.t1 * k
        if isinstance(seg, Hold):
            return Hold(t0, t1, seg.value)
        if isinstance(seg, Linear):
            return Linear(t0, t1, seg.v0, seg.v1)
        if isinstance(seg, Bezier):
            return Bezier(
                t0, t1, seg.v0, seg.v1,
                d + seg.x1 * k, seg.y1,
                d + seg.x2 * k, seg.y2,
            )
    else:
        t0 = d + (length - seg.t1) * k
        t1 = d + (length - seg.t0) * k
        if isinstance(seg, Hold):
            return Hold(t0, t1, seg.value)
        if isinstance(seg, Linear):
            return Linear(t0, t1, seg.v1, seg.v0)
        if isinstance(seg, Bezier):
            return Bezier(
                t0, t1, seg.v1, seg.v0,
                t0 + (seg.t1 - seg.x2) * k, seg.y2,
                t0 + (seg.t1 - seg.x1) * k, seg.y1,
            )
    raise TypeError(f"unknown segment type: {type(seg)!r}")
