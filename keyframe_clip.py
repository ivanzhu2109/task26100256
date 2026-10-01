"""In-memory keyframe curve clipboard for a single scalar animation track.

A track covers [0, T) with finitely many contiguous, non-overlapping
segments. Each segment includes its left endpoint and excludes its right
endpoint, so at a jump the right-hand segment wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class Hold:
    t0: float
    t1: float
    v0: float


@dataclass(frozen=True)
class Linear:
    t0: float
    t1: float
    v0: float
    v1: float


@dataclass(frozen=True)
class Bezier:
    # 2D cubic bezier with control points
    # (t0, v0), (x1, y1), (x2, y2), (t1, v1) and t0 <= x1 <= x2 <= t1.
    t0: float
    t1: float
    v0: float
    v1: float
    x1: float
    y1: float
    x2: float
    y2: float


Segment = Union[Hold, Linear, Bezier]


@dataclass(frozen=True)
class Track:
    segments: Tuple[Segment, ...]

    @property
    def duration(self) -> float:
        return self.segments[-1].t1


# A clip is just a track whose local time starts at 0.
Clip = Track


def evaluate(track: Track, t: float) -> float:
    return _eval_segment(_find_segment(track, t), t)


def copy_range(track: Track, a: float, b: float) -> Clip:
    return Clip(tuple(_slice_segments(track, a, b, -a)))


def paste(track: Track, d: float, e: float, clip: Clip, reverse: bool = False) -> Track:
    source = _reverse_clip(clip) if reverse else clip
    k = (e - d) / source.duration
    middle = [_scale_segment(seg, d, k) for seg in source.segments]
    left = _slice_segments(track, 0.0, d, 0.0)
    right = _slice_segments(track, e, track.duration, 0.0)
    return Track(tuple(left) + tuple(middle) + tuple(right))
# --- evaluate implementation, appended into keyframe_clip.py ---

def _bezier_x(seg: Bezier, u: float) -> float:
    w = 1.0 - u
    return (w * w * w * seg.t0 + 3.0 * w * w * u * seg.x1
            + 3.0 * w * u * u * seg.x2 + u * u * u * seg.t1)


def _bezier_y(seg: Bezier, u: float) -> float:
    w = 1.0 - u
    return (w * w * w * seg.v0 + 3.0 * w * w * u * seg.y1
            + 3.0 * w * u * u * seg.y2 + u * u * u * seg.v1)


def _bezier_dx(seg: Bezier, u: float) -> float:
    w = 1.0 - u
    return (3.0 * w * w * (seg.x1 - seg.t0)
            + 6.0 * w * u * (seg.x2 - seg.x1)
            + 3.0 * u * u * (seg.t1 - seg.x2))


def _solve_u(seg: Bezier, t: float) -> float:
    """Find u in [0, 1] with _bezier_x(seg, u) == t (x is monotonic)."""
    if t <= seg.t0:
        return 0.0
    if t >= seg.t1:
        return 1.0
    lo, hi = 0.0, 1.0
    u = (t - seg.t0) / (seg.t1 - seg.t0)
    tol = 1e-13 * max(1.0, seg.t1 - seg.t0)
    for _ in range(100):
        x = _bezier_x(seg, u)
        err = x - t
        if abs(err) <= tol:
            return u
        if err > 0.0:
            hi = u
        else:
            lo = u
        dx = _bezier_dx(seg, u)
        if dx > 0.0:
            candidate = u - err / dx
            if lo < candidate < hi:
                u = candidate
                continue
        u = 0.5 * (lo + hi)
    return u


def _eval_segment(seg: Segment, t: float) -> float:
    if isinstance(seg, Hold):
        return seg.v0
    if isinstance(seg, Linear):
        return seg.v0 + (seg.v1 - seg.v0) * (t - seg.t0) / (seg.t1 - seg.t0)
    u = _solve_u(seg, t)
    return _bezier_y(seg, u)


def _find_segment(track: Track, t: float) -> Segment:
    lo, hi = 0, len(track.segments)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if track.segments[mid].t0 <= t:
            lo = mid
        else:
            hi = mid
    seg = track.segments[lo]
    if not (seg.t0 <= t < seg.t1):
        raise ValueError(f"t={t} outside track domain")
    return seg
# --- copy_range implementation ---

def _split_bezier_points(pts, u):
    """de Casteljau split of control points at parameter u."""
    p0, p1, p2, p3 = pts
    q0 = (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
    q1 = (p1[0] + (p2[0] - p1[0]) * u, p1[1] + (p2[1] - p1[1]) * u)
    q2 = (p2[0] + (p3[0] - p2[0]) * u, p2[1] + (p3[1] - p2[1]) * u)
    r0 = (q0[0] + (q1[0] - q0[0]) * u, q0[1] + (q1[1] - q0[1]) * u)
    r1 = (q1[0] + (q2[0] - q1[0]) * u, q1[1] + (q2[1] - q1[1]) * u)
    s0 = (r0[0] + (r1[0] - r0[0]) * u, r0[1] + (r1[1] - r0[1]) * u)
    return [p0, q0, r0, s0], [s0, r1, q2, p3]


def _trim_bezier(seg: Bezier, lo: float, hi: float) -> Bezier:
    ua = _solve_u(seg, lo)
    ub = _solve_u(seg, hi)
    pts = [(seg.t0, seg.v0), (seg.x1, seg.y1), (seg.x2, seg.y2), (seg.t1, seg.v1)]
    _, right = _split_bezier_points(pts, ua)
    if ub < 1.0 and ua < ub:
        frac = (ub - ua) / (1.0 - ua)
        sub, _ = _split_bezier_points(right, frac)
    else:
        sub = right
    (p0, p1, p2, p3) = sub
    return Bezier(lo, hi, p0[1], p3[1], p1[0], p1[1], p2[0], p2[1])


def _trim_segment(seg: Segment, lo: float, hi: float) -> Segment:
    if isinstance(seg, Hold):
        return Hold(lo, hi, seg.v0)
    if isinstance(seg, Linear):
        return Linear(lo, hi, _eval_segment(seg, lo), _eval_segment(seg, hi))
    return _trim_bezier(seg, lo, hi)


def _slice_segments(track: Track, a: float, b: float, shift: float):
    out = []
    for seg in track.segments:
        lo = max(seg.t0, a)
        hi = min(seg.t1, b)
        if hi <= lo:
            continue
        out.append(_shift_segment(_trim_segment(seg, lo, hi), shift))
    return out


def _shift_segment(seg: Segment, shift: float) -> Segment:
    if shift == 0.0:
        return seg
    if isinstance(seg, Hold):
        return Hold(seg.t0 + shift, seg.t1 + shift, seg.v0)
    if isinstance(seg, Linear):
        return Linear(seg.t0 + shift, seg.t1 + shift, seg.v0, seg.v1)
    return Bezier(seg.t0 + shift, seg.t1 + shift, seg.v0, seg.v1,
                  seg.x1 + shift, seg.y1, seg.x2 + shift, seg.y2)
# --- paste implementation ---

def _scale_segment(seg: Segment, offset: float, k: float) -> Segment:
    if isinstance(seg, Hold):
        return Hold(offset + k * seg.t0, offset + k * seg.t1, seg.v0)
    if isinstance(seg, Linear):
        return Linear(offset + k * seg.t0, offset + k * seg.t1, seg.v0, seg.v1)
    return Bezier(offset + k * seg.t0, offset + k * seg.t1, seg.v0, seg.v1,
                  offset + k * seg.x1, seg.y1, offset + k * seg.x2, seg.y2)


def _reverse_segment(seg: Segment, length: float) -> Segment:
    if isinstance(seg, Hold):
        return Hold(length - seg.t1, length - seg.t0, seg.v0)
    if isinstance(seg, Linear):
        return Linear(length - seg.t1, length - seg.t0, seg.v1, seg.v0)
    return Bezier(length - seg.t1, length - seg.t0, seg.v1, seg.v0,
                  length - seg.x2, seg.y2, length - seg.x1, seg.y1)


def _reverse_clip(clip: Clip) -> Clip:
    length = clip.duration
    return Clip(tuple(_reverse_segment(seg, length)
                      for seg in reversed(clip.segments)))
