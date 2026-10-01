"""Segment types for a scalar animation track.

A track covers [0, T) with finitely many explicit, contiguous,
non-overlapping segments. Each segment contains its left endpoint and
excludes its right endpoint. Time is measured in frames and may be
non-integer.
"""

from __future__ import annotations

from dataclasses import dataclass


class Segment:
    """Base class. A segment covers [t0, t1)."""

    t0: float
    t1: float

    def value_at(self, t: float) -> float:
        raise NotImplementedError

    def shifted(self, dt: float) -> "Segment":
        raise NotImplementedError


@dataclass(frozen=True)
class Hold(Segment):
    t0: float
    t1: float
    value: float

    def value_at(self, t: float) -> float:
        return self.value

    def shifted(self, dt: float) -> "Hold":
        return Hold(self.t0 + dt, self.t1 + dt, self.value)


@dataclass(frozen=True)
class Linear(Segment):
    t0: float
    t1: float
    v0: float
    v1: float

    def value_at(self, t: float) -> float:
        u = (t - self.t0) / (self.t1 - self.t0)
        return self.v0 + (self.v1 - self.v0) * u

    def shifted(self, dt: float) -> "Linear":
        return Linear(self.t0 + dt, self.t1 + dt, self.v0, self.v1)


@dataclass(frozen=True)
class Bezier(Segment):
    """2D cubic bezier with control points (t0,v0),(x1,y1),(x2,y2),(t1,v1).

    Requires t0 <= x1 <= x2 <= t1 so time is monotonic along the curve.
    Vertical control points may overshoot.
    """

    t0: float
    t1: float
    v0: float
    v1: float
    x1: float
    y1: float
    x2: float
    y2: float

    def value_at(self, t: float) -> float:
        return _bezier_value(self, t)

    def shifted(self, dt: float) -> "Bezier":
        return Bezier(
            self.t0 + dt, self.t1 + dt, self.v0, self.v1,
            self.x1 + dt, self.y1, self.x2 + dt, self.y2,
        )


def split_bezier(seg: Bezier, t: float) -> tuple[Bezier, Bezier]:
    """Split a bezier segment at time t into left and right parts."""
    tau = _solve_tau(seg, t)
    p0 = (seg.t0, seg.v0)
    p1 = (seg.x1, seg.y1)
    p2 = (seg.x2, seg.y2)
    p3 = (seg.t1, seg.v1)

    q0 = _lerp2(p0, p1, tau)
    q1 = _lerp2(p1, p2, tau)
    q2 = _lerp2(p2, p3, tau)
    r0 = _lerp2(q0, q1, tau)
    r1 = _lerp2(q1, q2, tau)
    s0 = _lerp2(r0, r1, tau)

    # Snap the split point's time coordinate to the exact split time so
    # chained restrict/shift operations keep clean boundaries.
    left = Bezier(p0[0], t, p0[1], s0[1], q0[0], q0[1], r0[0], r0[1])
    right = Bezier(t, p3[0], s0[1], p3[1], r1[0], r1[1], q2[0], q2[1])
    return left, right


def _lerp2(p: tuple[float, float], q: tuple[float, float], u: float) -> tuple[float, float]:
    return (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)


def _bezier_time(seg: Bezier, tau: float) -> float:
    mt = 1.0 - tau
    return (
        mt * mt * mt * seg.t0
        + 3.0 * mt * mt * tau * seg.x1
        + 3.0 * mt * tau * tau * seg.x2
        + tau * tau * tau * seg.t1
    )


def _bezier_value(seg: Bezier, t: float) -> float:
    tau = _solve_tau(seg, t)
    mt = 1.0 - tau
    return (
        mt * mt * mt * seg.v0
        + 3.0 * mt * mt * tau * seg.y1
        + 3.0 * mt * tau * tau * seg.y2
        + tau * tau * tau * seg.v1
    )


def _solve_tau(seg: Bezier, t: float) -> float:
    """Find tau in [0, 1] with bezier time coordinate equal to t.

    Time is monotonic non-decreasing in tau because t0 <= x1 <= x2 <= t1,
    so a hybrid Newton/bisection solve converges reliably.
    """
    if t <= seg.t0:
        return 0.0
    if t >= seg.t1:
        return 1.0
    lo, hi = 0.0, 1.0
    tau = (t - seg.t0) / (seg.t1 - seg.t0)
    for _ in range(64):
        f = _bezier_time(seg, tau) - t
        if f > 0.0:
            hi = tau
        else:
            lo = tau
        # Newton step, clamped to the current bracket.
        mt = 1.0 - tau
        deriv = 3.0 * (
            mt * mt * (seg.x1 - seg.t0)
            + 2.0 * mt * tau * (seg.x2 - seg.x1)
            + tau * tau * (seg.t1 - seg.x2)
        )
        candidate = tau - f / deriv if deriv > 1e-300 else 0.5 * (lo + hi)
        if not lo < candidate < hi:
            candidate = 0.5 * (lo + hi)
        if candidate == tau:
            break
        tau = candidate
        if hi - lo < 1e-15:
            break
    return tau
