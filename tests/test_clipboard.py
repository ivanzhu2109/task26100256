"""Tests for the in-memory keyframe curve clipboard."""

import math

import pytest

from keyframe_clipboard import Bezier, Hold, Linear, Track, copy_range, evaluate, paste

TOL = 1e-9


def ref_bezier_value(p0, p1, p2, p3, t):
    """Independent reference: value of a 2D cubic bezier at time t.

    Solves the time parameter by pure bisection (the library uses a
    Newton/bisection hybrid), then evaluates the value polynomial.
    """

    def cubic(a, b, c, d, u):
        m = 1.0 - u
        return m * m * m * a + 3 * m * m * u * b + 3 * m * u * u * c + u * u * u * d

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cubic(p0[0], p1[0], p2[0], p3[0], mid) < t:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return cubic(p0[1], p1[1], p2[1], p3[1], tau)


# Control points of the source bezier used in scenario 1: coincident time
# control points (x1 == x2 == 4) and vertical overshoot (y1=20 > v1,
# y2=-8 < v0).
BZ = ((0.0, 0.0), (4.0, 20.0), (4.0, -8.0), (10.0, 5.0))


def test_copy_then_time_scaled_paste_matches_reference():
    src = Track((Bezier(0.0, 10.0, 0.0, 5.0, 4.0, 20.0, 4.0, -8.0),), 10.0)
    a, b = 2.5, 6.5  # both endpoints inside the bezier, non-integer
    clip = copy_range(src, a, b)
    assert clip.length == pytest.approx(b - a)
    assert clip.segments[0] is not src.segments[0]  # independent copy

    dst = Track((Hold(0.0, 3.0, 1.0), Linear(3.0, 9.0, 2.0, 8.0)), 9.0)
    d, e = 1.5, 7.5  # e - d = 6 != L = 4: time-scaled paste, k = 1.5
    out = paste(dst, d, e, clip, reverse=False)
    k = (e - d) / clip.length

    # Non-keyframe times inside the pasted interval, checked against the
    # independent reference evaluated on the original curve.
    for t in (1.5, 1.7, 2.3, 3.0, 3.3, 4.9, 6.1, 7.4):
        s = (t - d) / k
        expected = ref_bezier_value(BZ[0], BZ[1], BZ[2], BZ[3], a + s)
        assert evaluate(out, t) == pytest.approx(expected, abs=TOL)

    # Right end of the target interval restores the original track value.
    assert evaluate(out, e) == pytest.approx(evaluate(dst, e), abs=TOL)
    # Outside the interval the curve is unchanged, on both sides.
    for t in (0.0, 0.5, 1.4, 7.6, 8.0, 8.999):
        assert evaluate(out, t) == pytest.approx(evaluate(dst, t), abs=TOL)

    # Inputs were not modified.
    assert src.segments == (Bezier(0.0, 10.0, 0.0, 5.0, 4.0, 20.0, 4.0, -8.0),)
    assert dst.segments == (Hold(0.0, 3.0, 1.0), Linear(3.0, 9.0, 2.0, 8.0))
    assert clip.length == pytest.approx(4.0)


def make_jump_track():
    # Jumps at t=4 (hold 3.0 -> bezier start 0.0) and t=10 (bezier end
    # 6.0 -> linear start 1.0).
    return Track(
        (
            Hold(0.0, 4.0, 3.0),
            Bezier(4.0, 10.0, 0.0, 6.0, 6.0, 12.0, 8.0, -4.0),
            Linear(10.0, 14.0, 1.0, 5.0),
        ),
        14.0,
    )


def test_reverse_paste_with_hold_and_bezier_jumps():
    src = make_jump_track()
    clip = copy_range(src, 1.0, 13.0)  # L = 12: hold [0,3), bezier [3,9), linear [9,12)

    dst = Track((Hold(0.0, 8.0, 100.0), Linear(8.0, 20.0, 50.0, -50.0)), 20.0)
    d, e = 2.0, 14.0  # k = 1
    out = paste(dst, d, e, clip, reverse=True)

    # Target left end takes the left limit of the clip's end: the reversed
    # linear's value there (linear runs 1.0 -> 4.0 over local [9, 12)).
    assert evaluate(out, d) == pytest.approx(4.0, abs=TOL)
    # Interior of the reversed linear (t < 5 maps to local u = 14 - t).
    assert evaluate(out, 3.0) == pytest.approx(3.0, abs=TOL)

    # Internal boundary t=5 (local u=9, jump linear -> bezier in the clip):
    # the jump belongs to the right-side segment, i.e. the reversed bezier,
    # whose value is the bezier's left limit at u=9 (its end value 6.0).
    assert evaluate(out, 5.0) == pytest.approx(6.0, abs=TOL)
    assert evaluate(out, 5.0 - 1e-9) == pytest.approx(1.0, abs=1e-6)

    # Interior of the reversed bezier: t=8 maps to local u=6, i.e. source
    # time 1 + 6 = 7 on the original bezier.
    assert evaluate(out, 8.0) == pytest.approx(evaluate(src, 7.0), abs=TOL)

    # Internal boundary t=11 (local u=3, jump bezier -> hold in the clip):
    # right side is the reversed hold (3.0); left side approaches the
    # bezier's left limit at u=3 (its start value 0.0).
    assert evaluate(out, 11.0) == pytest.approx(3.0, abs=TOL)
    assert evaluate(out, 11.0 - 1e-9) == pytest.approx(0.0, abs=1e-6)
    assert evaluate(out, 13.999) == pytest.approx(3.0, abs=TOL)

    # Right end restores the original track; outside is unchanged.
    assert evaluate(out, e) == pytest.approx(evaluate(dst, e), abs=TOL)
    for t in (0.0, 1.0, 1.999, 14.0, 15.5, 19.999):
        assert evaluate(out, t) == pytest.approx(evaluate(dst, t), abs=TOL)

    # Inputs were not modified.
    assert src == make_jump_track()
    assert dst.segments == (Hold(0.0, 8.0, 100.0), Linear(8.0, 20.0, 50.0, -50.0))


def test_paste_with_e_equal_T():
    clip = copy_range(make_jump_track(), 0.0, 4.0)  # hold 3.0, L = 4
    dst = Track((Linear(0.0, 10.0, 0.0, 100.0),), 10.0)
    out = paste(dst, 6.0, 10.0, clip, reverse=False)  # e == T
    assert evaluate(out, 6.0) == pytest.approx(3.0, abs=TOL)
    assert evaluate(out, 9.999) == pytest.approx(3.0, abs=TOL)
    assert evaluate(out, 5.999) == pytest.approx(evaluate(dst, 5.999), abs=TOL)
    with pytest.raises(ValueError):
        evaluate(out, 10.0)  # T is outside the domain
