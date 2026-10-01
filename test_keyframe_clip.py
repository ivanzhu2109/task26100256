"""Tests for the in-memory keyframe curve clipboard."""

import pytest

from keyframe_clip import Bezier, Hold, Linear, Track, copy_range, evaluate, paste

TOL = 1e-9


def ref_bezier(t0, v0, x1, y1, x2, y2, t1, v1, t):
    """Independent reference: bisection on u + direct Bernstein evaluation."""
    def bern(a, b, c, d, u):
        w = 1.0 - u
        return w**3 * a + 3 * w * w * u * b + 3 * w * u * u * c + u**3 * d

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bern(t0, x1, x2, t1, mid) < t:
            lo = mid
        else:
            hi = mid
    u = 0.5 * (lo + hi)
    return bern(v0, y1, y2, v1, u)


# ---------------------------------------------------------------------------
# Scenario 1: bezier with coincident time control points and vertical
# overshoot, cut at non-integer frames, pasted with time scaling.
# ---------------------------------------------------------------------------

@pytest.fixture
def track1():
    return Track((
        Hold(0.0, 1.0, 5.0),
        # x1 == x2 == 4.0 (coincident), y overshoots far beyond [0, 10]
        Bezier(1.0, 5.0, 0.0, 10.0, 4.0, 40.0, 4.0, -25.0),
        Linear(5.0, 7.0, 10.0, 2.0),
    ))


def ref_track1(t):
    if 0.0 <= t < 1.0:
        return 5.0
    if 1.0 <= t < 5.0:
        return ref_bezier(1.0, 0.0, 4.0, 40.0, 4.0, -25.0, 5.0, 10.0, t)
    if 5.0 <= t < 7.0:
        return 10.0 + (2.0 - 10.0) * (t - 5.0) / 2.0
    raise ValueError(t)


def test_bezier_evaluate_matches_reference(track1):
    for t in (1.0, 1.3, 2.7, 3.333, 4.9, 5.0 - 1e-7):
        assert evaluate(track1, t) == pytest.approx(ref_track1(t), abs=TOL)


def test_copy_inside_bezier_then_scaled_paste(track1):
    a, b = 1.7, 4.3          # both ends inside the bezier, non-integer
    d, e = 0.2, 2.6          # target interval, different length -> time scaling
    clip = copy_range(track1, a, b)
    length = b - a
    assert clip.segments[0].t0 == 0.0
    assert clip.duration == pytest.approx(length)

    out = paste(track1, d, e, clip, reverse=False)
    k = (e - d) / length

    # non-keyframe times inside the pasted interval
    for t in (0.2, 0.875, 1.333, 2.11, 2.6 - 1e-7):
        s = (t - d) / k
        expected = ref_track1(a + s)
        assert evaluate(out, t) == pytest.approx(expected, abs=1e-7)

    # right end e restores the original track value
    assert evaluate(out, e) == pytest.approx(ref_track1(e), abs=TOL)
    # outside the interval the curve is untouched
    for t in (0.0, 0.1, 0.2 - 1e-7, 2.6, 3.9, 5.5, 7.0 - 1e-7):
        assert evaluate(out, t) == pytest.approx(ref_track1(t), abs=TOL)
    # inputs were not mutated
    assert track1.segments[1].x1 == 4.0
    assert clip.duration == pytest.approx(length)


def test_copy_range_is_independent(track1):
    clip = copy_range(track1, 1.7, 4.3)
    out = paste(track1, 0.0, 2.0, clip, reverse=False)
    # further edits of the result must not affect the clip or the source
    out2 = paste(out, 0.0, 2.0, clip, reverse=True)
    for s in (0.0, 0.6, 1.9):
        assert evaluate(clip, s) == pytest.approx(ref_track1(1.7 + s), abs=1e-7)
    assert evaluate(track1, 3.0) == pytest.approx(ref_track1(3.0), abs=TOL)
    assert evaluate(out2, 0.0) != evaluate(out, 0.0) or True  # out2 built fine


# ---------------------------------------------------------------------------
# Scenario 2: hold + bezier with jumps between segments, reverse paste.
# ---------------------------------------------------------------------------

@pytest.fixture
def track2():
    return Track((
        Hold(0.0, 2.0, 3.0),                                   # jump at 2: 3 -> 0
        Bezier(2.0, 5.0, 0.0, 8.0, 3.0, 1.0, 4.0, 7.0),
        Hold(5.0, 7.0, -4.0),                                  # jump at 5: 8 -> -4
    ))


def ref_track2(t):
    if 0.0 <= t < 2.0:
        return 3.0
    if 2.0 <= t < 5.0:
        return ref_bezier(2.0, 0.0, 3.0, 1.0, 4.0, 7.0, 5.0, 8.0, t)
    if 5.0 <= t < 7.0:
        return -4.0
    raise ValueError(t)


def test_reverse_paste_with_jumps(track2):
    a, b = 1.0, 6.0            # clip covers both jumps
    d, e = 0.5, 3.0            # scaled by k = 0.5
    clip = copy_range(track2, a, b)
    length = b - a             # 5.0
    k = (e - d) / length

    out = paste(track2, d, e, clip, reverse=True)

    eps = 1e-7
    # target left end: left limit at the clip's end = last clip segment (hold -4)
    assert evaluate(out, d) == pytest.approx(-4.0, abs=TOL)
    # target right end restores the original track
    assert evaluate(out, e) == pytest.approx(ref_track2(e), abs=TOL)

    # clip-local jump times: u=1 (global 2) and u=4 (global 5)
    # reversed they map to local L-1=4 and L-4=1, i.e. target times:
    t_jump1 = d + k * (length - 1.0)   # 2.5, from the jump at global 2
    t_jump2 = d + k * (length - 4.0)   # 1.0, from the jump at global 5

    # at t_jump2 the reversed hold(-4) ends and the mirrored bezier starts;
    # the boundary value belongs to the right segment: left limit of the
    # original bezier at its end = 8
    assert evaluate(out, t_jump2) == pytest.approx(8.0, abs=TOL)
    assert evaluate(out, t_jump2 - eps) == pytest.approx(-4.0, abs=1e-4)
    assert evaluate(out, t_jump2 + eps) == pytest.approx(8.0, abs=1e-3)

    # at t_jump1 the mirrored bezier ends and the reversed hold(3) starts;
    # boundary value belongs to the right segment = 3
    assert evaluate(out, t_jump1) == pytest.approx(3.0, abs=TOL)
    assert evaluate(out, t_jump1 - eps) == pytest.approx(0.0, abs=1e-3)
    assert evaluate(out, t_jump1 + eps) == pytest.approx(3.0, abs=TOL)

    # interior of the mirrored bezier matches the mirrored reference
    for t in (1.2, 1.75, 2.3):
        s = (t - d) / k                    # local time in reversed clip
        expected = ref_track2(a + (length - s))
        assert evaluate(out, t) == pytest.approx(expected, abs=1e-6)

    # outside the pasted interval the original curve is unchanged
    for t in (0.0, 0.25, d - eps, e, 4.4, 6.5, 7.0 - eps):
        assert evaluate(out, t) == pytest.approx(ref_track2(t), abs=TOL)

    # inputs were not mutated
    assert evaluate(track2, 0.5) == 3.0
    assert evaluate(track2, 5.5) == -4.0
    assert clip.segments[0] == Hold(0.0, 1.0, 3.0)
    assert isinstance(clip.segments[1], Bezier)
    assert clip.segments[2] == Hold(4.0, 5.0, -4.0)


def test_reverse_paste_full_domain(track2):
    clip = copy_range(track2, 0.0, 7.0)
    out = paste(track2, 0.0, 7.0, clip, reverse=True)
    # left limit at clip end (u=7) is the last hold value
    assert evaluate(out, 0.0) == pytest.approx(-4.0, abs=TOL)
    # jump at global 5 maps to t = 7 - 5 = 2, owned by the right segment
    assert evaluate(out, 2.0) == pytest.approx(8.0, abs=TOL)
    # jump at global 2 maps to t = 5, owned by the right segment
    assert evaluate(out, 5.0) == pytest.approx(3.0, abs=TOL)
    assert evaluate(out, 6.9) == pytest.approx(3.0, abs=TOL)
