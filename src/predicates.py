"""
Robust geometric predicates: orient2d and incircle.

The floating-point versions are fast but can return the wrong SIGN when the
answer is close to zero (nearly collinear / nearly cocircular points). A
wrong sign corrupts the triangulation - that was the real root cause of the
"missing triangle" bug in delaunay2d.py.

Fix used here (Shewchuk-style adaptive filter):
  1. Evaluate in floating point.
  2. Compare |result| against a rigorous forward error bound.
  3. If the result is inside the error bound, recompute EXACTLY with
     Python Fractions (every float converts exactly to a Fraction).

Degenerate cases are rare, so almost every call stays on the fast path, but
the sign is always correct. Only the SIGN of the return value is meaningful.
"""

from fractions import Fraction

_EPS = 2.0 ** -53
_CCW_ERRBOUND = (3.0 + 16.0 * _EPS) * _EPS
_ICC_ERRBOUND = (10.0 + 96.0 * _EPS) * _EPS


def orient2d(ax, ay, bx, by, cx, cy):
    """> 0 if a, b, c are counter-clockwise, < 0 if clockwise, 0 if collinear."""
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright
    errbound = _CCW_ERRBOUND * (abs(detleft) + abs(detright))
    if det > errbound or -det > errbound:
        return det
    return _orient2d_exact(ax, ay, bx, by, cx, cy)


def _orient2d_exact(ax, ay, bx, by, cx, cy):
    ax, ay, bx, by, cx, cy = (Fraction(v) for v in (ax, ay, bx, by, cx, cy))
    d = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)


def incircle(ax, ay, bx, by, cx, cy, dx, dy):
    """> 0 if d is strictly inside the circle through CCW a, b, c;
    < 0 if outside; 0 if exactly on the circle."""
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    alift = adx * adx + ady * ady

    cdxady = cdx * ady
    adxcdy = adx * cdy
    blift = bdx * bdx + bdy * bdy

    adxbdy = adx * bdy
    bdxady = bdx * ady
    clift = cdx * cdx + cdy * cdy

    det = (alift * (bdxcdy - cdxbdy)
           + blift * (cdxady - adxcdy)
           + clift * (adxbdy - bdxady))

    permanent = ((abs(bdxcdy) + abs(cdxbdy)) * alift
                 + (abs(cdxady) + abs(adxcdy)) * blift
                 + (abs(adxbdy) + abs(bdxady)) * clift)
    errbound = _ICC_ERRBOUND * permanent
    if det > errbound or -det > errbound:
        return det
    return _incircle_exact(ax, ay, bx, by, cx, cy, dx, dy)


def _incircle_exact(ax, ay, bx, by, cx, cy, dx, dy):
    ax, ay, bx, by, cx, cy, dx, dy = (
        Fraction(v) for v in (ax, ay, bx, by, cx, cy, dx, dy))
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy
    det = ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
           + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
           + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))
    return 1.0 if det > 0 else (-1.0 if det < 0 else 0.0)
