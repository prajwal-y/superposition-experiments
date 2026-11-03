"""Synthetic V / rounded-V / O stimuli, drawn at 4x and downsampled to 224 px (anti-aliased).

A V is two arms of length L meeting at an apex with half-opening angle theta, rotated by phi
(phi = 0: apex at the bottom, arms opening upward, like a chin or a snout seen head-on).
Rounding replaces the apex with a circular arc tangent to both arms (fillet radius r):
r = 0 is a sharp V; large r turns it into a U.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from PIL import Image, ImageDraw

S = 224
K = 4  # supersampling


def _v_points(L, theta, r, phi, cx, cy, n_arc=48):
    """Polyline for a (possibly rounded) V with apex near (cx, cy), in supersampled pixels."""
    # Arms leave the apex upward at +-theta from vertical (y grows downward in images).
    d1 = np.array([-math.sin(theta), -math.cos(theta)])
    d2 = np.array([math.sin(theta), -math.cos(theta)])
    if r <= 0:
        pts = [d1 * L, np.zeros(2), d2 * L]
    else:
        t = min(r / math.tan(theta), L * 0.98)        # tangent distance from apex along each arm
        r_eff = t * math.tan(theta)
        c = np.array([0.0, -r_eff / math.sin(theta)])  # fillet centre on the bisector
        p1, p2 = d1 * t, d2 * t
        a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
        a2 = math.atan2(p2[1] - c[1], p2[0] - c[0])
        # sweep the short way through the bottom (angle pi/2 in image coordinates)
        if a2 < a1:
            a2 += 2 * math.pi
        arc = [c + r_eff * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(a1, a2, n_arc)]
        if not (a1 <= math.pi / 2 <= a2):
            arc = [c + r_eff * np.array([math.cos(a), math.sin(a)])
                   for a in np.linspace(a1, a2 - 2 * math.pi, n_arc)]
        pts = [d1 * L] + arc + [d2 * L]
    # centre the shape's bounding box on (cx, cy), then rotate by phi about the centre
    P = np.array(pts)
    P -= (P.max(0) + P.min(0)) / 2
    R = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    P = P @ R.T
    return [(cx + x, cy + y) for x, y in P]


def draw_v(L, theta_deg, r_frac, phi_deg, width, fill, dark_on_light):
    bg, fg = (217, 0) if dark_on_light else (38, 255)
    img = Image.new("L", (S * K, S * K), bg)
    d = ImageDraw.Draw(img)
    th = math.radians(theta_deg)
    # r_frac in [0, 1]: fraction of the largest fillet (one that would eat 90% of each arm)
    pts = _v_points(L * K, th, r_frac * 0.9 * L * K * math.tan(th), math.radians(phi_deg), S * K / 2, S * K / 2)
    if fill:
        d.polygon(pts, fill=fg)
    else:
        d.line(pts, fill=fg, width=int(width * K), joint="curve")
        h = width * K / 2  # round dots over every joint hide the seams PIL leaves between segments
        for x, y in pts[1:-1]:
            d.ellipse([x - h, y - h, x + h, y + h], fill=fg)
    return img.resize((S, S), Image.LANCZOS).convert("RGB")


def draw_o(radius, width, fill, dark_on_light):
    bg, fg = (217, 0) if dark_on_light else (38, 255)
    img = Image.new("L", (S * K, S * K), bg)
    d = ImageDraw.Draw(img)
    c, R = S * K / 2, radius * K
    if fill:
        d.ellipse([c - R, c - R, c + R, c + R], fill=fg)
    else:
        d.ellipse([c - R, c - R, c + R, c + R], outline=fg, width=int(width * K))
    return img.resize((S, S), Image.LANCZOS).convert("RGB")


V_GRID = dict(
    L=[40, 70, 100],
    theta=[15, 25, 35, 50],
    phi=list(range(0, 360, 15)),
    style=[("line", 3), ("line", 6), ("line", 10), ("fill", 0)],
    dark_on_light=[True, False],
)
ROUND = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]


def v_specs():
    for L, th, phi, (style, w), pol in itertools.product(*V_GRID.values()):
        yield dict(L=L, theta=th, phi=phi, style=style, width=w, dark_on_light=pol)


def render_v(spec, r_frac=0.0):
    return draw_v(spec["L"], spec["theta"], r_frac, spec["phi"], spec["width"], spec["style"] == "fill",
                  spec["dark_on_light"])


def o_specs():
    for R, (style, w), pol in itertools.product([20, 35, 50, 70], [("line", 3), ("line", 6), ("line", 10), ("fill", 0)],
                                               [True, False]):
        yield dict(R=R, style=style, width=w, dark_on_light=pol)


def render_o(spec):
    return draw_o(spec["R"], spec["width"], spec["style"] == "fill", spec["dark_on_light"])


if __name__ == "__main__":
    import sys
    out = sys.argv[1]
    tiles = []
    for r in ROUND:
        tiles.append(np.asarray(draw_v(80, 30, r, 0, 6, False, True)))
    for r in ROUND:
        tiles.append(np.asarray(draw_v(80, 30, r, 180, 0, True, False)))
    n = len(ROUND)
    row1, row2 = np.concatenate(tiles[:n], 1), np.concatenate(tiles[n:], 1)
    Image.fromarray(np.concatenate([row1, row2], 0)).save(out)
    print(sum(1 for _ in v_specs()), "V specs x", len(ROUND), "roundings;", sum(1 for _ in o_specs()), "O specs")
