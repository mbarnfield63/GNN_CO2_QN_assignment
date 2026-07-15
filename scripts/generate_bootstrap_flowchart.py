"""Renders the per-state assignment logic tree for the paper (sec:bootstrap_loop).

Static diagram of pipeline logic, not data-driven — thresholds below are
transcribed from main.tex and must be kept in sync with it by hand if the
pipeline's gates ever change. (bootstrap.py: PROB_THRESHOLD=0.80,
build_polyad_class_map: margin_threshold=0.95, final_assignment.py:
DUMMY_PENALTY=0.85, paper_utils.py reporting threshold=0.75.)

Two-pass rendering: Graphviz lays out the nodes and the label-free edges
(labels on dot edges either kink the line or land unpredictably), then PIL
draws every edge label and the rectangular retry loop at exact coordinates
computed from the layout. Label sides follow the "outside" rule: left of a
down-left branch, right of a down-right branch.

The "unpublished" terminal (assigned_prob < 0.75 in the final pass) shares the
"Discarded (no assignment)" label with the trash-can terminal: both are the
same fate, they differ only in which gate rejected the state.
"""

import io
import math
import os

import graphviz
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = "data"
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 96  # graphviz PNG default; converts "plain" layout inches to pixels
LABEL_GAP = 14  # px between a line and the nearest edge of its label
FONT = ImageFont.truetype("arial.ttf", 21)  # Helvetica maps to Arial on Windows

# Okabe-Ito hues at a ~15% tint (light pastel fill, same aesthetic as before):
# PROCESS=sky blue, DECISION=orange, TERMINAL_GOOD=bluish green, TERMINAL_BAD=vermillion
PROCESS = {"shape": "box", "style": "rounded,filled", "fillcolor": "#E6F4FC", "fontname": "Helvetica"}
DECISION = {"shape": "diamond", "style": "filled", "fillcolor": "#FBF1D9", "fontname": "Helvetica"}
TERMINAL_GOOD = {"shape": "box", "style": "rounded,filled", "fillcolor": "#D9F0EA", "fontname": "Helvetica"}
TERMINAL_BAD = {"shape": "box", "style": "rounded,filled", "fillcolor": "#F9E7D9", "fontname": "Helvetica"}


def build():
    g = graphviz.Digraph("bootstrap_logic")
    g.attr(rankdir="TB", fontname="Helvetica", nodesep="0.8", ranksep="0.6", splines="polyline")

    g.node("start", "Unassigned Ca state\n(pool, generation g)", **PROCESS)
    g.node("infer", "GNN + Hungarian solver\n(grouped by isotope, polyad, J, parity)", **PROCESS)
    g.node("harvest", "Harvest threshold:\nassigned prob\n≥ 0.80 ?", **DECISION)
    g.node("rcheck", "r-ordering\nconsistent with\nAFGL convention?", **DECISION)
    g.node("locked", "Locked as pseudo-ground-truth;\nincluded for next bootstrap training.\n(if prob ≥ 0.95: class also extends\nthe solver's valid-class pool)", **TERMINAL_GOOD)
    g.node("gen_limit", "Returned to pool", **PROCESS)
    g.node("final_pass", 'Final relaxed pass\n(grouped by isotope, J, parity;\n"trash-can" cost = 0.85)', **PROCESS)
    g.node("trashed", 'Routed to "trash-can"\ncolumn', **DECISION)
    g.node("report", "Reporting threshold:\nassigned prob\n≥ 0.75 ?", **DECISION)
    g.node("published", "Published assignment", **TERMINAL_GOOD)
    g.node("unpublished", "Discarded\n(no assignment)", **TERMINAL_BAD)
    g.node("discarded", "Discarded\n(no assignment)", **TERMINAL_BAD)

    # Invisible waypoints reserving space for the hand-drawn retry loop:
    # loop_helper east of gen_limit, loop_top east of start.
    g.node("loop_helper", "", shape="point", width="0.01", style="invis")
    g.node("loop_top", "", shape="point", width="0.01", style="invis")

    # rcheck and gen_limit share a rank so the inversion edge is a straight
    # horizontal line into the side of "Returned to pool".
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("rcheck")
        s.node("gen_limit")
        s.node("loop_helper")
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("start")
        s.node("loop_top")

    # The six diamond branch edges are invisible: they only shape the layout.
    # dot cannot attach two edges to exactly the same :s port point (even with
    # sametail they land a few px apart and cross), so annotate() hand-draws
    # them from the diamond's true bottom vertex instead.
    g.edge("start:s", "infer:n")
    g.edge("infer:s", "harvest:n")
    g.edge("harvest:s", "rcheck:n", style="invis")
    g.edge("harvest:s", "gen_limit:n", style="invis")
    g.edge("rcheck:s", "locked:n")
    g.edge("rcheck:e", "gen_limit:w")
    g.edge("gen_limit:s", "final_pass:n")
    g.edge("final_pass:s", "trashed:n")
    g.edge("trashed:s", "discarded:n", style="invis")
    g.edge("trashed:s", "report:n", style="invis")
    g.edge("report:s", "published:n", style="invis")
    g.edge("report:s", "unpublished:n", style="invis")

    # Invisible edges: keep the loop waypoints east of the flow and reserve
    # the right margin the hand-drawn loop will occupy.
    g.edge("gen_limit:e", "loop_helper:w", style="invis")
    g.edge("start", "loop_top", style="invis")  # orders loop_top right of start
    g.edge("loop_helper", "loop_top", style="invis", constraint="false")

    return g


def node_geometry(g):
    """Node name -> (cx, cy, w, h) in pixels, y down, from dot's plain output.

    The rendered PNG has dot's default 4pt pad on every side that the plain
    coordinates lack, so both axes are shifted by it.
    """
    pad = 4 / 72 * DPI
    geo = {}
    height = None
    for line in g.pipe(format="plain").decode().splitlines():
        parts = line.split()
        if parts[0] == "graph":
            height = float(parts[3])
        elif parts[0] == "node":
            x, y, w, h = (float(v) for v in parts[2:6])
            geo[parts[1]] = (x * DPI + pad, (height - y) * DPI + pad, w * DPI, h * DPI)
    return geo


def bottom(p):
    return (p[0], p[1] + p[3] / 2)


def top(p):
    return (p[0], p[1] - p[3] / 2)


def east(p):
    return (p[0] + p[2] / 2, p[1])


def west(p):
    return (p[0] - p[2] / 2, p[1])


def seg_label(draw, text, a, b, side, t=0.5, gap=LABEL_GAP):
    """Draw text beside segment a->b: 'left'/'right' of travel as seen by the
    viewer, or 'below' the segment. Anchoring keeps the whole text block on
    the chosen side regardless of its width."""
    mx, my = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    if side == "below":
        draw.text((mx, my + gap), text, font=FONT, fill="black", anchor="ma", align="center")
        return
    ux, uy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(ux, uy)
    ux, uy = ux / norm, uy / norm
    nx, ny = -uy, ux  # one of the two perpendiculars
    if (side == "left") != (nx < 0):
        nx, ny = -nx, -ny
    if nx == 0:  # horizontal-ish fallback shouldn't happen for side labels
        nx = -1 if side == "left" else 1
    anchor = "rm" if side == "left" else "lm"
    align = "right" if side == "left" else "left"
    draw.text((mx + nx * gap, my + ny * gap), text, font=FONT, fill="black", anchor=anchor, align=align)


SS = 4  # supersampling factor for hand-drawn lines (PIL draws them aliased)
ARROW_LEN = 13
ARROW_HALF_W = 4.7


def annotate(png_bytes, geo):
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", (img.width * SS, img.height * SS), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    def line(points, width=1.5):
        od.line([(x * SS, y * SS) for x, y in points], fill=(0, 0, 0, 255), width=round(width * SS))

    def arrow(tip, u):
        """Filled arrowhead at tip; u = unit direction the edge travels."""
        bx, by = tip[0] - u[0] * ARROW_LEN, tip[1] - u[1] * ARROW_LEN
        nx, ny = -u[1] * ARROW_HALF_W, u[0] * ARROW_HALF_W
        od.polygon(
            [(tip[0] * SS, tip[1] * SS), ((bx + nx) * SS, (by + ny) * SS), ((bx - nx) * SS, (by - ny) * SS)],
            fill=(0, 0, 0, 255),
        )

    def branch(a, b):
        """Straight arrow from a to b, e.g. diamond bottom vertex -> node top.

        Starts 3px before a (inside the source shape) so sub-pixel mismatch
        between dot's drawn border and the plain-format coordinates can't
        leave a visible gap or crossing at the vertex.
        """
        ux, uy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(ux, uy)
        ux, uy = ux / norm, uy / norm
        line([(a[0] - ux * 3, a[1] - uy * 3), (b[0] - ux * (ARROW_LEN - 2), b[1] - uy * (ARROW_LEN - 2))])
        arrow(b, (ux, uy))

    branch(bottom(geo["harvest"]), top(geo["rcheck"]))
    branch(bottom(geo["harvest"]), top(geo["gen_limit"]))
    branch(bottom(geo["trashed"]), top(geo["discarded"]))
    branch(bottom(geo["trashed"]), top(geo["report"]))
    branch(bottom(geo["report"]), top(geo["published"]))
    branch(bottom(geo["report"]), top(geo["unpublished"]))

    # Rectangular retry loop: gen_limit -> right -> up -> arrow into start:e.
    x_loop = max(geo["loop_helper"][0], geo["loop_top"][0])
    ax, ay = east(geo["gen_limit"])
    ex, ey = east(geo["start"])
    line([(ax, ay), (x_loop, ay), (x_loop, ey), (ex + ARROW_LEN - 2, ey)])
    arrow((ex, ey), (-1, 0))

    img = Image.alpha_composite(img, overlay.resize(img.size, Image.LANCZOS)).convert("RGB")
    draw = ImageDraw.Draw(img)

    seg_label(
        draw,
        "Bootstrap\ngeneration < 5",
        (x_loop, ey),
        (x_loop, ay),
        side="left",
        t=0.30,
        gap=LABEL_GAP - 2,
    )

    seg_label(draw, "yes", bottom(geo["harvest"]), top(geo["rcheck"]), "left", t=0.45)
    seg_label(draw, "no", bottom(geo["harvest"]), top(geo["gen_limit"]), "right", t=0.45)
    seg_label(draw, "yes", bottom(geo["rcheck"]), top(geo["locked"]), "left")
    # Right-aligned by hand: the inversion segment is shorter than the text,
    # so a centred "below" label would run under the gen_limit node's corner.
    wx, wy = west(geo["gen_limit"])
    draw.text((wx - 10, wy + 31), "no (inversion)", font=FONT, fill="black", anchor="ra")
    seg_label(
        draw,
        "Bootstrap\ngeneration = 5",
        bottom(geo["gen_limit"]),
        top(geo["final_pass"]),
        "right",
        t=0.35,
        gap=LABEL_GAP + 2,
    )
    seg_label(draw, "no (real class)", bottom(geo["trashed"]), top(geo["report"]), "left")
    seg_label(draw, "yes (fake class)", bottom(geo["trashed"]), top(geo["discarded"]), "right")
    seg_label(draw, "yes", bottom(geo["report"]), top(geo["published"]), "left")
    seg_label(draw, "no", bottom(geo["report"]), top(geo["unpublished"]), "right")
    return img


if __name__ == "__main__":
    g = build()
    img = annotate(g.pipe(format="png"), node_geometry(g))
    out_path = os.path.join(FIGURES_DIR, "bootstrap_logic_flow.png")
    img.save(out_path)
    print(f"Saved {out_path}")
