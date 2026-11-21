"""
Figure 1 (study design) and Figure 4 (structural prior injection) of the paper as editable SVG drawings, converted to PDF.

The SVG files are the masters (one group per panel) and can be edited in Inkscape or Illustrator. Text that mixes styles
(italic symbols, subscripts) is placed run by run at positions measured with the font files in FONTS, so that it renders the
same in every viewer. Text is set in Times New Roman (the metrically identical Liberation Serif is used for measuring); the
element-of sign uses DejaVu Serif. PDF conversion calls LibreOffice (soffice) in headless mode. Adjust FONTS if the fonts are
installed elsewhere. The SVG files of the paper are in figures/ next to this script.
Usage: python fig_framework_svg.py [out_dir]            # write both SVGs and convert them to PDF (default out_dir: figures)
       python fig_framework_svg.py [out_dir] --pdf-only # convert edited SVGs without regenerating them
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import ImageFont

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FIG = Path(ARGS[0]) if ARGS else Path(__file__).resolve().parent / "figures"
SANS = "'Times New Roman', 'Liberation Serif', Times, serif"
MATH = "'DejaVu Serif', 'Cambria Math', serif"
FONTS = {(False, False): "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
         (True, False): "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
         (False, True): "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
         (True, True): "/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf",
         "math": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"}
_cache = {}
FS = 1.0   # global font-size factor (sizes below 6 pt are raised to 6 pt)

# palette: accent colour for borders and title bars, pale tint for panel fills
INK, MUTED, LINE = "#1C232D", "#5C6676", "#3B4654"
NEUTRAL = dict(border="#5C6676", fill="#EEF1F5", title="#5C6676", soft="#B9C1CB")
LLM = dict(border="#B03A3A", fill="#F8E9E7", title="#B03A3A", soft="#F0D0CB")
PRIOR = dict(border="#2B6A9E", fill="#E6EFF7", title="#2B6A9E", soft="#C7DBEC")
EA = dict(border="#3D7A46", fill="#E7F2E8", title="#3D7A46", soft="#CBE4CE")
EVAL = dict(border="#9A6A1F", fill="#F7EFDF", title="#9A6A1F", soft="#ECDAB5")


def width(text, size, bold=False, italic=False, math=False):
    key = "math" if math else (bold, italic)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(FONTS[key], 1000)
    return _cache[key].getlength(text) * size / 1000


class Canvas:
    def __init__(self, w, h):
        self.w, self.h, self.out = w, h, []

    def add(self, s):
        self.out.append(s)

    def group(self, gid):
        self.add(f'<g id="{gid}">')

    def end(self):
        self.add("</g>")

    def rect(self, x, y, w, h, rx=3, fill="#FFFFFF", stroke=LINE, sw=0.6, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        st = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else 'stroke="none"'
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}"{o} {st}{d}/>')

    def panel(self, x, y, w, h, pal, title_runs, title_size=7.4):
        r, hb = 5, 12.5
        self.rect(x, y, w, h, rx=r, fill=pal["fill"], stroke=pal["border"], sw=0.9)
        self.add(f'<path d="M{x:.2f},{y + hb:.2f} V{y + r:.2f} Q{x:.2f},{y:.2f} {x + r:.2f},{y:.2f} H{x + w - r:.2f} '
                 f'Q{x + w:.2f},{y:.2f} {x + w:.2f},{y + r:.2f} V{y + hb:.2f} Z" fill="{pal["title"]}" stroke="{pal["border"]}" stroke-width="0.9"/>')
        self.rich(x + w / 2, y + 9.2, title_runs, title_size, color="#FFFFFF", anchor="middle", bold=True)

    def line(self, pts, color=LINE, sw=0.7, arrow=True, dash=None):
        d = "M" + " L".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        m = f' marker-end="url(#ah-{color[1:]})"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.markers.add(color) if arrow else None
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{ds}{m}/>')

    def circle(self, cx, cy, r, fill, stroke="#FFFFFF", sw=0.5):
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=7, color=INK, anchor="start", bold=False, italic=False, rotate=None, scaled=False):
        size = size if scaled else max(size * FS, 6.0)
        tr = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}"{tr} font-family="{SANS}" font-size="{size}" '
                 f'font-weight="{"bold" if bold else "normal"}" font-style="{"italic" if italic else "normal"}" '
                 f'fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')

    def rich(self, x, y, runs, size=7, color=INK, anchor="start", bold=False):
        """runs: list of (text, flags) with flags from b (bold), i (italic), m (math font), sub, sup, c=#rrggbb."""
        if isinstance(runs, str):
            runs = [(runs, "")]
        size = max(size * FS, 6.0)
        items = []
        for t, flags in runs:
            f = flags.split()
            b = bold or "b" in f
            it = "i" in f
            m = "m" in f
            sz = size * (0.72 if ("sub" in f or "sup" in f) else 1)
            dy = size * 0.28 if "sub" in f else (-size * 0.38 if "sup" in f else 0)
            col = next((g[2:] for g in f if g.startswith("c=")), color)
            items.append((t, b, it, m, sz, dy, col, width(t, sz, b, it, m)))
        total = sum(i[-1] for i in items)
        cx = x - total / 2 if anchor == "middle" else (x - total if anchor == "end" else x)
        if len(items) == 1 and anchor in ("middle", "start", "end"):
            t, b, it, m, sz, dy, col, w = items[0]
            if not m:
                self.text(x, y + dy, t, sz, col, anchor, b, it, scaled=True)
                return total
        for t, b, it, m, sz, dy, col, w in items:
            fam = MATH if m else SANS
            self.add(f'<text x="{cx:.2f}" y="{y + dy:.2f}" font-family="{fam}" font-size="{sz:.2f}" '
                     f'font-weight="{"bold" if b else "normal"}" font-style="{"italic" if it else "normal"}" '
                     f'fill="{col}" xml:space="preserve">{esc(t)}</text>')
            cx += w
        return total

    def svg(self):
        defs = "".join(
            f'<marker id="ah-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
            f'orient="auto"><path d="M0,0.8 L10,5 L0,9.2 z" fill="{c}"/></marker>' for c in sorted(self.markers))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}pt" height="{self.h}pt" '
                f'viewBox="0 0 {self.w} {self.h}">\n<defs>{defs}</defs>\n' + "\n".join(self.out) + "\n</svg>\n")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def new_canvas(w, h):
    c = Canvas(w, h)
    c.markers = set()
    return c


def template_grid(c, x, y, cw, ch, weights, size=5.4, labels=True):
    """2 x 4 template: attention row on top, MLP row below; shade grows with the weight."""
    for r, name in enumerate(("Attn", "MLP")):
        if labels:
            c.text(x - 3, y + r * ch + ch * 0.68, name, size, MUTED, "end")
        for k in range(4):
            w = weights[r * 4 + k]
            shade = ["#FFFFFF", "#E3EDF6", "#C7DBEC", "#9CC0DC", "#5F97C4"][min(4, [0, .25, .5, 1, 2].index(w) if w in (0, .25, .5, 1, 2) else 4)]
            c.rect(x + k * cw, y + r * ch, cw, ch, rx=0, fill=shade, stroke=PRIOR["border"], sw=0.5)
            lab = "0" if w == 0 else ("1" if w == 1 else (f"{w:g}".lstrip("0") if w < 1 else f"{w:g}"))
            c.text(x + k * cw + cw / 2, y + r * ch + ch * 0.68, lab, size, INK, "middle")


def mini_llm(c, x, y, removed_attn=None, removed_mlp=None, label=""):
    """Four layers, each an attention block and an MLP block; the removed part is left white."""
    aw, mw, h, gap = 15, 28, 6.5, 3.6
    c.text(x + aw / 2, y - 3, "Attn", 5.6, MUTED, "middle")
    c.text(x + aw + 4 + mw / 2, y - 3, "MLP", 5.6, MUTED, "middle")
    for i in range(4):
        yy = y + i * (h + gap)
        for bx, bw, rem in ((x, aw, removed_attn[i] if removed_attn else 0), (x + aw + 4, mw, removed_mlp[i] if removed_mlp else 0)):
            c.rect(bx, yy, bw, h, rx=1, fill="#FFFFFF", stroke=NEUTRAL["border"], sw=0.5)
            c.rect(bx, yy, bw * (1 - rem), h, rx=1, fill=NEUTRAL["soft"], stroke=None)
    c.text(x + (aw + 4 + mw) / 2, y + 4 * (h + gap) + 6, label, 6.2, INK, "middle")



def lines_box(c, x, y, w, lines, pal, size=6.3, lead=8.4, pad=6, header=None):
    """White box with left-aligned lines (operation name, then its parameters)."""
    n = len(lines) + (1 if header else 0)
    h = pad + n * lead + 3
    c.rect(x, y, w, h, rx=3, fill="#FFFFFF", stroke=pal["border"], sw=0.6)
    yy = y + pad + lead * 0.72
    if header:
        c.rich(x + 6, yy, header, size, bold=True)
        yy += lead
    for ln in lines:
        c.rich(x + 6, yy, ln, size)
        yy += lead
    return h


# ----------------------------------------------------------------------------------------------------------------
def figure_study_design():
    c = new_canvas(504, 270)
    c.group("shared-problem")
    c.panel(6, 6, 492, 94, NEUTRAL, [("Shared problem: how much to prune each module of an LLM", "")])
    mini_llm(c, 18, 34, label="dense LLM")
    c.line([(68, 53), (92, 53)])
    c.rich(80, 49, [("x", "b i")], 7.5, anchor="middle")
    mini_llm(c, 98, 34, [0, .35, .6, .2], [.3, .5, .6, .4], label="pruned LLM")
    tx, sz = 162, 6.9
    c.rich(tx, 35, [("allocation ", "b"), ("x", "b i"), (": a pruning level for every attention and MLP module,", "")], sz)
    c.rich(tx + 8, 45, "taken from a database of pre-pruned modules", sz)
    c.rich(tx, 57, [("constraint", "b"), (": removed compute ", ""), ("r", "i"), ("(", ""), ("x", "b i"), (") ", ""), ("∈", "m"),
                    (" [", ""), ("ρ", "i"), (", ", ""), ("ρ", "i"), (" + ", ""), ("δ", "i"), ("]", "")], sz)
    c.rich(tx, 69, [("objective", "b"), (": KL divergence from the dense model", "")], sz)
    c.rich(tx, 81, [("budget", "b"), (": ", ""), ("B", "i"), (" evaluations, a few hundred", "")], sz)
    lines_box(c, 388, 26, 102, [[("same evaluator and budget ", ""), ("B", "i")], "16 random starts per seed", "seeds paired across methods"],
              NEUTRAL, 6.5, 9.4, 6, header="Fixed for every search")
    c.end()

    py, ph = 114, 118
    xa, xb, xc, pw = 6, 173, 340, 158
    c.group("same-llm")
    c.line([(xa + pw / 2, py), (xa + pw / 2, 106), (xc + pw / 2, 106), (xc + pw / 2, py)], LLM["border"], 0.7, arrow=False, dash="2.5 1.8")
    tw = width("the same LLM", 6.4)
    c.rect(252 - tw / 2 - 4, 101.5, tw + 8, 9, rx=2, fill="#FFFFFF", stroke=None)
    c.text(252, 108.3, "the same LLM", 6.4, LLM["title"], "middle")
    c.end()

    c.group("panel-a")
    c.panel(xa, py, pw, ph, LLM, [("(a) LLM as the optimizer", "")])
    cx = xa + 73
    c.rect(cx - 55, 132, 110, 24, rx=3, fill="#FFFFFF", stroke=LLM["border"], sw=0.6)
    c.text(cx, 142, "history, with KL", 6.7, INK, "middle")
    c.text(cx, 151, "best, worst and recent", 6.2, MUTED, "middle")
    c.line([(cx, 156), (cx, 164)])
    c.rect(cx - 55, 164, 110, 17, rx=3, fill="#FFFFFF", stroke=LLM["border"], sw=0.8)
    c.text(cx, 175.3, "LLM proposes 8 allocations", 6.9, LLM["title"], "middle", bold=True)
    c.line([(cx, 181), (cx, 189)])
    c.rect(cx - 55, 189, 110, 17, rx=3, fill=EVAL["fill"], stroke=EVAL["border"], sw=0.8)
    c.text(cx, 200.3, "evaluate KL", 6.9, EVAL["title"], "middle", bold=True)
    c.line([(cx + 55, 197.5), (cx + 67, 197.5), (cx + 67, 144), (cx + 55.6, 144)])
    c.rich(cx, 222, [("every round, until ", ""), ("B", "i")], 6.3, color=MUTED, anchor="middle")
    c.end()

    c.group("panel-b")
    c.panel(xb, py, pw, ph, NEUTRAL, [("(b) Conventional optimizers", "")])
    lx, lw_ = xb + 10, 138
    c.rect(lx, 131, lw_, 53, rx=3, fill="#FFFFFF", stroke=NEUTRAL["border"], sw=0.6)
    names = [["EA", "guided EA", "GP-BO"], ["TPE", "DE", "PSO"], ["CMA-ES", "GA", "random"]]
    for r, row in enumerate(names):
        for k, n in enumerate(row):
            c.text(lx + 10 + k * 45, 144 + r * 14, n, 6.8, EA["title"] if n == "EA" else INK, bold=(n == "EA"))
    bx = lx + lw_ / 2
    c.line([(bx, 184), (bx, 191)])
    c.rect(bx - 55, 191, 110, 17, rx=3, fill=EVAL["fill"], stroke=EVAL["border"], sw=0.8)
    c.text(bx, 202.3, "evaluate KL", 6.9, EVAL["title"], "middle", bold=True)
    c.line([(bx + 55, 199.5), (lx + lw_ + 7, 199.5), (lx + lw_ + 7, 157.5), (lx + lw_ + 0.6, 157.5)])
    c.text(bx, 222, "no prior", 6.3, MUTED, "middle")
    c.end()

    c.group("panel-c")
    c.panel(xc, py, pw, ph, PRIOR, [("(c) Structural prior + EA", "")])
    template_grid(c, xc + 22, 131, 13, 10, [0, .25, .5, 0, .5, 1, 1, .5], 6.0)
    sx = xc + 84
    c.text(sx, 135, "template source", 6.2, MUTED)
    c.text(sx, 143.5, "LLM", 6.6, LLM["title"], bold=True)
    c.text(sx, 151.5, "transferred", 6.4, INK)
    c.text(sx, 159.5, "hand-designed", 6.4, INK)
    c.text(sx, 167.5, "random", 6.4, INK)
    c.line([(xc + 48, 151), (xc + 48, 174)])
    c.rect(xc + 10, 174, 70, 24, rx=3, fill="#FFFFFF", stroke=PRIOR["border"], sw=0.6)
    c.text(xc + 45, 184, "decode, refine", 6.7, INK, "middle")
    c.text(xc + 45, 193, "(evals. 17–80)", 6.2, MUTED, "middle")
    c.line([(xc + 80, 186), (xc + 92, 186)])
    c.rect(xc + 92, 174, 57, 24, rx=3, fill=EA["soft"], stroke=EA["border"], sw=0.8)
    c.text(xc + 120.5, 184, "EA as in (b)", 6.7, EA["title"], "middle", bold=True)
    c.rich(xc + 120.5, 193, [("until ", "b c=" + EA["title"]), ("B", "b i c=" + EA["title"])], 6.7, anchor="middle")
    c.line([(xc + 120.5, 198), (xc + 120.5, 204)])
    c.rect(xc + 92, 204, 57, 16, rx=3, fill=EVAL["fill"], stroke=EVAL["border"], sw=0.8)
    c.text(xc + 120.5, 214.8, "evaluate KL", 6.7, EVAL["title"], "middle", bold=True)
    c.text(xc + 45, 211, "the LLM writes templates", 6.1, MUTED, "middle")
    c.text(xc + 45, 219, "in two early rounds", 6.1, MUTED, "middle")
    c.end()

    c.group("comparisons")
    ca, cb, ccc = xa + 73, bx, xc + 79
    c.line([(cb, 236), (cb, 241), (ccc - 6, 241), (ccc - 6, 236)], NEUTRAL["border"], 0.7, arrow=False)
    c.text((cb + ccc - 6) / 2, 250, "(c) against the EA: what the prior adds", 6.5, INK, "middle")
    c.line([(ca, 236), (ca, 257), (ccc + 6, 257), (ccc + 6, 236)], NEUTRAL["border"], 0.7, arrow=False)
    c.text((ca + ccc) / 2, 266, "(c) against (a), with the same LLM: its knowledge used by the EA or by the LLM itself", 6.5, INK, "middle")
    c.end()
    return c


# ----------------------------------------------------------------------------------------------------------------
def figure_pipeline():
    c = new_canvas(504, 306)
    c.group("budget")
    segs = [(8, 50, ["16 random", "allocations"], NEUTRAL), (58, 80, ["round 1: two templates,", "steps 1–3 for each"], PRIOR),
            (138, 80, ["round 2: two templates,", "steps 1–3 for each"], PRIOR), (218, 172, ["step 4: evolutionary search"], EA)]
    for x, w, lines, pal in segs:
        c.rect(x, 9, w, 19, rx=0, fill=pal["fill"] if pal is not NEUTRAL else "#FFFFFF", stroke=pal["border"], sw=0.6)
        for k, t in enumerate(lines):
            c.text(x + w / 2, 9 + (13 if len(lines) == 1 else 8.3 + 7 * k), t, 6.1, INK, "middle")
    for x, lab in [(8, "0"), (58, "16"), (138, "48"), (218, "80"), (390, "B")]:
        c.line([(x, 28), (x, 31)], NEUTRAL["border"], 0.6, arrow=False)
        c.text(x, 38, lab, 6.2, INK, "middle", italic=(lab == "B"))
    c.text(304, 38.5, "evaluations used", 6.2, MUTED, "middle")
    c.end()
    c.group("legend")
    for k, (pal, lab) in enumerate([(LLM, "LLM"), (PRIOR, "template and decoder"), (EA, "evolutionary search"), (EVAL, "evaluator (KL)")]):
        yy = 10 + k * 8.4
        c.rect(406, yy, 8, 5.5, rx=1, fill=pal["soft"], stroke=pal["border"], sw=0.6)
        c.text(418, yy + 4.9, lab, 6.2, INK)
    c.end()

    py, ph = 48, 150
    c.group("step-1-template")
    x1, w1 = 6, 164
    c.panel(x1, py, w1, ph, LLM, [("(1) Template from the LLM", "")])
    cx = x1 + w1 / 2
    c.text(x1 + 9, 70, "request (abridged)", 6.1, MUTED)
    lines_box(c, x1 + 8, 73, w1 - 16, ["round 1, Llama-3.1-8B, 16 evaluations used", "best KL so far: 2.00",
                                        "removable compute per cell: 0.05 … 0.20", "strong, weak and recent allocations"], LLM, 6.3, 8.4, 5)
    c.line([(cx, 112.6), (cx, 121)])
    c.rect(cx - 42, 121, 84, 15, rx=3, fill="#FFFFFF", stroke=LLM["border"], sw=0.8)
    c.text(cx, 131.4, "LLM (gpt-5-mini)", 6.9, LLM["title"], "middle", bold=True)
    c.line([(cx, 136), (cx, 144)])
    c.rect(x1 + 8, 144, w1 - 16, 41, rx=3, fill="#FFFFFF", stroke=LLM["border"], sw=0.6)
    template_grid(c, x1 + 32, 150, 12, 10.5, [0, .25, .5, 0, .5, 1, 1, .5], 6.0)
    for k, t in enumerate(["two templates", "8 cell weights each", "with a rationale"]):
        c.text(x1 + 90, 158 + 8.4 * k, t, 6.3, INK)
    c.text(cx, 206, "other sources: transferred, hand-designed or random", 6.1, MUTED, "middle")
    c.end()

    c.group("step-2-decoder")
    x2, w2 = 176, 162
    c.panel(x2, py, w2, ph, PRIOR, [("(2) Feasibility decoder (Alg. 1)", "")])
    cx2 = x2 + w2 / 2
    template_grid(c, x2 + 30, 65, 14, 10.5, [0, .25, .5, 0, .5, 1, 1, .5], 6.0)
    c.rich(x2 + 94, 73.5, [("template ", ""), ("w", "b")], 6.7)
    c.text(x2 + 94, 82.5, "one weight per cell", 6.2, MUTED)
    c.line([(x2 + 58, 86), (x2 + 58, 93)])
    bw = w2 - 18
    c.rect(x2 + 9, 93, bw, 29, rx=3, fill="#FFFFFF", stroke=PRIOR["border"], sw=0.6)
    c.rich(cx2, 103.5, [("strength ", ""), ("γ", "i"), (" by bisection, so that", "")], 6.6, anchor="middle")
    c.rich(cx2, 115.5, [("Σ", "m"), ("j", "i sub"), (" ", ""), ("β", "i"), ("j", "i sub"), (" min(", ""), ("γ", "i"), (" ", ""),
                        ("w", "i"), ("c(j)", "i sub"), (", ", ""), ("φ", "i"), ("j", "i sub"), ("(", ""), ("K", "i"), ("j", "i sub"),
                        (")) = ", ""), ("ρ", "i"), ("*", "sup")], 7.0, anchor="middle")
    c.line([(cx2, 122), (cx2, 129)])
    c.rect(x2 + 9, 129, bw, 16, rx=3, fill="#FFFFFF", stroke=PRIOR["border"], sw=0.6)
    c.text(cx2, 139.6, "stochastic rounding to levels", 6.6, INK, "middle")
    c.line([(cx2, 145), (cx2, 152)])
    c.rect(x2 + 9, 152, bw, 16, rx=3, fill="#FFFFFF", stroke=PRIOR["border"], sw=0.6)
    c.rich(cx2, 162.6, [("repair until ", ""), ("r", "i"), ("(", ""), ("x", "b i"), (") ", ""), ("∈", "m"), (" [", ""), ("ρ", "i"),
                        (", ", ""), ("ρ", "i"), (" + ", ""), ("δ", "i"), ("]", "")], 6.6, anchor="middle")
    c.text(cx2, 184, "no evaluation is used", 6.2, MUTED, "middle")
    c.end()

    c.group("step-3-refinement")
    x3, w3 = 344, 154
    c.panel(x3, py, w3, ph, EA, [("(3) Short refinement", "")])
    c.rect(x3 + 9, 64, 74, 74, rx=3, fill="#FFFFFF", stroke=PRIOR["border"], sw=0.6)
    c.rich(x3 + 46, 74, [("anchor ", ""), ("x", "b i"), ("a", "i sub")], 6.6, anchor="middle")
    attn = [0, 0, .175, .175, .35, .35, 0, 0]
    mlp = [.35, .35, .7, .7, .7, .7, .35, .35]
    bx0 = x3 + 29
    for vals, base, lab in ((attn, 100, "Attn"), (mlp, 126, "MLP")):
        c.text(bx0 - 3, base - 1, lab, 6.0, MUTED, "end")
        c.line([(bx0 - 1, base), (bx0 + 8 * 6.4, base)], NEUTRAL["border"], 0.5, arrow=False)
        for k, v in enumerate(vals):
            if v > 0:
                c.rect(bx0 + k * 6.4, base - v * 25, 5.0, v * 25, rx=0, fill=PRIOR["border"], stroke=None)
    c.text(x3 + 46, 134.5, "removed per layer", 6.0, MUTED, "middle")
    c.line([(x2 + 9 + bw, 160), (x3 + 4, 160), (x3 + 4, 101), (x3 + 9, 101)], PRIOR["title"], 0.7)
    root = (x3 + 118, 75)
    g1 = [(x3 + 100 + 12 * i, 95) for i in range(4)]
    g2 = [(x3 + 94 + 12 * i, 115) for i in range(4)]
    for p in g1:
        c.line([root, p], EA["border"], 0.6, arrow=False)
    for i, p in enumerate(g2):
        c.line([g1[[0, 0, 2, 3][i]], p], EA["border"], 0.6, arrow=False)
    c.circle(*root, 3.6, PRIOR["border"])
    for p in g1 + g2:
        c.circle(*p, 3.0, EA["border"])
    c.text(x3 + 118, 127, "…", 7, EA["title"], "middle")
    lines_box(c, x3 + 9, 143, w3 - 18, ["15 more evaluations within the lineage", "local population: top 8",
                                        "4 children per generation"], EA, 6.3, 8.4, 5)
    c.end()

    c.group("step-4-ea")
    y4, h4 = 210, 90
    c.panel(6, y4, 492, h4, EA, [("(4) Evolutionary search until ", ""), ("B", "i"), (" evaluations", "")])
    yc = 256
    c.rect(16, yc - 12, 52, 24, rx=3, fill="#FFFFFF", stroke=EA["border"], sw=0.6)
    c.text(42, yc - 2, "4 best", 6.7, INK, "middle")
    c.text(42, yc + 7, "as parents", 6.7, INK, "middle")
    c.line([(68, yc), (80, yc)])
    c.rect(80, yc - 28, 140, 56, rx=3, fill="#FFFFFF", stroke=EA["border"], sw=0.6)
    c.text(150, yc - 18, "budget-transfer mutation", 6.8, INK, "middle", bold=True)
    c.text(150, yc - 9.5, "moves 1.5% of the dense compute", 6.2, MUTED, "middle")
    yb = yc + 17
    for k, (before, after) in enumerate(((22, 13), (10, 19))):
        bxk = 108 + k * 44
        c.rect(bxk, yb - after, 12, after, rx=0, fill=EA["border"], stroke=None)
        c.rect(bxk, yb - before, 12, before, rx=0, fill="none", stroke=LINE, sw=0.6, dash="1.8 1.2")
        c.text(bxk + 6, yb + 7.5, "module " + "ij"[k], 6.0, INK, "middle")
    c.line([(122, yb - 15), (150, yb - 15)], EA["title"], 0.8)
    c.text(176, yb - 10, "removed", 6.0, MUTED)
    c.text(176, yb - 3, "compute", 6.0, MUTED)
    c.line([(220, yc), (232, yc)])
    c.rect(232, yc - 10, 54, 20, rx=3, fill=EVAL["fill"], stroke=EVAL["border"], sw=0.8)
    c.text(259, yc + 2.4, "evaluate KL", 6.9, EVAL["title"], "middle", bold=True)
    c.line([(286, yc), (298, yc)])
    c.rect(298, yc - 12, 70, 24, rx=3, fill="#FFFFFF", stroke=EA["border"], sw=0.6)
    c.rich(333, yc - 2, [("keep best 8 of ", ""), ("P", "i")], 6.6, anchor="middle")
    c.text(333, yc + 7, "and children", 6.6, INK, "middle")
    c.line([(368, yc), (380, yc)])
    c.rect(380, yc - 15, 102, 30, rx=3, fill=EA["soft"], stroke=EA["border"], sw=0.8)
    c.rich(431, yc - 3, [("global population ", "b c=" + EA["title"]), ("P", "b i c=" + EA["title"])], 6.9, anchor="middle")
    c.rich(431, yc + 7, [("best ", ""), ("μ", "i"), (" = 8 so far", "")], 6.5, anchor="middle")
    c.line([(431, yc + 15), (431, 292), (42, 292), (42, yc + 12.6)])
    lw = width("while fewer than B evaluations", 6.3)
    c.rect(236 - lw / 2 - 4, 287.5, lw + 8, 9, rx=2, fill=EA["fill"], stroke=None)
    c.rich(236, 294, [("while fewer than ", ""), ("B", "i"), (" evaluations", "")], 6.3, color=INK, anchor="middle")
    c.end()
    c.group("survivors")
    sxv = x3 + 128
    c.line([(sxv, 173.2), (sxv, 203.5), (501, 203.5), (501, yc), (482.6, yc)], EA["title"], 0.7)
    c.rich(sxv - 4, 190, [("survivors join ", "c=" + EA["title"]), ("P", "i c=" + EA["title"])], 6.2, anchor="end")
    c.end()
    return c


def to_pdf(svg_path):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp, str(svg_path)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        pdf = Path(tmp) / (svg_path.stem + ".pdf")
        target = svg_path.with_suffix(".pdf")
        target.write_bytes(pdf.read_bytes())
    print("saved", target)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    names = {"fig_study_design": figure_study_design, "fig_pipeline": figure_pipeline}
    for name, make in names.items():
        svg = FIG / f"{name}.svg"
        if "--pdf-only" not in sys.argv:
            svg.write_text(make().svg())
            print("saved", svg)
        to_pdf(svg)
