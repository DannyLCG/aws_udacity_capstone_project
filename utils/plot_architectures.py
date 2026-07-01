'''Generator for horizontal dataflow architecture diagrams (slab style).
Renders the benchmark CNN and the CNN+biLSTM ensemble to PNGs.

Each slab is a feature map flowing through the network, labeled with its tensor
shape as PyTorch sees it: channels x length. Stages follow the exact order of the
forward() methods in models/cnn_model.py and models/ensemble_model.py.'''
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrow

# Palette (matching the reference figure)
C_INPUT = "#D9D9D9"  # gray        - encoded input (not a learned layer)
C_CONV = "#AEE0F5"   # light blue  - convolution
C_POOL = "#F58CE8"   # magenta     - maxpool
C_LSTM = "#FFD27F"   # amber       - recurrent (biLSTM)
C_DENSE = "#8BE08B"  # green       - dense / FC
EDGE = "#1f3a5f"

# Isometric depth offset for the 3D look
DX, DY = 0.18, 0.30

def _h(length):
    '''Slab height encodes the sequence length of the feature map.'''
    return 0.16 * length + 0.2

def _w(channels):
    '''Slab width encodes the channel count of the feature map.'''
    return 0.28 + 0.0016 * channels


def slab(ax, cx, h, w, color):
    '''Draw one isometric feature-map slab centered horizontally at cx.
    -------------------------------------------------
    Params:
        cx: float, horizontal center of the slab.
        h: float, slab height (encodes sequence length).
        w: float, slab width (encodes channel count).
        color: str, fill color.
    '''
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = -h / 2, h / 2
    # Front face
    ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                         closed=True, fc=color, ec=EDGE, lw=1.2, zorder=3))
    # Top face
    ax.add_patch(Polygon([(x0, y1), (x1, y1), (x1 + DX, y1 + DY), (x0 + DX, y1 + DY)],
                         closed=True, fc=color, ec=EDGE, lw=1.2, zorder=2))
    # Right face
    ax.add_patch(Polygon([(x1, y0), (x1 + DX, y0 + DY), (x1 + DX, y1 + DY), (x1, y1)],
                         closed=True, fc=color, ec=EDGE, lw=1.2, zorder=2))

def bar(ax, cx, h, w, color, label):
    '''Draw a flat FC / recurrent bar with a vertical label.'''
    ax.add_patch(FancyBboxPatch((cx - w / 2, -h / 2), w, h,
                 boxstyle="round,pad=0.02", fc=color, ec=EDGE, lw=1.2, zorder=3))
    ax.text(cx, -h / 2 - 0.25, label, ha="center", va="top", rotation=90, fontsize=8)

def arrow(ax, x0, x1):
    ax.add_patch(FancyArrow(x0, 0, x1 - x0, 0, width=0.04, head_width=0.22,
                head_length=0.18, length_includes_head=True, color="black", zorder=4))

def dim_label(ax, cx, ymin, text):
    '''Vertical dimension label under a stage.'''
    ax.text(cx, ymin - 0.2, text, ha="center", va="top", rotation=90, fontsize=8)


def _draw_slabs(ax, stages, x):
    '''Lay out a run of isometric slabs left to right; return the running x.
    stages: list of (label, color, length, channels).'''
    centers = []
    for lab, col, length, ch in stages:
        h, w = _h(length), _w(ch)
        slab(ax, x, h, w, col)
        centers.append((x, h, lab))
        x += w + 0.45
    for cx, h, lab in centers:
        dim_label(ax, cx, -h / 2, lab)
    return x


def draw_benchmark(path):
    '''DeepCNN: conv1 conv2 pool1 | conv3 conv4 pool2 | conv5 conv6 pool3 | fc1 fc2 fc3.'''
    fig, ax = plt.subplots(figsize=(13, 4.2))
    # (label channels x length, color, length, channels)
    stages = [("51×20", C_INPUT, 20, 51),
              ("32×18", C_CONV, 18, 32), ("64×16", C_CONV, 16, 64),
              ("64×15", C_POOL, 15, 64),
              ("128×13", C_CONV, 13, 128), ("128×11", C_CONV, 11, 128),
              ("128×10", C_POOL, 10, 128),
              ("256×8", C_CONV, 8, 256), ("256×6", C_CONV, 6, 256),
              ("256×5", C_POOL, 5, 256)]
    x = _draw_slabs(ax, stages, 0.0)
    # Flatten -> three FC layers -> single regression output
    x += 0.4; arrow(ax, x - 0.5, x)
    bar(ax, x, 3.0, 0.30, C_DENSE, "Flatten 1280"); x += 0.9
    arrow(ax, x - 0.5, x); bar(ax, x, 2.2, 0.30, C_DENSE, "FC 512"); x += 0.9
    arrow(ax, x - 0.5, x); bar(ax, x, 1.4, 0.30, C_DENSE, "FC 128"); x += 0.9
    arrow(ax, x - 0.5, x)
    ax.add_patch(Polygon([(x, -0.25), (x + 0.4, -0.25), (x + 0.4, 0.25), (x, 0.25)],
                 closed=True, fc=C_DENSE, ec=EDGE, lw=1.2))
    ax.text(x + 0.9, 0, "pMIC", ha="left", va="center", fontsize=11)
    _legend(ax, x); _finish(ax, fig, path, "Benchmark CNN", x)


def draw_ensemble(path):
    '''CNN_biLSTM_Model: conv1 conv2 pool | biLSTM1 biLSTM2 last-step | fc1 fc2 fc3.'''
    fig, ax = plt.subplots(figsize=(12, 4.2))
    stages = [("51×20", C_INPUT, 20, 51),
              ("64×18", C_CONV, 18, 64), ("64×16", C_CONV, 16, 64),
              ("64×8", C_POOL, 8, 64)]
    x = _draw_slabs(ax, stages, 0.0)
    # Recurrent stage (biLSTM outputs seq × features), then last time step
    x += 0.4; arrow(ax, x - 0.5, x); bar(ax, x, 2.4, 0.34, C_LSTM, "biLSTM  8×128"); x += 1.0
    arrow(ax, x - 0.5, x); bar(ax, x, 2.0, 0.34, C_LSTM, "biLSTM  8×64"); x += 1.0
    arrow(ax, x - 0.5, x); bar(ax, x, 1.6, 0.30, C_LSTM, "last step 64"); x += 0.9
    # Three FC layers -> single regression output
    arrow(ax, x - 0.5, x); bar(ax, x, 1.5, 0.30, C_DENSE, "FC 100"); x += 0.9
    arrow(ax, x - 0.5, x); bar(ax, x, 1.2, 0.30, C_DENSE, "FC 20"); x += 0.9
    arrow(ax, x - 0.5, x)
    ax.add_patch(Polygon([(x, -0.25), (x + 0.4, -0.25), (x + 0.4, 0.25), (x, 0.25)],
                 closed=True, fc=C_DENSE, ec=EDGE, lw=1.2))
    ax.text(x + 0.9, 0, "pMIC", ha="left", va="center", fontsize=11)
    _legend(ax, x, lstm=True); _finish(ax, fig, path, "CNN + biLSTM Ensemble", x)


def _legend(ax, xmax, lstm=False):
    '''Horizontal legend strip along the bottom, centered under the content.'''
    items = [(C_INPUT, "Input"), (C_CONV, "Convolution"),
             (C_POOL, "MaxPool"), (C_DENSE, "Dense")]
    if lstm:
        items.insert(3, (C_LSTM, "biLSTM"))
    spacing, y = 2.1, -4.1
    # Center under the content, but never let the first swatch run off the left.
    start = max(0.3, xmax / 2 - spacing * len(items) / 2)
    for i, (col, lab) in enumerate(items):
        sx = start + i * spacing
        ax.add_patch(Polygon([(sx, y), (sx + 0.3, y), (sx + 0.3, y + 0.3), (sx, y + 0.3)],
                     closed=True, fc=col, ec=EDGE, lw=1.0))
        ax.text(sx + 0.42, y + 0.15, lab, va="center", fontsize=8)

def _finish(ax, fig, path, title, xmax):
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlim(-0.5, xmax + 2.0)
    ax.set_ylim(-5.0, 2.6)
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    # Write the figures into the images/ directory (run from the repo root)
    draw_benchmark("images/benchmark_arch.png")
    draw_ensemble("images/ensemble_arch.png")
    print("done")
