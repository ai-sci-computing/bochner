#!/usr/bin/env python3
"""Animate per-frame vortex-filament polylines written by ring_demo.

Usage: animate_filaments.py [frames.txt] [out.gif]
Reads the FRAME blocks (frame, closed, npoints, then xyz lines) and renders a
rotating 3D animation of the extracted filament loop inside the simulation box.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

infile = sys.argv[1] if len(sys.argv) > 1 else "frames.txt"
outfile = sys.argv[2] if len(sys.argv) > 2 else "ring_filament.gif"

frames, lo, hi = [], -1.0, 1.0
cur = None
for line in open(infile):
    if line.startswith("#"):
        t = line.split()
        if "box" in t:
            lo, hi = float(t[t.index("box") + 1]), float(t[t.index("box") + 2])
        continue
    if line.startswith("FRAME"):
        _, i, closed, n = line.split()
        cur = {"idx": int(i), "closed": closed == "1", "pts": []}
        frames.append(cur)
    else:
        cur["pts"].append([float(x) for x in line.split()])

for fr in frames:
    fr["pts"] = np.array(fr["pts"]) if fr["pts"] else np.empty((0, 3))
# Animate only frames that actually carry a filament.
frames = [fr for fr in frames if len(fr["pts"])]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

# Box edges for spatial reference.
import itertools
corners = np.array(list(itertools.product([lo, hi], repeat=3)))
def box():
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b]) > 1e-9) == 1:
                ax.plot(*zip(corners[a], corners[b]), color="0.85", lw=0.8)

def draw(k):
    ax.cla()
    box()
    fr = frames[k]
    p = fr["pts"]
    if len(p):
        q = np.vstack([p, p[0]]) if fr["closed"] else p
        ax.plot(q[:, 0], q[:, 1], q[:, 2], "-", color="crimson", lw=3)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color="crimson", s=8)
        mz = p[:, 2].mean()
        mr = np.hypot(p[:, 0], p[:, 1]).mean()
        ax.set_title(f"frame {fr['idx']}   meanZ={mz:+.2f}  meanR={mr:.2f}", fontsize=11)
    else:
        ax.set_title(f"frame {fr['idx']}   (no filament)", fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=30 + 3 * k)  # slow rotation

anim = FuncAnimation(fig, draw, frames=len(frames), interval=180)
anim.save(outfile, writer=PillowWriter(fps=6))
print(f"wrote {outfile} ({len(frames)} frames)")
