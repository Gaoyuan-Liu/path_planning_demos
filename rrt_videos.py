"""
Render 5-second MP4 videos of RRT growing on the demo map, for slides.

Each video is one random run of rrt_demo.RRT (Δ = 50, goal bias 0.05): the tree grows
over the first 4.5 s and the found path (green) is held for the last 0.5 s. The map is
drawn in the demo's colors, without the pseudocode panel.

Usage (ffmpeg comes from the imageio-ffmpeg package):
  uv run --with imageio-ffmpeg python rrt_videos.py [seed ...]     -> videos/rrt_seed<seed>.mp4
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import LineCollection
import imageio_ffmpeg

from rrt_demo import (RRT, W, H, WALLS, START, GOAL, GOAL_RADIUS, WALL_COLOR, MARKER_COLOR, LINE_W, NODE_MS,
                      star_vertices)

FPS = 30
GROW_S, HOLD_S = 4.5, 0.5
SIZE = (1280, 788)   # pixels; even sides for H.264
DPI = 160            # with SIZE this keeps the map as wide as in the demo window, so lines match
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")


def run(seed):
    """Run RRT to the goal; return it and the node count after each iteration."""
    np.random.seed(seed)
    rrt = RRT(START, GOAL)
    counts = [1]
    while rrt.advance():
        if rrt.iteration_ended or rrt.phase == "connect":
            counts.append(len(rrt.nodes))
    return rrt, counts


def render(seed):
    rrt, counts = run(seed)
    fig = plt.figure(figsize=(SIZE[0] / DPI, SIZE[1] / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    for x0, y0, x1, y1 in WALLS:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL_COLOR))
    tree = LineCollection([], colors="0.6", linewidths=LINE_W)
    ax.add_collection(tree)
    nodes_plot, = ax.plot([], [], "o", color="0.45", ms=NODE_MS)
    path_plot, = ax.plot([], [], "-", color=MARKER_COLOR, lw=LINE_W)
    ax.add_patch(Circle(rrt.start, GOAL_RADIUS, color=MARKER_COLOR, zorder=5))
    ax.add_patch(Polygon(star_vertices(rrt.goal), color=MARKER_COLOR, zorder=5))

    nodes, parent = np.array(rrt.nodes), rrt.parent
    n_grow, n_hold = round(GROW_S * FPS), round(HOLD_S * FPS)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"rrt_seed{seed}.mp4")
    writer = imageio_ffmpeg.write_frames(path, SIZE, fps=FPS, quality=9, macro_block_size=1,
                                         pix_fmt_out="yuv420p")
    writer.send(None)
    for f in range(n_grow + n_hold):
        n = counts[round(min(f / (n_grow - 1), 1) * (len(counts) - 1))]
        tree.set_segments([[nodes[parent[i]], nodes[i]] for i in range(1, n)])
        nodes_plot.set_data(nodes[:n, 0], nodes[:n, 1])
        if f >= n_grow:
            path_plot.set_data(rrt.path[:, 0], rrt.path[:, 1])
        fig.canvas.draw()
        writer.send(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    writer.close()
    plt.close(fig)
    length = np.sum(np.linalg.norm(np.diff(rrt.path, axis=0), axis=1))
    print(f"{path}: {len(counts) - 1} iterations, {len(nodes)} nodes, path {length:.0f} px")


if __name__ == "__main__":
    for s in map(int, sys.argv[1:] or ["1", "2", "3", "4"]):
        render(s)
