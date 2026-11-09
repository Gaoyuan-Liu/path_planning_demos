# Path planning demos

Interactive teaching demos for path planning, built with numpy and matplotlib. Each demo
runs an algorithm one pseudocode step at a time on the same 2D map, highlights the current
line of a LaTeX-rendered pseudocode figure, and explains the step in words.

| Demo | What it shows |
|---|---|
| `rrt_demo.py` | RRT with random sampling (and an optional goal bias) |
| `rrt_manual.py` | RRT where you click to choose each sample point |
| `rrt_draw.py` | RRT where you first drag extra obstacles onto the map |
| `other_methods_demo/astar_demo.py` | A* search on a grid |
| `other_methods_demo/dijkstra_demo.py` | Dijkstra's algorithm on the same grid |
| `other_methods_demo/randomwalk_demo.py` | A random walk that wanders until it reaches the goal (baseline) |

## Setup

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python rrt_demo.py
```

The demos open a matplotlib window, so they need a desktop session (not a headless server).
Run the other demos the same way, for example `uv run python other_methods_demo/astar_demo.py`.

## Controls

All demos share the same layout: the map on the left, the pseudocode on the right, and
sliders and buttons below.

| Control | Action |
|---|---|
| Step | Run one pseudocode step |
| Iteration / Expand | Run to the end of the current iteration (RRT, random walk) or cell expansion (A*, Dijkstra) |
| Run / Pause | Play the steps continuously, at the rate set by the Speed slider |
| Reset | Start again, keeping the start, goal and obstacles |
| Capture | Save only the map as a 300 dpi PNG in `captures/` (RRT and random walk demos) |
| Left click / right click | Move the start (green dot) / goal (green star) |

Demo-specific controls:

- `rrt_manual.py`: left click chooses the sample point q_rand. Without a click, Step and
  Run pick a random one. Shift + left click moves the start.
- `rrt_draw.py`: left drag draws a block, right click on a block removes it, and Clear
  removes all drawn blocks. Otherwise it behaves like `rrt_manual.py`.
- A* and Dijkstra: sliders set the cell size and, for A*, the heuristic weight w
  (w = 0 behaves like Dijkstra).

## Videos for slides

`rrt_videos.py` renders 5-second MP4s of RRT runs (map only, tree growing, then the found
path in green), one per random seed:

```bash
uv run --with imageio-ffmpeg python rrt_videos.py 1 2 3 4   # -> videos/rrt_seed<seed>.mp4
```

## Project layout

```
rrt_demo.py            map, collision checks, RRT algorithm and the shared GUI
rrt_manual.py          click-to-sample RRT (subclass of rrt_demo)
rrt_draw.py            draw-your-own-obstacles RRT (subclass of rrt_manual)
rrt_videos.py          MP4 export
other_methods_demo/    A*, Dijkstra and random walk demos
figures/               pseudocode figures shown in the demos
overleaf/              LaTeX sources of the pseudocode figures
RRT.pdf                problem definition and RRT pseudocode handout
```

To rebuild a pseudocode figure after editing its `.tex` file (needs
[tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd overleaf
tectonic astar.tex && uv run --with pymupdf python render_png.py astar.pdf ../figures/astar.png
```

`figures/method.png` has no LaTeX source in this repository.
