# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A single-file interactive RRT (Rapidly-exploring Random Tree) path-planning demo, `rrt_demo.py`, built with numpy and matplotlib widgets. Dependencies are managed with `uv` (`pyproject.toml` + `uv.lock`, Python >= 3.11). There are no tests, linter config, or build step.

## Commands

```bash
uv sync                    # install deps into .venv
uv run python rrt_demo.py    # random sampling (needs a GUI matplotlib backend)
uv run python rrt_manual.py  # the user clicks to choose each x_rand
uv run python rrt_draw.py    # the user drags extra blocks onto the map, then plans (subclass of rrt_manual.py)
uv run --with imageio-ffmpeg python rrt_videos.py [seed ...]  # 5 s MP4s of RRT runs -> videos/rrt_seed<seed>.mp4
# both RRT demos (and the random walk) have a Capture button: it saves only the map axes to captures/ at 300 dpi
uv run python other_methods_demo/astar_demo.py     # grid A* in the same style
uv run python other_methods_demo/dijkstra_demo.py  # grid Dijkstra (subclass of the A* demo)
uv run python other_methods_demo/randomwalk_demo.py  # random walk baseline (subclass of the RRT demo)
```

`randomwalk_demo.py` stores the walk as an `RRT` chain (each node's parent is the previous one, `i_near` = current position) and overrides `_run`. The `RandomWalkApp` subclass uses the `SHOW_RAND`, `SPEED_MAX_LOG` and `draw_tree` hooks of `App`, and its figure is `randomwalk.png` (`WALK_LINES`).

`astar_demo.py` imports the map, geometry and `load_pseudocode_figure` / `wrap_math` from `rrt_demo.py`, but has its own `AStar` generator and `App`, built the same way (phases → `PHASE_LINES` → highlighted lines of `astar.png`). A cell is blocked if its center is not `point_free`. A move is valid only if the segment between cell centers is `segment_free`, which also prevents diagonal corner-cutting. Changing the cell size or w restarts the search. `dijkstra_demo.py` subclasses it: `Dijkstra` overrides `f`/`_key` (priority = g only, no heuristic tie-break) and `DijkstraApp` sets `NAME` / `SEARCH_CLASS` / `PSEUDO_IMAGE`. `dijkstra.png` has the same 21 lines as `astar.png` (only lines 1 and 4 differ), so `PHASE_LINES` is shared.

`overleaf/` holds the LaTeX pseudocode figures (`method_manual.tex`, `astar.tex`, `dijkstra.tex`; `main.tex` is an older standalone version). To rebuild a demo's panel image, run this in that folder: `tectonic X.tex && uv run --with pymupdf python render_png.py X.pdf ../figures/X.png`. `method.png` was provided by the user and has no source here.

`rrt_draw.py` subclasses `ManualApp`. `point_free` / `segment_free` read the module-level `WALLS` / `INFLATED` lists of `rrt_demo.py`, so `DrawApp.update_walls` edits those lists in place (map walls + drawn blocks). A left press becomes a block or a click on release, depending on `DRAG_MIN`.

The demo images live in `figures/`. `astar_demo.py`, `dijkstra_demo.py` and `randomwalk_demo.py` live in `other_methods_demo/` and add the parent folder to `sys.path` to import `rrt_demo`.

`rrt_manual.py` reuses `rrt_demo.py` by subclassing (`ManualRRT` overrides `_sample` to use the clicked point, or a random one if nothing was clicked, and sets `reach_step = True` to add a `reach` phase that shows the Δ circle before `steer`; `ManualApp` sets the `RRT_CLASS` / `PSEUDO_IMAGE` class attributes).

## Architecture

`rrt_demo.py` has three layers:

- **Map / collision**: Hard-coded rectangular `WALLS` in image pixel coordinates (661×407, **y points down**). Collision checks use `INFLATED` walls (each grown by `MARGIN`), not `WALLS` directly. `point_free` tests a point; `segment_free` tests an edge via Liang-Barsky clipping (`segment_hits_rect`). `WALLS` is used only for drawing.
- **`RRT`**: Pure algorithm with no plotting. The tree is stored as parallel lists `nodes` / `parent` (parent index, `-1` for the root). The main loop is the generator `_run()`, which yields a phase name after every pseudocode step (`sample` → `nearest` → `steer` → `check` → `discard` | `add` → `goal_check` → `connect` → `done`). `advance()` runs one step and stores the step's results on the instance (`phase`, `x_rand`, `i_near`, `x_new`, `ok`). `reset()` recreates the generator.
- **Pseudocode panel**: `rrt_demo.py` shows the image `method.png` (Algorithm 1, LaTeX-rendered). `load_pseudocode_figure` finds its numbered lines automatically, and `METHOD_LINES` maps each phase to a line index (line k is numbered k+1 in the figure). If you re-export the figure with different lines, update `METHOD_LINES`. `rrt_manual.py` shows `method_manual.png` instead, built from `overleaf/method_manual.tex` (render the PDF to PNG at 300 dpi and crop it). It has the same 16 lines with line 4 changed, so `METHOD_LINES` applies to both. Setting `PSEUDO_IMAGE = None` falls back to the plain-text `PSEUDO` / `PHASE_LINES` panel, which nothing uses now. The status text uses the Computer Modern font (`cmr10` + mathtext `cm`) and `$q_{\mathrm{...}}$` notation to match the figure. Use mathtext for any non-ASCII symbol there, since `cmr10` lacks Unicode glyphs.
- **`App`**: matplotlib GUI with the map on the left and the pseudocode on the right. A 30 ms timer (`on_timer`) turns the log-scale Speed slider into a number of steps per tick. `redraw()` rebuilds the tree from `nodes`/`parent`, then draws the overlays for the current phase (`draw_overlays`), highlights the pseudocode line and shows the explanation text (`describe`). Slider values are pushed into the `RRT` before each step via `sync_params`. The axes use `set_ylim(H, 0)` so they match the image orientation. Left/right clicks move start/goal (rejected if inside an inflated wall) and reset the tree.
