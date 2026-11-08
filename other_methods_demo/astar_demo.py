"""
Interactive A* demo on the example map, with step-by-step pseudocode.

The map is split into square cells; a cell is blocked if its center is inside a
wall (inflated by the safety margin). Moves go to the 8 neighboring cells.

Controls
  Run / Pause : execute pseudocode steps continuously (rate set by "Speed")
  Step        : execute one pseudocode step (pick s, close s, look at a neighbor, ...)
  Expand      : execute steps until the current cell s has been fully expanded
  Reset       : restart the search (keeps start/goal)
  Left click  : move start (green dot)
  Right click : move goal (green star)
  Sliders     : cell size, heuristic weight w (0 = Dijkstra), speed

Reuses the map, geometry and pseudocode-figure panel from rrt_demo.py. The panel shows
astar.png, built from overleaf/astar.tex.
"""
import heapq
import itertools
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # rrt_demo.py is one folder up

from rrt_demo import (W, H, WALLS, START, GOAL, GOAL_RADIUS, WALL_COLOR, MARKER_COLOR, TICK_MS, LINE_W,
                      point_free, segment_free, star_vertices, load_pseudocode_figure, wrap_math)

FREE, BLOCKED, OPEN, CLOSED = range(4)
CELL_COLORS = np.array([to_rgba(c) for c in ("white", "#e8e8e8", "#cfe6ff", "#ffdcaa")])
S_COLOR, SN_COLOR = "darkorange", "#1f6fd1"
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

# ---------------- Pseudocode ----------------
FIGURES = os.path.join(ROOT, "figures")
PSEUDO_IMAGE = os.path.join(FIGURES, "astar.png")
PHASE_LINES = {  # phase -> highlighted lines of the figure (line k is numbered k + 1 there)
    "init": [1],
    "pop": [3],
    "goal_test": [4],
    "done": [4, 5],
    "close": [7],
    "neighbor": [8],
    "skip": [9, 10],
    "cost": [12],
    "update": [13, 14, 15],
    "no_update": [13],
    "fail": [19],
}


# ---------------- A* ----------------
class AStar:
    """Grid A* whose main loop is a generator that pauses after every pseudocode step.

    Cells are (col, row) tuples. After each `advance()`, `phase` names the step just
    executed and s / s_n / cost / g_new / g_old / ok describe its result.
    """

    def __init__(self, start, goal, cell=20, w=1.0):
        self.start, self.goal = start.copy(), goal.copy()
        self.cell, self.w = cell, w
        self.reset()

    def reset(self):
        cs = self.cell
        self.cols, self.rows = int(np.ceil(W / cs)), int(np.ceil(H / cs))
        self.state = np.full((self.rows, self.cols), FREE)
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = self.center((c, r))
                if not (0 < x < W and 0 < y < H and point_free((x, y))):
                    self.state[r, c] = BLOCKED
        self.s_start, self.s_goal = self.cell_of(self.start), self.cell_of(self.goal)
        self.g, self.parent = {}, {}
        self.open, self.closed, self.heap = set(), set(), []
        self.tie = itertools.count()
        self.s = self.s_n = self.cost = self.g_new = self.g_old = self.ok = None
        self.last_nb = False
        self.path = None
        self.failed = False
        if self.is_free(self.s_start) and self.is_free(self.s_goal):
            self.g[self.s_start] = 0.0
            self.parent[self.s_start] = None
            self._push(self.s_start)
        self.phase = "init"
        self._steps = self._run()

    def cell_of(self, p):
        return int(p[0] // self.cell), int(p[1] // self.cell)

    def center(self, s):
        return np.array([(s[0] + 0.5) * self.cell, (s[1] + 0.5) * self.cell])

    def is_free(self, s):
        c, r = s
        return 0 <= c < self.cols and 0 <= r < self.rows and self.state[r, c] != BLOCKED

    def h(self, s):
        return float(np.linalg.norm(self.center(s) - self.center(self.s_goal)))

    def f(self, s):
        return self.g[s] + self.w * self.h(s)

    def _key(self, s):
        """Priority of s in OPEN; ties go to the cell closer to the goal."""
        return self.f(s), self.h(s)

    def _push(self, s):
        self.open.add(s)
        self.state[s[1], s[0]] = OPEN
        heapq.heappush(self.heap, (*self._key(s), next(self.tie), s))

    def advance(self):
        """Execute one pseudocode step. Returns False once the search has finished."""
        if self.path is not None or self.failed:
            return False
        self.phase = next(self._steps)
        return True

    @property
    def iteration_ended(self):
        return self.phase in ("done", "fail") or \
            (self.phase in ("skip", "update", "no_update") and self.last_nb)

    def _run(self):
        while self.heap:
            s = heapq.heappop(self.heap)[-1]
            if s in self.closed:      # outdated heap entry (s was improved later)
                continue
            self.s, self.s_n = s, None
            yield "pop"
            if s == self.s_goal:
                self.path = self._extract_path(s)
                yield "done"
                return
            yield "goal_test"
            self.open.discard(s)
            self.closed.add(s)
            self.state[s[1], s[0]] = CLOSED
            yield "close"
            nbs = [(s[0] + dc, s[1] + dr) for dc, dr in MOVES]
            nbs = [n for n in nbs if 0 <= n[0] < self.cols and 0 <= n[1] < self.rows]
            for k, n in enumerate(nbs):
                self.s_n, self.last_nb = n, k == len(nbs) - 1
                self.cost = float(np.linalg.norm(self.center(n) - self.center(s)))
                self.g_new = self.g_old = None
                yield "neighbor"
                if n in self.closed or not self.is_free(n) or \
                        not segment_free(self.center(s), self.center(n)):
                    self.ok = False
                    yield "skip"
                    continue
                self.g_new = self.g[s] + self.cost
                yield "cost"
                self.g_old = self.g.get(n)
                self.ok = n not in self.open or self.g_new < self.g_old
                if self.ok:
                    self.g[n], self.parent[n] = self.g_new, s
                    self._push(n)
                    yield "update"
                else:
                    yield "no_update"
        self.s = self.s_n = None
        self.failed = True
        yield "fail"

    def _extract_path(self, s):
        path = []
        while s is not None:
            path.append(self.center(s))
            s = self.parent[s]
        return np.array(path[::-1])


# ---------------- GUI ----------------
class App:
    NAME = "A*"
    SEARCH_CLASS = AStar
    PSEUDO_IMAGE = PSEUDO_IMAGE
    PHASE_LINES = PHASE_LINES

    def __init__(self):
        self.astar = self.SEARCH_CLASS(START, GOAL)
        self.running = False
        self.credit = 0.0   # fractional steps carried between timer ticks

        self.fig = plt.figure(figsize=(14, 7.5))
        self.ax = self.fig.add_axes([0.02, 0.28, 0.62, 0.66])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)          # flip y so it looks like the image
        self.ax.set_aspect("equal")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.cells_img = self.ax.imshow(np.zeros((1, 1, 4)), interpolation="nearest", zorder=0)
        self.grid_lines = LineCollection([], colors="0.88", linewidths=0.5, zorder=0.5)
        self.ax.add_collection(self.grid_lines)
        for x0, y0, x1, y1 in WALLS:
            self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL_COLOR))

        self.path_plot, = self.ax.plot([], [], "-", color=MARKER_COLOR, lw=LINE_W * 1.5, zorder=4)
        self.start_dot = Circle((0, 0), GOAL_RADIUS, color=MARKER_COLOR, zorder=5)
        self.goal_dot = Polygon(star_vertices(np.zeros(2)), color=MARKER_COLOR, zorder=5)
        self.ax.add_patch(self.start_dot); self.ax.add_patch(self.goal_dot)

        # overlays for the current pseudocode step
        self.s_box = Rectangle((0, 0), 1, 1, fill=False, ec=S_COLOR, lw=LINE_W * 1.5, visible=False, zorder=7)
        self.sn_box = Rectangle((0, 0), 1, 1, fill=False, ec=SN_COLOR, lw=LINE_W * 1.5, visible=False, zorder=7)
        self.ax.add_patch(self.s_box); self.ax.add_patch(self.sn_box)
        self.edge, = self.ax.plot([], [], "-", lw=LINE_W * 1.5, zorder=6)

        # pseudocode panel: the figure, with highlight bars drawn behind its lines
        rgba, centers, left, pitch = load_pseudocode_figure(self.PSEUDO_IMAGE)
        self.ax_code = self.fig.add_axes([0.655, 0.37, 0.34, 0.6])
        self.ax_code.set_axis_off()
        self.line_marks = []
        for y in centers:
            mark = Rectangle((left, y - pitch / 2), rgba.shape[1] - left, pitch, color="#ffe680", visible=False)
            self.ax_code.add_patch(mark)
            self.line_marks.append(mark)
        self.ax_code.imshow(rgba, zorder=2)
        status_ax = self.fig.add_axes([0.66, 0.02, 0.33, 0.33])
        status_ax.set_axis_off()
        status_ax.set_xlim(0, 1); status_ax.set_ylim(0, 1)
        self.status = status_ax.text(0.02, 0.97, "", fontsize=12, family="cmr10", va="top", linespacing=1.5)
        # legend
        legend = [("OPEN", dict(color=CELL_COLORS[OPEN])), ("CLOSED", dict(color=CELL_COLORS[CLOSED])),
                  ("blocked", dict(color=CELL_COLORS[BLOCKED])),
                  ("$s$", dict(fill=False, ec=S_COLOR, lw=2)), ("$s'$", dict(fill=False, ec=SN_COLOR, lw=2))]
        for k, (label, style) in enumerate(legend):
            x = 0.02 + k * 0.2
            status_ax.add_patch(Rectangle((x, 0.02), 0.04, 0.07, **style))
            status_ax.text(x + 0.055, 0.055, label, fontsize=11, family="cmr10", va="center")

        # sliders (changing cell size or w restarts the search)
        self.s_cell = Slider(self.fig.add_axes([0.12, 0.18, 0.45, 0.03]), "Cell size", 10, 40,
                             valinit=20, valstep=1)
        self.s_w = Slider(self.fig.add_axes([0.12, 0.13, 0.45, 0.03]), "Heuristic w", 0.0, 3.0, valinit=1.0)
        self.s_speed = Slider(self.fig.add_axes([0.12, 0.08, 0.45, 0.03]), "Speed", 0, 3, valinit=np.log10(3))
        self.s_cell.on_changed(lambda v: self.reset())
        self.s_w.on_changed(lambda v: self.reset())
        self.s_speed.on_changed(self.speed_label)
        self.speed_label(self.s_speed.val)

        # buttons
        self.b_run = Button(self.fig.add_axes([0.05, 0.015, 0.12, 0.045]), "Run")
        self.b_step = Button(self.fig.add_axes([0.19, 0.015, 0.12, 0.045]), "Step")
        self.b_iter = Button(self.fig.add_axes([0.33, 0.015, 0.12, 0.045]), "Expand")
        self.b_reset = Button(self.fig.add_axes([0.47, 0.015, 0.12, 0.045]), "Reset")
        self.b_run.on_clicked(self.toggle_run)
        self.b_step.on_clicked(lambda e: self.advance(1))
        self.b_iter.on_clicked(lambda e: self.advance_iteration())
        self.b_reset.on_clicked(lambda e: self.reset())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.timer = self.fig.canvas.new_timer(interval=TICK_MS)
        self.timer.add_callback(self.on_timer)
        self.timer.start()
        self.reset()

    def speed_label(self, v):
        self.s_speed.valtext.set_text(f"{10 ** v:.0f} steps/s")

    def step_once(self):
        if not self.astar.advance():
            return False
        if self.astar.path is not None or self.astar.failed:
            self.set_running(False)
        return True

    def advance(self, n):
        for _ in range(int(n)):
            if not self.step_once():
                break
        self.redraw()

    def advance_iteration(self):
        while self.step_once() and not self.astar.iteration_ended:
            pass
        self.redraw()

    def on_timer(self):
        if not self.running:
            return
        self.credit += 10 ** self.s_speed.val * TICK_MS / 1000
        n = int(self.credit)
        self.credit -= n
        if n:
            self.advance(n)

    def set_running(self, running):
        self.running = running
        self.credit = 1.0 if running else 0.0   # react immediately on Run
        self.b_run.label.set_text("Pause" if running else "Run")
        self.fig.canvas.draw_idle()

    def toggle_run(self, _):
        a = self.astar
        self.set_running(not self.running and a.path is None and not a.failed)

    def reset(self):
        self.set_running(False)
        a = self.astar
        a.cell, a.w = int(self.s_cell.val), self.s_w.val
        a.reset()
        cs = a.cell
        self.cells_img.set_extent((0, a.cols * cs, a.rows * cs, 0))
        self.grid_lines.set_segments([[(c * cs, 0), (c * cs, H)] for c in range(a.cols + 1)] +
                                     [[(0, r * cs), (W, r * cs)] for r in range(a.rows + 1)])
        self.start_dot.center = a.center(a.s_start)
        self.goal_dot.set_xy(star_vertices(a.center(a.s_goal)))
        self.redraw()

    def on_click(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        p = np.array([e.xdata, e.ydata])
        if not self.astar.is_free(self.astar.cell_of(p)):
            return
        if e.button == 1:
            self.astar.start = p
        elif e.button == 3:
            self.astar.goal = p
        self.reset()

    def redraw(self):
        a = self.astar
        self.cells_img.set_data(CELL_COLORS[a.state])
        if a.path is not None:
            self.path_plot.set_data(a.path[:, 0], a.path[:, 1])
        else:
            self.path_plot.set_data([], [])
        self.draw_overlays()
        for k, mark in enumerate(self.line_marks):
            mark.set_visible(k in self.PHASE_LINES[a.phase])
        self.status.set_text(wrap_math(self.describe(), 52))
        self.update_title()
        self.fig.canvas.draw_idle()

    def draw_overlays(self):
        a, ph = self.astar, self.astar.phase
        cs = a.cell
        self.s_box.set_visible(a.s is not None and ph not in ("init", "done", "fail"))
        if a.s is not None:
            self.s_box.set_bounds(a.s[0] * cs, a.s[1] * cs, cs, cs)
        show_n = a.s_n is not None and ph in ("neighbor", "skip", "cost", "update", "no_update")
        self.sn_box.set_visible(show_n)
        self.edge.set_data([], [])
        if show_n:
            self.sn_box.set_bounds(a.s_n[0] * cs, a.s_n[1] * cs, cs, cs)
            color = {"skip": "red", "update": "#00a000", "no_update": "0.4"}.get(ph, S_COLOR)
            p, q = a.center(a.s), a.center(a.s_n)
            self.edge.set_color(color)
            self.edge.set_data([p[0], q[0]], [p[1], q[1]])

    def describe(self):
        a, ph = self.astar, self.astar.phase
        cell = lambda s: f"({s[0]}, {s[1]})"
        if ph == "init":
            return r"OPEN holds only $s_{\mathrm{start}}$, with $g(s_{\mathrm{start}})=0$. Press Step."
        if ph == "fail":
            if not (a.is_free(a.s_start) and a.is_free(a.s_goal)):
                return "The start or goal cell is blocked at this cell size. Move it or change the cell size."
            return "OPEN is empty: every reachable cell was expanded and no path exists."
        if ph == "pop":
            return rf"Pick the OPEN cell with the smallest $g+w \cdot h$: $s={cell(a.s)}$ with " \
                   rf"$g={a.g[a.s]:.0f}$, $h={a.h(a.s):.0f}$, $g+{a.w:.1f} \cdot h={a.f(a.s):.0f}$."
        if ph == "goal_test":
            return r"$s \neq s_{\mathrm{goal}}$, so keep searching."
        if ph == "done":
            return rf"$s=s_{{\mathrm{{goal}}}}$! ReconstructPath follows the parents back to the start. " \
                   f"Path length {a.g[a.s]:.0f} px, {len(a.closed)} cells expanded."
        if ph == "close":
            return "$s$ moves from OPEN to CLOSED: its $g$ is final and it will not be expanded again."
        diag = a.s_n[0] != a.s[0] and a.s_n[1] != a.s[1]
        if ph == "neighbor":
            return f"Look at neighbor $s'={cell(a.s_n)}$ ({'diagonal' if diag else 'straight'} move, " \
                   f"cost {a.cost:.0f})."
        if ph == "skip":
            if a.s_n in a.closed:
                return r"$s' \in$ CLOSED: continue."
            return "$s'$ is blocked (inside or too close to a wall), or the move cuts a wall corner: " \
                   "CollisionFree is false, continue."
        if ph == "cost":
            return rf"Cost of reaching $s'$ through $s$: $g'=g(s)+\mathrm{{Cost}}(s,s')=" \
                   rf"{a.g[a.s]:.0f}+{a.cost:.0f}={a.g_new:.0f}$."
        if ph == "update":
            if a.g_old is None:
                return rf"$s' \notin$ OPEN: set $g(s')={a.g_new:.0f}$, parent$(s')=s$ and add $s'$ to OPEN."
            return rf"$g'={a.g_new:.0f}<g(s')={a.g_old:.0f}$: a shorter way to $s'$. " \
                   "Update its $g$ and parent."
        return rf"$g'={a.g_new:.0f} \geq g(s')={a.g_old:.0f}$: the route $s'$ already has is at least " \
               "as good. Keep it."

    def update_title(self):
        a = self.astar
        msg = f"{self.NAME}   expanded: {len(a.closed)}   OPEN: {len(a.open)}"
        if a.path is not None:
            msg += f"   path found, length = {a.g[a.s]:.0f} px"
        self.ax.set_title(msg)


if __name__ == "__main__":
    app = App()
    plt.show()
