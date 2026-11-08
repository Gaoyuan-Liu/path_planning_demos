"""
Interactive RRT demo on the example map, with step-by-step pseudocode.

Controls
  Run / Pause : execute pseudocode steps continuously (rate set by "Speed")
  Step        : execute one pseudocode step (sample, nearest, steer, check, ...)
  Iteration   : execute steps until the current iteration ends
  Reset       : clear the tree (keeps start/goal)
  Left click  : move start (green dot)
  Right click : move goal (green star)
  Sliders     : step size, goal bias, speed (pseudocode steps per second)

Requires: numpy, matplotlib   ->  pip install numpy matplotlib
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import LineCollection

# ---------------- Map (pixel coordinates of the image, y points down) ----------------
W, H = 661, 407
WALLS = [  # (x0, y0, x1, y1)
    (0, 0, W, 18), (0, 388, W, H), (0, 0, 18, H), (643, 0, W, H),  # outer border
    (130, 198, 268, 216),   # horizontal wall, left
    (250, 198, 268, 388),   # vertical wall from bottom, left
    (383, 0, 401, 213),     # vertical wall from top, middle
    (509, 198, 527, 388),   # vertical wall from bottom, right
]
START = np.array([90.0, 325.0])
GOAL = np.array([580.0, 82.0])
MARGIN = 6          # clearance kept from walls (pixels)
GOAL_RADIUS = 18    # radius of the start dot
STAR_RADIUS = 24    # outer radius of the goal star
WALL_COLOR = "#5c5c5c"
MARKER_COLOR = "#00a200"
TICK_MS = 30        # GUI timer interval
LINE_W = 2          # width of every line (edges, path, rays, rings)
NODE_MS = 8         # marker size of every tree node (same as the new node during exploration)

INFLATED = [(x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN) for x0, y0, x1, y1 in WALLS]


def star_vertices(center, r_out=STAR_RADIUS, r_in=STAR_RADIUS * 0.45):
    """Five-pointed star pointing up on screen (the y axis points down)."""
    ang = -np.pi / 2 + np.arange(10) * np.pi / 5
    r = np.where(np.arange(10) % 2 == 0, r_out, r_in)
    return center + np.c_[r * np.cos(ang), r * np.sin(ang)]


def point_free(p):
    return not any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in INFLATED)


def segment_hits_rect(p, q, rect):
    """Liang-Barsky clipping: does segment p->q touch the rectangle?"""
    x0, y0, x1, y1 = rect
    d = q - p
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], p[0] - x0), (d[0], x1 - p[0]), (-d[1], p[1] - y0), (d[1], y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return False
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def segment_free(p, q):
    return not any(segment_hits_rect(p, q, r) for r in INFLATED)


# ---------------- Pseudocode ----------------
PSEUDO = [
    "RRT(x_start, x_goal):",
    "  T ← { x_start }",
    "  repeat",
    "    x_rand ← random point in whole map",
    "             (x_goal with prob. p_goal)",
    "    x_near ← Nearest(T, x_rand)",
    "    x_new  ← Steer(x_near, x_rand, Δ)",
    "    if not CollisionFree(x_near, x_new):",
    "      discard x_new; continue",
    "    T.add(x_new, parent = x_near)",
    "    if ‖x_new − x_goal‖ ≤ Δ and",
    "       CollisionFree(x_new, x_goal):",
    "      T.add(x_goal, parent = x_new)",
    "      return path x_start → x_goal",
]
PHASE_LINES = {  # phase -> highlighted pseudocode lines
    "init": [1],
    "sample": [3, 4],
    "nearest": [5],
    "steer": [6],
    "check": [7],
    "discard": [8],
    "add": [9],
    "goal_check": [10, 11],
    "connect": [12],
    "done": [13],
}


# ---------------- RRT ----------------
class RRT:
    """RRT whose main loop is a generator that pauses after every pseudocode step.

    After each `advance()`, `phase` names the step just executed and the
    attributes x_rand / i_near / x_new / ok describe its result.
    """

    def __init__(self, start, goal, step=25.0, goal_bias=0.05):
        self.start, self.goal = start.copy(), goal.copy()
        self.step_size, self.goal_bias = step, goal_bias
        self.reset()

    def reset(self):
        self.nodes = [self.start.copy()]
        self.parent = [-1]
        self.iters = 0
        self.path = None
        self.x_rand = self.i_near = self.x_new = self.ok = None
        self.goal_sample = False
        self.phase = "init"
        self._steps = self._run()

    def advance(self):
        """Execute one pseudocode step. Returns False once the path has been found."""
        if self.path is not None:
            return False
        self.phase = next(self._steps)
        return True

    @property
    def iteration_ended(self):
        return self.phase in ("discard", "done") or (self.phase == "goal_check" and not self.ok)

    def _run(self):
        while True:
            self.x_rand = self.i_near = self.x_new = self.ok = None
            self.iters += 1
            # 1. sample anywhere in the map (with goal bias)
            self.goal_sample = np.random.rand() < self.goal_bias
            if self.goal_sample:
                self.x_rand = self.goal.copy()
            else:
                self.x_rand = np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
            yield "sample"
            # 2. nearest tree node
            pts = np.array(self.nodes)
            self.i_near = int(np.argmin(np.sum((pts - self.x_rand) ** 2, axis=1)))
            yield "nearest"
            # 3. steer at most step_size toward the sample
            near = self.nodes[self.i_near]
            d = self.x_rand - near
            dist = np.linalg.norm(d)
            self.x_new = near + d / dist * min(self.step_size, dist) if dist > 1e-9 else near.copy()
            yield "steer"
            # 4. collision check: keep or discard x_new
            self.ok = dist > 1e-9 and point_free(self.x_new) and segment_free(near, self.x_new)
            yield "check"
            if not self.ok:
                yield "discard"
                continue
            # 5. grow the tree
            self.nodes.append(self.x_new)
            self.parent.append(self.i_near)
            yield "add"
            # 6. try to connect to the goal
            self.ok = bool(np.linalg.norm(self.x_new - self.goal) <= self.step_size
                           and segment_free(self.x_new, self.goal))
            yield "goal_check"
            if self.ok:
                self.nodes.append(self.goal.copy())
                self.parent.append(len(self.nodes) - 2)
                yield "connect"
                self.path = self._extract_path()
                yield "done"

    def _extract_path(self):
        path, i = [], len(self.nodes) - 1
        while i != -1:
            path.append(self.nodes[i])
            i = self.parent[i]
        return np.array(path[::-1])


# ---------------- GUI ----------------
class App:
    def __init__(self):
        self.rrt = RRT(START, GOAL)
        self.running = False
        self.credit = 0.0   # fractional steps carried between timer ticks

        self.fig = plt.figure(figsize=(14, 7.5))
        self.ax = self.fig.add_axes([0.02, 0.28, 0.62, 0.66])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)          # flip y so it looks like the image
        self.ax.set_aspect("equal")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for x0, y0, x1, y1 in WALLS:
            self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL_COLOR))

        self.tree = LineCollection([], colors="0.6", linewidths=LINE_W)
        self.ax.add_collection(self.tree)
        self.nodes_plot, = self.ax.plot([], [], "o", color="0.45", ms=NODE_MS)
        self.path_plot, = self.ax.plot([], [], "-", color="red", lw=LINE_W)
        self.start_dot = Circle(self.rrt.start, GOAL_RADIUS, color=MARKER_COLOR, zorder=5)
        self.goal_dot = Polygon(star_vertices(self.rrt.goal), color=MARKER_COLOR, zorder=5)
        self.ax.add_patch(self.start_dot); self.ax.add_patch(self.goal_dot)

        # overlays for the current pseudocode step
        self.ray, = self.ax.plot([], [], "--", color="orange", lw=LINE_W, zorder=6)
        self.rand_plot, = self.ax.plot([], [], "x", color="orange", ms=10, mew=LINE_W, zorder=7)
        self.cand_edge, = self.ax.plot([], [], "-", lw=LINE_W, zorder=6)
        self.new_plot, = self.ax.plot([], [], "o", ms=NODE_MS, zorder=8)
        self.goal_ray, = self.ax.plot([], [], "--", lw=LINE_W, zorder=6)
        self.goal_ring = Circle((0, 0), 1, fill=False, ls="--", lw=LINE_W, visible=False, zorder=6)
        self.ax.add_patch(self.goal_ring)
        self.delta_ring = Circle((0, 0), 1, fill=False, ls="--", lw=LINE_W, color="orange", visible=False, zorder=6)
        self.ax.add_patch(self.delta_ring)
        self.delta_text = self.ax.text(0, 0, "", color="darkorange", fontsize=11, fontweight="bold",
                                       ha="center", va="bottom", visible=False, zorder=9)

        # pseudocode panel
        self.ax_code = self.fig.add_axes([0.66, 0.08, 0.33, 0.86])
        self.ax_code.set_axis_off()
        self.ax_code.set_xlim(0, 1); self.ax_code.set_ylim(0, 1)
        self.line_marks = []
        lh = 0.045
        for k, line in enumerate(PSEUDO):
            y = 0.97 - k * lh
            mark = Rectangle((0, y - lh / 2), 1, lh, color="#ffe680", visible=False)
            self.ax_code.add_patch(mark)
            self.line_marks.append(mark)
            self.ax_code.text(0.02, y, line, family="monospace", fontsize=10, va="center")
        self.status = self.ax_code.text(0.02, 0.97 - len(PSEUDO) * lh - 0.03, "",
                                        fontsize=10, va="top", linespacing=1.4)

        # sliders
        self.s_step = Slider(self.fig.add_axes([0.12, 0.18, 0.45, 0.03]), "Step size Δ", 5, 80, valinit=25)
        self.s_bias = Slider(self.fig.add_axes([0.12, 0.13, 0.45, 0.03]), "Goal bias", 0.0, 0.5, valinit=0.05)
        self.s_speed = Slider(self.fig.add_axes([0.12, 0.08, 0.45, 0.03]), "Speed", 0, 3, valinit=np.log10(3))
        self.s_speed.on_changed(self.speed_label)
        self.speed_label(self.s_speed.val)

        # buttons
        self.b_run = Button(self.fig.add_axes([0.05, 0.015, 0.12, 0.045]), "Run")
        self.b_step = Button(self.fig.add_axes([0.19, 0.015, 0.12, 0.045]), "Step")
        self.b_iter = Button(self.fig.add_axes([0.33, 0.015, 0.12, 0.045]), "Iteration")
        self.b_reset = Button(self.fig.add_axes([0.47, 0.015, 0.12, 0.045]), "Reset")
        self.b_run.on_clicked(self.toggle_run)
        self.b_step.on_clicked(lambda e: self.advance(1))
        self.b_iter.on_clicked(lambda e: self.advance_iteration())
        self.b_reset.on_clicked(lambda e: self.reset())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.timer = self.fig.canvas.new_timer(interval=TICK_MS)
        self.timer.add_callback(self.on_timer)
        self.timer.start()
        self.redraw()

    def speed_label(self, v):
        self.s_speed.valtext.set_text(f"{10 ** v:.0f} steps/s")

    def sync_params(self):
        self.rrt.step_size = self.s_step.val
        self.rrt.goal_bias = self.s_bias.val

    def step_once(self):
        self.sync_params()
        if not self.rrt.advance():
            return False
        if self.rrt.path is not None:
            self.running = False
            self.b_run.label.set_text("Run")
        return True

    def advance(self, n):
        for _ in range(int(n)):
            if not self.step_once():
                break
        self.redraw()

    def advance_iteration(self):
        while self.step_once() and not self.rrt.iteration_ended:
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

    def toggle_run(self, _):
        self.running = not self.running and self.rrt.path is None
        self.credit = 1.0 if self.running else 0.0   # react immediately on Run
        self.b_run.label.set_text("Pause" if self.running else "Run")
        self.fig.canvas.draw_idle()

    def reset(self):
        self.running = False
        self.b_run.label.set_text("Run")
        self.rrt.reset()
        self.redraw()

    def on_click(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        p = np.array([e.xdata, e.ydata])
        if not point_free(p):
            return
        if e.button == 1:
            self.rrt.start = p; self.start_dot.center = p
        elif e.button == 3:
            self.rrt.goal = p; self.goal_dot.set_xy(star_vertices(p))
        self.reset()

    def redraw(self):
        r = self.rrt
        self.tree.set_segments([(r.nodes[r.parent[i]], r.nodes[i]) for i in range(1, len(r.nodes))])
        pts = np.array(r.nodes)
        self.nodes_plot.set_data(pts[:, 0], pts[:, 1])
        if r.path is not None:
            self.path_plot.set_data(r.path[:, 0], r.path[:, 1])
        else:
            self.path_plot.set_data([], [])
        self.draw_overlays()
        for k, mark in enumerate(self.line_marks):
            mark.set_visible(k in PHASE_LINES[r.phase])
        self.status.set_text(textwrap.fill(self.describe(), 48))
        self.update_title()
        self.fig.canvas.draw_idle()

    def draw_overlays(self):
        r, ph = self.rrt, self.rrt.phase
        for artist in (self.ray, self.rand_plot, self.cand_edge, self.new_plot, self.goal_ray):
            artist.set_data([], [])
        self.goal_ring.set_visible(False)
        self.delta_ring.set_visible(False)
        self.delta_text.set_visible(False)
        if ph in ("init", "connect", "done"):
            return
        near = r.nodes[r.i_near] if r.i_near is not None else None
        if ph in ("sample", "nearest", "steer", "check", "discard", "add"):
            self.rand_plot.set_data([r.x_rand[0]], [r.x_rand[1]])
        if near is not None and ph != "goal_check":
            self.ray.set_data([near[0], r.x_rand[0]], [near[1], r.x_rand[1]])
        if ph in ("steer", "check", "discard"):    # exploration: reach of Δ around x_near
            self.delta_ring.center, self.delta_ring.radius = near, r.step_size
            self.delta_ring.set_visible(True)
            self.delta_text.set_position((near[0], near[1] - r.step_size - 2))
            self.delta_text.set_text(f"Δ = {r.step_size:.0f}")
            self.delta_text.set_visible(True)
        if r.x_new is not None:
            color = {"steer": "orange", "check": "#00a000" if r.ok else "red",
                     "discard": "red"}.get(ph, "#00a000")
            self.new_plot.set_marker("X" if ph == "discard" else "o")
            self.new_plot.set_color(color)
            self.new_plot.set_data([r.x_new[0]], [r.x_new[1]])
            if ph != "goal_check":
                self.cand_edge.set_color(color)
                self.cand_edge.set_data([near[0], r.x_new[0]], [near[1], r.x_new[1]])
        if ph == "goal_check":
            color = "#00a000" if r.ok else "0.4"
            self.goal_ring.center, self.goal_ring.radius = r.x_new, r.step_size
            self.goal_ring.set_edgecolor(color)
            self.goal_ring.set_visible(True)
            self.goal_ray.set_color(color)
            self.goal_ray.set_data([r.x_new[0], r.goal[0]], [r.x_new[1], r.goal[1]])

    def describe(self):
        r, ph = self.rrt, self.rrt.phase
        if ph == "init":
            return "The tree contains only the start node. Press Step to sample a point."
        if ph == "sample":
            how = "goal bias: picked x_goal itself" if r.goal_sample else "uniform over the whole map"
            return f"Sampled x_rand = ({r.x_rand[0]:.0f}, {r.x_rand[1]:.0f}) ({how}). " \
                   "It may lie inside a wall: it only gives a direction to grow."
        if ph == "nearest":
            n = r.nodes[r.i_near]
            return f"Nearest tree node to x_rand: node #{r.i_near} at ({n[0]:.0f}, {n[1]:.0f})."
        if ph == "steer":
            return f"Move from x_near toward x_rand by at most Δ = {r.step_size:.0f} px " \
                   "(dashed circle). That gives the candidate x_new."
        if ph == "check":
            return "x_new and the edge x_near → x_new are clear of walls: keep x_new." if r.ok \
                else "x_new or the edge x_near → x_new hits a wall: reject x_new."
        if ph == "discard":
            return "x_new discarded; the tree is unchanged. Next: sample again."
        if ph == "add":
            return f"x_new added to the tree with parent x_near (now {len(r.nodes)} nodes)."
        if ph == "goal_check":
            if r.ok:
                return "The goal is within Δ of x_new and visible: connect it!"
            d = np.linalg.norm(r.x_new - r.goal)
            why = "but blocked by a wall" if d <= r.step_size else f"> Δ = {r.step_size:.0f} px"
            return f"The goal is {d:.0f} px from x_new ({why}). Next iteration."
        if ph == "connect":
            return "x_goal added to the tree with parent x_new."
        length = np.sum(np.linalg.norm(np.diff(r.path, axis=0), axis=1))
        return f"Done: follow parents back from the goal. Path length {length:.0f} px " \
               f"after {r.iters} iterations."

    def update_title(self):
        msg = f"RRT   iteration: {self.rrt.iters}   nodes: {len(self.rrt.nodes)}"
        if self.rrt.path is not None:
            length = np.sum(np.linalg.norm(np.diff(self.rrt.path, axis=0), axis=1))
            msg += f"   path found, length = {length:.0f} px"
        self.ax.set_title(msg)


if __name__ == "__main__":
    app = App()
    plt.show()
