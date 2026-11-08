"""
RRT demo where YOU add obstacles to the map, then plan.

The map starts with the same walls as the other demos. Everything else works as in
rrt_manual.py: click to choose q_rand, or let Step / Iteration / Run sample at random.

Controls
  Left drag         : draw a rectangular block (restarts the tree)
  Right click block : remove that block (restarts the tree)
  Clear             : remove all drawn blocks (restarts the tree)
  Left click        : choose q_rand anywhere in the map (walls included)
  Step              : show the next step of the iteration
  Iteration         : run the rest of the iteration
  Run / Pause       : auto-play the steps (rate set by "Speed")
  Shift + left click: move start (green dot)
  Right click       : move goal (green star), when not on a block
  Reset             : clear the tree (keeps the blocks, start and goal)
  Capture           : save the map panel only as a PNG in captures/

A block that would cover the start or the goal is refused.

Reuses rrt_manual.py. The collision checks of rrt_demo.py read its WALLS / INFLATED
lists, so this demo edits those lists in place.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from rrt_demo import WALLS, INFLATED, MARGIN, WALL_COLOR, QR
from rrt_manual import ManualApp

MAP_WALLS = WALLS[:]   # the walls of the map, which stay in place
DRAG_MIN = 8         # a press that moves less than this (pixels of the map) is a click
BLOCK_MIN = 12       # thinnest block side (pixels of the map)


def inflate(rect):
    x0, y0, x1, y1 = rect
    return x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN


def contains(rect, p):
    x0, y0, x1, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def block_rect(p, q):
    """Rectangle with corners p and q, widened around its center up to BLOCK_MIN."""
    lo, hi = np.minimum(p, q), np.maximum(p, q)
    grow = np.maximum(BLOCK_MIN - (hi - lo), 0) / 2
    return (*(lo - grow), *(hi + grow))


class DrawApp(ManualApp):
    CAPTURE_PREFIX = "rrt_draw"

    def __init__(self):
        self.blocks = []       # (rect, patch) of each drawn block
        self.press = None      # left-button press that may become a drag
        self.drag_end = None
        self.notice = ""       # one-off message added to the explanation text
        self.update_walls()
        super().__init__()

        self.preview = Rectangle((0, 0), 0, 0, color=WALL_COLOR, alpha=0.5, visible=False, zorder=4)
        self.ax.add_patch(self.preview)
        self.b_clear = Button(self.fig.add_axes([0.5, 0.015, 0.1, 0.045]), "Clear")
        self.b_clear.on_clicked(lambda e: self.clear_blocks())
        for i, b in enumerate((self.b_run, self.b_step, self.b_iter, self.b_reset, self.b_capture, self.b_clear)):
            b.ax.set_position([0.05 + 0.092 * i, 0.015, 0.083, 0.045])
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    def update_walls(self):
        WALLS[:] = MAP_WALLS + [rect for rect, _ in self.blocks]
        INFLATED[:] = [inflate(rect) for rect in WALLS]

    def block_allowed(self, rect):
        return not (contains(inflate(rect), self.rrt.start) or contains(inflate(rect), self.rrt.goal))

    def add_block(self, rect):
        if not self.block_allowed(rect):
            self.notice = "That block would cover the start or the goal, so it was not added."
            self.redraw()
            self.notice = ""
            return
        x0, y0, x1, y1 = rect
        patch = Rectangle((x0, y0), x1 - x0, y1 - y0, color=WALL_COLOR)
        self.ax.add_patch(patch)
        self.blocks.append((rect, patch))
        self.update_walls()
        self.reset()

    def remove_block(self, block):
        block[1].remove()
        self.blocks.remove(block)
        self.update_walls()
        self.reset()

    def clear_blocks(self):
        for _, patch in self.blocks:
            patch.remove()
        self.blocks.clear()
        self.update_walls()
        self.reset()

    def on_click(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        p = np.array([e.xdata, e.ydata])
        if e.button == 1 and e.key != "shift":
            self.press, self.drag_end = e, p   # click or drag: decided on release
            return
        if e.button == 3:
            for block in reversed(self.blocks):   # the most recent block is on top
                if contains(inflate(block[0]), p):
                    self.remove_block(block)
                    return
        super().on_click(e)

    def on_motion(self, e):
        if self.press is None or e.inaxes is not self.ax or e.xdata is None:
            return
        self.drag_end = np.array([e.xdata, e.ydata])
        start = np.array([self.press.xdata, self.press.ydata])
        if np.abs(self.drag_end - start).max() >= DRAG_MIN:
            x0, y0, x1, y1 = block_rect(start, self.drag_end)
            self.preview.set_bounds(x0, y0, x1 - x0, y1 - y0)
            self.preview.set_color(WALL_COLOR if self.block_allowed((x0, y0, x1, y1)) else "red")
            self.preview.set_visible(True)
            self.fig.canvas.draw_idle()

    def on_release(self, e):
        if self.press is None:
            return
        press, self.press = self.press, None
        self.preview.set_visible(False)
        start = np.array([press.xdata, press.ydata])
        if np.abs(self.drag_end - start).max() >= DRAG_MIN:
            self.add_block(block_rect(start, self.drag_end))
        else:
            super().on_click(press)
        self.fig.canvas.draw_idle()

    def describe(self):
        r = self.rrt
        if r.phase == "init":
            msg = f"Drag on the map to draw blocks. Then click to choose {QR}, or press Step to sample it at random."
        else:
            msg = super().describe()
        return f"{msg} {self.notice}" if self.notice else msg


if __name__ == "__main__":
    app = DrawApp()
    plt.show()
