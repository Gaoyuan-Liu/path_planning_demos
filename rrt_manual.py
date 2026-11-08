"""
RRT demo where YOU pick each sample point q_rand with the mouse.

Controls
  Left click        : choose q_rand anywhere in the map (walls included)
  Step              : show the next step of the iteration: nearest node,
                      Δ circle, candidate q_new, accept/reject, add, goal check.
                      At the sampling step without a click, q_rand is chosen at random.
  Iteration         : run the rest of the iteration (or a whole one with a random
                      q_rand if no point was clicked)
  Run / Pause       : auto-play the steps instead (rate set by "Speed")
  Shift + left click: move start (green dot)
  Right click       : move goal (green star)
  Reset             : clear the tree (keeps start/goal)
  Capture           : save the map panel only as a PNG in captures/

Reuses the algorithm and GUI from rrt_demo.py. The pseudocode panel shows
method_manual.png (built from overleaf/method_manual.tex): method.png with line 4 changed.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from rrt_demo import W, H, RRT, App, QR, point_free, star_vertices

# same 16 lines as method.png, so the inherited METHOD_LINES highlighting applies unchanged
MANUAL_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "method_manual.png")


class ManualRRT(RRT):
    """RRT that samples the clicked point, or a random one when nothing was clicked."""
    reach_step = True

    def reset(self):
        self.clicked = None
        super().reset()

    @property
    def awaiting_click(self):
        return self.path is None and (self.phase == "init" or self.iteration_ended)

    def _sample(self):
        self.goal_sample = False
        self.random_sample = self.clicked is None
        if self.random_sample:   # no click: uniform sample, no goal bias
            return np.array([np.random.uniform(0, W), np.random.uniform(0, H)])
        p, self.clicked = self.clicked, None
        return p


class ManualApp(App):
    RRT_CLASS = ManualRRT
    PSEUDO_IMAGE = MANUAL_IMAGE
    CAPTURE_PREFIX = "rrt_manual"

    def __init__(self):
        super().__init__()
        self.s_bias.ax.set_visible(False)   # goal bias has no meaning when the user samples

    def on_click(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        p = np.array([e.xdata, e.ydata])
        if e.button == 1 and not e.key == "shift":
            # a click during an iteration is kept and used as the next x_rand
            self.rrt.clicked = p
            if self.rrt.awaiting_click:
                self.credit = 0.0
                self.advance(1)          # show x_rand right away
            return
        if not point_free(p):
            return
        if e.button == 1:
            self.rrt.start = p; self.start_dot.center = p
        elif e.button == 3:
            self.rrt.goal = p; self.goal_dot.set_xy(star_vertices(p))
        self.reset()

    def describe(self):
        r, ph = self.rrt, self.rrt.phase
        if ph == "init":
            return f"The tree contains only the start node. Click anywhere on the map to choose {QR}, " \
                   "or press Step to sample it at random."
        if ph == "sample":
            who = "No click, so a random" if r.random_sample else "You chose"
            return f"{who} {QR} = ({r.x_rand[0]:.0f}, {r.x_rand[1]:.0f}). " \
                   "It may lie inside a wall: it only gives a direction to grow. Press Step."
        msg = super().describe()
        if r.awaiting_click:
            msg += f" Click to choose the next {QR}, or press Step for a random one."
        return msg


if __name__ == "__main__":
    app = ManualApp()
    plt.show()
