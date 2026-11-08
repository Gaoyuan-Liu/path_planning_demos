"""
Random-walk "planner" on the example map: the extreme baseline.

The walker takes a step of length Δ in a uniformly random direction; a step that
would hit a wall is refused and the walker tries again. It stops once the goal is
within Δ and visible. There is no memory, no tree and no sense of where the goal
is, so the resulting path is the whole wandering trajectory.

Controls
  Run / Pause : execute pseudocode steps continuously (rate set by "Speed", up to 10000/s)
  Step        : execute one pseudocode step (direction, candidate, check, move, goal check)
  Iteration   : execute steps until the current step of the walk ends
  Reset       : back to the start
  Capture     : save the map panel only as a PNG in captures/
  Left click  : move start (green dot)
  Right click : move goal (green star)

Reuses the map, GUI and figure panel from rrt_demo.py. The panel shows
randomwalk.png, built from overleaf/randomwalk.tex.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # rrt_demo.py is one folder up

from rrt_demo import RRT, App, QS, QG, QW, DELTA, point_free, segment_free

PSEUDO_IMAGE = os.path.join(ROOT, "figures", "randomwalk.png")
WALK_LINES = {  # phase -> highlighted lines of randomwalk.png (line k is numbered k + 1 there)
    "init": [1],
    "reach": [3],
    "steer": [4],
    "check": [5],
    "discard": [6],
    "add": [8],
    "goal_check": [9],
    "connect": [10],
    "done": [10],
}
Q, THETA = "$q$", r"$\theta$"


class RandomWalk(RRT):
    """The walk is stored as an RRT whose every node is the child of the previous one.

    i_near is always the walker's current position (the last node); x_rand is the
    point Δ away in the random direction, used only to draw that direction.
    """

    def _run(self):
        while True:
            self.x_rand = self.x_new = self.ok = None
            self.iters += 1
            self.i_near = len(self.nodes) - 1
            q = self.nodes[self.i_near]
            # 1. random direction
            self.theta = np.random.uniform(0, 2 * np.pi)
            self.x_rand = q + self.step_size * np.array([np.cos(self.theta), np.sin(self.theta)])
            yield "reach"
            # 2. candidate position one step of length Δ away
            self.x_new = self.x_rand.copy()
            yield "steer"
            # 3. refuse moves into walls
            self.ok = point_free(self.x_new) and segment_free(q, self.x_new)
            yield "check"
            if not self.ok:
                yield "discard"
                continue
            # 4. move
            self.nodes.append(self.x_new)
            self.parent.append(self.i_near)
            yield "add"
            # 5. stop when the goal is within reach
            self.ok = bool(np.linalg.norm(self.x_new - self.goal) <= self.step_size
                           and segment_free(self.x_new, self.goal))
            yield "goal_check"
            if self.ok:
                self.nodes.append(self.goal.copy())
                self.parent.append(len(self.nodes) - 2)
                yield "connect"
                self.path = np.array(self.nodes)   # the whole walk
                yield "done"


class RandomWalkApp(App):
    RRT_CLASS = RandomWalk
    PSEUDO_IMAGE = PSEUDO_IMAGE
    CAPTURE_PREFIX = "randomwalk"
    PHASE_LINES = WALK_LINES
    SHOW_RAND = False
    SPEED_MAX_LOG = 4

    def __init__(self):
        super().__init__()
        self.s_bias.ax.set_visible(False)   # no sampling, so no goal bias

    def draw_tree(self):
        # one polyline for the whole walk, and a dot only where the walker stands
        pts = np.array(self.rrt.nodes)
        self.tree.set_segments([pts])
        self.nodes_plot.set_data(pts[-1:, 0], pts[-1:, 1])

    def draw_overlays(self):
        super().draw_overlays()
        self.delta_ring.set_visible(False)   # no Δ circle: the walker steps exactly Δ in one direction

    def describe(self):
        r, ph = self.rrt, self.rrt.phase
        step = f"{DELTA} = {r.step_size:.0f} px"
        if ph == "init":
            return f"The walker stands at {QS}. Press Step to pick a random direction."
        if ph == "reach":
            deg = np.degrees(r.theta) % 360
            return f"Random direction {THETA} = {deg:.0f} degrees (dashed line). " \
                   f"Each move has length {step}."
        if ph == "steer":
            return f"Candidate {QW}: one step of length {DELTA} from {Q} in direction {THETA}."
        if ph == "check":
            return f"The move from {Q} to {QW} is clear of walls." if r.ok \
                else f"{QW} or the move from {Q} to {QW} hits a wall."
        if ph == "discard":
            return f"Continue: the walker stays at {Q} and tries another random direction."
        if ph == "add":
            return f"The walker moves to {QW}. $P$ now has {len(r.nodes)} points."
        if ph == "goal_check":
            if r.ok:
                return f"{QG} is within {DELTA} and visible: step onto it!"
            d = np.linalg.norm(r.x_new - r.goal)
            why = "but blocked by a wall" if d <= r.step_size else f"more than {step}"
            return f"{QG} is {d:.0f} px away ({why}). Keep walking."
        if ph == "connect":
            return f"{QG} appended to $P$."
        moves = len(r.nodes) - 2
        length = np.sum(np.linalg.norm(np.diff(r.path, axis=0), axis=1))
        return f"Goal reached after {r.iters} tries ({moves} moves, {r.iters - moves} refused). " \
               f"The path $P$ is the whole walk, every detour and loop included: {length:.0f} px long."

    def update_title(self):
        r = self.rrt
        moves = len(r.nodes) - (2 if r.path is not None else 1)
        msg = f"Random walk   tries: {r.iters}   moves: {moves}"
        if r.path is not None:
            length = np.sum(np.linalg.norm(np.diff(r.path, axis=0), axis=1))
            msg += f"   goal reached, path length = {length:.0f} px"
        self.ax.set_title(msg)


if __name__ == "__main__":
    app = RandomWalkApp()
    plt.show()
