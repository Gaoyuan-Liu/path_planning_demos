"""
Interactive Dijkstra demo on the example map, with step-by-step pseudocode.

Same grid, controls and colors as astar_demo.py, but the OPEN cell with the smallest
g (cost from the start) is expanded next: no heuristic, so the search grows evenly in
all directions instead of being pulled toward the goal.

Controls
  Run / Pause : execute pseudocode steps continuously (rate set by "Speed")
  Step        : execute one pseudocode step (pick s, close s, look at a neighbor, ...)
  Expand      : execute steps until the current cell s has been fully expanded
  Reset       : restart the search (keeps start/goal)
  Left click  : move start (green dot)
  Right click : move goal (green star)
  Sliders     : cell size, speed

The pseudocode panel shows dijkstra.png, built from overleaf/dijkstra.tex.
"""
import os

import matplotlib.pyplot as plt

from astar_demo import AStar, App, FIGURES

# same lines as astar.png except 1 and 4, so the A* PHASE_LINES apply unchanged
PSEUDO_IMAGE = os.path.join(FIGURES, "dijkstra.png")


class Dijkstra(AStar):
    """A* without the heuristic: the priority is g alone, ties broken by insertion order."""

    def f(self, s):
        return self.g[s]

    def _key(self, s):
        return (self.g[s],)


class DijkstraApp(App):
    NAME = "Dijkstra"
    SEARCH_CLASS = Dijkstra
    PSEUDO_IMAGE = PSEUDO_IMAGE

    def __init__(self):
        super().__init__()
        self.s_w.ax.set_visible(False)   # no heuristic weight in Dijkstra

    def describe(self):
        a, ph = self.astar, self.astar.phase
        if ph == "pop":
            return f"Pick the OPEN cell with the smallest $g$: $s=({a.s[0]}, {a.s[1]})$ with " \
                   f"$g={a.g[a.s]:.0f}$. Where the goal is plays no role."
        return super().describe()


if __name__ == "__main__":
    app = DijkstraApp()
    plt.show()
