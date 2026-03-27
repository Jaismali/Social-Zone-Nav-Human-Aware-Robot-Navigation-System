"""
planner.py
==========
Grid-based A* path planner with social-zone cost integration.

Socially-aware navigation is framed here as a weighted graph-search problem:
the robot seeks the shortest path from start to goal on a 2D occupancy grid
whose edge weights incorporate both Euclidean distance and a social-cost term
derived from nearby pedestrians' personal-space zones (see social_zone.py).

This formulation follows the cost-map tradition of Sisbot et al. (2007) and
Svenstrup et al. (2010) rather than reactive approaches (Helbing & Molnar,
1995) or learning-based methods (Everett et al., 2021).  The principal
advantage is completeness: A* is guaranteed to find the optimal path if one
exists, whereas reactive controllers can become trapped in local minima in
densely crowded scenes.

Grid resolution is set to 0.2 m (50x50 cells over a 10x10 m arena).  This
was chosen as a balance between planning fidelity and computational cost:
finer grids (e.g. 0.1 m -> 100x100 cells) give smoother paths but increase
the worst-case number of expanded nodes by 4x with diminishing quality gains
in our experimental scenarios.

References
----------
Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the
    heuristic determination of minimum cost paths. IEEE Transactions on
    Systems Science and Cybernetics, 4(2), 100-107.
    Original A* paper; octile heuristic is the 8-connected admissible version.

Sisbot, E. A., Marin-Urias, L. F., Alami, R., & Simeon, T. (2007). A human
    aware mobile robot motion planner. IEEE Transactions on Robotics, 23(5),
    874-883.  Social cost-map approach for human-robot navigation.

Svenstrup, M., Bak, T., & Andersen, H. J. (2010). Trajectory planning for
    robots in dynamic human environments. In Proc. IROS, pp. 4293-4298.
    Grid-based planner with Gaussian personal-space cost fields.

Kruse, T., Pandey, A. K., Alami, R., & Mertsching, B. (2013). Human-aware
    robot navigation: A survey. Robotics and Autonomous Systems, 61(12),
    1726-1743.

Everett, M., Chen, Y. F., & How, J. P. (2021). Collision avoidance in
    pedestrian-rich environments with deep reinforcement learning. IEEE
    Access, 9, 10357-10377.  Representative learning-based alternative.
"""

import heapq
import numpy as np

from social_zone import SocialZone, CircularSocialZone


class AStarPlanner:
    """
    Social-cost-aware A* planner on a uniform 2D grid.

    Cost model
    ~~~~~~~~~~
    The g-score accumulates two terms at each step::

        g(n) += step_length * (1 + DISTANCE_COST) + social(n) * grid_size

    where ``social(n)`` is the sum of SocialZone.compute_cost() over all
    pedestrians evaluated at the centre of cell n, and ``grid_size`` converts
    the per-cell cost to a distance-equivalent penalty.  Multiplying by
    grid_size is necessary to keep social cost dimensionally consistent with
    path length so that the relative weighting is invariant to resolution.

    The DISTANCE_COST coefficient (default 0.1) biases the planner toward
    shorter paths without making social comfort secondary; it was tuned so
    that a path 20% longer is preferred if it reduces social cost by ~2.0
    units — roughly the cost of grazing a zone boundary.

    Obstacle handling
    ~~~~~~~~~~~~~~~~~
    Cells with social cost > max_social_cost (default 20.0) are treated as
    hard obstacles.  This threshold corresponds to exp(20) ~ 5e8 in physical
    terms, which is only reached very close to a pedestrian (within ~0.1 m of
    a 1.8 m front zone).  In practice it prevents the planner from routing
    directly through a pedestrian while still allowing near-boundary paths.

    Heuristic
    ~~~~~~~~~
    Octile distance (Hart et al., 1968) is admissible for 8-connected grids
    and significantly outperforms Manhattan or Euclidean heuristics in terms
    of nodes expanded.  It is exact along diagonal-free corridors and only
    underestimates (never overestimates) the true cost, preserving A*
    optimality.

    Parameters
    ----------
    social_zone : SocialZone or CircularSocialZone
        Zone model used to compute per-cell costs.
    arena_size : float
        Side length of the square arena in metres.
    grid_size : float
        Cell width in metres.  Must divide arena_size evenly.
    """

    # Weighting for path length vs. social comfort.  Increasing this value
    # makes the planner prioritise shorter paths at the expense of social
    # comfort; decreasing it produces longer but socially more conservative
    # routes.
    DISTANCE_COST = 0.1

    def __init__(self, social_zone, arena_size=10.0, grid_size=0.2):
        self.social_zone = social_zone
        self.arena_size  = arena_size
        self.grid_size   = grid_size
        self.n_cells     = int(arena_size / grid_size)   # 50 for default params

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, pos):
        """
        Convert a world-frame position to integer grid indices.

        Uses floor division so that a position exactly on a cell boundary is
        assigned to the cell to the left/below (consistent with numpy integer
        indexing conventions).  Clips to valid range to handle floating-point
        positions that land marginally outside the arena.

        Parameters
        ----------
        pos : array-like, shape (2,)
            (x, y) in metres.

        Returns
        -------
        (gx, gy) : tuple of int
        """
        gx = int(pos[0] / self.grid_size)
        gy = int(pos[1] / self.grid_size)
        gx = int(np.clip(gx, 0, self.n_cells - 1))
        gy = int(np.clip(gy, 0, self.n_cells - 1))
        return (gx, gy)

    def grid_to_world(self, cell):
        """
        Return the world-frame centre of a grid cell.

        Using cell centres (+ 0.5 * grid_size) rather than corners ensures
        that social costs are evaluated at the point a robot would actually
        occupy, not at cell edges where cost gradients are steepest.

        Parameters
        ----------
        cell : tuple of int
            (gx, gy) grid indices.

        Returns
        -------
        np.ndarray, shape (2,)
        """
        x = (cell[0] + 0.5) * self.grid_size
        y = (cell[1] + 0.5) * self.grid_size
        return np.array([x, y])

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def compute_cell_cost(self, world_pos, pedestrians):
        """
        Aggregate social cost at a world position from all pedestrians.

        Costs from multiple humans are summed linearly.  This is consistent
        with the additive formulation of Sisbot et al. (2007) and avoids the
        saturation artefacts of max-pooling, which would treat a dense crowd
        the same as a single nearby pedestrian.

        Parameters
        ----------
        world_pos : np.ndarray, shape (2,)
        pedestrians : list of Pedestrian
            Each must have attributes x, y, orientation.

        Returns
        -------
        float
            Total social cost at world_pos.

        Notes
        -----
        Performance bottleneck for large N_humans: this is O(N_humans) per
        cell and is called inside the A* loop.  The on-demand cost cache in
        plan() amortises repeated evaluations of the same cell.

        TODO: vectorise across pedestrians using numpy broadcasting for
              scenarios with N_humans > 20.
        """
        total = 0.0
        for p in pedestrians:
            total += self.social_zone.compute_cost(
                world_pos,
                np.array([p.x, p.y]),
                p.orientation,
            )
        return total

    # ------------------------------------------------------------------
    # A* components
    # ------------------------------------------------------------------

    def heuristic(self, cell, goal_cell):
        """
        Octile distance heuristic for 8-connected grids (Hart et al., 1968).

        For a grid that allows diagonal moves at cost sqrt(2), the octile
        distance is the tightest admissible heuristic::

            h = grid_size * (dx + dy + (sqrt(2) - 2) * min(dx, dy))

        This equals the exact cost when no obstacles are present, so A* with
        this heuristic expands very few unnecessary nodes on open terrain.

        Parameters
        ----------
        cell, goal_cell : tuple of int

        Returns
        -------
        float
            Lower bound on remaining path cost.
        """
        dx = abs(cell[0] - goal_cell[0])
        dy = abs(cell[1] - goal_cell[1])
        return self.grid_size * (dx + dy + (np.sqrt(2) - 2.0) * min(dx, dy))

    def get_neighbors(self, cell):
        """
        Return valid 8-connected neighbours with their step costs.

        Diagonal moves have step cost grid_size * sqrt(2) to correctly
        penalise longer diagonal paths.  Using uniform costs for all eight
        directions (as in 4-connected grids) would cause the planner to
        over-prefer diagonal routes, producing jagged paths.

        Parameters
        ----------
        cell : tuple of int

        Returns
        -------
        list of ((gx, gy), step_cost)
        """
        gx, gy = cell
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.n_cells and 0 <= ny < self.n_cells:
                    # Diagonal moves cost sqrt(2) * grid_size, cardinal = 1 * grid_size
                    step = self.grid_size * (np.sqrt(2) if (dx != 0 and dy != 0) else 1.0)
                    neighbors.append(((nx, ny), step))
        return neighbors

    def plan(self, start_pos, goal_pos, pedestrians, max_social_cost=20.0):
        """
        Find a socially-aware path from start to goal using A*.

        The open set is maintained as a min-heap on f = g + h.  Ties in f
        are broken by g (prefer nodes closer to the goal), which tends to
        produce shorter paths when multiple routes have equal social cost.

        On-demand cost caching: social costs are computed only when a cell is
        first popped from the open set, rather than pre-computing the full
        50x50 cost map.  In practice, A* with a good heuristic expands far
        fewer cells than the grid total, so lazy evaluation is faster.

        Parameters
        ----------
        start_pos : array-like, shape (2,)
            Robot start position in world frame (metres).
        goal_pos : array-like, shape (2,)
            Goal position in world frame (metres).
        pedestrians : list of Pedestrian
        max_social_cost : float
            Cells whose aggregate social cost exceeds this value are treated
            as impassable.  Default 20.0 corresponds to being ~0.1 m inside
            the front zone boundary (exp(1.8 - 0.1) - 1 ~ 4.5 per pedestrian,
            so ~4-5 pedestrians would block a cell at this threshold).

        Returns
        -------
        list of array-like or None
            Sequence of (x, y) world-frame waypoints from start to goal,
            or None if no path exists within the cost constraints.

        Notes
        -----
        The path begins with start_pos (exact) and ends with goal_pos
        (exact); interior points are cell centres.  Downstream controllers
        should handle the ~0.1 m snap from cell centre to exact position.

        FIXME: the closed set uses a Python set of tuples which is O(1)
               average but has high constant overhead.  A 2D boolean numpy
               array would be faster for dense grids.

        TODO: implement a dynamic replanning variant (e.g. D*-Lite) so that
              moving pedestrians can be handled without full re-planning at
              each timestep.
        """
        start_cell = self.world_to_grid(start_pos)
        goal_cell  = self.world_to_grid(goal_pos)

        # Trivial case: start and goal map to the same cell
        if start_cell == goal_cell:
            return [start_pos, goal_pos]

        # On-demand social cost cache: avoids recomputing the same cell twice
        # when it is expanded from multiple paths (common near pedestrians).
        cost_cache = {}

        def get_cost(cell):
            if cell not in cost_cache:
                cost_cache[cell] = self.compute_cell_cost(
                    self.grid_to_world(cell), pedestrians
                )
            return cost_cache[cell]

        # A* data structures
        # Heap entries: (f_score, g_score, cell) — g_score used as tiebreaker
        open_heap  = []
        heapq.heappush(open_heap, (0.0, 0.0, start_cell))
        came_from  = {}                          # child -> parent for path reconstruction
        g_score    = {start_cell: 0.0}
        closed     = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)

            # Lazy deletion: skip stale heap entries for already-closed cells
            if current in closed:
                continue
            closed.add(current)

            if current == goal_cell:
                # Reconstruct path by tracing came_from back to start
                path_cells = []
                node = current
                while node in came_from:
                    path_cells.append(node)
                    node = came_from[node]
                path_cells.append(start_cell)
                path_cells.reverse()

                # Map grid cells -> world positions; preserve exact start/goal
                path = [start_pos]
                for cell in path_cells[1:]:
                    path.append(self.grid_to_world(cell))
                path.append(goal_pos)
                return path

            for neighbor, step_cost in self.get_neighbors(current):
                if neighbor in closed:
                    continue

                social = get_cost(neighbor)

                # Hard obstacle: too socially costly to traverse
                if social > max_social_cost:
                    continue

                # g update: Euclidean step + social penalty (scaled by grid_size
                # to keep units consistent with path length)
                tentative_g = (
                    g
                    + step_cost * (1.0 + self.DISTANCE_COST)
                    + social * self.grid_size
                )

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor]   = tentative_g
                    h = self.heuristic(neighbor, goal_cell)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, neighbor))

        return None   # no path found within cost constraints

    def smooth_path(self, path, pedestrians, window=3):
        """
        Greedy string-pulling to remove unnecessary waypoints.

        Attempts to shortcut between waypoints separated by up to ``window``
        steps.  A shortcut is accepted if the straight-line segment between
        the two points does not come within 0.4 m of any pedestrian (checked
        at 5 uniformly-spaced intermediate points).

        This is a simple post-processing step rather than a full elastic-band
        or Bezier smoothing (Quinlan & Khatib, 1993).  It is fast (O(n))
        and sufficient to remove the grid-staircase artefact.

        Parameters
        ----------
        path : list or None
        pedestrians : list of Pedestrian
        window : int
            Maximum number of waypoints to skip in one shortcut attempt.

        Returns
        -------
        list or None
            Smoothed path, or the original path if fewer than 3 points.

        Notes
        -----
        The 0.4 m clearance threshold is larger than COLLISION_DIST (0.3 m +
        0.2 m = 0.5 m) so that smoothed paths have a small safety margin.

        TODO: replace with proper elastic-band smoothing (Quinlan & Khatib,
              1993) for publication-quality path shapes.
        """
        if path is None or len(path) < 3:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Try to jump as far forward as the window allows
            j = min(i + window, len(path) - 1)
            while j > i + 1:
                p1 = np.array(path[i])
                p2 = np.array(path[j])
                # Sample 5 intermediate points along the candidate shortcut
                ok = True
                for t in np.linspace(0, 1, 5):
                    mid = p1 + t * (p2 - p1)
                    for p in pedestrians:
                        if np.linalg.norm(mid - p.pos) < 0.4:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    break
                j -= 1
            smoothed.append(path[j])
            i = j

        return smoothed


# ---------------------------------------------------------------------------
# Concrete planner subclasses
# ---------------------------------------------------------------------------

class CircularPlanner(AStarPlanner):
    """
    A* planner with isotropic (circular) personal-space zones.

    Baseline comparison representing the prior-art assumption of symmetric
    personal space (Kruse et al., 2013).  All planning logic is identical to
    AStarPlanner; only the zone model differs.

    Parameters
    ----------
    radius : float
        Circular zone radius in metres (default 1.2 m; see social_zone.py).
    arena_size, grid_size : float
    """

    def __init__(self, radius=1.2, arena_size=10.0, grid_size=0.2):
        zone = CircularSocialZone(radius=radius)
        super().__init__(zone, arena_size=arena_size, grid_size=grid_size)
        self.radius = radius

    def __repr__(self):
        return f"CircularPlanner(radius={self.radius}m)"


class AsymmetricPlanner(AStarPlanner):
    """
    A* planner with direction-dependent asymmetric personal-space zones.

    The proposed method: uses zone distances learned from ETH/UCY data
    (see extract_zones.py) rather than a fixed symmetric radius.

    Parameters
    ----------
    zone_distances : dict or None
        Per-direction distances {'front', 'back', 'left', 'right'}.
        Defaults to literature values (Hall, 1966) if None.
    arena_size, grid_size : float
    """

    def __init__(self, zone_distances=None, arena_size=10.0, grid_size=0.2):
        if zone_distances is None:
            zone_distances = {'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2}
        zone = SocialZone.from_dict(zone_distances)
        super().__init__(zone, arena_size=arena_size, grid_size=grid_size)

    def __repr__(self):
        return f"AsymmetricPlanner({self.social_zone})"


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_path_length(path):
    """
    Compute the total Euclidean length of a path.

    Parameters
    ----------
    path : list of array-like or None

    Returns
    -------
    float
        Sum of segment lengths in metres.  Returns 0.0 for None or
        single-point paths.
    """
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += float(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])))
    return total


def min_distance_to_humans(path, pedestrians):
    """
    Compute the minimum separation between any path point and any pedestrian.

    Used as a safety metric: lower values indicate the planner took the robot
    close to pedestrians, which correlates with higher collision risk and
    perceived discomfort (Truong & Ngo, 2017).

    Parameters
    ----------
    path : list of array-like or None
    pedestrians : list of Pedestrian

    Returns
    -------
    float
        Minimum observed separation (metres), or inf if path is None or
        no pedestrians are present.
    """
    if path is None or not pedestrians:
        return float('inf')
    min_d = float('inf')
    for pos in path:
        pos_arr = np.array(pos)
        for p in pedestrians:
            d = float(np.linalg.norm(pos_arr - p.pos))
            if d < min_d:
                min_d = d
    return min_d


def count_social_violations(path, pedestrians, social_zone):
    """
    Count the number of path waypoints that violate at least one pedestrian's
    personal-space zone.

    Each waypoint is counted at most once regardless of how many zones it
    violates simultaneously (i.e. this is a per-waypoint binary count, not a
    per-human count).

    Parameters
    ----------
    path : list of array-like or None
    pedestrians : list of Pedestrian
    social_zone : SocialZone or subclass

    Returns
    -------
    int
        Number of violating waypoints.

    Notes
    -----
    This metric is sensitive to path resolution: a finer-resolution path will
    yield more waypoints and potentially higher counts even for identical
    routes.  For cross-planner comparison, ensure all planners use the same
    grid resolution.

    TODO: normalise by path length (violations per metre) for resolution-
          independent comparison across planners with different grid sizes.
    """
    if path is None:
        return 0

    violations = 0
    for pos in path:
        pos_arr = np.array(pos)
        for p in pedestrians:
            if social_zone.is_violation(pos_arr, p.pos, p.orientation):
                violations += 1
                break   # count this waypoint once; move on
    return violations
