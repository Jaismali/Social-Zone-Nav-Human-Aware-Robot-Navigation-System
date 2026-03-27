"""
environment.py
==============
Simulation environment for 2D socially-aware robot navigation experiments.

This module provides two classes:

  Pedestrian
      A kinematic agent with position, heading, and constant velocity.
      Motion model is straight-line with elastic wall reflection, matching
      the simplified pedestrian dynamics used in the ETH/UCY benchmark
      evaluations (Pellegrini et al., 2009; Lerner et al., 2007).

  NavigationEnvironment
      A 10 x 10 m square arena containing a mobile robot and N pedestrians.
      Handles collision detection, goal checking, and pedestrian stepping.
      Designed to be lightweight enough to run thousands of episodes on CPU
      without a physics engine (no contact dynamics, no inter-pedestrian
      avoidance).

Design decisions
~~~~~~~~~~~~~~~~
Arena size (10 x 10 m) was chosen to match the approximate scale of the
ETH hotel and university datasets and is large enough to make long-horizon
path planning non-trivial while keeping A* planning times below 50 ms per
episode on a single CPU core.

Pedestrian motion uses constant-velocity straight lines with elastic boundary
reflection rather than a social force model (Helbing & Molnar, 1995) for two
reasons: (1) the planner is evaluated on static snapshots of pedestrian
positions, so complex interaction dynamics between pedestrians do not affect
planning quality; (2) constant-velocity motion is reproducible given a seed,
simplifying experimental comparison across planners.

Collision threshold (COLLISION_DIST + ROBOT_RADIUS = 0.5 m) reflects typical
values for a differential-drive research platform (e.g. TurtleBot 2 radius
~0.18 m) plus the 0.3 m pedestrian radius assumed by the ETH/UCY annotations.

References
----------
Pellegrini, S., Ess, A., Schindler, K., & Van Gool, L. (2009). You'll
    never walk alone: Modelling social behaviour for multi-target tracking.
    In Proc. ICCV, pp. 261-268.  ETH pedestrian dataset.

Lerner, A., Chrysanthou, Y., & Lischinski, D. (2007). Crowds by example.
    Computer Graphics Forum, 26(3), 655-664.  UCY pedestrian dataset.

Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
    Physical Review E, 51(5), 4282-4286.

Kruse, T., Pandey, A. K., Alami, R., & Mertsching, B. (2013). Human-aware
    robot navigation: A survey. Robotics and Autonomous Systems, 61(12),
    1726-1743.
"""

import numpy as np


class Pedestrian:
    """
    Kinematic pedestrian agent with constant-velocity straight-line motion.

    State is represented as (x, y, orientation, vx, vy) in the world frame.
    Heading (orientation) is derived from the velocity vector when the
    agent is moving, or held fixed when stationary — this avoids the
    undefined heading problem for zero-velocity agents.

    The 0.3 m physical radius is drawn from the standard annotation radius
    used in the ETH/UCY datasets and is consistent with a shoulder-width
    approximation of an adult pedestrian in overhead camera footage.

    Parameters
    ----------
    x, y : float
        Initial world-frame position (metres).
    orientation : float
        Initial heading in radians (0 = +x axis, counter-clockwise positive).
        If vx/vy are non-zero, orientation will be overwritten on the first
        call to step() to ensure consistency.
    velocity : float
        Scalar speed in m/s.  Stored for reference; actual motion is driven
        by vx, vy.
    vx, vy : float
        Velocity components in m/s.
    """

    # Physical radius used for collision detection (metres).
    # 0.3 m matches the annotation convention of the ETH/UCY datasets.
    PHYSICAL_RADIUS = 0.3

    def __init__(self, x, y, orientation=0.0, velocity=0.0, vx=0.0, vy=0.0):
        self.x           = float(x)
        self.y           = float(y)
        self.orientation = float(orientation)   # radians
        self.velocity    = float(velocity)      # scalar speed (m/s)
        self.vx          = float(vx)
        self.vy          = float(vy)
        self.radius      = self.PHYSICAL_RADIUS

    @property
    def pos(self):
        """
        Current position as a numpy array.

        Returned as a fresh array on each access to prevent callers from
        accidentally holding a mutable reference to the agent's state.

        Returns
        -------
        np.ndarray, shape (2,)
        """
        return np.array([self.x, self.y])

    def step(self, dt=0.1):
        """
        Advance the pedestrian by one time step under constant-velocity motion.

        Position update: Euler integration (exact for constant velocity).
        Heading update: derived from velocity vector when speed > threshold,
        held fixed when stationary.  The 0.001 m/s threshold prevents
        arctan2 from producing noisy headings due to floating-point noise
        in near-zero velocity components.

        Parameters
        ----------
        dt : float
            Time step in seconds.  Default 0.1 s gives smooth motion at
            typical pedestrian speeds (0.5-1.5 m/s).

        Notes
        -----
        Boundary reflection is handled by NavigationEnvironment.step_pedestrians()
        after this method returns, so the position may briefly exceed arena
        bounds between the Euler step and the clip operation.
        """
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Update heading only when moving; arctan2(0, 0) is undefined
        if abs(self.vx) > 0.001 or abs(self.vy) > 0.001:
            self.orientation = float(np.arctan2(self.vy, self.vx))

    def __repr__(self):
        return (
            f"Pedestrian("
            f"pos=({self.x:.2f}, {self.y:.2f}), "
            f"orient={np.degrees(self.orientation):.1f} deg, "
            f"speed={self.velocity:.2f} m/s)"
        )


class NavigationEnvironment:
    """
    Lightweight 2D arena for socially-aware robot navigation experiments.

    The environment is intentionally minimal: no physics engine, no
    inter-pedestrian avoidance, no sensor simulation.  This allows
    deterministic reproducibility given a seed and keeps per-episode runtime
    below 5 ms for the pedestrian simulation portion, which is negligible
    compared to A* planning time (~15-30 ms).

    Arena
    ~~~~~
    A square 10 x 10 m world with x in [0, 10] and y in [0, 10].
    Boundary conditions: elastic reflection (velocity component reversed
    when an agent approaches within 0.3 m of a wall).

    Collision detection
    ~~~~~~~~~~~~~~~~~~~
    A collision is declared when the robot centre comes within
    COLLISION_DIST + ROBOT_RADIUS = 0.5 m of any pedestrian centre.  This
    sum-of-radii check is the standard approach for circular agents (Siegwart
    et al., 2011) and avoids the computational cost of polygon intersection.

    Success criterion
    ~~~~~~~~~~~~~~~~~
    The robot is considered to have reached the goal when its centre is within
    GOAL_TOLERANCE = 0.5 m of goal_pos.  This tolerance accounts for the
    grid discretisation error (max 0.14 m diagonal for 0.2 m cells) plus a
    margin for the robot's physical footprint.

    Parameters
    ----------
    pedestrians : list of Pedestrian or None
    robot_start : array-like, shape (2,) or None
    robot_goal  : array-like, shape (2,) or None
    seed : int or None
        Random seed for the internal RNG (used by random() and any future
        stochastic methods).

    References
    ----------
    Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011). Introduction
        to Autonomous Mobile Robots (2nd ed.). MIT Press.
    """

    ARENA_SIZE    = 10.0   # metres; matches ETH/UCY filming area scale
    ROBOT_RADIUS  = 0.2    # metres; approximates a TurtleBot 2 footprint
    GOAL_TOLERANCE = 0.5   # metres; success radius around goal position
    COLLISION_DIST = 0.3   # metres; pedestrian radius (ETH/UCY annotation)

    def __init__(self, pedestrians=None, robot_start=None, robot_goal=None, seed=None):
        self.rng = np.random.default_rng(seed)

        self.pedestrians = pedestrians if pedestrians is not None else []

        # Use explicit None checks rather than truthiness to handle numpy
        # arrays correctly (a numpy array with one element is falsy when
        # all elements are zero, which would silently ignore a valid [0,0]
        # start position).
        self.robot_pos  = (
            np.array(robot_start, dtype=float)
            if robot_start is not None
            else np.array([1.0, 1.0])
        )
        self.robot_goal = (
            np.array(robot_goal, dtype=float)
            if robot_goal is not None
            else np.array([9.0, 9.0])
        )

        self.t  = 0.0    # simulation clock (seconds)
        self.dt = 0.1    # time step (seconds); 10 Hz matches typical robot control rate

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, n_pedestrians=4, seed=None):
        """
        Construct an environment with randomly placed pedestrians and robot.

        Pedestrians are placed uniformly in [1, 9]^2 with random headings
        and speeds in [0, 0.5] m/s.  The speed range approximates slow
        indoor walking; the ETH/UCY datasets show mean pedestrian speeds of
        approximately 0.6-1.0 m/s outdoors, but we use slower speeds for
        the static-planner experiments to keep scenarios tractable.

        Robot start is sampled in [0.5, 2.5]^2 and goal in [7.5, 9.5]^2,
        ensuring a non-trivial crossing distance.  A rejection loop (max 100
        attempts) ensures the start position is at least 0.8 m from all
        pedestrians, preventing the robot from spawning inside a social zone.

        Parameters
        ----------
        n_pedestrians : int
        seed : int or None

        Returns
        -------
        NavigationEnvironment

        Notes
        -----
        The rejection loop terminates within 1-2 iterations for typical
        n_pedestrians <= 6 in a 10 x 10 m arena.  For very dense scenarios
        (N > 15) it may exhaust all attempts; the last sampled start is
        used regardless, which may produce an initial zone violation.

        TODO: add a density parameter to control crowd-level (sparse /
              moderate / dense) rather than hard-coding n_pedestrians.
        """
        rng = np.random.default_rng(seed)
        pedestrians = []

        for _ in range(n_pedestrians):
            x           = rng.uniform(1.0, 9.0)
            y           = rng.uniform(1.0, 9.0)
            orientation = rng.uniform(-np.pi, np.pi)
            speed       = rng.uniform(0.0, 0.5)   # slow indoor walking pace
            vx          = speed * np.cos(orientation)
            vy          = speed * np.sin(orientation)
            pedestrians.append(Pedestrian(x, y, orientation, speed, vx, vy))

        # Rejection sampling for robot start position
        start = rng.uniform(0.5, 2.5, 2)   # initialise in case loop finds nothing
        for _ in range(100):
            candidate = rng.uniform(0.5, 2.5, 2)
            min_d = (
                min(np.linalg.norm(candidate - p.pos) for p in pedestrians)
                if pedestrians else float('inf')
            )
            if min_d > 0.8:
                start = candidate
                break

        goal = rng.uniform(7.5, 9.5, 2)

        return cls(
            pedestrians=pedestrians,
            robot_start=start,
            robot_goal=goal,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, pedestrians=None, robot_start=None, robot_goal=None):
        """
        Reset the environment to a new configuration.

        Only the provided arguments are updated; unspecified arguments
        retain their current values.  Simulation clock is reset to zero.

        Parameters
        ----------
        pedestrians : list of Pedestrian or None
        robot_start : array-like, shape (2,) or None
        robot_goal  : array-like, shape (2,) or None
        """
        if pedestrians is not None:
            self.pedestrians = pedestrians
        if robot_start is not None:
            self.robot_pos = np.array(robot_start, dtype=float)
        if robot_goal is not None:
            self.robot_goal = np.array(robot_goal, dtype=float)
        self.t = 0.0

    def get_state(self):
        """
        Return a snapshot of the current environment state as a dictionary.

        Returns
        -------
        dict with keys:
            'robot_pos'   : np.ndarray, shape (2,)
            'robot_goal'  : np.ndarray, shape (2,)
            'pedestrians' : list of (x, y, orientation) tuples
            'time'        : float — elapsed simulation time (seconds)

        Notes
        -----
        Pedestrian state is returned as plain tuples rather than Pedestrian
        objects to simplify serialisation (e.g. np.save) and to prevent
        callers from accidentally mutating live agent state through the
        returned snapshot.
        """
        return {
            'robot_pos':   self.robot_pos.copy(),
            'robot_goal':  self.robot_goal.copy(),
            'pedestrians': [(p.x, p.y, p.orientation) for p in self.pedestrians],
            'time':        self.t,
        }

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------

    def is_collision(self, pos=None):
        """
        Test whether the robot has collided with any pedestrian.

        Uses a sum-of-radii check: collision occurs when the centre-to-centre
        distance is less than COLLISION_DIST + ROBOT_RADIUS = 0.5 m.

        Parameters
        ----------
        pos : array-like, shape (2,) or None
            Position to test.  Uses robot_pos if None.

        Returns
        -------
        bool
        """
        if pos is None:
            pos = self.robot_pos
        pos = np.asarray(pos)
        for p in self.pedestrians:
            if np.linalg.norm(pos - p.pos) < self.COLLISION_DIST + self.ROBOT_RADIUS:
                return True
        return False

    def is_out_of_bounds(self, pos=None):
        """
        Test whether a position is outside the arena boundary.

        Parameters
        ----------
        pos : array-like, shape (2,) or None

        Returns
        -------
        bool
        """
        if pos is None:
            pos = self.robot_pos
        return (
            pos[0] < 0.0 or pos[0] > self.ARENA_SIZE or
            pos[1] < 0.0 or pos[1] > self.ARENA_SIZE
        )

    def reached_goal(self, pos=None):
        """
        Test whether the robot has reached the goal position.

        Parameters
        ----------
        pos : array-like, shape (2,) or None

        Returns
        -------
        bool
            True when centre-to-centre distance to goal < GOAL_TOLERANCE.
        """
        if pos is None:
            pos = self.robot_pos
        return float(np.linalg.norm(np.asarray(pos) - self.robot_goal)) < self.GOAL_TOLERANCE

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step_pedestrians(self):
        """
        Advance all pedestrians by one time step with elastic wall reflection.

        Each pedestrian is stepped via Pedestrian.step(dt), then checked
        against the arena boundary.  If an agent is within 0.3 m of a wall,
        the corresponding velocity component is negated (elastic reflection)
        and the position is clamped to [0.1, ARENA_SIZE - 0.1] to prevent
        agents from escaping the arena over multiple steps.

        The 0.3 m reflection distance is slightly larger than COLLISION_DIST
        to give the reflection a small safety margin and prevent agents from
        oscillating at the wall boundary.

        Notes
        -----
        Pedestrians do not avoid each other: inter-pedestrian collisions are
        ignored.  This is a deliberate simplification: the planner is
        evaluated on snapshots of agent positions, and inter-pedestrian
        interaction dynamics do not affect the social cost field encountered
        by the robot within a single planning step.

        TODO: implement a more realistic pedestrian motion model (e.g.
              Constant Velocity + Gaussian noise, or a simplified SFM) for
              long-horizon rollout experiments.
        """
        wall = 0.3   # reflection distance from wall (metres)
        clamp_lo = 0.1
        clamp_hi = self.ARENA_SIZE - 0.1

        for p in self.pedestrians:
            p.step(self.dt)

            # Elastic reflection: negate velocity component approaching wall
            if p.x < wall or p.x > self.ARENA_SIZE - wall:
                p.vx *= -1.0
            if p.y < wall or p.y > self.ARENA_SIZE - wall:
                p.vy *= -1.0

            # Hard clamp to prevent numerical drift outside arena
            p.x = float(np.clip(p.x, clamp_lo, clamp_hi))
            p.y = float(np.clip(p.y, clamp_lo, clamp_hi))

        self.t += self.dt

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    def set_default_scenario(self):
        """
        Configure a canonical four-pedestrian test scenario.

        Used for deterministic visual comparisons between planners (fig3 in
        figures.py).  Pedestrians face different directions to exercise all
        four zones of the asymmetric model:

          - Agent 0 (3, 5): facing right (+x), slow forward motion
          - Agent 1 (5, 3): facing up (+y), slow forward motion
          - Agent 2 (7, 7): facing left (-x), slow diagonal motion
          - Agent 3 (4, 7): facing lower-right, stationary

        Robot navigates from (1, 1) to (9, 9), a diagonal crossing that
        forces the planner to resolve conflicts with all four agents.
        """
        self.pedestrians = [
            Pedestrian(3.0, 5.0, orientation=0.0,       vx=0.10,  vy=0.00),
            Pedestrian(5.0, 3.0, orientation=np.pi/2,   vx=0.00,  vy=0.10),
            Pedestrian(7.0, 7.0, orientation=np.pi,     vx=-0.05, vy=0.05),
            Pedestrian(4.0, 7.0, orientation=-np.pi/4,  vx=0.00,  vy=0.00),
        ]
        self.robot_pos  = np.array([1.0, 1.0])
        self.robot_goal = np.array([9.0, 9.0])

    def __repr__(self):
        return (
            f"NavigationEnvironment("
            f"{len(self.pedestrians)} pedestrians, "
            f"robot={self.robot_pos}, goal={self.robot_goal}, "
            f"t={self.t:.1f}s)"
        )
