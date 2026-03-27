"""
social_zone.py
==============
Asymmetric personal-space model for socially-aware robot navigation.

This module operationalises the proxemics theory of Hall (1966), who observed
that humans maintain direction-dependent interpersonal distances rather than
the symmetric "bubble" commonly assumed in prior navigation work.  The four-
zone discretisation (front / left / right / back) mirrors the angular
partitioning used by Kirby et al. (2009) in the COMPANION system and is
consistent with the empirical findings of Kendon (1990) on F-formation
dynamics.

The exponential cost kernel is adapted from the Social Force Model of
Helbing & Molnár (1995): instead of a repulsive force field we produce a
scalar cost field suitable for integration into graph-search planners (see
planner.py).  An exponential kernel is preferred over a quadratic one because
it remains differentiable everywhere and grows sharply inside the violation
boundary, which steers the planner away from zone edges rather than treating
all sub-threshold distances equally.

References
----------
Hall, E. T. (1966). The Hidden Dimension. Doubleday, New York.
    Foundational proxemics work; defines personal (0.45-1.2 m), social
    (1.2-3.6 m) and public (3.6+ m) space.

Helbing, D., & Molnar, P. (1995). Social force model for pedestrian dynamics.
    Physical Review E, 51(5), 4282-4286.
    Exponential repulsion kernel; motivates our cost formulation.

Kirby, R., Simmons, R., & Forlizzi, J. (2009). COMPANION: A constraint-
    optimizing method for person-acceptable navigation. In Proc. IEEE RO-MAN,
    pp. 607-612.
    Introduces direction-dependent personal-space zones for HRI.

Kendon, A. (1990). Conducting Interaction. Cambridge University Press.
    Empirical study of F-formations; validates asymmetric interpersonal space.

Kruse, T., Pandey, A. K., Alami, R., & Mertsching, B. (2013). Human-aware
    robot navigation: A survey. Robotics and Autonomous Systems, 61(12),
    1726-1743.  Survey; Section 4.2 reviews proxemics-based cost functions.

Truong, X. T., & Ngo, T. D. (2017). Toward socially aware robot navigation
    in dynamic and crowded environments: A proactive social motion model.
    IEEE Transactions on Automation Science and Engineering, 14(4), 1743-1760.
    Empirically validates front-zone sensitivity using ETH/UCY data.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Zone angular boundaries (degrees, relative to human heading = 0 deg)
# ---------------------------------------------------------------------------
# The +/-30 deg front cone matches Kirby et al. (2009) and is narrower than
# Hall's original "personal front" concept (+/-45 deg).  A narrower front
# zone makes the planner prefer lateral passes, which aligns with observed
# human overtaking behaviour (Kruse et al., 2013, Table 1).
#
# TODO: make zone boundaries configurable so cone-width ablations can be run
#       without modifying this file.
_FRONT_HALF_ANGLE = 30.0    # degrees — boundary of front zone
_SIDE_HALF_ANGLE  = 150.0   # degrees — boundary between side and back zones


class SocialZone:
    """
    Asymmetric personal-space model with four directional zones.

    The zone geometry (from the human's egocentric frame, heading = 0 deg)::

        +-----------------------------------------+
        |            FRONT  (+/-30 deg)           |
        |       required distance:  ~1.8 m        |
        |  LEFT          [h]          RIGHT        |
        | (30-150 deg)               (-30 to -150) |
        |       required distance:  ~1.2 m        |
        |            BACK  (>150 deg)             |
        |       required distance:  ~0.6 m        |
        +-----------------------------------------+

    Zone distances are the 10th-percentile minimum observed inter-agent
    distances extracted from the ETH and UCY pedestrian datasets
    (Pellegrini et al., 2009; Lerner et al., 2007) via extract_zones.py.
    The 10th percentile captures the *minimum comfortable* distance rather
    than the typical one; using the median would overestimate comfortable
    proximity in the rear zone.

    Cost function
    ~~~~~~~~~~~~~
    For robot position r near a human at h facing direction theta::

        d_actual   = ||r - h||
        d_required = zone_distance( angle(r - h) - theta )

        cost(r) = exp(d_required - d_actual) - 1   if d_actual < d_required
                = 0                                  otherwise

    This one-sided exponential is zero outside the zone boundary and rises
    sharply inside it.  Subtracting 1 ensures cost = 0 exactly at the
    boundary, giving a continuous (though not C1-smooth) cost field.

    Parameters
    ----------
    front : float
        Minimum comfortable distance in the front zone (metres).
    back : float
        Minimum comfortable distance in the back zone (metres).
    left : float
        Minimum comfortable distance in the left zone (metres).
    right : float
        Minimum comfortable distance in the right zone (metres).

    References
    ----------
    Pellegrini, S., Ess, A., Schindler, K., & Van Gool, L. (2009).
        You'll never walk alone: Modelling social behaviour for multi-target
        tracking.  In Proc. ICCV, pp. 261-268.

    Lerner, A., Chrysanthou, Y., & Lischinski, D. (2007). Crowds by example.
        Computer Graphics Forum, 26(3), 655-664.
    """

    # Default distances from 10th-percentile analysis of ETH/UCY data (see
    # extract_zones.py).  Values match those reported by Truong & Ngo (2017)
    # and are consistent with Hall's (1966) personal-space bounds.
    DEFAULT_DISTANCES = {
        'front': 1.8,   # largest: humans are most sensitive to frontal approach
        'back':  0.6,   # smallest: rear approach is least disruptive
        'left':  1.2,   # lateral zones are intermediate
        'right': 1.2,
    }

    # Angular boundaries stored for external inspection and unit testing.
    ZONE_BOUNDARIES = {
        'front': (-_FRONT_HALF_ANGLE,  _FRONT_HALF_ANGLE),
        'left':  ( _FRONT_HALF_ANGLE,  _SIDE_HALF_ANGLE),
        'right': (-_SIDE_HALF_ANGLE,  -_FRONT_HALF_ANGLE),
        'back':  ( _SIDE_HALF_ANGLE,   180.0),   # wraps: also -180 to -150 deg
    }

    def __init__(self, front=1.8, back=0.6, left=1.2, right=1.2):
        """
        Initialise zone with per-direction minimum comfortable distances.

        Parameters
        ----------
        front, back, left, right : float
            Minimum comfortable separation for each directional zone (metres).
            Values outside [0.1, 5.0] are accepted but physically unreasonable.

        Notes
        -----
        Stored as Python floats rather than numpy scalars to avoid subtle
        type-promotion issues when zone distances are loaded from .npy files,
        which can return numpy.float64 objects that behave unexpectedly in
        some comparison contexts.
        """
        self.distances = {
            'front': float(front),
            'back':  float(back),
            'left':  float(left),
            'right': float(right),
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d):
        """
        Construct from a dictionary of zone distances.

        Parameters
        ----------
        d : dict
            Subset of {'front', 'back', 'left', 'right'}.
            Missing keys fall back to DEFAULT_DISTANCES.

        Returns
        -------
        SocialZone
        """
        return cls(
            front=d.get('front', cls.DEFAULT_DISTANCES['front']),
            back= d.get('back',  cls.DEFAULT_DISTANCES['back']),
            left= d.get('left',  cls.DEFAULT_DISTANCES['left']),
            right=d.get('right', cls.DEFAULT_DISTANCES['right']),
        )

    @classmethod
    def from_file(cls, filepath="zone_distances.npy"):
        """
        Load zone distances persisted by ``extract_zones.run_extraction()``.

        Parameters
        ----------
        filepath : str
            Path to a .npy file whose allow_pickle=True item is a dict with
            keys 'front', 'back', 'left', 'right'.

        Returns
        -------
        SocialZone
            Falls back to DEFAULT_DISTANCES if the file does not exist.
        """
        if os.path.exists(filepath):
            d = np.load(filepath, allow_pickle=True).item()
            return cls.from_dict(d)
        print(f"Warning: {filepath} not found. Using default distances (Hall, 1966).")
        return cls()

    @classmethod
    def circular(cls, radius=1.2):
        """
        Construct a symmetric (circular) zone for baseline comparison.

        The default radius of 1.2 m corresponds to the inner boundary of
        Hall's (1966) social space and is the most commonly cited value in
        HRI literature (Kruse et al., 2013).

        Parameters
        ----------
        radius : float

        Returns
        -------
        SocialZone
        """
        return cls(front=radius, back=radius, left=radius, right=radius)

    # ------------------------------------------------------------------
    # Core geometry
    # ------------------------------------------------------------------

    def get_zone(self, angle_deg):
        """
        Map a relative bearing to a zone label.

        Parameters
        ----------
        angle_deg : float
            Bearing from the human to the robot in the human's egocentric
            frame (degrees).  0 deg = directly ahead; 90 deg = left;
            +/-180 deg = directly behind.

        Returns
        -------
        str
            One of {'front', 'left', 'right', 'back'}.

        Notes
        -----
        Input is wrapped to (-180, 180] before classification so the function
        handles arbitrary winding numbers (e.g. from cumulative IMU headings).
        The wrapping formula ``(a + 180) % 360 - 180`` is numerically stable
        for all finite float inputs.
        """
        # Wrap to (-180, 180] — handles any winding number cleanly
        a = (angle_deg + 180.0) % 360.0 - 180.0

        if  -_FRONT_HALF_ANGLE <= a <= _FRONT_HALF_ANGLE:
            return 'front'
        elif _FRONT_HALF_ANGLE < a <= _SIDE_HALF_ANGLE:
            return 'left'
        elif a > _SIDE_HALF_ANGLE or a < -_SIDE_HALF_ANGLE:
            return 'back'
        elif -_SIDE_HALF_ANGLE <= a < -_FRONT_HALF_ANGLE:
            return 'right'

        return 'front'  # unreachable; silences static analysis warnings

    def required_distance(self, robot_pos, human_pos, human_orientation):
        """
        Return the zone-specific minimum comfortable distance.

        Computes the bearing from the human to the robot in the human's
        egocentric frame and returns the corresponding zone threshold.

        Parameters
        ----------
        robot_pos : array-like, shape (2,)
            Robot (x, y) in world frame (metres).
        human_pos : array-like, shape (2,)
            Human (x, y) in world frame (metres).
        human_orientation : float
            Human heading in radians (0 = +x axis, counter-clockwise positive).

        Returns
        -------
        float
            Required minimum separation distance in metres.

        Notes
        -----
        Two coordinate transforms are applied:
          1. World-frame displacement -> bearing angle (arctan2).
          2. Bearing -> egocentric angle by subtracting human_orientation.
        The wrap to (-pi, pi] after step 2 is mandatory; without it, arctan2
        outputs near +/-pi can straddle the zone boundary incorrectly.
        """
        dx = robot_pos[0] - human_pos[0]
        dy = robot_pos[1] - human_pos[1]

        # Step 1: absolute bearing from human to robot in world frame
        abs_angle = np.arctan2(dy, dx)

        # Step 2: transform to human egocentric frame
        rel_angle = abs_angle - human_orientation

        # Step 3: wrap to (-pi, pi] to keep degrees in expected range
        rel_angle = (rel_angle + np.pi) % (2.0 * np.pi) - np.pi

        zone = self.get_zone(np.degrees(rel_angle))
        return self.distances[zone]

    def compute_cost(self, robot_pos, human_pos, human_orientation):
        """
        Compute the social navigation cost at a candidate robot position.

        Uses a one-sided exponential kernel (Helbing & Molnar, 1995)::

            cost = exp(d_req - d_act) - 1    if d_act < d_req
                 = 0                          otherwise

        Subtracting 1 ensures continuity at the zone boundary.  The
        exponential shape means the gradient is largest close to the boundary
        and decays steeply inside the zone, which guides the planner to
        prefer paths that skim zone edges rather than cutting through the
        centre — consistent with observed human passing behaviour.

        Parameters
        ----------
        robot_pos : array-like, shape (2,)
        human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        float
            Non-negative social cost.

        Notes
        -----
        Performance: called O(N_cells * N_humans) times per plan() invocation.
        For N_humans > ~10 or grids finer than 0.1 m, consider vectorising
        this over a batch of positions using numpy broadcasting.

        TODO: add a tunable decay-rate sigma (currently fixed at 1.0) so the
              cost field sharpness can be ablated independently of the zone
              distances.

        FIXME: when d_actual is near zero (robot directly on human position),
               exp() can return very large values.  Add a minimum distance
               clamp of ~0.05 m to avoid overflow in degenerate cases.
        """
        dx = robot_pos[0] - human_pos[0]
        dy = robot_pos[1] - human_pos[1]
        actual_dist = float(np.sqrt(dx * dx + dy * dy))

        required = self.required_distance(robot_pos, human_pos, human_orientation)

        if actual_dist < required:
            return float(np.exp(required - actual_dist) - 1.0)
        return 0.0

    def is_violation(self, robot_pos, human_pos, human_orientation):
        """
        Test whether a robot position intrudes into the personal space zone.

        Parameters
        ----------
        robot_pos : array-like, shape (2,)
        human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        bool
            True if actual distance < zone's required distance.
        """
        dx = robot_pos[0] - human_pos[0]
        dy = robot_pos[1] - human_pos[1]
        actual_dist = float(np.sqrt(dx * dx + dy * dy))
        return actual_dist < self.required_distance(robot_pos, human_pos, human_orientation)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def get_zone_boundary_points(self, human_pos, human_orientation, n_points=360):
        """
        Sample the zone boundary as a closed polygon for matplotlib rendering.

        Sweeps n_points angles uniformly around the circle, looks up the
        required distance for each direction, and returns world-frame (x, y)
        coordinates forming a closed polygon.  The result is a step function
        (piecewise-constant radius) with visible corners at zone boundaries;
        use smooth_zone_boundary() for publication figures.

        Parameters
        ----------
        human_pos : array-like, shape (2,)
        human_orientation : float
        n_points : int
            Angular resolution.  360 gives 1 deg steps; sufficient for most
            visualisations.

        Returns
        -------
        xs, ys : np.ndarray, shape (n_points,)
            World-frame boundary coordinates.
        """
        angles = np.linspace(-np.pi, np.pi, n_points)
        xs = np.empty(n_points)
        ys = np.empty(n_points)

        for k, angle in enumerate(angles):
            zone = self.get_zone(np.degrees(angle))
            r    = self.distances[zone]
            abs_angle = human_orientation + angle
            xs[k] = human_pos[0] + r * np.cos(abs_angle)
            ys[k] = human_pos[1] + r * np.sin(abs_angle)

        return xs, ys

    def smooth_zone_boundary(self, human_pos, human_orientation, n_points=360):
        """
        Return a Gaussian-smoothed boundary for publication figures.

        The raw boundary has hard corners at the four zone transitions.  A
        circular Gaussian applied to the radial profile removes these
        discontinuities, producing an ellipse-like shape more representative
        of continuously varying personal space (Truong & Ngo, 2017, Fig. 3).

        Parameters
        ----------
        human_pos : array-like, shape (2,)
        human_orientation : float
        n_points : int

        Returns
        -------
        xs, ys : np.ndarray, shape (n_points,)

        Notes
        -----
        mode='wrap' is critical: the radius profile is periodic (the zone
        boundary is a closed curve in angle space), so non-wrapping padding
        introduces spurious edge effects at +/-180 deg.

        sigma=10 samples (= 10 deg at 360 pts) was chosen empirically to
        remove step artefacts while preserving the overall asymmetry shape.

        TODO: expose sigma as a parameter for controllable smoothness.
        """
        from scipy.ndimage import gaussian_filter1d

        angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
        radii  = np.array(
            [self.distances[self.get_zone(np.degrees(a))] for a in angles],
            dtype=float,
        )

        # Circular Gaussian: sigma in *samples*, mode='wrap' for periodicity
        radii_smooth = gaussian_filter1d(radii, sigma=10, mode='wrap')

        xs = human_pos[0] + radii_smooth * np.cos(human_orientation + angles)
        ys = human_pos[1] + radii_smooth * np.sin(human_orientation + angles)
        return xs, ys

    def __repr__(self):
        d = self.distances
        return (
            f"SocialZone("
            f"front={d['front']:.2f}m, back={d['back']:.2f}m, "
            f"left={d['left']:.2f}m, right={d['right']:.2f}m)"
        )


# ---------------------------------------------------------------------------
# Circular (symmetric) baseline
# ---------------------------------------------------------------------------

class CircularSocialZone(SocialZone):
    """
    Isotropic personal-space model for ablation and baseline comparison.

    Implements the circular-bubble assumption used in the majority of prior
    socially-aware navigation systems (Kruse et al., 2013, Section 4.1).
    Overrides required_distance() and compute_cost() to bypass the angular
    zone-lookup entirely, making the baseline behaviour explicit and ~2x
    faster per cost evaluation.

    Parameters
    ----------
    radius : float
        Uniform zone radius in metres.  Default 1.2 m is the most commonly
        cited value in HRI literature (Kruse et al., 2013) and corresponds
        to the inner boundary of Hall's (1966) social-space band.
    """

    def __init__(self, radius=1.2):
        # Initialise parent with uniform distances so inherited helpers
        # (get_zone_boundary_points, smooth_zone_boundary) still return a
        # geometrically correct circle without special-casing.
        super().__init__(front=radius, back=radius, left=radius, right=radius)
        self.radius = float(radius)

    def required_distance(self, robot_pos, human_pos, human_orientation):
        """Return the uniform radius; direction is irrelevant."""
        return self.radius

    def compute_cost(self, robot_pos, human_pos, human_orientation):
        """
        Isotropic exponential cost — identical kernel to SocialZone but
        without the egocentric angle computation.

        Parameters
        ----------
        robot_pos, human_pos : array-like, shape (2,)
        human_orientation : float
            Unused; present for interface compatibility with SocialZone.

        Returns
        -------
        float
        """
        dx = robot_pos[0] - human_pos[0]
        dy = robot_pos[1] - human_pos[1]
        actual_dist = float(np.sqrt(dx * dx + dy * dy))
        if actual_dist < self.radius:
            return float(np.exp(self.radius - actual_dist) - 1.0)
        return 0.0

    def __repr__(self):
        return f"CircularSocialZone(radius={self.radius:.2f}m)"
