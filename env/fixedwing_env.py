"""
Multi-UAV Fixed-Wing Environment with Box2D physics, point-cloud perception,
occupancy grid mapping, and dynamic object tracking.

Observations (per UAV):
  - pointcloud: 60-dim (normalized ray distances)
  - grid_map: 160×160 occupancy grid
  - self_state: 6-dim [v, psi, d_goal, theta_goal, delta, arrived_flag]
  - dynamic_obs: 25-dim (K=5 objects × [dx, dy, dvx, dvy, size])

Actions (per UAV, continuous):
  - [a_th, delta] ∈ [-1, 1]
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import math
from collections import deque
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

from config import CONFIG

# =============================================================================
#  Kalman Filter for Constant-Velocity Tracking
# =============================================================================

class KalmanFilterCV:
    """Constant-velocity Kalman filter for dynamic obstacle tracking.
    State: [x, y, vx, vy]. Measurement: [x, y].
    """
    def __init__(self, x: float, y: float, dt: float = 0.1):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        # Covariance
        self.P = np.eye(4, dtype=np.float64) * 1.0
        # Process noise
        self.Q = np.eye(4, dtype=np.float64) * CONFIG["tracking"]["kalman_Q"]
        self.Q[2:, 2:] *= 0.1
        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * CONFIG["tracking"]["kalman_R"]
        self.lost_count = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """z: measurement [x, y]"""
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.lost_count = 0

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_pos(self) -> np.ndarray:
        return self.x[:2].copy()


# =============================================================================
#  Tracked Object
# =============================================================================

@dataclass
class TrackedObject:
    kalman: KalmanFilterCV
    label: int
    age: int = 0
    last_seen: int = 0

    @property
    def pos(self) -> np.ndarray:
        return self.kalman.get_pos()

    @property
    def vel(self) -> np.ndarray:
        return self.kalman.x[2:4]

    @property
    def is_lost(self) -> bool:
        return self.kalman.lost_count > CONFIG["tracking"]["lost_threshold"]


# =============================================================================
#  Fixed-Wing Kinematics (Bicycle Model)
# =============================================================================

class FixedWingKinematics:
    """Fixed-wing UAV kinematics using bank-angle (roll) turn model.
    State: [x, y, psi, v] (position, heading, speed).
    Control: [a_th, delta] (throttle, bank-angle command).

    Turn dynamics (real fixed-wing physics):
      - Bank angle φ = delta * max_bank_angle
      - Turn rate ψ_dot = g * tan(φ) / v   (centripetal acceleration)
      - Turn radius r = v² / (g * tan(φ))  (∝ v², realistic)
    """
    def __init__(self, x: float, y: float, psi: float, v: float):
        self.x = x
        self.y = y
        self.psi = psi  # heading (rad)
        self.v = v      # speed (m/s)
        cfg = CONFIG["uav"]
        self.v_min = cfg["v_min"]
        self.v_max = cfg["v_max"]
        self.max_bank_angle = cfg["max_bank_angle"]
        self.a_th_max = cfg["a_th_max"]
        self.drag = cfg["drag_coef"]
        self.g = cfg["g"]

    def step(self, a_th: float, delta: float, dt: float):
        """Advance kinematics by dt seconds using bank-angle turn model."""
        # Clip controls
        a_th = np.clip(a_th, -1.0, 1.0)
        delta = np.clip(delta, -1.0, 1.0)

        # Thrust / drag → speed
        acceleration = a_th * self.a_th_max - self.drag * self.v
        self.v += acceleration * dt
        self.v = np.clip(self.v, self.v_min, self.v_max)

        # Bank-angle turn (fixed-wing physics)
        bank_angle = delta * self.max_bank_angle
        # Turn rate: centripetal acceleration g*tan(φ), divided by speed
        if self.v > 0.01:
            self.psi += (self.g * math.tan(bank_angle) / self.v) * dt
        # Normalize heading
        self.psi = math.atan2(math.sin(self.psi), math.cos(self.psi))

        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.v], dtype=np.float32)


# =============================================================================
#  Point-Cloud Perception
# =============================================================================

class PointCloudPerception:
    """Forward-facing fan-shaped point cloud with noise and ghost points."""
    def __init__(self):
        cfg = CONFIG["perception"]
        self.num_rays = cfg["num_rays"]
        self.fov = cfg["fov"]
        self.max_range = cfg["max_range"]
        self.D0 = cfg["D0"]
        self.noise_coef = cfg["noise_coef"]
        self.ghost_prob = cfg["ghost_prob"]
        # Precompute ray angles
        self.ray_angles = np.linspace(-self.fov / 2, self.fov / 2, self.num_rays)

    def sense(self, uav_x: float, uav_y: float, uav_psi: float,
              obstacles: List[np.ndarray], world_size: float = 50.0) -> np.ndarray:
        """Cast rays and return range readings (num_rays,)."""
        ranges = np.ones(self.num_rays) * self.max_range

        for i, angle in enumerate(self.ray_angles):
            ray_angle = uav_psi + angle
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)

            min_dist = self.max_range

            # Check obstacle collisions
            for obs in obstacles:
                dist = self._ray_circle_intersect(
                    uav_x, uav_y, dx, dy, obs[0], obs[1], obs[2]
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist

            # Check world boundary (walls)
            wall_dist = self._ray_wall_intersect(
                uav_x, uav_y, dx, dy, world_size
            )
            if wall_dist is not None and wall_dist < min_dist:
                min_dist = wall_dist

            # Detection probability (exponential falloff)
            if np.random.random() > math.exp(-min_dist / self.D0):
                min_dist = self.max_range

            # Gaussian noise
            noise = np.random.normal(0, self.noise_coef * min_dist)
            min_dist += noise
            min_dist = np.clip(min_dist, 0, self.max_range)

            ranges[i] = min_dist

        # Ghost points
        ghost_mask = np.random.random(self.num_rays) < self.ghost_prob
        ranges[ghost_mask] = np.random.uniform(0, self.max_range, ghost_mask.sum())

        return ranges.astype(np.float32)

    @staticmethod
    def _ray_wall_intersect(ox: float, oy: float, dx: float, dy: float,
                            world_size: float) -> Optional[float]:
        """Ray-wall intersection. Returns distance to nearest wall or None."""
        # Avoid division by zero
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return None

        t_min = float("inf")

        # Check x-walls (x=0, x=world_size)
        if abs(dx) > 1e-10:
            for wx in (0, world_size):
                # UAV exactly at this wall: t=0 if ray points into the wall
                if abs(wx - ox) < 1e-10:
                    if (wx == 0 and dx <= 0) or (wx == world_size and dx >= 0):
                        t = 0.0
                    else:
                        continue
                else:
                    t = (wx - ox) / dx
                    if t <= 1e-6:  # Must be strictly forward
                        continue
                wy = oy + t * dy
                if 0 <= wy <= world_size:
                    t_min = min(t_min, t)

        # Check y-walls (y=0, y=world_size)
        if abs(dy) > 1e-10:
            for wy in (0, world_size):
                # UAV exactly at this wall: t=0 if ray points into the wall
                if abs(wy - oy) < 1e-10:
                    if (wy == 0 and dy <= 0) or (wy == world_size and dy >= 0):
                        t = 0.0
                    else:
                        continue
                else:
                    t = (wy - oy) / dy
                    if t <= 1e-6:  # Must be strictly forward
                        continue
                wx = ox + t * dx
                if 0 <= wx <= world_size:
                    t_min = min(t_min, t)

        return t_min if t_min < float("inf") else None

    @staticmethod
    def _ray_circle_intersect(ox: float, oy: float, dx: float, dy: float,
                              cx: float, cy: float, r: float) -> Optional[float]:
        """Ray-circle intersection. Returns distance or None."""
        fx = ox - cx
        fy = oy - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t1 = (-b - math.sqrt(disc)) / (2 * a)
        t2 = (-b + math.sqrt(disc)) / (2 * a)
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        return None

    def get_ray_directions(self, uav_psi: float) -> np.ndarray:
        """Return (num_rays, 2) direction vectors."""
        angles = uav_psi + self.ray_angles
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


# =============================================================================
#  Occupancy Grid Mapping (Bresenham)
# =============================================================================

class OccupancyGrid:
    """Local occupancy grid that shifts with the UAV."""
    def __init__(self):
        cfg = CONFIG["grid"]
        self.width = cfg["width"]
        self.height = cfg["height"]
        self.resolution = cfg["resolution"]
        self.size = cfg["size"]
        self.free_inc = cfg["free_inc"]
        self.occ_inc = cfg["occ_inc"]
        self.clip_min = cfg["clip_min"]
        self.clip_max = cfg["clip_max"]
        # Grid origin in world coords (center of grid)
        self.origin_x = 0.0
        self.origin_y = 0.0
        # Occupancy values [0, 1]
        self.grid = np.ones((self.height, self.width), dtype=np.float32) * 0.5

    def reset(self, uav_x: float, uav_y: float):
        """Reset grid centered on UAV."""
        self.origin_x = uav_x - self.size / 2
        self.origin_y = uav_y - self.size / 2
        self.grid.fill(0.5)

    def shift(self, uav_x: float, uav_y: float):
        """Re-center grid on UAV. Preserves edge values instead of resetting to 0.5."""
        new_ox = uav_x - self.size / 2
        new_oy = uav_y - self.size / 2
        dx_pix = int(round((new_ox - self.origin_x) / self.resolution))
        dy_pix = int(round((new_oy - self.origin_y) / self.resolution))

        if abs(dx_pix) > 0 or abs(dy_pix) > 0:
            # Roll the grid (edges wrap around, preserving values)
            self.grid = np.roll(self.grid, -dy_pix, axis=0)
            self.grid = np.roll(self.grid, -dx_pix, axis=1)
            # Note: edges that come from the opposite side of the grid via np.roll
            # are preserved (containing historical values), not reset to 0.5.
            # This helps maintain map consistency as the UAV moves.

            self.origin_x = new_ox
            self.origin_y = new_oy

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """World coords to grid indices."""
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Grid indices to world coords."""
        wx = gx * self.resolution + self.origin_x + self.resolution / 2
        wy = gy * self.resolution + self.origin_y + self.resolution / 2
        return wx, wy

    def update(self, uav_x: float, uav_y: float,
               ranges: np.ndarray, ray_angles: np.ndarray, uav_psi: float,
               max_range: float):
        """Bresenham line-of-sight update from point cloud."""
        uav_gx, uav_gy = self.world_to_grid(uav_x, uav_y)

        for i, (r, angle) in enumerate(zip(ranges, ray_angles)):
            if r >= max_range - 0.01:
                continue  # No detection
            ray_angle = uav_psi + angle
            end_x = uav_x + r * math.cos(ray_angle)
            end_y = uav_y + r * math.sin(ray_angle)
            egx, egy = self.world_to_grid(end_x, end_y)

            # Bresenham line
            for gx, gy in self._bresenham(uav_gx, uav_gy, egx, egy):
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    if (gx, gy) == (egx, egy) or abs(gx - egx) + abs(gy - egy) <= 1:
                        # Endpoint: occupied
                        self.grid[gy, gx] = np.clip(
                            self.grid[gy, gx] + self.occ_inc,
                            self.clip_min, self.clip_max
                        )
                    else:
                        # Ray interior: free
                        self.grid[gy, gx] = np.clip(
                            self.grid[gy, gx] + self.free_inc,
                            self.clip_min, self.clip_max
                        )

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int):
        """Bresenham line algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

        return points

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()


# =============================================================================
#  UAV Agent (internal state for one UAV)
# =============================================================================

class UAVAgent:
    """Internal state for a single UAV."""
    def __init__(self, uav_id: int):
        self.id = uav_id
        cfg = CONFIG["uav"]
        self.kinematics = FixedWingKinematics(
            x=np.random.uniform(1, CONFIG["world_size"] - 1),
            y=np.random.uniform(1, CONFIG["world_size"] - 1),
            psi=np.random.uniform(0, 2 * math.pi),
            v=cfg["v_cruise"],
        )
        self.goal = np.array([0.0, 0.0])
        self.perception = PointCloudPerception()
        self.occupancy = OccupancyGrid()
        self.arrived = False
        self.collided = False
        self.steps = 0
        self.path_history = deque(maxlen=CONFIG["demo"]["trail_length"])
        self.success_count = 0
        self.prev_goal_dist = 0.0

    def reset(self, world_size: float):
        """Reset UAV at random position."""
        self.kinematics = FixedWingKinematics(
            x=np.random.uniform(1, world_size - 1),
            y=np.random.uniform(1, world_size - 1),
            psi=np.random.uniform(0, 2 * math.pi),
            v=CONFIG["uav"]["v_cruise"],
        )
        self.goal = np.random.uniform(1, world_size - 1, size=2)
        self.arrived = False
        self.collided = False
        self.steps = 0
        self.path_history.clear()
        self.occupancy.reset(self.kinematics.x, self.kinematics.y)
        self.prev_goal_dist = math.hypot(self.goal[0] - self.kinematics.x,
                                         self.goal[1] - self.kinematics.y)

    def get_observation(self, obstacles: List[np.ndarray],
                        tracked_objects: List[TrackedObject],
                        world_size: float) -> Dict[str, np.ndarray]:
        """Build the observation dict for this UAV."""
        k = self.kinematics

        # Point cloud
        ranges = self.perception.sense(k.x, k.y, k.psi, obstacles, world_size)
        pc_normalized = ranges / CONFIG["perception"]["max_range"]

        # Occupancy grid (shift to UAV center)
        self.occupancy.shift(k.x, k.y)
        self.occupancy.update(
            k.x, k.y, ranges, self.perception.ray_angles,
            k.psi, CONFIG["perception"]["max_range"]
        )

        # Self state: [v, psi, d_goal, theta_goal, delta, arrived_flag]
        dx_goal = self.goal[0] - k.x
        dy_goal = self.goal[1] - k.y
        d_goal = math.hypot(dx_goal, dy_goal)
        theta_goal = math.atan2(dy_goal, dx_goal) - k.psi
        theta_goal = math.atan2(math.sin(theta_goal), math.cos(theta_goal))

        # Normalize speed
        v_norm = (k.v - CONFIG["uav"]["v_min"]) / (CONFIG["uav"]["v_max"] - CONFIG["uav"]["v_min"])
        self_state = np.array([
            v_norm * 2 - 1,           # normalized speed to [-1, 1]
            math.sin(k.psi),
            math.cos(k.psi),
            math.tanh(d_goal / 10.0),  # normalized distance
            theta_goal / math.pi,      # normalized heading error
            1.0 if self.arrived else -1.0,
        ], dtype=np.float32)

        # Dynamic obstacles (K nearest tracked objects)
        k_obj = CONFIG["obs"]["k_objects"]
        n_feats = CONFIG["obs"]["object_feats"]
        dyn_obs = np.zeros((k_obj, n_feats), dtype=np.float32)

        if tracked_objects:
            obj_states = []
            for obj in tracked_objects:
                dx = obj.pos[0] - k.x
                dy = obj.pos[1] - k.y
                dvx = obj.vel[0] - k.v * math.cos(k.psi)
                dvy = obj.vel[1] - k.v * math.sin(k.psi)
                dist = math.hypot(dx, dy)
                size = 0.5  # default size
                obj_states.append((dist, [dx, dy, dvx, dvy, size]))

            # Sort by distance, take K nearest
            obj_states.sort(key=lambda x: x[0])
            for i, (_, feats) in enumerate(obj_states[:k_obj]):
                # Normalize
                feats[0] /= 20.0   # dx
                feats[1] /= 20.0   # dy
                feats[2] /= 10.0   # dvx
                feats[3] /= 10.0   # dvy
                dyn_obs[i] = feats

        return {
            "pointcloud": pc_normalized,
            "grid_map": self.occupancy.get_grid(),
            "self_state": self_state,
            "dynamic_obs": dyn_obs.flatten(),
        }

    def get_collision_penalty(self) -> float:
        """Collision penalty with increasing beta."""
        base = CONFIG["reward"]["collision_penalty_base"]
        beta = CONFIG["reward"]["beta_init"] + self.success_count * CONFIG["reward"]["beta_increment"]
        return -base * beta


# =============================================================================
#  Multi-UAV Environment
# =============================================================================

class MultiFixedWingEnv(gym.Env):
    """
    Gym environment for multi-UAV fixed-wing navigation with point-cloud perception.

    Observation Space (per UAV):
        Dict with:
        - pointcloud: Box(0, 1, (60,))
        - grid_map: Box(0, 1, (160, 160))
        - self_state: Box(-1, 1, (6,))
        - dynamic_obs: Box(-1, 1, (25,))

    Action Space (per UAV):
        Box(-1, 1, (2,)) — [a_th, delta]
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, num_uavs: int = None, world_size: float = None):
        super().__init__()
        self.num_uavs = num_uavs or CONFIG["num_uavs"]
        self.world_size = world_size or CONFIG["world_size"]
        self.max_steps = CONFIG["max_steps"]
        self.dt = CONFIG["dt"]
        self.goal_radius = CONFIG["reward"]["goal_radius"]
        self.uav_radius = CONFIG["collision"]["uav_radius"]

        # Obstacles (static)
        self.static_obstacles = []  # list of [x, y, radius]
        self._generate_obstacles()

        # UAVs
        self.uavs = [UAVAgent(i) for i in range(self.num_uavs)]

        # Dynamic tracking
        self.tracked_objects: List[TrackedObject] = []
        self.next_track_label = 0
        self.global_step = 0

        # Observation space
        obs_cfg = CONFIG["obs"]
        self.observation_space = spaces.Dict({
            "pointcloud": spaces.Box(0.0, 1.0, (obs_cfg["pointcloud_dim"],), dtype=np.float32),
            "grid_map": spaces.Box(0.0, 1.0, (obs_cfg["grid_h"], obs_cfg["grid_w"]), dtype=np.float32),
            "self_state": spaces.Box(-1.0, 1.0, (obs_cfg["state_dim"],), dtype=np.float32),
            "dynamic_obs": spaces.Box(-1.0, 1.0, (obs_cfg["dyn_obs_dim"],), dtype=np.float32),
        })

        # Action space
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        # Rendering
        self.render_buf = None
        self.fig = None
        self.ax = None

    def _generate_safe_goal(self, min_dist_from_obs: float = None, margin: float = None) -> np.ndarray:
        """Generate a goal position that's safe from obstacles and boundaries.

        Tries 50 times to find a position away from obstacles, falls back to center of world.
        """
        if min_dist_from_obs is None:
            min_dist_from_obs = CONFIG["safe_goal"]["min_dist_from_obs"]
        if margin is None:
            margin = CONFIG["safe_goal"]["margin"]
        for _ in range(50):
            gx = np.random.uniform(margin, self.world_size - margin)
            gy = np.random.uniform(margin, self.world_size - margin)
            safe = True
            for obs in self.static_obstacles:
                if math.hypot(gx - obs[0], gy - obs[1]) < obs[2] + min_dist_from_obs:
                    safe = False
                    break
            if safe:
                return np.array([gx, gy], dtype=np.float32)
        # Fallback: center of world
        return np.array([self.world_size / 2, self.world_size / 2], dtype=np.float32)

    def _generate_obstacles(self):
        """Generate random static obstacles."""
        self.static_obstacles = []
        rng = np.random.RandomState(42)
        num_obs = rng.randint(8, 15)
        for _ in range(num_obs):
            x = rng.uniform(5, self.world_size - 5)
            y = rng.uniform(5, self.world_size - 5)
            r = rng.uniform(0.8, 2.0)
            self.static_obstacles.append([x, y, r])

    def reset(self, seed=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)

        self.global_step = 0
        self.tracked_objects.clear()
        self.next_track_label = 0

        # Generate new obstacles each reset (do this first so safe goals work)
        self._generate_obstacles()

        for uav in self.uavs:
            uav.reset(self.world_size)
            # Override with safe goal
            uav.goal = self._generate_safe_goal()
            uav.prev_goal_dist = math.hypot(uav.goal[0] - uav.kinematics.x,
                                             uav.goal[1] - uav.kinematics.y)

        # Sense initially to populate observations
        observations = []
        for uav in self.uavs:
            self._update_tracking()
            obs = uav.get_observation(
                self.static_obstacles + [o.pos for o in self.tracked_objects],
                self.tracked_objects,
                self.world_size,
            )
            observations.append(obs)

        return observations

    def step(self, actions: np.ndarray):
        """
        Take a step with all UAVs.
        actions: (num_uavs, 2) array of [a_th, delta]
        Returns: (obs, rewards, done, info)
        """
        actions = np.asarray(actions)
        rewards = np.zeros(self.num_uavs, dtype=np.float32)
        dones = np.zeros(self.num_uavs, dtype=bool)
        infos = [{} for _ in range(self.num_uavs)]

        # Step kinematics
        for i, uav in enumerate(self.uavs):
            if not uav.arrived:
                uav.kinematics.step(actions[i, 0], actions[i, 1], self.dt)
                # Boundary collision detection (after step, before clamping)
                if (uav.kinematics.x <= 0 or uav.kinematics.x >= self.world_size or
                    uav.kinematics.y <= 0 or uav.kinematics.y >= self.world_size):
                    uav.collided = True
                # Always clamp to world
                uav.kinematics.x = np.clip(uav.kinematics.x, 0, self.world_size)
                uav.kinematics.y = np.clip(uav.kinematics.y, 0, self.world_size)
                uav.steps += 1

            # Record path
            uav.path_history.append((uav.kinematics.x, uav.kinematics.y))

        # Check collisions between UAVs and obstacles
        uav_positions = [(uav.kinematics.x, uav.kinematics.y) for uav in self.uavs]

        for i, uav in enumerate(self.uavs):
            if uav.collided or uav.arrived:
                continue

            # Collision with static obstacles
            for obs in self.static_obstacles:
                dx = uav.kinematics.x - obs[0]
                dy = uav.kinematics.y - obs[1]
                dist = math.hypot(dx, dy)
                if dist < self.uav_radius + obs[2]:
                    uav.collided = True
                    break

            # Collision with other UAVs
            if not uav.collided:
                for j, other in enumerate(self.uavs):
                    if i == j or other.collided:
                        continue
                    dx = uav.kinematics.x - other.kinematics.x
                    dy = uav.kinematics.y - other.kinematics.y
                    if math.hypot(dx, dy) < self.uav_radius * 2:
                        uav.collided = True
                        break

        # Update dynamic tracking
        self._update_tracking()

        # Compute observations and rewards
        all_obstacles = self.static_obstacles + [np.array([o.pos[0], o.pos[1], 0.5]) for o in self.tracked_objects]

        for i, uav in enumerate(self.uavs):
            # Get observation
            obs = uav.get_observation(all_obstacles, self.tracked_objects, self.world_size)
            infos[i]["observation"] = obs

            # Compute reward
            reward = CONFIG["reward"]["step_penalty"]

            if uav.collided:
                reward += uav.get_collision_penalty()
                uav.collided = False
                # UAV continues from current position — no reset.
                # The model must learn to avoid collisions by itself.

            elif not uav.arrived:
                # Guidance reward: reward for moving closer to goal
                dx_curr = math.hypot(uav.goal[0] - uav.kinematics.x,
                                     uav.goal[1] - uav.kinematics.y)
                guidance_reward = CONFIG["reward"]["guidance_scale"] * (uav.prev_goal_dist - dx_curr)
                uav.prev_goal_dist = dx_curr
                reward += guidance_reward

                # Check goal arrival
                dx = uav.goal[0] - uav.kinematics.x
                dy = uav.goal[1] - uav.kinematics.y
                if math.hypot(dx, dy) < self.goal_radius:
                    uav.success_count += 1
                    reward += CONFIG["reward"]["goal_reward"]
                    # New random goal (safe placement)
                    uav.goal = self._generate_safe_goal()
                    uav.arrived = False
                    uav.prev_goal_dist = math.hypot(uav.goal[0] - uav.kinematics.x,
                                                     uav.goal[1] - uav.kinematics.y)

            # Step limit
            if uav.steps >= self.max_steps:
                dones[i] = True

            rewards[i] = reward

        self.global_step += 1
        infos[0]["global_step"] = self.global_step

        # Collect observations
        observations = [infos[i]["observation"] for i in range(self.num_uavs)]

        # Episode done when all UAVs done or max global steps
        global_done = self.global_step >= self.max_steps

        return observations, rewards, dones, infos

    def _update_tracking(self):
        """DBSCAN clustering → Hungarian matching → Kalman filter update."""
        # Collect all UAV positions as measurements
        measurements = []
        for uav in self.uavs:
            # Only track UAVs that are not too close to each other
            measurements.append([uav.kinematics.x, uav.kinematics.y])

        if not measurements:
            return

        measurements = np.array(measurements)

        # Predict all tracked objects
        for obj in self.tracked_objects:
            obj.kalman.predict()
            obj.kalman.lost_count += 1

        if len(measurements) == 0:
            # Remove lost objects
            self.tracked_objects = [o for o in self.tracked_objects if not o.is_lost]
            return

        # DBSCAN clustering on measurements
        if len(measurements) >= CONFIG["tracking"]["dbscan_min_samples"]:
            clustering = DBSCAN(
                eps=CONFIG["tracking"]["dbscan_eps"],
                min_samples=CONFIG["tracking"]["dbscan_min_samples"]
            ).fit(measurements)
            labels = clustering.labels_
        else:
            labels = np.zeros(len(measurements), dtype=int)

        # Cluster centers
        unique_labels = set(labels)
        clusters = []
        for label in unique_labels:
            if label == -1:  # noise
                continue
            mask = labels == label
            cluster_pts = measurements[mask]
            center = cluster_pts.mean(axis=0)
            clusters.append(center)

        if not clusters:
            return

        clusters = np.array(clusters)

        if len(self.tracked_objects) == 0:
            # Initialize new tracks
            for c in clusters:
                obj = TrackedObject(
                    kalman=KalmanFilterCV(c[0], c[1], self.dt),
                    label=self.next_track_label,
                )
                self.tracked_objects.append(obj)
                self.next_track_label += 1
        else:
            # Hungarian matching
            track_pos = np.array([o.pos for o in self.tracked_objects])
            cost_matrix = np.zeros((len(self.tracked_objects), len(clusters)))
            for i, tp in enumerate(track_pos):
                for j, cp in enumerate(clusters):
                    cost_matrix[i, j] = math.hypot(tp[0] - cp[0], tp[1] - cp[1])

            row_idx, col_idx = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            matched_tracks = set()
            matched_clusters = set()
            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] < CONFIG["tracking"]["dbscan_eps"] * 3:
                    self.tracked_objects[r].kalman.update(clusters[c])
                    self.tracked_objects[r].age += 1
                    matched_tracks.add(r)
                    matched_clusters.add(c)

            # Create new tracks for unmatched clusters
            for j, cp in enumerate(clusters):
                if j not in matched_clusters:
                    obj = TrackedObject(
                        kalman=KalmanFilterCV(cp[0], cp[1], self.dt),
                        label=self.next_track_label,
                    )
                    self.tracked_objects.append(obj)
                    self.next_track_label += 1

        # Remove lost tracks
        self.tracked_objects = [o for o in self.tracked_objects if not o.is_lost]
        # Limit max tracked objects
        max_track = CONFIG["tracking"]["max_tracked"]
        if len(self.tracked_objects) > max_track:
            self.tracked_objects = self.tracked_objects[:max_track]

    def render(self, mode: str = "human"):
        """Render the environment (placeholder for matplotlib/Pygame integration)."""
        pass

    def close(self):
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None
