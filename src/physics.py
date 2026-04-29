from dataclasses import dataclass, field

import numpy as np

GRAVITY = np.array([0.0, -9.81])
DT = 1.0 / 60.0


@dataclass
class Body:
    width: float
    height: float

    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    angular_vel: float = 0.0

    force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torque: float = 0.0

    mass: float = 0.0
    inertia: float = 0.0

    def __post_init__(self):
        density = 1.0
        self.mass = self.width * self.height * density
        self.inertia = self.mass * (self.width**2 + self.height**2) / 12.0

    def apply_force(self, force: np.ndarray):
        """Apply a force to the center of mass."""
        self.force += force

    def apply_force_at(self, force: np.ndarray, point: np.ndarray):
        """Apply a force at a specific point, generating torque."""
        self.force += force
        r = point - self.pos
        self.torque += np.cross(r, force)

    def integrate(self, dt: float):
        """Semi-implicit Euler integration."""

        acc = self.force / self.mass + GRAVITY
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        angular_acc = self.torque / self.inertia
        self.angular_vel = self.angular_vel + angular_acc * dt
        self.angle = self.angle + self.angular_vel * dt

        self.force = np.zeros(2)
        self.torque = 0.0

    def local_to_world(self, local_point: np.ndarray) -> np.ndarray:
        """Convert a point from local space to world space."""
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return self.pos + rotation_matrix @ local_point

    def get_corners(self) -> np.ndarray:
        """Get the world coordinates of the corners of the body."""
        half_width = self.width / 2
        half_height = self.height / 2
        local_corners = np.array(
            [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height],
            ]
        )
        return np.array([self.local_to_world(corner) for corner in local_corners])
