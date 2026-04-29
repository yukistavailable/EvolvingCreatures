from dataclasses import dataclass, field

import numpy as np

GRAVITY = np.array([0.0, -9.81])
DT = 1.0 / 60.0
RESTITUTION = 0.5


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

    def velocity_at(self, world_point: np.ndarray) -> np.ndarray:
        """Calculate the velocity at a specific world point."""
        r = world_point - self.pos
        return self.vel + self.angular_vel * np.array([-r[1], r[0]])

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


def resolve_ground_collision(body: Body, ground_y: float):
    """Simple collision response with the ground."""
    corners = body.get_corners()
    for corner in corners:
        if corner[1] < ground_y:
            penetration = ground_y - corner[1]
            body.pos[1] += penetration
            contact_vel = body.velocity_at(corner)

            contact_vel_y = contact_vel[1]
            if contact_vel_y > 0:
                continue

            r = corner - body.pos
            # j = -(1 + e) * vn / (1/m + (r×n)² / I)
            denominator = (1 / body.mass) + (
                np.cross(r, np.array([0, 1])) ** 2
            ) / body.inertia
            if denominator == 0:
                continue
            j = -(1 + RESTITUTION) * contact_vel_y / denominator

            body.vel[1] += j / body.mass
            body.angular_vel += np.cross(r, np.array([0, j])) / body.inertia
