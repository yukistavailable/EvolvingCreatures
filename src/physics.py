from dataclasses import dataclass, field

import numpy as np

GRAVITY = np.array([0.0, -9.81])
DT = 1.0 / 60.0
RESTITUTION = 0.5  # coefficient of restitution
FRICTION = 0.5  # coefficient of friction
VELOCITY_THRESHOLD = 0.01  # threshold for considering a collision as resting contact
SLEEP_LINEAR_THRESHOLD = 0.05
SLEEP_ANGULAR_THRESHOLD = 0.1


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
    sleeping: bool = False

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
        if self.sleeping:
            self.force = np.zeros(2)
            self.torque = 0.0
            return

        acc = self.force / self.mass + GRAVITY
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt

        angular_acc = self.torque / self.inertia
        self.angular_vel = self.angular_vel + angular_acc * dt
        self.angle = self.angle + self.angular_vel * dt

        self.force = np.zeros(2)
        self.torque = 0.0

        if self.vel is not None and np.linalg.norm(self.vel) < VELOCITY_THRESHOLD:
            self.vel = np.zeros(2)

        if abs(self.angular_vel) < VELOCITY_THRESHOLD:
            self.angular_vel = 0.0

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
    if body.sleeping:
        body.vel = np.zeros(2)
        body.angular_vel = 0.0
        body.force = np.zeros(2)
        body.torque = 0.0
        return

    corners = body.get_corners()
    max_penetration = 0.0
    for corner in corners:
        penetration = ground_y - corner[1]
        if penetration > max_penetration:
            max_penetration = penetration
    if max_penetration > 0:
        body.pos[1] += max_penetration

    # Recalculate corners after position correction
    corners = body.get_corners()

    for corner in corners:
        if corner[1] < ground_y + 0.01:
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
            j: float = -(1 + RESTITUTION) * contact_vel_y / denominator

            body.vel[1] += j / body.mass
            body.angular_vel += np.cross(r, np.array([0, j])) / body.inertia

            # Friction
            contact_vel = body.velocity_at(corner)
            contact_vel_x = contact_vel[0]

            if abs(contact_vel_x) > VELOCITY_THRESHOLD:
                denominator_friction = (1 / body.mass) + (
                    np.cross(r, np.array([1, 0])) ** 2
                ) / body.inertia
                if denominator_friction == 0:
                    continue
                jt = -contact_vel_x / denominator_friction
                max_friction = FRICTION * abs(j)
                jt = np.clip(jt, -max_friction, max_friction)

                body.vel[0] += jt / body.mass
                body.angular_vel += np.cross(r, np.array([jt, 0])) / body.inertia

    # Sleep check
    on_ground = any(corner[1] <= ground_y + 0.01 for corner in corners)
    if (
        on_ground
        and np.linalg.norm(body.vel) < SLEEP_LINEAR_THRESHOLD
        and abs(body.angular_vel) < SLEEP_ANGULAR_THRESHOLD
    ):
        body.sleeping = True
    else:
        body.sleeping = False
