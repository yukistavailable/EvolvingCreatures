from dataclasses import dataclass, field

import numpy as np

GRAVITY = np.array([0.0, -9.81])
DT = 1.0 / 60.0
RESTITUTION = 0.5  # coefficient of restitution
FRICTION = 0.5  # coefficient of friction
VELOCITY_THRESHOLD = 0.01  # threshold for considering a collision as resting contact
SLEEP_LINEAR_THRESHOLD = 0.1
SLEEP_ANGULAR_THRESHOLD = 0.1
JOINT_ITERATIONS = 8


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


@dataclass
class RevoluteJoint:
    """
    A simple revolute joint connecting two bodies at a pivot point.
    """

    body_a: Body
    body_b: Body
    anchor_a: np.ndarray  # Local anchor point on body_a
    anchor_b: np.ndarray  # Local anchor point on body_b

    angle_min: float = None
    angle_max: float = None

    sleeping: bool = False

    def get_world_anchor_a(self) -> np.ndarray:
        return self.body_a.local_to_world(self.anchor_a)

    def get_world_anchor_b(self) -> np.ndarray:
        return self.body_b.local_to_world(self.anchor_b)

    def get_angle(self) -> float:
        return self.body_b.angle - self.body_a.angle

    def check_sleeping(self):
        if self.body_a.sleeping or self.body_b.sleeping:
            self.sleeping = True
            self.body_a.sleeping = True
            self.body_b.sleeping = True
        else:
            self.sleeping = False


def solve_joint_constraint(joint: RevoluteJoint):
    """
    Solve the revolute joint constraint by applying corrective impulses to the connected bodies.
    Specifically, this function will ensure that the anchor points on both bodies coincide in world space and that the relative angle between the bodies stays within specified limits (if any).
    """
    joint.check_sleeping()
    if joint.sleeping:
        return

    body_a = joint.body_a
    body_b = joint.body_b

    # Position constraint
    wa = joint.get_world_anchor_a()
    wb = joint.get_world_anchor_b()
    delta = wb - wa
    distance = np.linalg.norm(delta)
    if distance < 1e-6:
        pass
    else:
        inv_mass_a = 1 / body_a.mass if body_a.mass > 0 else 0
        inv_mass_b = 1 / body_b.mass if body_b.mass > 0 else 0

        # Baumgarte stabilization
        # See https://box2d.org/posts/2024/02/solver2d/#:~:text=normal)%20%2B%20originalSeparation%3B-,Baumgarte%20Stabilization,-Baumgarte%20Stabilization%20comes
        beta = 0.3
        correction = delta * beta

        body_a.pos += correction * inv_mass_a / (inv_mass_a + inv_mass_b)
        body_b.pos -= correction * inv_mass_b / (inv_mass_a + inv_mass_b)

    # Velocity constraint
    wa = joint.get_world_anchor_a()
    wb = joint.get_world_anchor_b()

    ra = wa - body_a.pos
    rb = wb - body_b.pos

    vel_a = body_a.velocity_at(wa)
    vel_b = body_b.velocity_at(wb)
    relative_vel = vel_b - vel_a

    if np.linalg.norm(relative_vel) < VELOCITY_THRESHOLD:
        pass
    else:
        # Compute the impulse to make the relative velocity zero at the constraint point.
        inv_mass_a = 1 / body_a.mass if body_a.mass > 0 else 0
        inv_mass_b = 1 / body_b.mass if body_b.mass > 0 else 0

        # K * impulse = -relative_vel
        # K[0][0] = 1/ma + 1/mb + ra_y^2/Ia + rb_y^2/Ib
        # K[0][1] = -ra_x*ra_y/Ia - rb_x*rb_y/Ib
        # K[1][0] = K[0][1]
        # K[1][1] = 1/ma + 1/mb + ra_x^2/Ia + rb_x^2/Ib
        K = np.array(
            [
                [
                    inv_mass_a
                    + inv_mass_b
                    + ra[1] ** 2 / body_a.inertia
                    + rb[1] ** 2 / body_b.inertia,
                    -ra[0] * ra[1] / body_a.inertia - rb[0] * rb[1] / body_b.inertia,
                ],
                [
                    -ra[0] * ra[1] / body_a.inertia - rb[0] * rb[1] / body_b.inertia,
                    inv_mass_a
                    + inv_mass_b
                    + ra[0] ** 2 / body_a.inertia
                    + rb[0] ** 2 / body_b.inertia,
                ],
            ]
        )

        impulse = np.linalg.solve(K, -relative_vel)
        body_a.vel -= impulse * inv_mass_a
        body_a.angular_vel -= np.cross(ra, impulse) / body_a.inertia
        body_b.vel += impulse * inv_mass_b
        body_b.angular_vel += np.cross(rb, impulse) / body_b.inertia

    # Angle limits
    if joint.angle_min is not None and joint.angle_max is not None:
        relative_angle = joint.get_angle()
        total_inv_i = 1.0 / body_a.inertia + 1.0 / body_b.inertia

        if relative_angle < joint.angle_min:
            diff = joint.angle_min - relative_angle
            body_a.angle -= diff * (1.0 / body_a.inertia) / total_inv_i * 0.5
            body_b.angle += diff * (1.0 / body_b.inertia) / total_inv_i * 0.5
        elif relative_angle > joint.angle_max:
            diff = relative_angle - joint.angle_max
            body_a.angle += diff * (1.0 / body_a.inertia) / total_inv_i * 0.5
            body_b.angle -= diff * (1.0 / body_b.inertia) / total_inv_i * 0.5


def solve_joints(joints: list[RevoluteJoint], iterations: int = JOINT_ITERATIONS):
    for _ in range(iterations):
        for joint in joints:
            solve_joint_constraint(joint)


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
