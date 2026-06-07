from dataclasses import dataclass, field

import numpy as np

GRAVITY = np.array([0.0, -9.81])
DT = 1.0 / 60.0
RESTITUTION = 0.4  # coefficient of restitution
FRICTION = 0.5  # coefficient of friction
VELOCITY_THRESHOLD = 0.01  # threshold for considering a collision as resting contact
SLEEP_LINEAR_THRESHOLD = 0.2
SLEEP_ANGULAR_THRESHOLD = 0.2
JOINT_ITERATIONS = 8
PENETRATION_SLOP = 0.01
BAUMGARTE_BETA = 0.3
CONTACT_ITERATIONS = 10


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

    def integrate_velocity(self, dt: float):
        """Update velocity from forces (first half of semi-implicit Euler).

        Corresponds to: V2 = V1 + dt * M^{-1} * F_ext  (Eq. 33 in Catto)
        After this step, constraint impulses are applied to V2,
        then position is updated using the corrected velocity.
        """
        if self.sleeping:
            self.force = np.zeros(2)
            self.torque = 0.0
            return

        acc = self.force / self.mass + GRAVITY
        self.vel = self.vel + acc * dt

        angular_acc = self.torque / self.inertia
        self.angular_vel = self.angular_vel + angular_acc * dt

        self.force = np.zeros(2)
        self.torque = 0.0

    def integrate_position(self, dt: float):
        """Update position from (constraint-corrected) velocity.

        Corresponds to: x2 = x1 + dt * v2  (Eq. 36 in Catto)
        """
        if self.sleeping:
            return

        self.pos = self.pos + self.vel * dt
        self.angle = self.angle + self.angular_vel * dt

    def integrate(self, dt: float):
        """Legacy: combined integration (semi-implicit Euler).
        Use integrate_velocity + integrate_position for constraint-based flow.
        """
        self.integrate_velocity(dt)
        self.integrate_position(dt)

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

    motor_torque: float = 0.0
    max_motor_torque: float = 5.0

    lambda_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    lambda_angle: float = 0.0

    # Precomputed values
    _ra: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _rb: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _K: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    _K_inv: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    _bias_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _mass_angle: float = 0.0
    _bias_angle: float = 0.0

    def get_world_anchor_a(self) -> np.ndarray:
        return self.body_a.local_to_world(self.anchor_a)

    def get_world_anchor_b(self) -> np.ndarray:
        return self.body_b.local_to_world(self.anchor_b)

    def get_angle(self) -> float:
        return self.body_b.angle - self.body_a.angle

    def check_sleeping(self):
        a_can_sleep = (
            np.linalg.norm(self.body_a.vel) < SLEEP_LINEAR_THRESHOLD
            and abs(self.body_a.angular_vel) < SLEEP_ANGULAR_THRESHOLD
        )
        b_can_sleep = (
            np.linalg.norm(self.body_b.vel) < SLEEP_LINEAR_THRESHOLD
            and abs(self.body_b.angular_vel) < SLEEP_ANGULAR_THRESHOLD
        )

        if a_can_sleep and b_can_sleep:
            self.sleeping = True
            self.body_a.sleeping = True
            self.body_b.sleeping = True
        else:
            self.sleeping = False
            self.body_a.sleeping = False
            self.body_b.sleeping = False

    def precompute(self, dt: float):
        """Precompute effective mass and bias for PGS iterations.

        Called once per time step before the PGS loop begins.
        """
        # Reset accumulated impulses for this time step
        self.lambda_point = np.zeros(2)
        self.lambda_angle = 0.0

        body_a = self.body_a
        body_b = self.body_b

        wa = self.get_world_anchor_a()
        wb = self.get_world_anchor_b()

        self._ra = wa - body_a.pos
        self._rb = wb - body_b.pos
        ra = self._ra
        rb = self._rb

        inv_ma = 1.0 / body_a.mass
        inv_mb = 1.0 / body_b.mass
        inv_Ia = 1.0 / body_a.inertia
        inv_Ib = 1.0 / body_b.inertia

        # --- Point constraint effective mass (2×2) ---
        # K = J M⁻¹ Jᵀ, expanded for 2D revolute joint:
        #   K[0][0] = 1/ma + 1/mb + ra_y²/Ia + rb_y²/Ib
        #   K[0][1] = -ra_x*ra_y/Ia - rb_x*rb_y/Ib
        #   K[1][0] = K[0][1]
        #   K[1][1] = 1/ma + 1/mb + ra_x²/Ia + rb_x²/Ib
        self._K = np.array(
            [
                [
                    inv_ma + inv_mb + ra[1] ** 2 * inv_Ia + rb[1] ** 2 * inv_Ib,
                    -ra[0] * ra[1] * inv_Ia - rb[0] * rb[1] * inv_Ib,
                ],
                [
                    -ra[0] * ra[1] * inv_Ia - rb[0] * rb[1] * inv_Ib,
                    inv_ma + inv_mb + ra[0] ** 2 * inv_Ia + rb[0] ** 2 * inv_Ib,
                ],
            ]
        )
        self._K_inv = np.linalg.inv(self._K)

        # Baumgarte bias for position drift
        # Paper Eq. 20: JV = -βC, where C = wb - wa (position error)
        # We want Cdot to drive bodies back together: bias = -(β/dt) * C
        position_error = wb - wa  # C = x_b + r_b - x_a - r_a
        self._bias_point = -(BAUMGARTE_BETA / dt) * position_error

        # Angle limit effective mass (scalar)
        # For relative angle constraint, J = [0, 0, -1, 0, 0, +1]
        # So J M⁻¹ Jᵀ = 1/Ia + 1/Ib
        self._mass_angle = inv_Ia + inv_Ib

        # Angle limit bias: drive angle back within limits
        self._bias_angle = 0.0
        if self.angle_min is not None and self.angle_max is not None:
            relative_angle = self.get_angle()
            if relative_angle < self.angle_min:
                # C = relative_angle - angle_min < 0, want to push angle up
                self._bias_angle = -(BAUMGARTE_BETA / dt) * (
                    relative_angle - self.angle_min
                )
            elif relative_angle > self.angle_max:
                # C = relative_angle - angle_max > 0, want to push angle down
                self._bias_angle = -(BAUMGARTE_BETA / dt) * (
                    relative_angle - self.angle_max
                )

    def solve_point_constraint(self):
        """Solve the 2D point constraint (anchor coincidence) — one PGS step.

        Velocity constraint: Ċ = v_b + ω_b × r_b - v_a - ω_a × r_a = -bias
        """
        body_a = self.body_a
        body_b = self.body_b
        ra = self._ra
        rb = self._rb

        # Relative velocity at anchor: v_b + ω_b × r_b - v_a - ω_a × r_a
        vel_a = body_a.velocity_at(body_a.pos + ra)
        vel_b = body_b.velocity_at(body_b.pos + rb)
        Cdot = vel_b - vel_a

        # Δλ = K⁻¹ * -(Ċ - bias)
        delta_lambda = self._K_inv @ -(Cdot - self._bias_point)

        # Point constraint is equality: λ ∈ (-∞, +∞), no clamping needed
        self.lambda_point += delta_lambda

        # Apply impulse
        inv_ma = 1.0 / body_a.mass
        inv_mb = 1.0 / body_b.mass

        body_a.vel -= delta_lambda * inv_ma
        body_a.angular_vel -= np.cross(ra, delta_lambda) / body_a.inertia

        body_b.vel += delta_lambda * inv_mb
        body_b.angular_vel += np.cross(rb, delta_lambda) / body_b.inertia

    def solve_angle_constraint(self):
        """Solve the angle limit constraint — one PGS step.

        This is an inequality constraint on the relative angle.
        The constraint is: angle_min ≤ θ_b - θ_a ≤ angle_max

        We split this into two one-sided constraints:
            - Lower limit: C_lo = θ_rel - angle_min ≥ 0  →  λ_lo ≥ 0
            - Upper limit: C_hi = angle_max - θ_rel ≥ 0  →  λ_hi ≥ 0

        Combined as a single λ with bounds:
            If below min: λ ≥ 0 (push angle up)
            If above max: λ ≤ 0 (push angle down)
            If within range: no constraint active
        """
        if self.angle_min is None or self.angle_max is None:
            return
        if self._bias_angle == 0.0:
            # Within limits, no constraint active
            self.lambda_angle = 0.0
            return

        body_a = self.body_a
        body_b = self.body_b

        # Relative angular velocity: ω_b - ω_a
        Cdot_angle = body_b.angular_vel - body_a.angular_vel

        delta_lambda = -(Cdot_angle - self._bias_angle) / self._mass_angle

        old_lambda = self.lambda_angle
        self.lambda_angle += delta_lambda

        # Clamp based on which limit is violated
        relative_angle = self.get_angle()
        if relative_angle < self.angle_min:
            # Need positive torque to push angle up: λ ≥ 0
            self.lambda_angle = max(0.0, self.lambda_angle)
        elif relative_angle > self.angle_max:
            # Need negative torque to push angle down: λ ≤ 0
            self.lambda_angle = min(0.0, self.lambda_angle)

        actual_delta = self.lambda_angle - old_lambda

        # J = [0, 0, -1, 0, 0, +1] → impulse on angular velocity only
        body_a.angular_vel -= actual_delta / body_a.inertia
        body_b.angular_vel += actual_delta / body_b.inertia


def solve_joints(
    joints: list[RevoluteJoint], dt: float, iterations: int = JOINT_ITERATIONS
):
    """
    Solve all joint constraints using PGS.
    """
    # Precompute all joints
    for joint in joints:
        joint.check_sleeping()
        if joint.sleeping:
            continue
        joint.precompute(dt)

        # Apply motor torque as external impulse (before constraint solve)
        if joint.motor_torque != 0.0:
            torque = np.clip(joint.motor_torque, -1.0, 1.0) * joint.max_motor_torque
            # Apply as angular velocity change (impulse / inertia)
            motor_impulse = torque * dt
            joint.body_a.angular_vel -= motor_impulse / joint.body_a.inertia
            joint.body_b.angular_vel += motor_impulse / joint.body_b.inertia

    # PGS iterations
    for _ in range(iterations):
        for joint in joints:
            if joint.sleeping:
                continue
            joint.solve_point_constraint()
            joint.solve_angle_constraint()

def step_world(
    bodies: list[Body],
    joints: list["RevoluteJoint"],
    ground_y: float,
    dt: float = DT,
    iterations: int = max(JOINT_ITERATIONS, CONTACT_ITERATIONS),
    allow_sleeping: bool = False,
) -> list["GroundContactConstraint"]:
    """Advance an entire multi-body world by one time step.

    This is the single entry point for simulating a creature (root body plus
    its jointed parts) sitting on the ground. It folds together three things
    that the two-body drop_joint.py demo handled manually and separately:

    1. Sleeping is OFF by default. An actuated creature that momentarily slows
    down must NOT be put to sleep, or its motors stop being applied and it
    freezes mid-stride. Pass allow_sleeping=True only for passive scenes.

    2. Joints and ground contacts are solved in ONE interleaved PGS loop, so
    the foot-vs-ground reaction and the joint chain see each other every
    iteration. Solving them in separate passes (joints fully, then contacts
    fully) lets joints drift and feet sink on articulated bodies.

    3. All bodies, all joints, and all contacts are stepped together, instead
    of integrating two named bodies by hand.

    Step order (semi-implicit Euler with a velocity-level constraint solve):
        integrate velocities -> generate + precompute constraints
        -> interleaved PGS -> integrate positions.

    Returns the contact constraints generated this step (handy for rendering
    or for contact sensors).
    """
    # Sleeping management
    if not allow_sleeping:
        for b in bodies:
            b.sleeping = False
        for j in joints:
            j.sleeping = False

    # Integrate velocities (gravity + any accumulated forces)
    for b in bodies:
        b.integrate_velocity(dt)

    # Generate ground contact constraints for every body
    contacts: list[GroundContactConstraint] = []
    for b in bodies:
        if b.sleeping:
            continue
        contacts.extend(generate_ground_contact_constraints(b, ground_y))

    # Precompute joints (+ apply motors) and contacts
    active_joints: list[RevoluteJoint] = []
    for j in joints:
        if allow_sleeping:
            j.check_sleeping()
        if j.sleeping:
            continue
        j.precompute(dt)

        # Motor torque is applied once, before the solve, as a direct angular
        # velocity change on the two connected bodies (same scheme as
        # solve_joints). This is where the brain's effector output enters.
        if j.motor_torque != 0.0:
            torque = np.clip(j.motor_torque, -1.0, 1.0) * j.max_motor_torque
            motor_impulse = torque * dt
            j.body_a.angular_vel -= motor_impulse / j.body_a.inertia
            j.body_b.angular_vel += motor_impulse / j.body_b.inertia

        active_joints.append(j)

    for c in contacts:
        c.precompute(dt)

    # Interleaved velocity solve (joint <-> contact coupling)
    for _ in range(iterations):
        for j in active_joints:
            j.solve_point_constraint()
            j.solve_angle_constraint()
        for c in contacts:
            c.solve_velocity()

    # Integrate positions using the corrected velocities
    for b in bodies:
        b.integrate_position(dt)

    # Optional sleeping (passive scenes only)
    if allow_sleeping:
        for b in bodies:
            check_sleeping(b, ground_y)

    return contacts


@dataclass
class GroundContactConstraint:
    body: Body
    contact_point: np.ndarray
    penetration: float

    lambda_n: float = 0.0
    lambda_t: float = 0.0

    # r is the vector from the body's center of mass to the contact point
    r: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mass_n: float = 0.0
    mass_t: float = 0.0
    bias: float = 0.0

    def precompute(self, dt: float):
        self.r = self.contact_point - self.body.pos
        n = np.array([0, 1])  # Ground normal
        t = np.array([1, 0])  # Ground tangent

        inv_mass = 1 / self.body.mass if self.body.mass > 0 else 0
        inv_inertia = 1 / self.body.inertia if self.body.inertia > 0 else 0

        r_cross_n = np.cross(self.r, n)
        r_cross_t = np.cross(self.r, t)

        # Let's say j is the impulse magnitude in the normal direction
        # delta_v = j / mass
        # delta_omega = (r × j) / inertia
        # delta_v_n = j * (1/mass + (r×n)² / inertia)
        self.mass_n = inv_mass + (r_cross_n**2) * inv_inertia
        self.mass_t = inv_mass + (r_cross_t**2) * inv_inertia

        # Equation 20
        penetration_correction = max(self.penetration - PENETRATION_SLOP, 0.0)
        self.bias = (BAUMGARTE_BETA / dt) * penetration_correction

        # Equation 15
        v_contact = self.body.velocity_at(self.contact_point)
        v_n = np.dot(v_contact, n)
        if v_n < -VELOCITY_THRESHOLD:
            self.bias += -RESTITUTION * v_n

    def solve_velocity(self):
        """Solve this contact (normal + friction) — one PGS step.

        Extracted from the inner loop of solve_ground_contact_constraints so
        the same single step can be interleaved with joint solving inside
        step_world. Assumes precompute() has already been called this frame.
        """
        body = self.body
        n = np.array([0.0, 1.0])  # Ground normal
        t = np.array([1.0, 0.0])  # Ground tangent

        # Normal constraint (non-penetration): λ_n ≥ 0
        v_contact = body.velocity_at(self.contact_point)
        v_n = np.dot(v_contact, n)
        # target: v_n = bias  →  Δλ = -(v_n - bias) / mass_n
        delta_lambda_n = -(v_n - self.bias) / self.mass_n

        old_lambda_n = self.lambda_n
        self.lambda_n = max(0.0, self.lambda_n + delta_lambda_n)
        impulse_n = (self.lambda_n - old_lambda_n) * n

        body.vel += impulse_n / body.mass
        body.angular_vel += np.cross(self.r, impulse_n) / body.inertia

        # Friction constraint: |λ_t| ≤ μ λ_n
        # Recompute contact velocity after the normal impulse.
        v_contact = body.velocity_at(self.contact_point)
        v_t = np.dot(v_contact, t)
        delta_lambda_t = -v_t / self.mass_t

        old_lambda_t = self.lambda_t
        max_friction = FRICTION * self.lambda_n
        self.lambda_t = np.clip(
            self.lambda_t + delta_lambda_t, -max_friction, max_friction
        )
        impulse_t = (self.lambda_t - old_lambda_t) * t

        body.vel += impulse_t / body.mass
        body.angular_vel += np.cross(self.r, impulse_t) / body.inertia


def generate_ground_contact_constraints(
    body: Body, ground_y: float
) -> list[GroundContactConstraint]:
    constraints = []
    corners = body.get_corners()
    for corner in corners:
        penetration = ground_y - corner[1]
        if penetration > 0.01:
            constraints.append(
                GroundContactConstraint(
                    body=body, contact_point=corner, penetration=penetration
                )
            )
    return constraints


def solve_ground_contact_constraints(
    constraints: list[GroundContactConstraint],
    dt: float,
):
    """Solve contact constraints by building and solving a linear system.

    Following the paper's formulation (Eq. 34):
        A λ = b
    where:
        A = J M⁻¹ Jᵀ       (2k × 2k matrix for k contact points)
        b = bias − J V*     (V* = velocity after external forces)

    After solving for λ, the velocity is updated:
        V_new = V* + M⁻¹ Jᵀ λ

    For body-vs-ground in 2D:
        State vector V = [vx, vy, ω]  (3 unknowns)
        M⁻¹ = diag(1/m, 1/m, 1/I)    (3×3)

    Each contact point i generates 2 rows in J:
        Normal:   J_ni = [0,  1,  rᵢ × n]  = [0,  1,  rᵢₓ]
        Friction: J_ti = [1,  0,  rᵢ × t]  = [1,  0, −rᵢᵧ]
    """
    if not constraints:
        return

    for c in constraints:
        c.precompute(dt)

    # PGS iterations
    for _ in range(CONTACT_ITERATIONS):
        for c in constraints:
            c.solve_velocity()


def check_sleeping(body: Body, ground_y: float):
    """Check if the body should go to sleep."""
    corners = body.get_corners()
    on_ground = any(corner[1] <= ground_y + PENETRATION_SLOP for corner in corners)
    if (
        on_ground
        and np.linalg.norm(body.vel) < SLEEP_LINEAR_THRESHOLD
        and abs(body.angular_vel) < SLEEP_ANGULAR_THRESHOLD
    ):
        body.sleeping = True
    else:
        body.sleeping = False


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
        if corner[1] < ground_y + PENETRATION_SLOP:
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
    on_ground = any(corner[1] <= ground_y + PENETRATION_SLOP for corner in corners)
    if (
        on_ground
        and np.linalg.norm(body.vel) < SLEEP_LINEAR_THRESHOLD
        and abs(body.angular_vel) < SLEEP_ANGULAR_THRESHOLD
    ):
        body.sleeping = True
    else:
        body.sleeping = False
