from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.physics import Body, RevoluteJoint


@dataclass
class ConnectionGene:
    child_idx: int = 0

    attach_side: int = 0  # 0=bottom, 1=right, 2=top, 3=left
    attach_pos: float = 0.0  # -1.0-1.0, -1.0=left or bottom, 1.0=right or top
    angle_offset: float = (
        0.0  # 0.0=straight, positive=clockwise, negative=counterclockwise
    )
    scale: float = 1.0
    reflection: bool = False


@dataclass
class MorphNode:
    width: float = 0.3
    height: float = 0.15

    # joint angle limits for the revolute joint connecting this part to its parent
    joint_angle_min: float = -np.pi / 4
    joint_angle_max: float = np.pi / 4

    recursive_limit: int = 1

    connections: list[ConnectionGene] = field(default_factory=list)

    hue: float = 0.5

    # --- Controller / effector parameters ---------------------------------
    # A non-root part drives the single revolute joint connecting it to its parent.
    # The motor command for that joint is produced by
    # a sine oscillator plus an optional proprioceptive (joint-angle) feedback
    # term, then clipped to [-1, 1] and fed to RevoluteJoint.motor_torque.
    #
    #   cmd(t) = osc_amplitude * sin(2π · osc_frequency · t + phase)
    #            + osc_offset
    #            + sensor_gain * normalized_joint_angle
    #
    # where `phase` is the part's effective phase (osc_phase, with +π added to
    # reflected copies so symmetric limbs move in anti-phase).
    osc_amplitude: float = 0.5  # 0..1, sine amplitude of the motor command
    osc_frequency: float = 1.0  # Hz
    osc_phase: float = 0.0  # radians
    osc_offset: float = 0.0  # -1..1, tonic bias
    sensor_gain: float = 0.0  # joint-angle proprioceptive feedback gain


@dataclass
class Genotype:
    nodes: list[MorphNode] = field(default_factory=list)


def hue_to_rgb(hue: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.6, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255))


@dataclass
class PartInfo:
    body: Body
    joint: Optional[RevoluteJoint] = None
    parent: Optional["PartInfo"] = None
    children: list["PartInfo"] = field(default_factory=list)
    node_index: int = 0
    phase_offset: float = 0.0  # effective oscillator phase (including reflection)


def compute_anchor_and_child_pos(
    parent_body: Body,
    connection: ConnectionGene,
    child_width: float,
    child_height: float,
    reflect: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the local anchor positions and the child's world position based on the parent's body and the connection gene.
    Returns:
        (parent_anchor_local, child_anchor_local, child_world_pos)
    """
    pw, ph = parent_body.width, parent_body.height
    cw, ch = child_width, child_height

    side = connection.attach_side % 4
    t = np.clip(connection.attach_pos, -0.9, 0.9)

    if reflect:
        t = -t

    if side == 0:  # bottom
        parent_anchor = np.array([t * pw / 2, -ph / 2])
        child_anchor = np.array([0.0, ch / 2])
    elif side == 1:  # right
        parent_anchor = np.array([pw / 2, t * ph / 2])
        child_anchor = np.array([-cw / 2, 0.0])
    elif side == 2:  # top
        parent_anchor = np.array([t * pw / 2, ph / 2])
        child_anchor = np.array([0.0, -ch / 2])
    else:  # left
        parent_anchor = np.array([-pw / 2, t * ph / 2])
        child_anchor = np.array([cw / 2, 0.0])

    if reflect and side == 1:
        parent_anchor = np.array([-pw / 2, t * ph / 2])
        child_anchor = np.array([cw / 2, 0.0])
    elif reflect and side == 3:
        parent_anchor = np.array([pw / 2, t * ph / 2])
        child_anchor = np.array([-cw / 2, 0.0])

    parent_anchor_world = parent_body.local_to_world(parent_anchor)

    angle_offset = connection.angle_offset
    if reflect:
        angle_offset = -angle_offset
    child_angle = parent_body.angle + angle_offset

    c, s = np.cos(child_angle), np.sin(child_angle)
    rot = np.array([[c, -s], [s, c]])
    child_world_pos = parent_anchor_world - rot @ child_anchor

    return parent_anchor, child_anchor, child_world_pos, child_angle


class Creature:
    MAX_PARTS = 12

    # Effector max strength is proportional to the size of the parts a joint connects.
    # A single global torque value # would be far too strong for small/light parts and fling them, so each
    # joint's max_motor_torque is scaled by part area at build time.
    MOTOR_TORQUE_PER_AREA = 2.0

    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.parts: list[PartInfo] = []
        self.bodies: list[Body] = []
        self.joints: list[RevoluteJoint] = []

    def build(self, start_x: float = 0.0, start_y: float = 2.0):
        if not self.genotype.nodes:
            return

        root_node = self.genotype.nodes[0]

        root_body = Body(
            width=root_node.width,
            height=root_node.height,
            pos=np.array([start_x, start_y]),
        )

        root_part = PartInfo(body=root_body, node_index=0)
        self.parts.append(root_part)
        self.bodies.append(root_body)

        recursion_count: dict[int, int] = {0: 1}
        self._build_children(root_part, root_node, recursion_count)

        for part in self.parts:
            node = self.genotype.nodes[part.node_index]
            part.body.color = hue_to_rgb(node.hue)

    def _build_children(
        self,
        parent_part: PartInfo,
        parent_node: MorphNode,
        recursion_count: dict[int, int],
    ):

        for conn in parent_node.connections:
            if len(self.parts) >= self.MAX_PARTS:
                return

            child_idx = conn.child_idx % len(self.genotype.nodes)
            child_node = self.genotype.nodes[child_idx]

            current_count = recursion_count.get(child_idx, 0)
            if current_count >= child_node.recursive_limit:
                continue

            self._spawn_child(
                parent_part, conn, child_idx, child_node, recursion_count, reflect=False
            )

            if conn.reflection and len(self.parts) < self.MAX_PARTS:
                self._spawn_child(
                    parent_part,
                    conn,
                    child_idx,
                    child_node,
                    recursion_count,
                    reflect=True,
                )

    def _spawn_child(
        self,
        parent_part: PartInfo,
        conn: ConnectionGene,
        child_idx: int,
        child_node: MorphNode,
        recursion_count: dict[int, int],
        reflect: bool,
    ):

        scale = np.clip(conn.scale, 0.3, 1.5)
        child_width = child_node.width * scale
        child_height = child_node.height * scale

        parent_anchor, child_anchor, child_pos, child_angle = (
            compute_anchor_and_child_pos(
                parent_part.body, conn, child_width, child_height, reflect
            )
        )

        child_body = Body(
            width=child_width,
            height=child_height,
            pos=child_pos.copy(),
            angle=child_angle,
        )

        joint = RevoluteJoint(
            body_a=parent_part.body,
            body_b=child_body,
            anchor_a=parent_anchor.copy(),
            anchor_b=child_anchor.copy(),
            angle_min=child_node.joint_angle_min,
            angle_max=child_node.joint_angle_max,
        )

        # Muscle strength scales with the parts it joins (Sims §3.3): area is
        # the 2D analog of cross-section, so small light parts get weak motors
        # and are not catapulted by an over-strong global torque.
        area_parent = parent_part.body.width * parent_part.body.height
        area_child = child_body.width * child_body.height
        joint.max_motor_torque = self.MOTOR_TORQUE_PER_AREA * max(
            area_parent, area_child
        )

        child_part = PartInfo(
            body=child_body,
            joint=joint,
            parent=parent_part,
            node_index=child_idx,
            phase_offset=child_node.osc_phase + (np.pi if reflect else 0.0),
        )
        parent_part.children.append(child_part)
        self.parts.append(child_part)
        self.bodies.append(child_body)
        self.joints.append(joint)

        new_count = recursion_count.copy()
        new_count[child_idx] = new_count.get(child_idx, 0) + 1
        self._build_children(child_part, child_node, new_count)

    @staticmethod
    def _normalized_joint_angle(joint: RevoluteJoint) -> float:
        """Map a joint's current relative angle into roughly [-1, 1].

        This is the joint-angle sensor signal. When the joint has limits, the
        angle is normalized against them; otherwise a +/- pi/2 range is used.
        """
        angle = joint.get_angle()
        lo, hi = joint.angle_min, joint.angle_max
        if lo is not None and hi is not None and hi > lo:
            return float(np.clip(2.0 * (angle - lo) / (hi - lo) - 1.0, -1.0, 1.0))
        return float(np.clip(angle / (np.pi / 2), -1.0, 1.0))

    def apply_control(self, t: float):
        """Compute and write motor commands for every joint at time t.

        Each non-root part drives its single joint-to-parent with a sine
        oscillator (the effector) plus optional joint-angle feedback.
        """
        for part in self.parts:
            if part.joint is None:
                continue
            node = self.genotype.nodes[part.node_index]

            cmd = node.osc_amplitude * np.sin(
                2.0 * np.pi * node.osc_frequency * t + part.phase_offset
            )
            cmd += node.osc_offset
            if node.sensor_gain != 0.0:
                cmd += node.sensor_gain * self._normalized_joint_angle(part.joint)

            part.joint.motor_torque = float(np.clip(cmd, -1.0, 1.0))

    def total_mass(self) -> float:
        return float(sum(b.mass for b in self.bodies))

    def center_of_mass(self) -> np.ndarray:
        total = self.total_mass()
        if total == 0.0:
            return np.zeros(2)
        c = np.zeros(2)
        for b in self.bodies:
            c = c + b.mass * b.pos
        return c / total

    def snapshot(self) -> list[np.ndarray]:
        """Per-body corner arrays, for rendering a recorded run."""
        return [b.get_corners() for b in self.bodies]


def _bodies_finite(bodies: list[Body], bound: float = 1e3) -> bool:
    for b in bodies:
        if not np.all(np.isfinite(b.pos)) or np.any(np.abs(b.pos) > bound):
            return False
    return True
