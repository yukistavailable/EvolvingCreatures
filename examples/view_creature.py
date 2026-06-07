import numpy as np
import pygame

import src.creature as cr
from src.creature import Genotype, MorphNode, ConnectionGene, Creature
from src.physics import DT, step_world


def _crawler(n=5, freq=1.2, amp=0.9, ang=0.6, wave=1.2, hue_root=0.55):
    """Multi-segment chain driven by a traveling-wave phase gradient."""
    nodes = [
        MorphNode(
            width=0.32,
            height=0.12,
            hue=hue_root,
            connections=[ConnectionGene(child_idx=1, attach_side=1)],
        )
    ]
    for i in range(1, n):
        conns = [ConnectionGene(child_idx=i + 1, attach_side=1)] if i < n - 1 else []
        nodes.append(
            MorphNode(
                width=0.32,
                height=0.12,
                hue=(hue_root + 0.07 * i) % 1.0,
                joint_angle_min=-ang,
                joint_angle_max=ang,
                osc_amplitude=amp,
                osc_frequency=freq,
                osc_phase=i * wave,
                connections=conns,
            )
        )
    return Genotype(nodes=nodes)


def _biped(freq=1.6, amp=1.0, ang=0.5):
    """Wide body with two reflected hind-legs."""
    body = MorphNode(
        width=0.55,
        height=0.18,
        hue=0.62,
        connections=[
            ConnectionGene(child_idx=1, attach_side=0, attach_pos=0.55, reflection=True)
        ],
    )
    leg = MorphNode(
        width=0.12,
        height=0.42,
        hue=0.08,
        joint_angle_min=-ang,
        joint_angle_max=ang,
        osc_amplitude=amp,
        osc_frequency=freq,
    )
    return Genotype(nodes=[body, leg])


def _triped(freq=1.4, amp=0.9, ang=0.6):
    """Elongated body with two outer reflected legs + one central front leg."""
    body = MorphNode(
        width=0.60,
        height=0.16,
        hue=0.35,
        connections=[
            # outer legs (reflected pair, node 1)
            ConnectionGene(
                child_idx=1, attach_side=0, attach_pos=-0.55, reflection=True
            ),
            # central front leg (node 2, shifted phase in the node itself)
            ConnectionGene(child_idx=2, attach_side=0, attach_pos=0.30),
        ],
    )
    outer_leg = MorphNode(
        width=0.10,
        height=0.38,
        hue=0.12,
        joint_angle_min=-ang,
        joint_angle_max=ang,
        osc_amplitude=amp,
        osc_frequency=freq,
        osc_phase=0.0,
    )
    front_leg = MorphNode(
        width=0.10,
        height=0.38,
        hue=0.18,
        joint_angle_min=-ang,
        joint_angle_max=ang,
        osc_amplitude=amp,
        osc_frequency=freq,
        osc_phase=np.pi,  # anti-phase
    )
    return Genotype(nodes=[body, outer_leg, front_leg])


# Each preset: (display name, genotype factory, MOTOR_TORQUE_PER_AREA)
PRESETS = [
    ("Crawler  (5 seg · traveling wave)", lambda: _crawler(), 3.5),
    ("Biped    (reflected hind legs)", lambda: _biped(), 3.5),
    ("Triped   (2 outer + 1 front leg)", lambda: _triped(), 3.5),
]

WIDTH, HEIGHT = 1000, 600
GROUND_Y = 0.0  # physics world y of ground plane
START_Y = 1.2  # creature root spawn height

PPM = 150.0  # pixels per metre
GROUND_SCREEN_Y = int(HEIGHT * 0.73)  # screen y that maps to GROUND_Y

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Creature Viewer")
clock = pygame.time.Clock()
font_m = pygame.font.SysFont("monospace", 15)
font_s = pygame.font.SysFont("monospace", 12)

# Colours
BG_COLOUR = (22, 22, 32)
GROUND_FILL = (42, 55, 38)
GROUND_LINE = (80, 105, 68)
GRID_COLOUR = (40, 45, 55)
GRID_TXT = (70, 80, 95)
ANCHOR_A_COL = (255, 200, 50)
ANCHOR_B_COL = (255, 100, 50)
CONTACT_ACT = (255, 80, 80)
CONTACT_IDLE = (255, 200, 80)
TRAIL_COLOUR = (80, 140, 220)
COM_COLOUR = (255, 220, 60)
HUD_COLOUR = (200, 205, 220)
HUD_DIM = (110, 115, 130)
PAUSE_COLOUR = (220, 180, 60)
KEY_COLOUR = (140, 160, 185)

TRAIL_LEN = 600  # frames of COM history to keep

# Coordinate helpers
def w2s(world_pos, cam_x: float) -> tuple[int, int]:
    """World coords → screen pixel coords."""
    sx = int(WIDTH / 2 + (world_pos[0] - cam_x) * PPM)
    sy = int(GROUND_SCREEN_Y - world_pos[1] * PPM)
    return sx, sy


def draw_body(surf, body, cam_x: float):
    corners = body.get_corners()
    pts = [w2s(c, cam_x) for c in corners]
    r, g, b = body.color
    pygame.draw.polygon(surf, (r, g, b), pts)
    # bright edge
    light = (min(r + 50, 255), min(g + 50, 255), min(b + 50, 255))
    pygame.draw.polygon(surf, light, pts, 2)


def draw_ground(surf, cam_x: float):
    gy = GROUND_SCREEN_Y
    pygame.draw.rect(surf, GROUND_FILL, (0, gy, WIDTH, HEIGHT - gy))
    pygame.draw.line(surf, GROUND_LINE, (0, gy), (WIDTH, gy), 2)

    # Scrolling metre-grid with labels
    left_m = cam_x - WIDTH / (2 * PPM) - 1
    right_m = cam_x + WIDTH / (2 * PPM) + 1
    step = 1.0
    m = np.floor(left_m / step) * step
    while m <= right_m:
        sx, _ = w2s(np.array([m, 0.0]), cam_x)
        pygame.draw.line(surf, GRID_COLOUR, (sx, 0), (sx, HEIGHT), 1)
        lbl = font_s.render(f"{m:.0f}m", True, GRID_TXT)
        surf.blit(lbl, (sx + 2, gy + 4))
        m += step


def draw_joint_anchors(surf, joint, cam_x: float):
    wa = joint.get_world_anchor_a()
    wb = joint.get_world_anchor_b()
    pygame.draw.circle(surf, ANCHOR_A_COL, w2s(wa, cam_x), 5)
    pygame.draw.circle(surf, ANCHOR_B_COL, w2s(wb, cam_x), 3)


def draw_contacts(surf, contacts, cam_x: float):
    for c in contacts:
        col = CONTACT_ACT if c.lambda_n > 0 else CONTACT_IDLE
        pygame.draw.circle(surf, col, w2s(c.contact_point, cam_x), 4)


def draw_trail(surf, trail: list, cam_x: float):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        a = w2s(trail[i - 1], cam_x)
        b = w2s(trail[i], cam_x)
        alpha = int(60 + 140 * i / len(trail))
        pygame.draw.line(surf, (*TRAIL_COLOUR, alpha), a, b, 1)


# Simulation state
preset_idx = 0
creature: Creature | None = None
sim_time: float = 0.0
paused: bool = False
cam_x: float = 0.0
trail: list = []
start_com: np.ndarray | None = None
all_contacts: list = []


def reset(idx: int):
    global creature, sim_time, cam_x, trail, start_com, all_contacts

    _, gen_fn, torque_k = PRESETS[idx]
    cr.Creature.MOTOR_TORQUE_PER_AREA = torque_k

    g = gen_fn()
    creature = Creature(g)
    creature.build(start_x=0.0, start_y=START_Y)

    sim_time = 0.0
    trail = []
    all_contacts = []

    if creature.bodies:
        com = creature.center_of_mass()
        cam_x = float(com[0])
        start_com = com.copy()
    else:
        cam_x = 0.0
        start_com = np.zeros(2)

    pygame.display.set_caption(f"Creature Viewer — {PRESETS[idx][0]}")


reset(preset_idx)

# Main loop
running = True
while running:
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                reset(preset_idx)
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_1:
                preset_idx = 0
                reset(0)
            elif event.key == pygame.K_2:
                preset_idx = 1
                reset(1)
            elif event.key == pygame.K_3:
                preset_idx = 2
                reset(2)

    # Physics step
    if not paused and creature and creature.bodies:
        creature.apply_control(sim_time)
        all_contacts = step_world(creature.bodies, creature.joints, GROUND_Y, DT)
        sim_time += DT

        com = creature.center_of_mass()
        trail.append(com.copy())
        if len(trail) > TRAIL_LEN:
            trail.pop(0)

        # Smooth camera follow
        cam_x += (float(com[0]) - cam_x) * 0.06

    # Render
    screen.fill(BG_COLOUR)

    draw_ground(screen, cam_x)

    if creature and creature.bodies:
        draw_trail(screen, trail, cam_x)

        # Bodies
        for part in creature.parts:
            draw_body(screen, part.body, cam_x)

        # Joint anchors
        for j in creature.joints:
            draw_joint_anchors(screen, j, cam_x)

        # Contact points
        draw_contacts(screen, all_contacts, cam_x)

        # COM marker
        com = creature.center_of_mass()
        pygame.draw.circle(screen, COM_COLOUR, w2s(com, cam_x), 5)
        pygame.draw.circle(screen, COM_COLOUR, w2s(com, cam_x), 5, 1)

        # Start-position marker on ground
        if start_com is not None:
            sx0, sy0 = w2s(np.array([start_com[0], 0.0]), cam_x)
            pygame.draw.line(screen, (90, 110, 180), (sx0, sy0 - 14), (sx0, sy0 + 2), 2)
            lbl = font_s.render("start", True, (90, 110, 180))
            screen.blit(lbl, (sx0 + 3, sy0 - 13))

    # HUD
    if creature and creature.bodies and start_com is not None:
        com = creature.center_of_mass()
        dist = abs(com[0] - start_com[0])
        speed = dist / sim_time if sim_time > 0.1 else 0.0
        hud_lines = [
            (f"Preset : {PRESETS[preset_idx][0]}", HUD_COLOUR),
            (
                f"Parts  : {len(creature.bodies)} bodies  "
                f"{len(creature.joints)} joints",
                HUD_COLOUR,
            ),
            (f"Time   : {sim_time:6.2f} s", HUD_COLOUR),
            (f"COM x  : {com[0]:+6.3f} m", HUD_COLOUR),
            (f"Dist   : {dist:6.3f} m  ({speed:.3f} m/s avg)", HUD_COLOUR),
            ("", HUD_DIM),
            ("[1] Crawler  [2] Biped  [3] Triped", KEY_COLOUR),
            ("[R] Reset   [SPACE] Pause   [ESC] Quit", KEY_COLOUR),
        ]
        for i, (line, col) in enumerate(hud_lines):
            surf = font_m.render(line, True, col)
            screen.blit(surf, (10, 10 + i * 20))

    if paused:
        lbl = font_m.render("  PAUSED  ", True, PAUSE_COLOUR)
        r = lbl.get_rect(center=(WIDTH // 2, 24))
        pygame.draw.rect(screen, (30, 28, 20), r.inflate(10, 4))
        screen.blit(lbl, r)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
