import numpy as np
import pygame

from src.physics import (
    DT,
    Body,
    RevoluteJoint,
    generate_ground_contact_constraints,
    solve_ground_contact_constraints,
    solve_joints,
    check_sleeping,
)

WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Revolute Joint")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 16)

PIXELS_PER_METER = 100.0
ORIGIN_X = WIDTH // 2
ORIGIN_Y = HEIGHT * 0.4


def world_to_screen(world_pos):
    sx = ORIGIN_X + world_pos[0] * PIXELS_PER_METER
    sy = ORIGIN_Y - world_pos[1] * PIXELS_PER_METER
    return (int(sx), int(sy))


use_angle_limits = False


def create_scene():
    body_a = Body(
        width=1.0,
        height=0.4,
        pos=np.array([0.0, 2.0]),
    )
    body_b = Body(
        width=0.6,
        height=0.3,
        pos=np.array([0.0, 1.2]),
    )

    joint = RevoluteJoint(
        body_a=body_a,
        body_b=body_b,
        anchor_a=np.array([body_a.width / 2, -body_a.height / 2]),
        anchor_b=np.array([-body_b.width / 2, body_b.height / 2]),
        angle_min=-np.pi / 3 if use_angle_limits else None,
        angle_max=np.pi / 3 if use_angle_limits else None,
    )

    body_b.vel = np.array([3.0, 0.0])

    return body_a, body_b, joint


body_a, body_b, joint = create_scene()
ground_y = -1.5

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                body_a, body_b, joint = create_scene()
            if event.key == pygame.K_l:
                use_angle_limits = not use_angle_limits
                body_a, body_b, joint = create_scene()

    # ===== Constraint-based simulation step =====
    # 1. Integrate velocities (apply gravity)
    body_a.integrate_velocity(DT)
    body_b.integrate_velocity(DT)

    # 2. Generate contact constraints
    contacts_a = generate_ground_contact_constraints(body_a, ground_y)
    contacts_b = generate_ground_contact_constraints(body_b, ground_y)

    # 3. Solve all constraints (joints + contacts)
    solve_joints([joint], DT)
    solve_ground_contact_constraints(contacts_a, DT)
    solve_ground_contact_constraints(contacts_b, DT)

    # 4. Integrate positions using corrected velocities
    body_a.integrate_position(DT)
    body_b.integrate_position(DT)

    # 5. Sleep check
    check_sleeping(body_a, ground_y)
    check_sleeping(body_b, ground_y)

    # ===== Rendering =====
    screen.fill((30, 30, 40))

    gy = world_to_screen(np.array([0, ground_y]))[1]
    pygame.draw.rect(screen, (50, 65, 45), (0, gy, WIDTH, HEIGHT - gy))
    pygame.draw.line(screen, (90, 110, 80), (0, gy), (WIDTH, gy), 2)

    # Body A
    corners_a = body_a.get_corners()
    pts_a = [world_to_screen(c) for c in corners_a]
    pygame.draw.polygon(screen, (80, 130, 200), pts_a)
    pygame.draw.polygon(screen, (120, 170, 240), pts_a, 2)

    # Body B
    corners_b = body_b.get_corners()
    pts_b = [world_to_screen(c) for c in corners_b]
    pygame.draw.polygon(screen, (80, 180, 100), pts_b)
    pygame.draw.polygon(screen, (120, 220, 140), pts_b, 2)

    # Joint anchors (draw both to visualize drift)
    anchor_a_world = joint.get_world_anchor_a()
    anchor_b_world = joint.get_world_anchor_b()
    anchor_a_screen = world_to_screen(anchor_a_world)
    anchor_b_screen = world_to_screen(anchor_b_world)
    pygame.draw.circle(screen, (255, 200, 50), anchor_a_screen, 6)
    pygame.draw.circle(screen, (200, 150, 30), anchor_a_screen, 6, 2)
    pygame.draw.circle(screen, (255, 100, 50), anchor_b_screen, 4)

    # Draw contact points
    for c in contacts_a + contacts_b:
        sp = world_to_screen(c.contact_point)
        color = (255, 80, 80) if c.lambda_n > 0 else (255, 200, 80)
        pygame.draw.circle(screen, color, sp, 5)

    # Body centers
    pygame.draw.circle(screen, (255, 100, 100), world_to_screen(body_a.pos), 3)
    pygame.draw.circle(screen, (255, 100, 100), world_to_screen(body_b.pos), 3)

    rel_angle = joint.get_angle()
    drift = np.linalg.norm(anchor_b_world - anchor_a_world)
    info_lines = [
        f"Body A pos: ({body_a.pos[0]:.2f}, {body_a.pos[1]:.2f})",
        f"Body A vel: ({body_a.vel[0]:.2f}, {body_a.vel[1]:.2f})",
        f"Body A angular vel: ({body_a.angular_vel:.2f})",
        f"Body A sleeping: {body_a.sleeping}",
        f"Body B pos: ({body_b.pos[0]:.2f}, {body_b.pos[1]:.2f})",
        f"Body B vel: ({body_b.vel[0]:.2f}, {body_b.vel[1]:.2f})",
        f"Body B angular vel: ({body_b.angular_vel:.2f})",
        f"Body B sleeping: {body_b.sleeping}",
        f"Joint angle: {np.degrees(rel_angle):.1f} deg",
        f"Joint drift: {drift:.4f} m",
        f"Angle limits: {'ON' if use_angle_limits else 'OFF'}",
        f"Contacts: A={len(contacts_a)} B={len(contacts_b)}",
        "",
        "[R] Reset  [L] Toggle limits  [ESC] Quit",
    ]
    for i, line in enumerate(info_lines):
        text = font.render(line, True, (200, 200, 210))
        screen.blit(text, (10, 10 + i * 22))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
