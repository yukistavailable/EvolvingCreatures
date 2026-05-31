import numpy as np
import pygame

import sys
sys.path.append("..")

from src.physics import (
    DT,
    Body,
    generate_ground_contact_constraints,
    solve_ground_contact_constraints,
    check_sleeping,
)

GROUND_Y = 0.0
WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 16)

PIXELS_PER_METER = 100.0
ORIGIN_X = WIDTH // 2
ORIGIN_Y = HEIGHT * 0.7


def world_to_screen(world_pos):
    sx = ORIGIN_X + world_pos[0] * PIXELS_PER_METER
    sy = ORIGIN_Y - world_pos[1] * PIXELS_PER_METER
    return (int(sx), int(sy))


def create_body():
    return Body(
        width=1.0,
        height=0.5,
        pos=np.array([0.0, 3.0]),
        angle=0.4,  # 傾けて落とす
    )


body = create_body()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:
                body = create_body()  # リセット

    # ===== Constraint-based simulation step =====
    # 1. Integrate velocity (apply gravity and external forces)
    body.integrate_velocity(DT)

    # 2. Generate contact constraints (discrete collision detection)
    contacts = generate_ground_contact_constraints(body, GROUND_Y)

    # 3. Solve contact constraints with PGS (modifies velocity)
    solve_ground_contact_constraints(contacts, DT)

    # 4. Integrate position using corrected velocity
    body.integrate_position(DT)

    # 5. Sleep check
    check_sleeping(body, GROUND_Y)

    # ===== Rendering =====
    screen.fill((30, 30, 40))

    ground_screen_y = world_to_screen(np.array([0, GROUND_Y]))[1]
    pygame.draw.rect(
        screen, (50, 65, 45), (0, ground_screen_y, WIDTH, HEIGHT - ground_screen_y)
    )
    pygame.draw.line(
        screen, (90, 110, 80), (0, ground_screen_y), (WIDTH, ground_screen_y), 2
    )

    corners = body.get_corners()
    screen_pts = [world_to_screen(c) for c in corners]
    pygame.draw.polygon(screen, (100, 160, 220), screen_pts)
    pygame.draw.polygon(screen, (140, 200, 255), screen_pts, 2)

    center_screen = world_to_screen(body.pos)
    pygame.draw.circle(screen, (255, 200, 50), center_screen, 4)

    # Draw contact points
    for c in contacts:
        sp = world_to_screen(c.contact_point)
        color = (255, 80, 80) if c.lambda_n > 0 else (255, 200, 80)
        pygame.draw.circle(screen, color, sp, 5)

    info_lines = [
        f"pos: ({body.pos[0]:.2f}, {body.pos[1]:.2f})",
        f"vel: ({body.vel[0]:.2f}, {body.vel[1]:.2f})",
        f"angle: {np.degrees(body.angle):.1f} deg",
        f"angular_vel: {np.degrees(body.angular_vel):.1f} deg/s",
        f"sleeping: {body.sleeping}",
        f"contacts: {len(contacts)}  (active: {sum(1 for c in contacts if c.lambda_n > 0)})",
        "",
        "[R] Reset  [ESC] Quit",
    ]
    for i, line in enumerate(info_lines):
        text = font.render(line, True, (200, 200, 210))
        screen.blit(text, (10, 10 + i * 22))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
