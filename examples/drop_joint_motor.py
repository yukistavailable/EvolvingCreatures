import numpy as np
import pygame

from src.physics import DT, Body, RevoluteJoint, resolve_ground_collision, solve_joints

WIDTH, HEIGHT = 800, 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Joint Motor Test")
clock = pygame.time.Clock()
font = pygame.font.SysFont("monospace", 16)

PIXELS_PER_METER = 100.0
ORIGIN_X = WIDTH // 2
ORIGIN_Y = HEIGHT * 0.7


def world_to_screen(world_pos):
    sx = ORIGIN_X + world_pos[0] * PIXELS_PER_METER
    sy = ORIGIN_Y - world_pos[1] * PIXELS_PER_METER
    return (int(sx), int(sy))


motor_freq = 2.0  # Hz
max_torque = 0.1
ground_y = 0.0


def create_scene():
    body_a = Body(
        width=0.8,
        height=0.3,
        pos=np.array([0.0, 0.5]),
    )
    body_b = Body(
        width=0.5,
        height=0.15,
        pos=np.array([0.6, 0.2]),
        angle=-0.3,
    )

    joint = RevoluteJoint(
        body_a=body_a,
        body_b=body_b,
        anchor_a=np.array([body_a.width / 2, -body_a.height / 2]),
        anchor_b=np.array([-body_b.width / 2, 0.0]),
        angle_min=-np.pi / 3,
        angle_max=np.pi / 3,
        max_motor_torque=max_torque,
    )

    return body_a, body_b, joint


body_a, body_b, joint = create_scene()
time = 0.0

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
                time = 0.0
            if event.key == pygame.K_UP:
                motor_freq = min(10.0, motor_freq + 0.5)
            if event.key == pygame.K_DOWN:
                motor_freq = max(0.5, motor_freq - 0.5)
            if event.key == pygame.K_RIGHT:
                max_torque = min(20.0, max_torque + 1.0)
                joint.max_torque = max_torque
            if event.key == pygame.K_LEFT:
                max_torque = max(1.0, max_torque - 1.0)
                joint.max_torque = max_torque

    joint.motor_torque = np.sin(2 * np.pi * motor_freq * time)

    body_a.sleeping = False
    body_b.sleeping = False

    body_a.integrate(DT)
    body_b.integrate(DT)

    solve_joints([joint])

    resolve_ground_collision(body_a, ground_y)
    resolve_ground_collision(body_b, ground_y)

    time += DT

    screen.fill((30, 30, 40))

    gy = world_to_screen(np.array([0, ground_y]))[1]
    pygame.draw.rect(screen, (50, 65, 45), (0, gy, WIDTH, HEIGHT - gy))
    pygame.draw.line(screen, (90, 110, 80), (0, gy), (WIDTH, gy), 2)

    corners_a = body_a.get_corners()
    pts_a = [world_to_screen(c) for c in corners_a]
    pygame.draw.polygon(screen, (80, 130, 200), pts_a)
    pygame.draw.polygon(screen, (120, 170, 240), pts_a, 2)

    corners_b = body_b.get_corners()
    pts_b = [world_to_screen(c) for c in corners_b]
    pygame.draw.polygon(screen, (80, 180, 100), pts_b)
    pygame.draw.polygon(screen, (120, 220, 140), pts_b, 2)

    anchor_world = joint.get_world_anchor_a()
    anchor_screen = world_to_screen(anchor_world)
    pygame.draw.circle(screen, (255, 200, 50), anchor_screen, 6)
    pygame.draw.circle(screen, (200, 150, 30), anchor_screen, 6, 2)

    torque_vis = joint.motor_torque
    color = (255, 120, 80) if torque_vis > 0 else (80, 180, 255)
    arc_radius = int(abs(torque_vis) * 20)
    if arc_radius > 2:
        pygame.draw.circle(screen, color, anchor_screen, arc_radius, 2)

    info_lines = [
        f"motor_torque: {joint.motor_torque:.2f}",
        f"motor_freq: {motor_freq:.1f} Hz",
        f"max_torque: {max_torque:.1f}",
        f"joint angle: {np.degrees(joint.get_angle()):.1f} deg",
        f"time: {time:.1f}s",
        "",
        "[UP/DOWN] Freq  [LEFT/RIGHT] Torque",
        "[R] Reset  [ESC] Quit",
    ]
    for i, line in enumerate(info_lines):
        text = font.render(line, True, (200, 200, 210))
        screen.blit(text, (10, 10 + i * 22))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
