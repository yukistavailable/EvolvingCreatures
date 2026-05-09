import numpy as np
import pygame

from src.physics import DT, Body, resolve_ground_collision

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

    body.integrate(DT)
    resolve_ground_collision(body, GROUND_Y)

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

    for corner in corners:
        if corner[1] <= GROUND_Y + 0.01:
            sp = world_to_screen(corner)
            pygame.draw.circle(screen, (255, 80, 80), sp, 5)

    info_lines = [
        f"pos: ({body.pos[0]:.2f}, {body.pos[1]:.2f})",
        f"vel: ({body.vel[0]:.2f}, {body.vel[1]:.2f})",
        f"angle: {np.degrees(body.angle):.1f} deg",
        f"angular_vel: {np.degrees(body.angular_vel):.1f} deg/s",
        f"sleeping: {body.sleeping}",
        "",
        "[R] Reset  [ESC] Quit",
    ]
    for i, line in enumerate(info_lines):
        text = font.render(line, True, (200, 200, 210))
        screen.blit(text, (10, 10 + i * 22))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
