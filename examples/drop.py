import numpy as np
import pygame

from src.physics import DT, Body

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


body = Body(
    width=1.0,
    height=0.5,
    pos=np.array([0.0, 3.0]),
    angle=0.3,
)

print("Start")
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    body.integrate(DT)

    screen.fill((30, 30, 40))

    ground_y = int(ORIGIN_Y)
    pygame.draw.line(screen, (80, 100, 70), (0, ground_y), (WIDTH, ground_y), 2)

    corners = body.get_corners()
    screen_pts = [world_to_screen(c) for c in corners]
    pygame.draw.polygon(screen, (100, 160, 220), screen_pts)
    pygame.draw.polygon(screen, (140, 200, 255), screen_pts, 2)

    center_screen = world_to_screen(body.pos)
    pygame.draw.circle(screen, (255, 200, 50), center_screen, 4)

    info_lines = [
        f"pos: ({body.pos[0]:.2f}, {body.pos[1]:.2f})",
        f"vel: ({body.vel[0]:.2f}, {body.vel[1]:.2f})",
        f"angle: {np.degrees(body.angle):.1f}°",
        "[ESC] to quit",
    ]
    for i, line in enumerate(info_lines):
        text = font.render(line, True, (200, 200, 210))
        screen.blit(text, (10, 10 + i * 22))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
