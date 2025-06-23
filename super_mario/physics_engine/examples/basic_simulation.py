import numpy as np
import pygame
import sys
sys.path.append('..')

from game_physics.rigid_body import RigidBody
from game_physics.world import PhysicsWorld
from game_physics.utils import Rectangle, Circle
from game_physics.constraints import SpringConstraint

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
PIXELS_PER_METER = 50  # Scale factor for converting physics units to pixels

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Setup display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Physics Engine Demo")
clock = pygame.time.Clock()

def physics_to_screen(position):
    """Convert physics coordinates to screen coordinates."""
    return (
        int(position[0] * PIXELS_PER_METER + WINDOW_WIDTH // 2),
        int(WINDOW_HEIGHT - (position[1] * PIXELS_PER_METER + WINDOW_HEIGHT // 2))
    )

# Create physics world
world = PhysicsWorld(gravity=np.array([0, -9.81]))

# Create ground
ground = RigidBody(
    mass=float('inf'),
    moment_of_inertia=float('inf'),
    position=np.array([0, -5])
)
world.add_body(ground)

# Create falling box
box = RigidBody(
    mass=1.0,
    moment_of_inertia=Rectangle(1.0, 1.0).get_moment_of_inertia(1.0),
    position=np.array([0, 5]),
    velocity=np.array([2, 0])
)
world.add_body(box)

# Create bouncing ball
ball = RigidBody(
    mass=0.5,
    moment_of_inertia=Circle(0.3).get_moment_of_inertia(0.5),
    position=np.array([-2, 3]),
    velocity=np.array([1, 2])
)
world.add_body(ball)

# Create two bodies connected by a spring
spring_body1 = RigidBody(
    mass=1.0,
    moment_of_inertia=1.0,
    position=np.array([2, 4])
)
spring_body2 = RigidBody(
    mass=1.0,
    moment_of_inertia=1.0,
    position=np.array([4, 4])
)
world.add_body(spring_body1)
world.add_body(spring_body2)

# Create spring constraint
spring = SpringConstraint(
    spring_body1,
    spring_body2,
    np.array([0, 0]),
    np.array([0, 0]),
    rest_length=2.0,
    stiffness=5.0,
    damping=0.5
)

# Main game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Add new ball at mouse position
            mouse_pos = pygame.mouse.get_pos()
            physics_pos = np.array([
                (mouse_pos[0] - WINDOW_WIDTH // 2) / PIXELS_PER_METER,
                (WINDOW_HEIGHT - mouse_pos[1] - WINDOW_HEIGHT // 2) / PIXELS_PER_METER
            ])
            new_ball = RigidBody(
                mass=0.5,
                moment_of_inertia=Circle(0.3).get_moment_of_inertia(0.5),
                position=physics_pos,
                velocity=np.array([np.random.uniform(-2, 2), np.random.uniform(0, 2)])
            )
            world.add_body(new_ball)

    # Physics update
    spring.solve()
    world.step()

    # Drawing
    screen.fill(BLACK)

    # Draw ground
    ground_pos = physics_to_screen(ground.position)
    pygame.draw.line(screen, WHITE,
                     (0, ground_pos[1]),
                     (WINDOW_WIDTH, ground_pos[1]), 2)

    # Draw box
    box_pos = physics_to_screen(box.position)
    box_size = int(1.0 * PIXELS_PER_METER)
    pygame.draw.rect(screen, RED,
                     (box_pos[0] - box_size//2,
                      box_pos[1] - box_size//2,
                      box_size, box_size))

    # Draw ball
    ball_pos = physics_to_screen(ball.position)
    ball_radius = int(0.3 * PIXELS_PER_METER)
    pygame.draw.circle(screen, BLUE, ball_pos, ball_radius)

    # Draw spring bodies and connection
    spring_pos1 = physics_to_screen(spring_body1.position)
    spring_pos2 = physics_to_screen(spring_body2.position)
    pygame.draw.circle(screen, RED, spring_pos1, 10)
    pygame.draw.circle(screen, RED, spring_pos2, 10)
    pygame.draw.line(screen, WHITE, spring_pos1, spring_pos2, 2)

    # Update display
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()