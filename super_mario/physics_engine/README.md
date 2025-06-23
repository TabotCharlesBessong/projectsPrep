# Game Physics Engine

A comprehensive 2D physics engine for game development, providing rigid body dynamics, collision detection, and constraint systems.

## Features

- Rigid body dynamics simulation
- Collision detection and response
- Various constraint types (Distance, Spring, Hinge)
- Shape primitives (Rectangle, Circle)
- Utility functions for common physics calculations

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from game_physics.rigid_body import RigidBody
from game_physics.world import PhysicsWorld
from game_physics.utils import Rectangle

# Create a physics world
world = PhysicsWorld(gravity=np.array([0, -9.81]))

# Create a ground body
ground = RigidBody(
    mass=float('inf'),  # Static body
    moment_of_inertia=float('inf'),
    position=np.array([0, 0])
)

# Create a dynamic box
box = RigidBody(
    mass=1.0,
    moment_of_inertia=Rectangle(1.0, 1.0).get_moment_of_inertia(1.0),
    position=np.array([0, 5]),
    velocity=np.array([1, 0])
)

# Add bodies to the world
world.add_body(ground)
world.add_body(box)

# Simulation loop
while True:
    world.step()
```

## Advanced Usage

### Creating Constraints

```python
from game_physics.constraints import SpringConstraint

# Create two bodies
body1 = RigidBody(mass=1.0, moment_of_inertia=1.0, position=np.array([0, 0]))
body2 = RigidBody(mass=1.0, moment_of_inertia=1.0, position=np.array([2, 0]))

# Create a spring constraint between them
spring = SpringConstraint(
    body1=body1,
    body2=body2,
    anchor1=np.array([1, 0]),  # Local coordinates
    anchor2=np.array([-1, 0]),  # Local coordinates
    rest_length=2.0,
    stiffness=10.0,
    damping=0.5
)

# Add bodies to world
world.add_body(body1)
world.add_body(body2)

# Solve constraint in simulation loop
while True:
    spring.solve()
    world.step()
```

### Custom Shapes and Collision Detection

```python
from game_physics.utils import Circle
from game_physics.collision import AABB, CollisionResolver

# Create circular body
circle = RigidBody(
    mass=1.0,
    moment_of_inertia=Circle(0.5).get_moment_of_inertia(1.0),
    position=np.array([0, 5])
)

# Create bounding boxes
box1 = AABB(np.array([-1, -1]), np.array([1, 1]))
box2 = AABB(np.array([0, 0]), np.array([2, 2]))

# Check for collision
collision = CollisionResolver.check_collision(box1, box2)
if collision:
    print(f"Collision detected! Penetration depth: {collision.penetration}")
```

## API Reference

### RigidBody
- Main class for physical objects
- Handles position, velocity, forces, and torques
- Methods for applying forces and updating state

### PhysicsWorld
- Manages all physical bodies and their interactions
- Handles collision detection and resolution
- Updates all bodies in the simulation

### Constraints
- DistanceConstraint: Maintains fixed distance between points
- SpringConstraint: Creates spring-like behavior
- HingeConstraint: Creates rotational joint

### Collision
- AABB: Axis-aligned bounding box for collision detection
- CollisionResolver: Handles collision detection and response

### Utils
- Shape classes (Rectangle, Circle)
- Physics calculation utilities
- Vector operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.