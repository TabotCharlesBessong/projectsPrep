import numpy as np
from typing import Tuple, Optional
from .rigid_body import RigidBody

class Constraint:
    """Base class for constraints between rigid bodies."""
    
    def __init__(self, body1: RigidBody, body2: RigidBody):
        self.body1 = body1
        self.body2 = body2

class DistanceConstraint(Constraint):
    """Maintains a fixed distance between two points on two bodies."""
    
    def __init__(self, body1: RigidBody, body2: RigidBody, 
                 anchor1: np.ndarray, anchor2: np.ndarray, 
                 target_distance: Optional[float] = None):
        super().__init__(body1, body2)
        self.anchor1 = np.array(anchor1, dtype=float)
        self.anchor2 = np.array(anchor2, dtype=float)
        self.target_distance = target_distance or np.linalg.norm(body2.position + anchor2 - (body1.position + anchor1))
        self.stiffness = 0.5
        self.damping = 0.1

    def solve(self) -> None:
        """Solve the distance constraint between the two bodies."""
        # Calculate world positions of anchor points
        world_anchor1 = self.body1.position + self.anchor1
        world_anchor2 = self.body2.position + self.anchor2

        # Calculate current distance and direction
        delta = world_anchor2 - world_anchor1
        current_distance = np.linalg.norm(delta)
        if current_distance == 0:
            return

        direction = delta / current_distance
        distance_error = current_distance - self.target_distance

        # Calculate relative velocity at constraint points
        relative_velocity = self.body2.velocity - self.body1.velocity
        velocity_projection = np.dot(relative_velocity, direction)

        # Calculate impulse magnitude
        impulse = (distance_error * self.stiffness + velocity_projection * self.damping) / \
                 (1/self.body1.mass + 1/self.body2.mass)

        # Apply impulse
        self.body1.velocity += (impulse * direction) / self.body1.mass
        self.body2.velocity -= (impulse * direction) / self.body2.mass

class SpringConstraint(Constraint):
    """Creates a spring connection between two bodies."""
    
    def __init__(self, body1: RigidBody, body2: RigidBody,
                 anchor1: np.ndarray, anchor2: np.ndarray,
                 rest_length: float, stiffness: float, damping: float):
        super().__init__(body1, body2)
        self.anchor1 = np.array(anchor1, dtype=float)
        self.anchor2 = np.array(anchor2, dtype=float)
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping

    def solve(self) -> None:
        """Apply spring forces to the connected bodies."""
        # Calculate world positions of anchor points
        world_anchor1 = self.body1.position + self.anchor1
        world_anchor2 = self.body2.position + self.anchor2

        # Calculate spring vector and length
        delta = world_anchor2 - world_anchor1
        current_length = np.linalg.norm(delta)
        if current_length == 0:
            return

        direction = delta / current_length

        # Calculate spring force using Hooke's law: F = -k(x - L)
        displacement = current_length - self.rest_length
        spring_force = self.stiffness * displacement

        # Calculate relative velocity for damping
        relative_velocity = self.body2.velocity - self.body1.velocity
        damping_force = self.damping * np.dot(relative_velocity, direction)

        # Total force
        total_force = (spring_force + damping_force) * direction

        # Apply forces
        self.body1.apply_force(total_force)
        self.body2.apply_force(-total_force)

class HingeConstraint(Constraint):
    """Creates a hinge joint between two bodies."""
    
    def __init__(self, body1: RigidBody, body2: RigidBody, 
                 pivot: np.ndarray, 
                 min_angle: Optional[float] = None,
                 max_angle: Optional[float] = None):
        super().__init__(body1, body2)
        self.pivot = np.array(pivot, dtype=float)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.position_correction_factor = 0.2

    def solve(self) -> None:
        """Solve the hinge constraint."""
        # Position constraint - keep pivot points together
        world_pivot1 = self.body1.position + self.pivot
        world_pivot2 = self.body2.position + self.pivot

        # Calculate position error
        position_error = world_pivot2 - world_pivot1

        # Apply position correction
        correction = position_error * self.position_correction_factor
        self.body1.position += correction * (1.0 / self.body1.mass)
        self.body2.position -= correction * (1.0 / self.body2.mass)

        # Angular constraints if specified
        if self.min_angle is not None and self.max_angle is not None:
            relative_angle = self.body2.angular_velocity - self.body1.angular_velocity
            if relative_angle < self.min_angle:
                correction = self.min_angle - relative_angle
                self.body1.angular_velocity -= correction * 0.5
                self.body2.angular_velocity += correction * 0.5
            elif relative_angle > self.max_angle:
                correction = relative_angle - self.max_angle
                self.body1.angular_velocity += correction * 0.5
                self.body2.angular_velocity -= correction * 0.5