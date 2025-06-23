import numpy as np
from typing import Tuple, List, Optional
from .rigid_body import RigidBody

class AABB:
    """Axis-Aligned Bounding Box for basic collision detection."""
    
    def __init__(self, min_point: np.ndarray, max_point: np.ndarray):
        self.min_point = np.array(min_point, dtype=float)
        self.max_point = np.array(max_point, dtype=float)

    @property
    def center(self) -> np.ndarray:
        return (self.min_point + self.max_point) / 2

    @property
    def half_extents(self) -> np.ndarray:
        return (self.max_point - self.min_point) / 2

class CollisionInfo:
    """Contains information about a collision between two objects."""
    
    def __init__(self, normal: np.ndarray, penetration: float, contact_point: np.ndarray):
        self.normal = np.array(normal, dtype=float)
        self.penetration = penetration
        self.contact_point = np.array(contact_point, dtype=float)

class CollisionResolver:
    """Handles collision detection and response between rigid bodies."""
    
    @staticmethod
    def check_collision(box1: AABB, box2: AABB) -> Optional[CollisionInfo]:
        """Check for collision between two AABBs.
        
        Args:
            box1: First AABB
            box2: Second AABB
            
        Returns:
            CollisionInfo if collision detected, None otherwise
        """
        # Calculate overlap on each axis
        overlap_x = min(box1.max_point[0], box2.max_point[0]) - \
                   max(box1.min_point[0], box2.min_point[0])
        overlap_y = min(box1.max_point[1], box2.max_point[1]) - \
                   max(box1.min_point[1], box2.min_point[1])

        # If there's no overlap on any axis, no collision
        if overlap_x < 0 or overlap_y < 0:
            return None

        # Find smallest overlap to determine collision normal
        if overlap_x < overlap_y:
            normal = np.array([1, 0]) if box1.center[0] < box2.center[0] else np.array([-1, 0])
            penetration = overlap_x
        else:
            normal = np.array([0, 1]) if box1.center[1] < box2.center[1] else np.array([0, -1])
            penetration = overlap_y

        # Calculate contact point (center of overlap region)
        contact_point = (box1.center + box2.center) / 2

        return CollisionInfo(normal, penetration, contact_point)

    @staticmethod
    def resolve_collision(body1: RigidBody, body2: RigidBody, collision: CollisionInfo) -> None:
        """Resolve collision between two rigid bodies using impulse-based resolution.
        
        Args:
            body1: First rigid body
            body2: Second rigid body
            collision: Collision information
        """
        # Calculate relative velocity at contact point
        relative_velocity = body2.velocity - body1.velocity

        # Calculate impulse magnitude using coefficient of restitution
        restitution = 0.5  # Could be made configurable
        impulse_numerator = -(1 + restitution) * np.dot(relative_velocity, collision.normal)
        impulse_denominator = (1/body1.mass + 1/body2.mass)
        impulse = impulse_numerator / impulse_denominator

        # Apply impulse to both bodies
        body1.velocity -= (impulse / body1.mass) * collision.normal
        body2.velocity += (impulse / body2.mass) * collision.normal

        # Positional correction to prevent sinking
        percent = 0.2  # Penetration percentage to resolve
        correction = (collision.penetration / (1/body1.mass + 1/body2.mass)) * percent * collision.normal
        
        body1.position -= correction / body1.mass
        body2.position += correction / body2.mass