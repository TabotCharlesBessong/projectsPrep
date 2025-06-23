from typing import List, Dict, Set, Tuple
import numpy as np
from .rigid_body import RigidBody
from .collision import AABB, CollisionResolver, CollisionInfo

class PhysicsWorld:
    """Manages and simulates physical objects and their interactions."""
    
    def __init__(self, gravity: np.ndarray = np.array([0, -9.81])):
        self.bodies: List[RigidBody] = []
        self.collision_pairs: Dict[int, Set[int]] = {}
        self.gravity = np.array(gravity, dtype=float)
        self.collision_resolver = CollisionResolver()
        self.time_step = 1/60  # 60 Hz simulation
        self.velocity_iterations = 8
        self.position_iterations = 3

    def add_body(self, body: RigidBody) -> None:
        """Add a rigid body to the physics world.
        
        Args:
            body: The rigid body to add
        """
        self.bodies.append(body)
        body_index = len(self.bodies) - 1
        self.collision_pairs[body_index] = set()

    def remove_body(self, body: RigidBody) -> None:
        """Remove a rigid body from the physics world.
        
        Args:
            body: The rigid body to remove
        """
        if body in self.bodies:
            body_index = self.bodies.index(body)
            self.bodies.remove(body)
            del self.collision_pairs[body_index]
            # Update collision pairs
            for pairs in self.collision_pairs.values():
                if body_index in pairs:
                    pairs.remove(body_index)

    def step(self) -> None:
        """Advance the physics simulation by one time step."""
        # Apply gravity to all bodies
        for body in self.bodies:
            body.apply_force(body.mass * self.gravity)

        # Broad phase collision detection
        self._broad_phase()

        # Velocity iterations for collision resolution
        for _ in range(self.velocity_iterations):
            self._solve_collisions()

        # Update body positions
        for body in self.bodies:
            body.update(self.time_step)

        # Position iterations for constraint solving
        for _ in range(self.position_iterations):
            self._solve_position_constraints()

    def _broad_phase(self) -> None:
        """Broad phase collision detection using sweep and prune algorithm."""
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                body1, body2 = self.bodies[i], self.bodies[j]
                
                # Create AABBs for the bodies
                aabb1 = AABB(body1.position - np.array([1, 1]), body1.position + np.array([1, 1]))
                aabb2 = AABB(body2.position - np.array([1, 1]), body2.position + np.array([1, 1]))
                
                # Check for potential collision
                if CollisionResolver.check_collision(aabb1, aabb2) is not None:
                    self.collision_pairs[i].add(j)
                    self.collision_pairs[j].add(i)

    def _solve_collisions(self) -> None:
        """Resolve all detected collisions."""
        for body1_idx, colliding_bodies in self.collision_pairs.items():
            for body2_idx in colliding_bodies:
                body1, body2 = self.bodies[body1_idx], self.bodies[body2_idx]
                
                # Create AABBs for precise collision check
                aabb1 = AABB(body1.position - np.array([1, 1]), body1.position + np.array([1, 1]))
                aabb2 = AABB(body2.position - np.array([1, 1]), body2.position + np.array([1, 1]))
                
                collision = CollisionResolver.check_collision(aabb1, aabb2)
                if collision:
                    CollisionResolver.resolve_collision(body1, body2, collision)

    def _solve_position_constraints(self) -> None:
        """Solve position-based constraints to prevent object penetration."""
        for body1_idx, colliding_bodies in self.collision_pairs.items():
            for body2_idx in colliding_bodies:
                body1, body2 = self.bodies[body1_idx], self.bodies[body2_idx]
                
                # Create AABBs
                aabb1 = AABB(body1.position - np.array([1, 1]), body1.position + np.array([1, 1]))
                aabb2 = AABB(body2.position - np.array([1, 1]), body2.position + np.array([1, 1]))
                
                collision = CollisionResolver.check_collision(aabb1, aabb2)
                if collision:
                    # Apply position correction
                    correction = collision.normal * (collision.penetration * 0.2)
                    body1.position -= correction * (1.0 / body1.mass)
                    body2.position += correction * (1.0 / body2.mass)