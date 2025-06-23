import numpy as np
from typing import Tuple, List, Optional

class RigidBody:
    """A class representing a rigid body in 2D space with physical properties."""
    
    def __init__(self, mass: float, moment_of_inertia: float, position: np.ndarray,
                 velocity: np.ndarray = np.zeros(2), angular_velocity: float = 0.0):
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.angular_velocity = angular_velocity
        self.force_accumulator = np.zeros(2)
        self.torque_accumulator = 0.0
        self.gravity = np.array([0, -9.81])

    def apply_force(self, force: np.ndarray, point: Optional[np.ndarray] = None) -> None:
        """Apply a force to the rigid body at a specific point.
        
        Args:
            force: The force vector to apply
            point: The point of application relative to center of mass (optional)
        """
        self.force_accumulator += force
        
        if point is not None:
            # Calculate torque: τ = r × F
            torque = np.cross(point, force)
            self.torque_accumulator += torque

    def update(self, dt: float) -> None:
        """Update the rigid body's state over a time step using semi-implicit Euler integration.
        
        Args:
            dt: Time step in seconds
        """
        # Add gravity force
        self.force_accumulator += self.mass * self.gravity

        # Update linear motion
        acceleration = self.force_accumulator / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Update angular motion
        angular_acceleration = self.torque_accumulator / self.moment_of_inertia
        self.angular_velocity += angular_acceleration * dt

        # Clear accumulators
        self.force_accumulator.fill(0)
        self.torque_accumulator = 0.0

    def get_kinetic_energy(self) -> float:
        """Calculate the total kinetic energy of the rigid body.
        
        Returns:
            The sum of translational and rotational kinetic energy
        """
        translational = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        rotational = 0.5 * self.moment_of_inertia * self.angular_velocity**2
        return translational + rotational

    def get_momentum(self) -> np.ndarray:
        """Calculate the linear momentum of the rigid body.
        
        Returns:
            The momentum vector
        """
        return self.mass * self.velocity

    def get_angular_momentum(self) -> float:
        """Calculate the angular momentum of the rigid body.
        
        Returns:
            The angular momentum scalar (in 2D)
        """
        return self.moment_of_inertia * self.angular_velocity