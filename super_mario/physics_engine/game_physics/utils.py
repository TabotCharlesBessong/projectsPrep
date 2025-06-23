import numpy as np
from typing import List, Tuple, Union

class Shape:
    """Base class for collision shapes."""
    def get_moment_of_inertia(self, mass: float) -> float:
        raise NotImplementedError

class Rectangle(Shape):
    """Rectangular shape for collision detection."""
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def get_moment_of_inertia(self, mass: float) -> float:
        """Calculate moment of inertia for a rectangle.
        
        Args:
            mass: Mass of the rectangle
            
        Returns:
            Moment of inertia about the center
        """
        return (mass / 12.0) * (self.width**2 + self.height**2)

class Circle(Shape):
    """Circular shape for collision detection."""
    def __init__(self, radius: float):
        self.radius = radius

    def get_moment_of_inertia(self, mass: float) -> float:
        """Calculate moment of inertia for a circle.
        
        Args:
            mass: Mass of the circle
            
        Returns:
            Moment of inertia about the center
        """
        return 0.5 * mass * self.radius**2

def calculate_center_of_mass(positions: List[np.ndarray], masses: List[float]) -> np.ndarray:
    """Calculate the center of mass for a system of particles.
    
    Args:
        positions: List of position vectors
        masses: List of corresponding masses
        
    Returns:
        Center of mass position vector
    """
    total_mass = sum(masses)
    weighted_sum = sum(mass * pos for mass, pos in zip(masses, positions))
    return weighted_sum / total_mass

def rotate_vector(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D vector by a given angle.
    
    Args:
        vector: 2D vector to rotate
        angle: Angle in radians
        
    Returns:
        Rotated vector
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                              [sin_theta, cos_theta]])
    return np.dot(rotation_matrix, vector)

def calculate_impulse(
    mass1: float, mass2: float,
    velocity1: np.ndarray, velocity2: np.ndarray,
    normal: np.ndarray,
    restitution: float = 0.5
) -> float:
    """Calculate collision impulse magnitude using conservation of momentum.
    
    Args:
        mass1: Mass of first object
        mass2: Mass of second object
        velocity1: Velocity of first object
        velocity2: Velocity of second object
        normal: Collision normal vector
        restitution: Coefficient of restitution
        
    Returns:
        Impulse magnitude
    """
    relative_velocity = velocity2 - velocity1
    velocity_along_normal = np.dot(relative_velocity, normal)
    
    # Early exit if objects are moving apart
    if velocity_along_normal > 0:
        return 0.0
    
    impulse = -(1 + restitution) * velocity_along_normal
    impulse /= (1/mass1 + 1/mass2)
    
    return impulse

def integrate_force(mass: float, force: np.ndarray, velocity: np.ndarray, 
                   position: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Perform semi-implicit Euler integration for force.
    
    Args:
        mass: Object mass
        force: Applied force vector
        velocity: Current velocity vector
        position: Current position vector
        dt: Time step
        
    Returns:
        Tuple of (new_velocity, new_position)
    """
    acceleration = force / mass
    new_velocity = velocity + acceleration * dt
    new_position = position + new_velocity * dt
    return new_velocity, new_position

def calculate_gravitational_force(mass1: float, mass2: float, 
                                pos1: np.ndarray, pos2: np.ndarray,
                                G: float = 6.67430e-11) -> np.ndarray:
    """Calculate gravitational force between two masses.
    
    Args:
        mass1: Mass of first object
        mass2: Mass of second object
        pos1: Position of first object
        pos2: Position of second object
        G: Gravitational constant
        
    Returns:
        Gravitational force vector on mass1
    """
    r = pos2 - pos1
    distance = np.linalg.norm(r)
    if distance == 0:
        return np.zeros(2)
    
    force_magnitude = G * mass1 * mass2 / (distance * distance)
    return force_magnitude * (r / distance)