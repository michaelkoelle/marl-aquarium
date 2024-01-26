"""A class representing a 2D vector."""
import math


class Vector:
    """A class representing a 2D vector."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def add(self, other_vector: "Vector"):
        """Adds the x and y components of the other vector to this vector."""
        self.x += other_vector.x
        self.y += other_vector.y

    def sub(self, other_vector: "Vector"):
        """Subtracts the x and y components of the other vector from this vector."""
        self.x -= other_vector.x
        self.y -= other_vector.y

    def mult(self, scalar: float):
        """Multiplies the x and y components of this vector by the scalar."""
        self.x *= scalar
        self.y *= scalar

    def div(self, scalar: float):
        """Divides the x and y components of this vector by the scalar."""
        self.x /= scalar
        self.y /= scalar

    def negate(self):
        """Negates the x and y components of this vector."""
        self.x = -self.x
        self.y = -self.y

    def mag(self):
        """Returns the magnitude of this vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def set_mag(self, mag: float):
        """Sets the magnitude of this vector to the given magnitude."""
        self.normalize()
        self.mult(mag)

    def limit(self, max_speed: float):
        """Limits the magnitude of this vector to the given max_speed."""
        if self.mag() > max_speed:
            self.set_mag(max_speed)

    def normalize(self):
        """Normalizes this vector."""
        m = self.mag()
        if m != 0:
            self.div(m)

    def to_string(self):
        """Returns a string representation of this vector."""
        return f"x: {self.x}, y: {self.y}"

    def copy(self):
        """Returns a copy of this vector."""
        return Vector(self.x, self.y)
