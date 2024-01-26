"""This module contains the Animal class."""

import math
from abc import ABC, abstractmethod
from typing import Tuple

from marl_aquarium.env.vector import Vector


class Entity(ABC):
    """An abstract class representing an entity."""

    def __init__(
        self,
        position: Vector,
        velocity: Vector,
        acceleration: Vector,
        radius: int,
        view_distance: int,
        max_speed: float,
        mass: int,
        max_acceleration: float,
        max_orientation_change: int,
        color: Tuple[int, int, int],
    ):
        self.max_speed = max_speed
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.orientation_angle = -math.atan2(self.velocity.y, self.velocity.x) * (180 / math.pi)
        self.color = Color(color[0], color[1], color[2])
        self.alive = True
        self.radius = radius
        self.view_distance = view_distance
        self.max_acceleration = max_acceleration
        self.max_orientation_change = max_orientation_change
        self.stunned = False
        self.stun_steps = 0
        self.age = 0

    def calculate_new_position(self):
        """Calculates the new position of this entity."""
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y

    def calculate_new_velocity(self):
        """Calculates the new velocity of this entity."""
        self.velocity.x += self.acceleration.x
        self.velocity.y += self.acceleration.y

    def change_orientation(self, orientation: float):
        """Changes the orientation of this entity."""
        self.orientation_angle = round(orientation, 1)

    def apply_force(self, force: Vector):
        """Applies the given force to this entity."""
        force.div(self.mass)
        self.acceleration.add(force)

    @abstractmethod
    def id(self) -> str:
        """Returns the id of this entity."""


class Color:
    """A class representing a color."""

    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b
