"""Predator class, inherits from Animal class."""

from typing import Tuple

from marl_aquarium.env.animal import Entity
from marl_aquarium.env.vector import Vector


class Predator(Entity):
    """Predator class, inherits from Animal class."""

    identifier = 0

    def __init__(
        self,
        position: Vector,
        velocity: Vector,
        acceleration: Vector,
        radius: int,
        view_distance: int,
        max_velocity: float,
        max_acceleration: float,
        mass: int = 10,  # Average mass of a Shark is 680 kg
        max_orientation_change: int = 10,
        color: Tuple[int, int, int] = (150, 207, 250),
    ):
        Entity.__init__(
            self,
            position,
            velocity,
            acceleration,
            radius,
            view_distance,
            max_velocity,
            mass,
            max_acceleration,
            max_orientation_change,
            color,
        )
        self.identifier = Predator.identifier
        Predator.identifier += 1

    def id(self):
        return "predator_" + str(self.identifier)
