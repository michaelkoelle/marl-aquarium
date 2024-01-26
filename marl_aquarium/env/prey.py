"""Prey class, inherits from Entity class."""

import random
from typing import Tuple

from marl_aquarium.env.animal import Entity
from marl_aquarium.env.vector import Vector


class Prey(Entity):
    """Prey class, inherits from Entity class."""

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
        mass: int = 3,  # Average mass of a fish is 2 kg
        max_orientation_change: int = 10,
        color: Tuple[int, int, int] = (255, 140, 0),
        replication_age: int = 200,
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
        self.recently_died = False
        self.death_count = 0
        self.identifier = Prey.identifier
        Prey.identifier += 1
        self.replication_age = random.randint(
            int(replication_age - replication_age / 4),
            int(replication_age + replication_age / 4),
        )

    def id(self):
        return "prey_" + str(self.identifier)

    def replicate(self):
        """Replicates this prey and returns the new prey."""
        parent_fish_location = self.position.copy()
        initial_velocity = Vector(0, 0)
        initial_acceleration = Vector(0, 0)
        return Prey(
            parent_fish_location,
            initial_velocity,
            initial_acceleration,
            self.radius,
            self.view_distance,
            self.max_speed,
            self.max_acceleration,
        )
