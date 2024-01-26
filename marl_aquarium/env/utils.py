"""Utility functions for the environment"""

import math
import os
from typing import Any, Collection, List, Tuple

import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.VideoClip import ImageClip

from marl_aquarium.env.animal import Entity
from marl_aquarium.env.predator import Predator
from marl_aquarium.env.prey import Prey
from marl_aquarium.env.vector import Vector


class Torus:
    """A class representing a torus environment"""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def get_distance_in_torus(self, position1: Vector, position2: Vector):
        """Returns the distance between two positions in a torus environment"""
        dist_x = min(abs(position1.x - position2.x), self.width - abs(position1.x - position2.x))
        dist_y = min(abs(position1.y - position2.y), self.height - abs(position1.y - position2.y))

        return math.sqrt(dist_x**2 + dist_y**2)

    def collision(self, animal1: Entity, animal2: Entity):
        """Returns True if the two animals collide, False otherwise"""
        collision_distance = animal1.radius + animal2.radius
        if self.get_distance_in_torus(animal1.position, animal2.position) < collision_distance:
            return True
        return False

    def get_colliding_animal(self, animal: Entity, animals: Collection[Entity]):
        """Returns the first animal that collides with the given animal"""
        for a in animals:
            if self.collision(animal, a) and not animal == a:
                return a
        return None

    def get_directional_vector_to_animal_in_torus(self, animal: Entity, destination: Vector):
        """Returns a directional vector to the destination in a torus environment"""
        dx, dy = destination.x - animal.position.x, destination.y - animal.position.y
        directional_vector = Vector(
            dx - self.width if abs(dx) > self.width / 2 else dx,
            dy - self.height if abs(dy) > self.height / 2 else dy,
        )
        return directional_vector

    def get_nearest_entity_coordinates(self, animal: Entity, animals: Collection[Entity]):
        """Returns the coordinates of the nearest animal"""
        nearest_fish = None
        smallest_distance = 100000
        if not animals:
            return None
        for a in animals:
            distance = self.get_distance_in_torus(animal.position, a.position)
            if distance < smallest_distance or nearest_fish is None:
                nearest_fish = a
                smallest_distance = distance
        if nearest_fish is not None:
            return Vector(nearest_fish.position.x, nearest_fish.position.y)

        return None

    def get_direction_in_torus(self, position1: Vector, position2: Vector):
        """Returns the direction from position1 to position2 in a torus environment"""
        acceleration_vector = position2.copy()
        distance = self.get_distance_in_torus(position1, position2)
        acceleration_vector.sub(position1)
        if distance > self.width / 2:
            acceleration_vector.negate()

        direction = get_angle_from_vector(acceleration_vector)

        return direction

    def check_if_animal_is_in_view_cone(
        self,
        view_cone_position: Vector,
        view_cone_direction: float,
        animal: Entity,
        view_distance: float,
        fov: float,
    ) -> bool:
        """Returns True if the animal is in the view cone of the view cone position, False otherwise"""
        angle_to_shark = get_angle_from_vector(
            Vector(
                animal.position.x - view_cone_position.x, animal.position.y - view_cone_position.y
            )
        )
        angle_to_shark = scale(angle_to_shark, -180, 180, 0, 360)
        angular_diff = abs(angle_to_shark - view_cone_direction)
        # If the object is within the field of view and within the view distance
        if (
            angular_diff <= fov / 2
            and get_distance(view_cone_position, animal.position) <= view_distance
        ):
            return True

        return False

    def check_if_entity_is_in_view_in_torus(
        self, observer: Entity, animal: Entity, view_distance: float, fov: float
    ) -> bool:
        """Returns True if the animal is in the view cone of the observer, False otherwise"""
        view_cone_direction = scale(observer.orientation_angle, -180, 180, 0, 360)
        is_in_view = False
        # TODO: Create a function for offsetting positions
        # We check 8 other view cones around the original view cone,
        # as the animals can be on the other side of the torus
        offsets = [
            Vector(self.width, 0),
            Vector(-self.width, 0),
            Vector(0, self.height),
            Vector(0, -self.height),
            Vector(self.width, self.height),
            Vector(-self.width, self.height),
            Vector(self.width, -self.height),
            Vector(-self.width, -self.height),
        ]
        for offset in offsets:
            view_cone_position = observer.position.copy()
            view_cone_position.add(offset)
            is_in_view = self.check_if_animal_is_in_view_cone(
                view_cone_position, view_cone_direction, animal, view_distance, fov
            )

        # If the animal is not in one of the view cone copies,
        # we check if it is in the view cone of the original view cone
        if not is_in_view:
            is_in_view = self.check_if_animal_is_in_view_cone(
                observer.position, view_cone_direction, animal, view_distance, fov
            )
        return is_in_view

    def get_sides_on_which_position_is_outside_screen(self, position: Vector) -> List[str]:
        """Returns a list of sides on which the position is outside the screen"""
        sides = []
        if position.x < 0:
            sides.append("left")
        elif position.x > self.width:
            sides.append("right")
        if position.y < 0:
            sides.append("top")
        elif position.y > self.height:
            sides.append("bottom")
        return sides

    def position_offset(self, animal: Entity, function: Any, *args: Any):
        """Calls the given function with the animal's position and the given arguments
        for all positions that are offset by the width or height of the torus"""
        offsets = []
        if animal.position.x < self.width / 2 and animal.position.y < self.height / 2:
            offsets = [
                Vector(self.width, 0),
                Vector(0, self.height),
                Vector(self.width, self.height),
            ]
        elif animal.position.x < self.width / 2 and animal.position.y > self.height / 2:
            offsets = [
                Vector(self.width, 0),
                Vector(0, -self.height),
                Vector(self.width, -self.height),
            ]
        elif animal.position.x > self.width / 2 and animal.position.y < self.height / 2:
            offsets = [
                Vector(-self.width, 0),
                Vector(0, self.height),
                Vector(-self.width, self.height),
            ]
        elif animal.position.x > self.width / 2 and animal.position.y > self.height / 2:
            offsets = [
                Vector(-self.width, 0),
                Vector(0, -self.height),
                Vector(-self.width, -self.height),
            ]
        # offsets = [Vector(self.width, 0), Vector(-self.width, 0), Vector(0, self.height), Vector(0, -self.height),
        #            Vector(self.width, self.height), Vector(-self.width, self.height), Vector(self.width, -self.height), Vector(-self.width, -self.height)]
        for offset in offsets:
            animal_copy_position = animal.position.copy()
            animal_copy_position.add(offset)
            function(animal_copy_position, *args)

    def check_if_animal_is_in_view(
        self, observer: Entity, animal: Entity, fov: float, view_distance: float
    ) -> bool:
        """Returns True if the animal is in the view cone of the observer, False otherwise"""
        angle_to_shark = math.atan2(
            animal.position.y - observer.position.y, animal.position.x - observer.position.x
        )
        # Normalize the angle between 0 and 2*pi
        angle_to_shark = (angle_to_shark + 2 * math.pi) % (2 * math.pi)
        # Calculate the absolute angular difference between the direction and the object 1 to object 2 line
        angular_diff = abs(angle_to_shark - observer.orientation_angle)
        # If the object is within the field of view and within the view distance
        if (
            angular_diff <= fov / 2
            and self.get_distance_in_torus(observer.position, animal.position) <= view_distance
        ):
            return True

        return False


def angle_diff(angle1: float, angle2: float):
    """Returns the smallest difference between two angles in degrees"""
    return min((angle1 - angle2) % 360, (angle2 - angle1) % 360)


def get_distance(position1: Vector, position2: Vector):
    """Returns the distance between two positions in a torus environment"""
    return np.sqrt((position1.x - position2.x) ** 2 + (position1.y - position2.y) ** 2)


def get_point_from_angle_and_distance(position: Vector, direction: int, distance: int) -> Vector:
    """Returns the point that is distance away from the position in the given direction"""
    # Calculate the x and y components of the new position
    dx = distance * math.cos(direction)
    dy = distance * math.sin(direction)

    # Calculate the new position by adding the components to position1
    x = position.x + dx
    y = position.y + dy

    return Vector(x, y)


def get_action_angle_from_vector(vector: Vector, action_count: int):
    """Returns the action angle from the given vector"""
    angle = math.atan2(vector.y, vector.x)
    angle = math.degrees(angle)
    if angle < 0:
        angle += 360
    angle += 90
    angle = angle % 360
    action_angle = angle_to_action(angle, action_count)
    return action_angle


def scale(
    value: float,
    input_min: float,
    input_max: float,
    output_min: float,
    output_max: float,
    tolerance: float = 1e-9,
):
    """Scales the given value from the input range to the output range"""

    assert (
        value >= input_min - tolerance
    ), f"Value {value} is smaller than input_min {input_min} considering tolerance"
    assert (
        value <= input_max + tolerance
    ), f"Value {value} is larger than input_max {input_max} considering tolerance"

    # Adjust the input value if it's within the tolerance of the boundaries
    if abs(value - input_min) < tolerance:
        value = input_min
    elif abs(value - input_max) < tolerance:
        value = input_max

    input_range = input_max - input_min
    output_range = output_max - output_min

    # Scale the value from the input range to the output range
    scaled_value = (((value - input_min) * output_range) / input_range) + output_min

    # Ensure the scaled value is within the output boundaries
    scaled_value = max(min(scaled_value, output_max), output_min)

    return scaled_value


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Converts cartesian coordinates to polar coordinates and returns the radius and angle"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def angle_to_action(angle: float, action_count: int):
    """Converts an angle to an action"""
    return round((angle * action_count) / 360)


def get_angle_from_vector(vector: Vector):
    """Returns the angle of the given vector in degrees"""
    return -math.atan2(vector.y, vector.x) * (180 / math.pi)


def get_vector_from_angle(angle: int):
    """Returns a vector from the given angle in degrees"""
    angle_radians = math.radians(angle)

    return Vector(math.sin(angle_radians), -math.cos(angle_radians))


def get_vector_from_action(action: int, action_count: int):
    """Returns a vector from the given action"""
    angle = (360 / action_count) * action
    radians = math.radians(angle)
    vector = Vector(math.sin(radians), -math.cos(radians))
    return vector


def get_prey_by_id(prey_id: str, prey: List[Prey]):
    """Returns the prey with the given id"""
    for p in prey:
        if p.id() == prey_id:
            return p
    return None


def get_predator_by_id(predator_id: str, predators: List[Predator]):
    """Returns the predator with the given id"""
    for p in predators:
        if p.id() == predator_id:
            return p
    return None


def save_frames_as_video(frames: Any, episode: int, log_dir: str):
    """Saves the given frames as a video"""
    frame_rate = 60
    run_dir = "run_videos"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # Convert the frames to a list of ImageClips
    image_clips = [ImageClip(frame, duration=1 / frame_rate) for frame in frames]
    print(f"Number of frames: {len(image_clips)}")
    # Create a video clip from the ImageClips
    video_clip = concatenate_videoclips(image_clips, method="compose")

    # Set the output file path
    log_dir = log_dir.split("/")[1]
    if not os.path.exists(f"{run_dir}/{log_dir}"):
        os.makedirs(f"{run_dir}/{log_dir}")
    output_file = f"{run_dir}/{log_dir}/episode_{episode}.mp4"
    print(f"Saving video to {output_file}")

    video_clip.fps = frame_rate

    # Write the video clip to an MP4 file
    video_clip.write_videofile(output_file, codec="libx264", bitrate="50000k", audio=False)
