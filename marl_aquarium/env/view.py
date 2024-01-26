"""View class, handles the drawing of the environment."""

import math
from importlib.resources import path  # Use importlib_resources for Python < 3.7
from typing import Tuple

import pygame

from marl_aquarium.env import utils
from marl_aquarium.env.animal import Entity
from marl_aquarium.env.predator import Predator
from marl_aquarium.env.prey import Prey
from marl_aquarium.env.vector import Vector


class View:
    """View class, handles the drawing of the environment."""

    def __init__(self, width: int, height: int, caption: str, fps: int):
        pygame.init()
        pygame.display.set_caption(caption)

        self.width = width
        self.height = height
        self.background_color = (172, 206, 231)

        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill(self.background_color)
        self.font = pygame.font.Font(None, 25)
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill(self.background_color)

        with path("marl_aquarium.env.assets", "prey.png") as img_path:
            self.fish_image = pygame.image.load(str(img_path))
            self.fish_image.convert()
        with path("marl_aquarium.env.assets", "predator.png") as img_path:
            self.shark_image = pygame.image.load(str(img_path))
            self.shark_image.convert()

    def draw_view_cone(
        self,
        position: Vector,
        orientation_angle: int,
        view_distance: int,
        fov: int,
        color: Tuple[int, int, int],
    ):
        """
        Draws a cone at the given position with the given orientation angle,
        view distance and field of view.
        """
        cone_pos = position

        cone_angle = math.radians(utils.scale(-orientation_angle, -180, 180, 0, 360)) + math.pi

        # Cone parameters
        cone_angle_span = math.radians(fov)
        cone_radius = view_distance

        # Calculate the start and end angles for the arc
        start_angle = math.degrees(cone_angle - cone_angle_span / 2)
        end_angle = math.degrees(cone_angle + cone_angle_span / 2)

        # Start list of polygon points
        points = [(cone_pos.x, cone_pos.y)]
        # Get points on arc
        for n in range(int(start_angle), int(end_angle)):
            x = cone_pos.x + int(cone_radius * math.cos(n * math.pi / 180))
            y = cone_pos.y + int(cone_radius * math.sin(n * math.pi / 180))
            points.append((x, y))
        points.append((cone_pos.x, cone_pos.y))

        # Draw the filled cone
        alpha = 80
        color_alpha = color + (alpha,)
        angles_surface = self.screen.convert_alpha()
        angles_surface.fill([0, 0, 0, 0])
        pygame.draw.polygon(angles_surface, color_alpha, points)
        self.screen.blit(angles_surface, (0, 0))

    def draw_background(self):
        """Draws the background of the pygame window."""
        self.screen.blit(self.background, (0, 0))
        self.clock.tick(self.fps)
        fps = self.clock.get_fps()
        fps_string = self.font.render(str(int(fps)), True, pygame.Color("black"))
        self.screen.blit(fps_string, (1, 1))

    def draw_circle_at_position(
        self, position: Vector, color: Tuple[int, int, int, int], size: float
    ):
        """Draws a circle at the given position."""
        circle_surface = self.screen.convert_alpha()
        circle_surface.fill([0, 0, 0, 0])
        pygame.draw.circle(circle_surface, color, (position.x, position.y), size)
        self.screen.blit(circle_surface, (0, 0))

    def draw_line_from_position_to_position(
        self, position1: Vector, position2: Vector, color: Tuple[int, int, int], width: int
    ):
        """Draws a line from position1 to position2."""
        pygame.draw.circle(self.screen, color, (position2.x, position2.y), 3)
        pygame.draw.line(
            self.screen, color, (position1.x, position1.y), (position2.x, position2.y), width
        )

    def draw_animal(self, position: Vector, animal: Entity):
        """Draws the given animal at the given position."""
        if animal.alive:
            if isinstance(animal, Prey):
                fish = animal
                angle = fish.orientation_angle
                fish_image_copy = pygame.transform.rotate(self.fish_image, angle)
                self.screen.blit(
                    fish_image_copy,
                    (
                        position.x - int(fish_image_copy.get_width() / 2),
                        position.y - int(fish_image_copy.get_height() / 2),
                    ),
                )

            elif isinstance(animal, Predator):
                shark = animal
                angle = shark.orientation_angle
                shark_image_copy = pygame.transform.rotate(self.shark_image, angle)
                self.screen.blit(
                    shark_image_copy,
                    (
                        position.x - int(shark_image_copy.get_width() / 2),
                        position.y - int(shark_image_copy.get_height() / 2),
                    ),
                )

    def get_frame(self):
        """Returns the current frame of the pygame window."""
        return pygame.surfarray.array3d(pygame.display.get_surface())

    @staticmethod
    def close():
        """Closes the pygame window."""
        pygame.quit()
