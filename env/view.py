import math

import pygame

from env import utils
from env.animal import Entity
from env.prey import Prey
from env.predator import Predator
from env.vector import Vector


class View:
    def __init__(self, width, height, caption, fps):
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

        self.fish_image = pygame.image.load("graphics/fish3.PNG")
        self.fish_image.convert()

        self.shark_image = pygame.image.load("graphics/shark3.PNG")
        self.shark_image.convert()

    def draw_view_cone(
        self, position: Vector, orientation_angle: int, view_distance: int, fov: int, color
    ):
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
        color = color + (alpha,)
        angles_surface = self.screen.convert_alpha()
        angles_surface.fill([0, 0, 0, 0])
        pygame.draw.polygon(angles_surface, color, points)
        self.screen.blit(angles_surface, (0, 0))

    def draw_background(self):
        self.screen.blit(self.background, (0, 0))
        self.clock.tick(self.fps)
        fps = self.clock.get_fps()
        fps_string = self.font.render(str(int(fps)), True, pygame.Color("black"))
        # self.screen.blit(fps_string, (1, 1))

    def draw_circle_at_position(self, position: Vector, color, size):
        circle_surface = self.screen.convert_alpha()
        circle_surface.fill([0, 0, 0, 0])
        pygame.draw.circle(circle_surface, color, (position.x, position.y), size)
        self.screen.blit(circle_surface, (0, 0))

    def draw_line_from_position_to_position(
        self, position1: Vector, position2: Vector, color, size
    ):
        pygame.draw.circle(self.screen, color, (position2.x, position2.y), 3)
        pygame.draw.line(
            self.screen, color, (position1.x, position1.y), (position2.x, position2.y), size
        )

    def draw_animal(self, position: Vector, animal: Entity):
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
        return pygame.surfarray.array3d(pygame.display.get_surface())

    @staticmethod
    def close():
        pygame.quit()
