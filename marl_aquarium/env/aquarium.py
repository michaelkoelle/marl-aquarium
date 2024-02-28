"""Aquarium environment"""

import copy
import datetime
import functools
import random
import sys
from typing import Any, Collection, Dict, List, Optional, Sequence

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

from marl_aquarium.env.animal import Entity
from marl_aquarium.env.predator import Predator
from marl_aquarium.env.prey import Prey
from marl_aquarium.env.utils import (
    Torus,
    get_angle_from_vector,
    get_predator_by_id,
    get_prey_by_id,
    get_vector_from_action,
    scale,
)
from marl_aquarium.env.vector import Vector
from marl_aquarium.env.view import View


class raw_env(ParallelEnv[str, Box, Discrete | None]):  # pylint: disable=C0103
    """The Aquarium environment"""

    metadata = {"name": "aquarium-v0", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "human",
        observable_walls: int = 2,
        width: int = 800,
        height: int = 800,
        caption: str = "Aquarium",
        fps: int = 60,
        max_time_steps: int = 3000,
        action_count: int = 16,
        predator_count: int = 1,
        prey_count: int = 16,
        predator_observe_count: int = 1,
        prey_observe_count: int = 3,
        draw_force_vectors: bool = False,
        draw_action_vectors: bool = False,
        draw_view_cones: bool = False,
        draw_hit_boxes: bool = False,
        draw_death_circles: bool = False,
        fov_enabled: bool = True,
        keep_prey_count_constant: bool = True,
        prey_radius: int = 20,
        prey_max_acceleration: float = 1,
        prey_max_velocity: float = 4,
        prey_view_distance: int = 100,
        prey_replication_age: int = 200,
        prey_max_steer_force: float = 0.6,
        prey_fov: int = 120,
        prey_reward: int = 1,
        prey_punishment: int = 1000,
        max_prey_count: int = 20,
        predator_max_acceleration: float = 0.6,
        predator_radius: int = 30,
        predator_max_velocity: float = 5,
        predator_view_distance: int = 200,
        predator_max_steer_force: float = 0.6,
        predator_max_age: int = 3000,
        predator_fov: int = 150,
        predator_reward: int = 10,
        catch_radius: int = 100,
        procreate: bool = False,
    ):
        self.render_mode = render_mode
        self.height = height
        self.width = width
        self.caption = caption
        self.fps = fps
        self.max_time_steps = max_time_steps
        self.action_count = action_count
        self.predator_count = predator_count
        self.prey_count = prey_count
        self.predator_observe_count = predator_observe_count
        self.prey_observe_count = prey_observe_count
        self.fov_enabled = fov_enabled
        self.keep_prey_count_constant = keep_prey_count_constant
        self.prey_radius = prey_radius
        self.prey_max_acceleration = prey_max_acceleration
        self.prey_max_velocity = prey_max_velocity
        self.prey_view_distance = prey_view_distance
        self.prey_replication_age = prey_replication_age
        self.prey_max_steer_force = prey_max_steer_force
        self.prey_fov = prey_fov
        self.prey_reward = prey_reward
        self.prey_punishment = prey_punishment
        self.max_prey_count = max_prey_count
        self.predator_max_acceleration = predator_max_acceleration
        self.predator_radius = predator_radius
        self.predator_max_velocity = predator_max_velocity
        self.predator_view_distance = predator_view_distance
        self.predator_max_steer_force = predator_max_steer_force
        self.predator_max_age = predator_max_age
        self.predator_fov = predator_fov
        self.predator_reward = predator_reward
        self.catch_radius = catch_radius
        self.procreate = procreate

        # DEBUG
        self.draw_force_vectors = draw_force_vectors
        self.draw_action_vectors = draw_action_vectors
        self.draw_view_cones = draw_view_cones
        self.draw_hit_boxes = draw_hit_boxes
        self.draw_death_circles = draw_death_circles

        self.time_step = 0
        # self.terminations = None
        # self.truncations = None

        # Init
        self.prey = self.create_prey()
        self.predators = self.create_predators()
        self.all_entities = self.prey + self.predators
        self.current_prey_count = len(self.prey)

        self.observable_walls = observable_walls
        self.dead_animals = {}
        self.death_positions = []
        self.possible_agents = ["predator_" + str(r) for r in range(self.predator_count)] + [
            "prey_" + str(i) for i in range(self.prey_count)
        ]
        self.obs_size = 6
        self.number_of_fish_observations = (
            5
            + self.predator_observe_count * self.obs_size
            + (self.prey_observe_count) * self.obs_size
        )

        self.number_of_predator_observations = (
            5
            + self.prey_observe_count * self.obs_size
            + (self.predator_observe_count) * self.obs_size
        )
        self.number_of_observations = (
            self.number_of_fish_observations + self.number_of_predator_observations
        )

        self.agents = copy.deepcopy(self.possible_agents)
        self.past_shark_positions = None

        self.torus = Torus(self.width, self.height)
        self.view = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ):
        self.agents = copy.deepcopy(self.possible_agents)
        Predator.identifier = 0
        Prey.identifier = 0
        self.time_step = 0
        self.prey = self.create_prey()
        self.predators = self.create_predators()
        self.current_prey_count = len(self.prey)
        self.all_entities = self.prey + self.predators
        infos = {agent: {} for agent in self.agents}
        return self.get_obs(), infos

    def state(self):
        return np.array(self.get_obs())

    def step(self, actions: Dict[Any, Any]):
        catches = []
        for entity in self.all_entities:
            desired_velocity = self.get_desired_velocity_from_action(actions[entity.id()], entity)
            entity.age += 1

            if isinstance(entity, Predator):
                self.update_predator(entity, desired_velocity)
            else:
                prey = self.update_prey(entity, self.predators, desired_velocity)
                if not prey.alive:
                    kill_event = {"killed": prey.id(), "position": prey.position}
                    catches.append(kill_event)

            # TODO: Move this to the render function
            if self.view and self.draw_action_vectors:
                self.view.draw_line_from_position_to_position(
                    entity.position,
                    Vector(
                        entity.position.x + desired_velocity.x * 15,
                        entity.position.y + desired_velocity.y * 15,
                    ),
                    (255, 0, 0),
                    4,
                )

        remaining_fish = [fish.id() for fish in self.prey]
        fishes_in_agents = [agent for agent in self.agents if agent.startswith("fish")]
        dead_fishes = list(set(fishes_in_agents) - set(remaining_fish))
        observations = self.get_obs()

        infos = {agent: {} for agent in self.agents}

        rewards = self.get_rewards(catches)
        for dead_fish in dead_fishes:
            rewards[dead_fish] = 0

        # Check termination conditions
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        for agent in self.agents:
            if agent.startswith("predator"):
                terminations[agent] = get_predator_by_id(agent, self.predators) is None
            elif agent.startswith("prey"):
                terminations[agent] = get_prey_by_id(agent, self.prey) is None

        if self.time_step > self.max_time_steps or len(self.prey) == 0 or len(self.predators) == 0:
            truncations = {a: True for a in self.agents}

        self.time_step += 1

        for agent in self.agents:
            if agent not in observations:
                if agent.startswith("predator"):
                    observations[agent] = [0] * self.number_of_predator_observations
                elif agent.startswith("prey"):
                    observations[agent] = [0] * self.number_of_fish_observations
            if agent not in rewards:
                rewards[agent] = 0.0
            if agent not in infos:
                infos[agent] = {}
            if agent not in terminations:
                terminations[agent] = True
            if agent not in truncations:
                truncations[agent] = False

        # Remove terminated or truncated agents
        to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                to_remove.append(agent)

        for agent in to_remove:
            self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def get_obs(self):
        """Returns the observations for all agents"""
        predator_observations = self.predator_observe()
        prey_observations = self.prey_observe()
        observations = prey_observations | predator_observations
        return observations

    def render(self, mode: str | None = None):
        if mode is not None:
            self.render_mode = mode

        if self.view is None:
            self.view = View(self.width, self.height, self.caption, self.fps)

        self.view.draw_background()

        screenshot_number = 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # pylint: disable=no-member
                pygame.quit()  # pylint: disable=no-member
                sys.exit()

        for entity in self.all_entities:
            if isinstance(entity, Prey):
                if self.draw_force_vectors:
                    self.draw_force_vectors_to_canvas(entity)
                if self.draw_view_cones:
                    self.draw_view_cone_in_torus(entity, self.prey_view_distance, self.prey_fov)
                if self.draw_hit_boxes:
                    self.draw_hit_box_in_torus(entity)
                if self.draw_death_circles:
                    self.draw_death_circle(entity)
                self.draw_entity_in_torus(entity)
            else:
                if self.draw_force_vectors:
                    self.draw_force_vectors_to_canvas(entity)
                if self.draw_view_cones:
                    self.draw_view_cone_in_torus(
                        entity, self.predator_view_distance, self.predator_fov
                    )
                if self.draw_hit_boxes:
                    self.draw_hit_box_in_torus(entity)
                self.draw_entity_in_torus(entity)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:  # pylint: disable=no-member
                if event.button == 1:  # Left mouse button clicked
                    # Generate a screenshot filename based on timestamp
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    screenshot_filename = f"screens/screenshot_{timestamp}_{screenshot_number}.png"
                    screenshot_number += 1

                    # Capture the screenshot
                    frame = self.view.get_frame()
                    # Convert the NumPy array back to a Pygame surface
                    screenshot_surface = pygame.surfarray.make_surface(frame)
                    pygame.image.save(screenshot_surface, screenshot_filename)
                    print(f"Screenshot saved as {screenshot_filename}")

        if self.render_mode == "rgb_array":
            # Get the current frame as RGB array
            frame = self.view.get_frame()
            return frame

    def close(self):
        pygame.quit()  # pylint: disable=no-member
        sys.exit()

    def create_random_prey(self):
        """Creates a prey with a random position and velocity"""
        random_initial_pos = Vector(
            random.randint(20, self.width - 20), random.randint(20, self.height - 20)
        )
        random_initial_vel = Vector(-1, 0)
        initial_acceleration = Vector(0, 0)
        return Prey(
            random_initial_pos,
            random_initial_vel,
            initial_acceleration,
            self.prey_radius,
            self.prey_view_distance,
            self.prey_max_velocity,
            self.prey_max_acceleration,
        )

    def create_random_predator(self):
        """Creates a predator with a random position and velocity"""
        random_initial_pos = Vector(random.randint(0, self.width), random.randint(0, self.height))
        random_initial_vel = Vector(0, -1)
        initial_acceleration = Vector(0, 0)
        return Predator(
            random_initial_pos,
            random_initial_vel,
            initial_acceleration,
            self.predator_radius,
            self.predator_view_distance,
            self.predator_max_velocity,
            self.predator_max_acceleration,
        )

    def create_predators(self):
        """Creates a list of predators"""
        return [self.create_random_predator() for _ in range(self.predator_count)]

    def create_prey(self):
        """Creates a list of prey"""
        return [self.create_random_prey() for _ in range(self.prey_count)]

    @functools.lru_cache(maxsize=None)  # type: ignore
    def observation_space(self, agent: str):
        # Predator
        if agent.startswith("predator"):
            return Box(
                low=0.0,
                high=1.0,
                shape=(self.number_of_predator_observations,),
                dtype=np.float64,
            )
        # Prey
        return Box(
            low=0.0,
            high=1.0,
            shape=(self.number_of_fish_observations,),
            dtype=np.float64,
        )

    @functools.lru_cache(maxsize=None)  # type: ignore
    def action_space(self, agent: str):
        return Discrete(self.action_count)

    def check_borders(self, animal: Entity):
        """Checks if the animal is outside the borders of the aquarium and moves it to the other side if it is"""
        if animal.position.x > self.width:
            animal.position.x = 0
        elif animal.position.x < 0:
            animal.position.x = self.width
        if animal.position.y > self.height:
            animal.position.y = 0
        elif animal.position.y < 0:
            animal.position.y = self.height

    @staticmethod
    def spawn_new_prey(parent_prey: Prey):
        """Spawns a new prey at the location of the parent prey"""
        parent_fish_location = parent_prey.position.copy()
        random_initial_vel = Vector(0, 0)
        return Prey(
            parent_fish_location,
            random_initial_vel,
            Vector(0, 0),
            parent_prey.radius,
            parent_prey.view_distance,
            parent_prey.max_speed,
            parent_prey.max_acceleration,
        )

    def update_prey(self, prey: Prey, predators: List[Predator], desired_velocity: Vector):
        """Updates the prey"""
        prey.recently_died = False
        if self.torus.get_colliding_animal(prey, predators) is not None:
            if self.keep_prey_count_constant:
                # TODO: Sharks now get no reward for eating fish
                prey.death_count += 1
                prey.recently_died = True
                prey.position = Vector(
                    random.randint(0, self.width), random.randint(0, self.height)
                )
            else:
                prey.alive = False
                # print(f'Fish {fish.id()} died')
                self.current_prey_count -= 1
                # print(self.number_of_fish)
        steer_force = desired_velocity.copy()
        steer_force.sub(prey.velocity)
        steer_force.limit(self.prey_max_steer_force)

        prey.apply_force(steer_force)

        prey.acceleration.normalize()
        prey.acceleration.mult(self.prey_max_acceleration)

        colliding_fish = self.torus.get_colliding_animal(prey, self.prey)
        if colliding_fish is None:
            prey.velocity.add(prey.acceleration)
        else:
            # Acceleration in opposite direction of other fish
            bounce_acceleration = self.torus.get_directional_vector_to_animal_in_torus(
                prey, colliding_fish.position
            )
            bounce_acceleration.negate()
            bounce_acceleration.normalize()
            prey.velocity.add(bounce_acceleration)
        prey.velocity.limit(self.prey_max_velocity)
        prey.position.add(prey.velocity)
        self.check_borders(prey)
        prey.change_orientation(get_angle_from_vector(prey.velocity))

        if self.procreate:
            if prey.age == prey.replication_age and self.current_prey_count < self.max_prey_count:
                self.prey.append(prey.replicate())
                prey.age = 0
                self.all_entities = self.prey + self.predators
                self.current_prey_count += 1
        if not prey.alive:
            self.dead_animals[prey.id()] = self.time_step
            self.all_entities.remove(prey)
            self.prey.remove(prey)
        return prey

    def update_predator(self, predator: Predator, desired_velocity: Vector):
        """Updates the predator"""
        if predator.age > self.predator_max_age:
            predator.alive = False

        steer_force = desired_velocity.copy()
        steer_force.sub(predator.velocity)
        steer_force.limit(self.predator_max_steer_force)

        predator.apply_force(steer_force)
        predator.acceleration.normalize()
        predator.acceleration.mult(self.predator_max_acceleration)

        colliding_shark = self.torus.get_colliding_animal(predator, self.predators)
        colliding_fish = self.torus.get_colliding_animal(predator, self.prey)

        if colliding_shark is not None:
            bounce_acceleration = self.torus.get_directional_vector_to_animal_in_torus(
                predator, colliding_shark.position
            )
            bounce_acceleration.negate()
            bounce_acceleration.set_mag(self.predator_max_velocity / 8)
            predator.acceleration.add(bounce_acceleration)

        if colliding_fish is not None:
            predator.age = 0

        predator.velocity.add(predator.acceleration)
        predator.velocity.limit(self.predator_max_velocity)
        predator.position.add(predator.velocity)
        self.check_borders(predator)
        predator.change_orientation(get_angle_from_vector(predator.velocity))
        if not predator.alive:
            self.all_entities.remove(predator)
            self.predators.remove(predator)
        return predator

    def get_predator_by_id(self, predator_id: str):
        """Returns the predator with the given id"""
        for shark in self.predators:
            if shark.id() == predator_id:
                return shark

    def get_prey_by_id(self, prey_id: str):
        """Returns the prey with the given id"""
        for fish in self.prey:
            if fish.id() == prey_id:
                return fish

    def get_entity_by_id(self, animal_id: str):
        """Returns the animal with the given id"""
        for animal in self.all_entities:
            if animal.id() == animal_id:
                return animal

    def draw_target(self, predator: Predator, agent_target_position: Vector):
        """Draws the target of the predator"""
        if not self.view:
            return

        self.view.draw_circle_at_position(agent_target_position, (255, 0, 0, 255), 5)
        # self.view.draw_line_from_position_to_position(shark.position, agent_target_position, color=(255, 0, 0))

    def draw_force_vectors_to_canvas(self, entity: Entity):
        """Draws the force vectors of the given entity"""
        if not self.view:
            return

        velocity_vector = entity.velocity.copy()
        velocity_vector.normalize()
        velocity_vector.mult(30)
        direction = Vector(
            entity.position.x + velocity_vector.x, entity.position.y + velocity_vector.y
        )

        acceleration_vector = entity.acceleration.copy()
        acceleration_vector.normalize()
        acceleration_vector.mult(50)
        acceleration = Vector(
            entity.position.x + acceleration_vector.x,
            entity.position.y + acceleration_vector.y,
        )

        # Draw acceleration vector
        self.view.draw_line_from_position_to_position(
            entity.position, acceleration, (0, 0, 200), 3
        )
        # Draw velocity vector
        self.view.draw_line_from_position_to_position(entity.position, direction, (0, 255, 0), 3)

    def draw_view_cone_in_torus(self, animal: Entity, view_distance: int, fov: int):
        """Draws the view cone of the given animal"""
        if not self.view:
            return

        if isinstance(animal, Predator):
            color = (82, 117, 172)
        else:
            color = (167, 98, 88)

        # Draw main view cone
        self.view.draw_view_cone(
            animal.position, int(animal.orientation_angle), view_distance, fov, color
        )
        # Draw view_cone multiple times, so it is visible when it is at the edge of the screen
        self.torus.position_offset(
            animal,
            self.view.draw_view_cone,
            animal.orientation_angle,
            view_distance,
            fov,
            color,
        )

    def draw_entity_in_torus(self, entity: Entity):
        """Draws the given einity in the torus"""
        if not self.view:
            return

        # Draw main animal
        self.view.draw_animal(entity.position, entity)
        # Draw animal multiple times, so it is visible when it is at the edge of the screen
        self.torus.position_offset(entity, self.view.draw_animal, entity)

    def draw_hit_box_in_torus(self, entity: Entity):
        """Draws the hit box of the given entity"""
        if not self.view:
            return

        alpha = 80
        if isinstance(entity, Predator):
            color = (82, 117, 172, alpha)
        else:
            color = (167, 98, 88, alpha)
        # Draw main animal
        self.view.draw_circle_at_position(entity.position, color, entity.radius)
        # Draw hit box multiple times, so it is visible when it is at the edge of the screen
        # utils.position_offset(animal, self.view.draw_circle_at_position, color, animal.radius)

    def draw_death_circle(self, prey: Prey):
        """Draws the death circle of the given prey"""
        if not self.view:
            return

        if prey.recently_died:
            self.death_positions.append(prey.position)
        color = (255, 0, 0, 255)
        for death_position in self.death_positions:
            self.view.draw_circle_at_position(death_position, color, 4)

    def get_desired_velocity_from_action(self, action: int, animal: Entity):
        """Returns the desired velocity from the given action"""
        max_velocity = 0
        if isinstance(animal, Predator):
            max_velocity = self.predator_max_velocity
        elif isinstance(animal, Prey):
            max_velocity = self.prey_max_velocity
        desired_velocity = get_vector_from_action(action, self.action_count)
        desired_velocity.normalize()
        desired_velocity.mult(max_velocity)
        return desired_velocity

    def get_rewards(self, catches: Collection[Any]):
        """Returns the rewards for all entities in the environment."""
        prey_rewards = {fish.id(): self.get_prey_rewards(fish) for fish in self.prey}
        predator_rewards = {
            predator.id(): self.get_predator_rewards(predator, catches)
            for predator in self.predators
        }

        rewards = prey_rewards | predator_rewards
        return rewards

    def get_predator_rewards(self, predator: Predator, catches: Collection[Any]) -> float:
        """Returns the reward for a predator."""
        total_reward = 0
        for catch in catches:
            catch_position = catch["position"]
            sharks_in_radius = [
                shark
                for shark in self.predators
                if self.torus.get_distance_in_torus(shark.position, catch_position)
                < self.catch_radius
            ]
            if predator in sharks_in_radius:
                total_reward += self.predator_reward / len(sharks_in_radius)

        return total_reward

    def get_prey_rewards(self, prey: Prey) -> float:
        """Returns the reward for a prey."""
        total_reward = 0
        # Reward for not being eaten
        if prey.alive:
            total_reward += self.prey_reward
        if prey.recently_died:
            total_reward -= self.prey_punishment
        # if len(fishes) == 0:
        #     total_reward = -200
        return total_reward

    def prey_observer_observation(
        self, observer: Entity, obs_min: float = 0, obs_max: float = 1
    ) -> List[float]:
        """Get the observations of the current state of the environment of an observer."""
        position = observer.position
        direction = observer.orientation_angle
        speed = observer.velocity.mag()
        scaled_position_x = scale(position.x, 0, self.width, obs_min, obs_max)
        scaled_position_y = scale(position.y, 0, self.height, obs_min, obs_max)
        scaled_direction = scale(direction, -180, 180, obs_min, obs_max)
        scaled_speed = scale(speed, 0, observer.max_speed, obs_min, obs_max)
        observation = [1, scaled_position_x, scaled_position_y, scaled_direction, scaled_speed]
        # print(f'Observer_observation: {len(observation)}')

        assert all(
            obs_min <= value <= obs_max for value in observation
        ), "prey_observer_observation: All values must be between -1 and 1"
        return observation

    def nearby_animal_observation(
        self, observer: Entity, animal: Entity, obs_min: float = 0, obs_max: float = 1
    ) -> List[float]:
        """Get the observations of the current state of the environment of an observer."""
        position = animal.position
        distance = self.torus.get_distance_in_torus(observer.position, animal.position)
        direction = self.torus.get_direction_in_torus(observer.position, animal.position)
        speed = animal.velocity.mag()
        scaled_position_x = scale(position.x, 0, self.width, obs_min, obs_max)
        scaled_position_y = scale(position.y, 0, self.height, obs_min, obs_max)
        scaled_distance = scale(distance, 0, observer.view_distance, obs_min, obs_max)
        scaled_direction = scale(direction, -180, 180, obs_min, obs_max)
        scaled_speed = scale(speed, 0, animal.max_speed, obs_min, obs_max)

        if isinstance(animal, Predator):
            entity_type = 0
        else:
            entity_type = 1

        observation = [
            entity_type,
            scaled_position_x,
            scaled_position_y,
            scaled_distance,
            scaled_direction,
            scaled_speed,
        ]

        # print(f'Nearby_animal_observation: {observation}')
        # print(distance, scaled_distance)
        # print(observation)
        assert all(
            obs_min <= value <= obs_max for value in observation
        ), "nearby_animal_observation: All values must be between -1 and 1"
        return observation

    def prey_get_n_closest_animals(
        self, observer: Entity, other_animals: Sequence[Entity], n_nearest_animals: int
    ) -> List[Entity]:
        """Get the n nearest animals to the observer."""
        distances = [
            (
                self.torus.get_distance_in_torus(observer.position, other_animal.position),
                other_animal,
            )
            for other_animal in other_animals
        ]
        # Sort the distances list based on the first element of each tuple (the number)
        sorted_distances = sorted(distances, key=lambda x: x[0])
        # Get the n elements with the smallest numbers
        n_nearest = sorted_distances[:n_nearest_animals]
        closest_animals = [other_animal for (_, other_animal) in n_nearest]
        return closest_animals

    def prey_nearby_sharks_observations(
        self, observer: Entity, all_sharks: Sequence[Entity], n_nearest_shark: int
    ) -> List[float]:
        """Get the observations of the current state of the environment of an observer."""
        observations = []
        if self.fov_enabled:
            for shark in all_sharks:
                if self.torus.check_if_entity_is_in_view_in_torus(
                    observer, shark, self.prey_view_distance, self.prey_fov
                ):
                    observation = self.nearby_animal_observation(observer, shark)
                    observations += observation
        else:
            closest_sharks = self.prey_get_n_closest_animals(
                observer, all_sharks, self.predator_observe_count
            )
            for shark in closest_sharks:
                observation = self.nearby_animal_observation(observer, shark)
                observations += observation
            # print(f'Fish Num diff: {fish_num - len(all_fishes)}')

        if len(observations) < n_nearest_shark * self.obs_size:
            observations += [0] * self.obs_size * (n_nearest_shark - len(observations))

        assert len(observations) == n_nearest_shark * self.obs_size
        # print(f'Shark observations: {len(observations)}')
        return observations

    def prey_nearby_fish_observations(
        self, observer: Entity, all_fishes: Sequence[Entity], n_nearest_fish: int
    ) -> List[float]:
        """Get the observations of the current state of the environment of an observer."""
        observations = []

        if self.fov_enabled:
            for fish in all_fishes:
                if (
                    self.torus.check_if_entity_is_in_view_in_torus(
                        observer, fish, self.prey_view_distance, self.prey_fov
                    )
                    and len(observations) < n_nearest_fish * self.obs_size
                ):
                    observation = self.nearby_animal_observation(observer, fish)
                    observations += observation
        else:
            closest_fish = self.prey_get_n_closest_animals(observer, all_fishes, n_nearest_fish)
            for fish in closest_fish:
                if fish is not observer:
                    observation = self.nearby_animal_observation(observer, fish)
                    observations += observation
                    # print(f'Fish Num diff: {fish_num - len(all_fishes)}')

        if len(observations) < (n_nearest_fish * self.obs_size):
            observations += [0] * (n_nearest_fish * self.obs_size - len(observations))

        assert len(observations) == n_nearest_fish * self.obs_size
        # print(f'Fish observations: {len(observations)}')
        return observations

    def get_prey_observations(self, observer: Entity):
        """Get the observations of the current state of the environment of an observer.
        The observations are a list of floats."""

        observed_observer = self.prey_observer_observation(observer)
        # print(f'Number of Observer Observations: {len(observed_observer)}')
        # observed_borders = border_observation(observer, aquarium, aquarium.observable_walls)
        # print(f'Number of Border Observations: {len(observed_borders)}')
        observed_sharks = self.prey_nearby_sharks_observations(
            observer, self.predators, self.predator_observe_count
        )

        observed_fishes = self.prey_nearby_fish_observations(
            observer, self.prey, self.prey_observe_count
        )
        # print(len(observed_observer), len(observed_sharks), len(observed_fishes))
        # print(f'Number of Shark Observations: {len(observed_sharks)}')
        observations = np.concatenate((observed_observer, observed_sharks, observed_fishes))
        # print(f'Total Number of Fish Observations: {len(observations)}')
        # print(f'Observations: {observations}')
        # print(len(observations), self.number_of_fish_observations)
        assert len(observations) == self.number_of_fish_observations
        return observations

    def prey_observe(self):
        """Returns the observations for all prey"""
        observations = {}
        for prey_id in range(self.prey_count):
            if len(self.prey) > prey_id:
                observations[self.prey[prey_id].id()] = self.get_prey_observations(
                    self.prey[prey_id]
                )
                # print(f'Fish {fish_id} observations: {observations[all_fishes[fish_id].id()]}')
            else:
                observations[f"fish_{prey_id}"] = np.zeros(self.number_of_fish_observations)
        # observations = {fish.id(): get_fish_observations(fish, self, SHARK_NUMBER) for fish in self.fishes}
        return observations

    def predator_observer_observation(
        self, observer: Entity, obs_min: float = 0, obs_max: float = 1
    ) -> Sequence[float]:
        """Get the observations of the current state of the environment of an observer."""
        position = observer.position
        direction = observer.orientation_angle
        speed = observer.velocity.mag()
        scaled_position_x = scale(position.x, 0, self.width, obs_min, obs_max)
        scaled_position_y = scale(position.y, 0, self.height, obs_min, obs_max)
        scaled_direction = scale(direction, -180, 180, obs_min, obs_max)
        scaled_speed = scale(speed, 0, observer.max_speed, obs_min, obs_max)
        observation = [0, scaled_position_x, scaled_position_y, scaled_direction, scaled_speed]

        assert all(
            obs_min <= value <= obs_max for value in observation
        ), "predator_observer_observation: All values must be between -1 and 1"
        # print(f'Observer_observation: {len(observation)}')
        return observation

    def predator_nearby_shark_observations(self, observer: Entity) -> Sequence[float]:
        """Get the observations of the current state of the environment of an observer."""
        observations = []
        if self.fov_enabled:
            for shark in self.predators:
                if (
                    self.torus.check_if_entity_is_in_view_in_torus(
                        observer, shark, self.prey_view_distance, self.prey_fov
                    )
                    and len(observations) < self.predator_observe_count * self.obs_size
                ):
                    observation = self.nearby_animal_observation(observer, shark)
                    observations += observation
        # print(f'Sharks : {len(all_sharks)}')
        for shark in self.predators:
            if shark is not observer:
                observation = self.nearby_animal_observation(observer, shark)
                observations += observation

        # print(f'Shark observations: {len(observations)}')

        if len(observations) < self.predator_observe_count * self.obs_size:
            observations += [0] * (self.predator_observe_count * self.obs_size - len(observations))

        return observations

    def predator_get_n_closest_fish(self, observer: Entity) -> Sequence[Entity]:
        """Get the n nearest animals to the observer."""
        distances = [
            (self.torus.get_distance_in_torus(observer.position, fish.position), fish)
            for fish in self.prey
        ]
        # Sort the distances list based on the first element of each tuple (the number)
        sorted_distances = sorted(distances, key=lambda x: x[0])
        # Get the n elements with the smallest numbers
        n_nearest = sorted_distances[: self.prey_observe_count]
        fishes = [fish for (_, fish) in n_nearest]

        return fishes

    def predator_nearby_fish_observations(self, observer: Entity) -> Sequence[float]:
        """Get the observations of the current state of the environment of an observer."""
        observations = []
        if self.fov_enabled:
            for fish in self.prey:
                if self.torus.check_if_entity_is_in_view_in_torus(
                    observer, fish, self.prey_view_distance, self.prey_fov
                ):
                    observation = self.nearby_animal_observation(observer, fish)
                    observations += observation
        else:
            closest_fish = self.predator_get_n_closest_fish(observer)
            for fish in closest_fish:
                if fish is not observer:
                    observation = self.nearby_animal_observation(observer, fish)
                    observations += observation
        # print(f'Fish Num diff: {fish_num - len(all_fishes)}')
        if len(observations) < self.prey_observe_count * self.obs_size:
            observations += [0] * (self.prey_observe_count * self.obs_size - len(observations))
        # print(f'Fish observations: {len(observations)}')
        return observations

    def get_predator_observations(self, observer: Entity):
        """Get the observations of the current state of the environment of an observer.
        The observations are a list of floats."""
        observed_observer = self.predator_observer_observation(observer)
        # observed_borders = border_observation(observer, aquarium, aquarium.observable_walls)
        observed_sharks = self.predator_nearby_shark_observations(observer)
        observed_fishes = self.predator_nearby_fish_observations(observer)

        observations = np.concatenate((observed_observer, observed_sharks, observed_fishes))
        # print(f'Total Number of Shark Observations: {len(observations)}')
        # print(f'Observations: {observations}')
        assert len(observations) == self.number_of_predator_observations
        return observations

    def predator_observe(self):
        """Returns the observations for all predators"""
        number_of_observations = self.number_of_predator_observations
        observations = {}
        for shark_id in range(self.predator_count):
            if len(self.predators) > shark_id:
                observations[self.predators[shark_id].id()] = self.get_predator_observations(
                    self.predators[shark_id]
                )
                # print(f'Fish {fish_id} observations: {observations[all_fishes[fish_id].id()]}')
            else:
                observations[f"shark_{shark_id}"] = np.zeros(number_of_observations)
        # if SHARK_MODEL == "single_agent_rl":
        #     observations = {shark.id(): get_shark_observations(shark, aquarium, FISH_NUMBER) for shark in aquarium.sharks}
        #     return observations
        # observations = {shark.id(): get_shark_observations(shark, aquarium, FISH_NUMBER) for shark in aquarium.sharks}
        return observations
