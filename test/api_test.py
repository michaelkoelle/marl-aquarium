"""Test the Aquarium environment with a random agent."""
from pettingzoo.test import api_test, parallel_api_test

from marl_aquarium.aquarium_v0 import env, parallel_env

parallel_api_test(parallel_env(), num_cycles=1_000_000)
api_test(env(), num_cycles=1_000_000, verbose_progress=True)
