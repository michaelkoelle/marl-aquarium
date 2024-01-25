from pettingzoo.test import api_test, parallel_api_test

from aquarium.env.aquarium import env, parallel_env

parallel_api_test(parallel_env(), num_cycles=1_000_000)
api_test(env(), num_cycles=1_000_000, verbose_progress=True)
