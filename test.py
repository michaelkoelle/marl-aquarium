from pettingzoo.test import parallel_api_test

from env.aquarium import Aquarium

env = Aquarium()
parallel_api_test(env, num_cycles=1_000_000)
