"""Example of how to use the Aquarium environment."""

from marl_aquarium import aquarium_v0

env = aquarium_v0.env(
    # draw_force_vectors=True,
    # draw_action_vectors=True,
    # draw_view_cones=True,
    # draw_hit_boxes=True,
    # draw_death_circles=True,
)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
    env.render()
env.close()
