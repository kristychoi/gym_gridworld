from gym.envs.registration import register

register(
    id='gym_gridworld-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
    timestep_limit=1000,
    nondeterministic = True,
)

register(
    id='gym_onehotgrid-v0',
    entry_point='gym_gridworld.envs:OneHotGridWorldEnv',
    timestep_limit=1000,
    nondeterministic = True,
)