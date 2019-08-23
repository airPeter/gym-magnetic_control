from gym.envs.registration import register
register(
        id='mag-v0',
        entry_point='gym_mag.envs:MagControlEnv'
)
