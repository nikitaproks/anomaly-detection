from gym.envs.registration import register

register(
    id='Anomaly-v4',
    entry_point='gym_env.env:CustomEnv',
    max_episode_steps=1000000,
)
