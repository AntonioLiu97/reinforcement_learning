from gym import register
from env.cartpole_control_env import CartPoleControlEnv
register(
    id='CartPoleControlEnv-v0',
    entry_point='env:CartPoleControlEnv'
)



register(
    id='NoisyCartPoleControlEnv-v0',
    entry_point='env:CartPoleControlEnv',
    kwargs = {"noisy": True}
)
