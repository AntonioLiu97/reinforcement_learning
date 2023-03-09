import gym
import numpy as np
from cartpole_controller import LocalLinearizationController, PIDController
from gym import wrappers

    
video_path = "./gym-results"
theta_ini = np.pi+0.1
theta_star = 0

# s_init = np.array([0, 0, 0, 0])
s_init = np.array([0, 0, theta_ini, 0])
s_star = np.array([0, 0, theta_star, 0], dtype=np.double)
a_star = np.array([0], dtype=np.double)
T = 500
T = 200
N = 50

# flag = 'LQR'
flag = 'iLQR'

if flag == 'LQR':
    
    env = gym.make("env:CartPoleControlEnv-v0")
    # a modified env that penalize high velocity
    controller = LocalLinearizationController(env)
    policies = controller.compute_local_policy(s_star, a_star, T)
    # print(policies[0])
    
if flag == 'iLQR':
    env = gym.make("env:CartPoleControlEnv-v0")
    controller = LocalLinearizationController(env)
    policies = controller.compute_global_policy(s_init, T, N)   
    # print(policies[0])
    # print(f"policies {policies}")
    
if flag == 'PID':
    controller = PIDController(1, 0., 0.)
    mask = np.array([0,0,1,0]) # PID only tracks angle error
    setpoint = 0


# For testing, we use a noisy environment which adds small Gaussian noise to
# state transition. Your controller only need to consider the env without noise.
# env = gym.make("env:NoisyCartPoleControlEnv-v0")
env = gym.make("env:CartPoleControlEnv-v0")

env = wrappers.Monitor(env, video_path, force = True)
total_cost = 0
observation = env.reset(state = s_init)

for t in range(T):
    env.render()
    if flag == 'LQR' or 'iLQR' :
        (K,k) = policies[t]
        action = np.ravel((K @ observation + k))
        # print(action.shape)
    else:
        err = np.dot(observation, mask) - setpoint
        action = controller.get_action(err).reshape(1,)
    observation, cost, done, info = env.step(action)
    total_cost += cost
    if done: # When the state is out of the range, the cost is set to inf and done is set to True
        break
env.close()
print("cost = ", total_cost)
