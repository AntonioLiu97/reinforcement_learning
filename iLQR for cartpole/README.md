# 2023 Spring CS4789 Programming Assignment 2: CartPole Control with LQR Method
## Modified by Toni Liu to add iterative LQR method, 2023 March 4th 

New methods:
lqr.py: ilqr(ABmQRMqrb_list)

cartpole_controller.py: 
  policy_mixture(l1, l2, alpha=1/2)
  linearized_traject(self, s_ini, policy_list, delta = 1e-7)
  compute_global_policy(self, s_ini, T, N)

cartpole.py:
  new option: if flag == 'iLQR'

This repository contains files that may help you get started with the programming assignments 2.
There are TODOs in the files `finite_difference_method.py`, `lqr.py` and `cartpole_controller.py`.
Please refer to the docstring in these files for more details.

## Usage

* run and show the cartpole environment. (also generate a video under directory ./gym-results)
```
  $ python cartpole.py
```
* test different initial states
```
  $ python test.py
```
