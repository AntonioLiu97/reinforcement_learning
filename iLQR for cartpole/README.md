## 2023 Spring CS4789 Programming Assignment 2: CartPole Control with LQR Method
## Refactored by Toni Liu, 2023 March 4th 

Apart from LQR from linear control, this package now supports iterative LQR for highly non-linear control tasks such as the swing-up problem.

New methods and functions:
lqr.py: ilqr(ABmQRMqrb_list)

cartpole_controller.py: 
  policy_mixture(l1, l2, alpha=1/2) <br>
  linearized_traject(self, s_ini, policy_list, delta = 1e-7)
  compute_global_policy(self, s_ini, T, N)

cartpole.py:
  new option: flag == 'iLQR'


## Usage

* run and show the cartpole environment. (also generate a video under directory ./gym-results)
```
  $ python cartpole.py
```
* test different initial states
```
  $ python test.py
```
