| Left panel                                                  | [IMDE paper](https://arxiv.org/abs/2206.07335) | How they are used during training, sampling                                                     |
| ----------------------------------------------------------- | -----------------------------------------------|-------------------------------------------------------------------------------------------------| 
| Number of sampling steps $`N`$, where $`t_k \Delta t = t`$  | S                                              | $`\frac1N = \Delta t = h`$                                                                      | 
| Start point $`X_0`$                                         | $x$                                            | $`X_0 := Z \sim \mathcal{N}(0, I_d)`$                                                           |
| Base Continuous trajectory, where $Sh = 1$                  | $\phi_{Sh,f}(x)$                               | $`X_t = \alpha_t X_0 + \beta_t X_1 \forall t \in [0,1] = X_0 + \int_0^t v(s, X_s) ds`$          |
| Base velocity field $v_B(t, x)$                             | $f$                                            | $`v_B(t, x) \approx \dot{X}_t`$                                                                 |
| Perturbed velocity field $v_P(t, x)$                        | $f_h$                                          |                                                                                                 |
| Base Euler sampling trajectory $`\hat{X}_{t_k}^{B, Euler}`$ | $(\Phi_{h,f})^S(x)$                            | $`\hat{X}_{t_k}^{B, Euler} = X_0 + \Delta t \sum_{k=0}^{N-1} v_B(t_k, x)`$                      |
| Perturbed Euler $`\hat{X}_{t_k}^{P, Euler}`$                | $(\Phi_{h,f_h})^S(x)$                          | $`\hat{X}_{t_k}^{P, Euler} = X_0 + \Delta t \sum_{k=0}^{N-1} v_P(t_k, x)`$                      |
| End point discretisation error of Base model $`e_B^{disc}`$ | $(\Phi_{h,f})^S(x) - \phi_{Sh,f}(x)$           | $`e_B^{disc} = \hat{X}_1^{Euler, B} - X_1`$                                                     |
| End point discretisation error counteraction $d_N$          | $(\Phi_{h,f_h})^S(x) - (\Phi_{h,f})^S(x)$      | $`d_N = \hat{X}_{t_N}^{Euler, P} - \hat{X}_{t_N}^{Euler, B}`$                                   |

# Goal
For each $`v_B(t, X_t)`$, we want to know its corresponding $`v_P(t, X_t^P)`$. Because once we have access to $`v_P(t, X_t^P)`$, we can just regress toward it during post-training like how Flow Matching framework regresses toward $`v_B(t, X_t)`$.

Ideally, we want Perturbed Euler to match Base Continuous at every sampling time step $`t_k`$. Take left panel for example, where $`N = 4`$, this means that $`X_{0.25} = \hat{X}_{t_1}^{B, Euler}`$, $`X_{0.5} = \hat{X}_{t_2}^{B, Euler}`$, and so on. Therefore, we can compute $`v_P(0.25, X_{0.25}) = \frac{X_{0.50} - X_{0.25}}{0.25}`$. Same rule applies for $`v_P(0, X_0)`$, $`v_P(0.5, X_{0.5})`$, and so on. Essentially, the formula is that $`v_P(t, X_t) = \frac{X_{t + \Delta t} - X_t}{\Delta t}`$. But $`v_P(t, X_t)`$ is not yet $`v_P(t, X_t^P)`$.


# Compromised goal
So what if we don't have access to $`v_P(t, X_t^P)`$? After all, $`v_P(t, X_t^P)`$ and $`v_P(t, X_t)`$ belong to the same Perturbed velocity field $`v_P(t,x)`$. Furthermore, given that we intent to treat post-training Flow Matching as another Flow Matching problem, but we regress toward $`v_P(t, x)`$ instead of $`v_B(t, x)`$. This means that we can evoke Lipschitz regularity assumption on $v_P(t, x)$ the same way Lipschitz regularity assumption had been invoked for $v_B(t, x)$. 

Formally, consider Lipschitz regularity assumption on $v_B(t, X_t)$: $`\left\| v_B(t, X_t^P) - v_B(t, X_t) \right\| \le L(t) \left\| X_t^P - X_t \right\|`$. Note, if you feel confused about this statement, just read it as $`\left\| v_B(t, x + c) - v_B(t, x) \right\| \le L(t) \left\| (x + c) - x \right\|`$ with $`x \in X_t`$ and $`(x + c) \in X_t^P`$; then when I add $`c \to 0`$, this statement is now the 1st principle of differentiation on $`v_B(t, x)`$. In other words, as $`c \to 0`$, $`L(t)`$ literally becomes the slope of $`v_B(t, x)`$. Anyway, section 3.2 in [Benton paper](https://openreview.net/forum?id=uqQPyWFDhY) shown that this Lipschitz regularity assumption is generally true. Now, given [one-sided linear interpolant](https://arxiv.org/abs/2303.08797) $`X_t = \alpha(t)X_0 + \beta(t)X_1, X_0 := Z \sim \mathcal{N}(0, I_d)`$, suppose we use a different choice of interpolant schedules $`\alpha(t)`$ and $`\beta(t)`$ such that $`v_P(t,x) = \mathbb{E}\left[\dot{X}_t \mid X_t = x \right]`$. For the same $`\left\{X_0,X_1\right\}`$, the difference between $`v_P(t,x)`$ and $`v_B(t,x)`$ is just the choice of interpolant schedules. So there's no reason that section 3.2 in Benton paper cannot also apply to $`v_P(t,x)`$ to prove that Lipschitz regularity assumption on $`v_P(t,x)`$ is also generally true. It's just that for the same time $`t`$, $`v_P(t,x)`$ will have a different value for its Lipschitz constant, i.e., $`\left\| v_P(t, X_t^P) - v_B(t, X_t) \right\| \le L(t)_P \left\| X_t^P - X_t \right\|`$.

From Lipschitz regularity assumption on $`v_P(t,x)`$ and the notion that $`v_P(t, X_t^P)`$ and $`v_P(t, X_t)`$ both belong to $`v_P(t,x)`$, we conjecture that if provided sufficient amount of regression signal $`v_P(t, X_t)`$, the neural network can learn to generalise to $`v_P(t, X_t^P)`$. It's the same as saying "provided sufficient amount of regression signal $`v_B(t, X_t)`$, the neural network can learn to generalise to $`v_B(t, X_t^P)`$. Therefore, it is entirely valid to say that it doesn't make a difference whether the neural network regresses toward $`v_P(t, X_t)`$ or $`v_P(t, X_t^P)`$ during post-training. And given that we actually have access to $`v_P(t, X_t)`$, why not pursue it instead.


# Loss function for Perturbed velocity field


# Sampling algorithm for Perturbed velocity field


# What happens if we use different integration solver
