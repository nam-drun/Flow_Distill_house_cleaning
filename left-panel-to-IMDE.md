| Left panel                            | [IMDE paper](https://arxiv.org/abs/2206.07335) | Meaning                                           | How to access them during training/sampling                        |
| ------------------------------------  | -----------------------------------------------| --------------------------------------------------|------------------------------------------------------------------- | 
| N                                     | S                                              | number of sampling steps. $h$ is time interval    | $\frac1N = \Delta t = h$                                           | 
| $X_0$                                 | $x$                                            | start point                                       | $X_0 \sim \mathcal{N}(0, I_d)$                                     |
| $X_1$                                 | $\phi_{Sh,f}(x)$                               | Base Continuous end point. $Sh$ is time interval  | $X_0 + \int_0^1 v(s, X_s) ds$. $Sh$ is time interval               |
| $v_B(t, x)$                           | $f$                                            | Base velocity field                               | $`v_B(t, x) \approx \dot{X}_t`$                                    |
| $v_P(t, x)$                           | $f_h$                                          | Perturbed velocity field                          | $`v_P(t, x) = v_B(t, x) + \Delta \theta`$. $\theta$ is NN weight   |
| $`\hat{X}_1^{Euler, B}`$              | $(\Phi_{h,f})^S(x)$                            | Base Euler end point.                             | $`\hat{X}_1^{Euler, B} = \Delta t \sum_{k=0}^{N-1} v_B(t_k, x)`$   |
| $`\hat{X}_1^{Euler, P}`$              | $(\Phi_{h,f_h})^S(x)$                          | Perturbed Euler end point                         | $`\hat{X}_1^{Euler, P} = \Delta t \sum_{k=0}^{N-1} v_P(t_k, x)`$   |
| $`e_B^{disc}`$                        | $(\Phi_{h,f})^S(x) - \phi_{Sh,f}(x)$           | discretisation error at $t=1$ of Base model       | $`e_B^{disc} = \hat{X}_1^{Euler, B} - X_1`$                        |
| $`d_N`$                               | $(\Phi_{h,f_h})^S(x) - (\Phi_{h,f})^S(x)$      | discretisation error counteraction at $t=1$       | $`d_N = \hat{X}_1^{Euler, P} - \hat{X}_1^{Euler, B}`$              |

# What to get out from IMDE paper
We want to quantify the relationship between $`v_P(t, X_t^P)`$ and $`v_B(t, X_t)`$. So that we can verify on whether distribution matching loss makes learned velocity field getting closer to $`v_P(t, X_t^P)`$. Here, the left panel also displays our challenge: We don't really know Perturbed Continuous trajectory $`X_t^P`$ really is, so we can't derive $`v_P(t, X_t^P)`$ like how Flow Matching typically does it. Fortunately, theorem 3.2 in IMDE paper can help us quantify $`v_P(t, X_t^P) - v_B(t, X_t)`$, so that if we know $`v_B(t, X_t)`$, then we can derive $`v_B(t, X_t)`$.


# IMDE theorem 3.2 + Gronwall's inequality (R.I.P)
Now, the first expression in theorem 3.2 is:
```math
\Phi_{h,f}(x) = \phi_{h,f}(x) + h^{p+1} \delta_f(x) + \mathcal{O}(h^{p+2})
```

Which is backed by Figure 2 in the paper. This expression implicates that $`v_P(t, X_t^P)`$ is affected by our intent to leverage discretisation error via $-\delta f$. Therefore, the power series expansion of $`v_P(t, X_t^P)`$ now becomes:
```math
f_h - f = -h^p \delta f  + \mathcal{O}(h^{p+1})
```

Rewrite the LHS in "left panel"'s notation and write $`\mathcal{O}(h^{p+1})`$ in full form:
```math
\begin{align*}
& v_P(t,x) - v_B(t,x) = -h^p \delta_{v_B}(t,x) + h^{p+1} f_{p+1} + h^{p+2} f_{p+2} + \cdots \\
& v_P(t,x) - v_B(t,x) = -h^p \delta_{v_B}(t,x) + h^{p+1}r_h(t,x) \\
\end{align*}
```

where $`r_h(t,x) = f_{p+1} + h f_{p+2} + h^2 f_{p+3} + \cdots`$. Although this is not yet $`v_P(t, X_t^P) - v_B(t, X_t)`$, but it nonetheless enables us to delegate the unknown to a simpler expression
```math
\begin{align*}
& v_P(t, X_t^P) - v_B(t, X_t) = v_P(t, X_t^P) - v_P(t, X_t) + v_P(t, X_t) - v_B(t, X_t) \\
& v_P(t, X_t^P) - v_B(t, X_t) = v_P(t, X_t^P) - v_P(t, X_t) -h^p \delta_{v_B}(t,x) + h^{p+1}r_h(t,x)
\end{align*}
```

Here, we invoke the Lipschitz regularity assumption on $v_B(t, X_t)$: $`\left\| v_B(t, X_t^P) - v_B(t, X_t) \right\| \le L(t) \left\| X_t^P - X_t \right\|`$. Note, if you feel confused about this statement, just read it as $`\left\| v_B(t, x + c) - v_B(t, x) \right\| \le L(t) \left\| (x + c) - x \right\|`$ with $`x \in X_t`$ and $`(x + c) \in X_t^P`$; then when I add $`c \to 0`$, this statement now reads like 1st principle of differentiation on $`v_B(t, x)`$. In other words, as $`c \to 0`$, $`L(t)`$ quite literally becomes the slope of $`v_B(t, x)`$. Anyway, this assumption generally holds true in Flow Matching, as shown by section 3.2 in [Error Bounds for Flow Matching Methods paper](https://openreview.net/forum?id=uqQPyWFDhY). 

Now we introduce another Lipschitz regularity assumption on $v_P(t,x)$: $`\left\| v_P(t, X_t^P) - v_P(t, X_t) \right\| \le L(t)_P \left\| X_t^P - X_t \right\|`$. Because this assumption comes out of nowhere so let's try to justify it: The core tenet of Flow Matching is that the ODE continuous trajectory $`X_t`$ needs to transport $X_0 \to X_1$. Whereas the core tenet of IMDE paper is that each continuous trajectory $`X_t`$ is explicitly linked to a corresponding perturbed continuous trajectory $`X_t^P`$. From initial experiments with post-trained Flow model, we observe that when the number of sampling steps during pretraining and post-training is $`[1, 4]`$, we observe that the post-trained models consistently have less discretisation compared to pretrained models. This is especially true for Euler solver. Thus, in term of Frechet distance and Wasserstein-2 distance, post-trained $P_1$ is closer to data distribution than pretrained $`P_1`$. At the very least, that seems to suggest that for every Base Continuous, there exists a set of corresponding Perturbed Continuous trajectories somewhere in $`\mathbb{R}^d`$ such that its Perturbed Euler's endpoint is closer to ideal $X_1$ than Base Euler's endpoint. 

With this Lipschitz regularity assumption on $v_P(t,x)$, we can bound $`v_P(t, X_t^P) - v_B(t, X_t)`$ as follows:
```math
\left\| v_P(t, X_t^P) - v_B(t, X_t) \right\| \le L(t)_P \left\| X_t^P - X_t \right\| + \left\| h^p \delta_{v_B}(t,x) + h^{p+1}r_h(t,x) \right\|
```

Now, we just need to figure out $`\left\| X_t^P - X_t \right\|`$ with Gronwall's inequality. Vanilla Gronwall's inequality has this basic structure:
```math
\begin{align*}
& \text{Given } \dot{f}(t) \le C(t)f(t) \Leftrightarrow f(t) \le f(0) + \int_0^t C(s)f(s) ds, \\
& \text{thus } f(t) \le f(0)\exp \left(\int_0^t C(s)ds \right) \\
\end{align*}
```
where $`C(s)`$ and $`f(t)`$ are some real-valued functions. But to deal with $`\left\| h^p \delta_{v_B}(t,x) + h^{p+1}r_h(t,x) \right\|`$, we'll use the extended version of Gronwall's inequality:
```math
\begin{align*}
& \text{Given } \dot{f}(t) \le C(t)f(t) + g(t) \Leftrightarrow f(t) \le f(0) + \int_0^t C(s)f(s) ds + \int_0^t g(s) ds, \\
& \text{thus } f(t) \le f(0) \exp \left(\int_0^t C(s)ds \right) + \int_0^t \exp \left(\int_0^t C(s)ds \right) g(s) ds \\
\end{align*}
```

Denote $`f(t) := \left\| X_t^P - X_t \right\|`$, thus $`\dot{f}(t) = \left\| v_P(t, X_t^P) - v_B(t, X_t) \right\|`$ and $`C(t) := L_p(t)`$. For $`f(0)`$, we'll set it $`f(0) := 0`$ since Perturbed Continuous and Base Continuous starts from the same $`X_0`$. This means that the remaining term to be denoted is $`g(t) := \left\| h^p \delta_{v_B}(t,x) + h^{p+1}r_h(t,x) \right\|`$. And with that, we now have access to the prerequisite condition of extended Gronwall's inequality. Applying extended Gronwall's inequality, we get:
```math
\begin{align*}
& \left\| X_t^P - X_t \right\| \le h^p \int_0^t \exp \left(\int_s^t L_P(r) dr \right) g(s)ds
\end{align*}
```

Plug $`\left\| X_t^P - X_t \right\|`$ back to the RHS of $`\left\| v_P(t, X_t^P) - v_B(t, X_t) \right\|`$ above, we get:
```math
\left\| v_P(t, X_t^P) - v_B(t, X_t) \right\| \le h^p \left[g(t) + L_p(t) \int_0^t \exp \left(\int_s^t L_P(r) dr \right) g(s)ds \right]
```

At this point, it is clear that IMDE paper with Gronwall's inequality won't have us deriving $`v_P(t, X_t^P)`$ exactly. So what is the remedy?

# Approximating Perturbed velocity field at small step size
In [Flow Matching recap](https://github.com/nam-drun/Flow_plus_Distribution-Matching-loss/blob/main/README.md#flow-matching-recap), we said that as number of sampling steps $`N \to \infty`$, $`\sum_{k=0}^{N-1} v(t, X_{t_k}) \approx \int_0^1 v(s, X_s) ds`$. In other words, as number of size step approaches 0, Base Euler becomes from approximately identical to Base Continuous $`\to`$ literally identical to Base Continuous. IMDE paper has similar expressions, too:
```math
\begin{align*}
& v_i = x + h \sum_{j=1}^I a_{ij} f(v_j), \\
& \Phi_h(x) = x + h \sum_{i=1}^I b_i f(v_i)
\end{align*}
```
