| Left panel                                                  | [IMDE paper](https://arxiv.org/abs/2206.07335) | How they are used during training, sampling                                                     |
| ----------------------------------------------------------- | -----------------------------------------------|-------------------------------------------------------------------------------------------------| 
| Number of sampling steps $`N`$                              | S                                              | $`\frac1N = \Delta t = h`$                                                                      | 
| Start point $`X_0`$                                         | $x$                                            | $`X_0 := Z \sim \mathcal{N}(0, I_d)`$                                                           |
| Base Continuous trajectory                                  | $\phi_{Sh,f}(x)$. $Sh = 1$                     | $`X_t = \alpha_t X_0 + \beta_t X_1 \forall t \in [0,1] = X_0 + \int_0^t v(s, X_s) ds`$          |
| Base velocity field $v_B(t, x)$                             | $f$                                            | $`v_B(t, x) \approx \dot{X}_t`$                                                                 |
| Perturbed velocity field $v_P(t, x)$                        | $f_h$                                          |                                                                                                 |
| Base Euler sampling trajectory $`\hat{X}_{t_k}^{B, Euler}`$ | $(\Phi_{h,f})^S(x)$                            | $`\hat{X}_{t_k}^{B, Euler} = X_0 + \Delta t \sum_{k=0}^{N-1} v_B(t_k, x)`$. $`t_k \Delta t = t`$|
| Perturbed Euler $`\hat{X}_{t_k}^{P, Euler}`$                | $(\Phi_{h,f_h})^S(x)$                          | $`\hat{X}_{t_k}^{P, Euler} = X_0 + \Delta t \sum_{k=0}^{N-1} v_P(t_k, x)`$.                     |
| End point discretisation error of Base model $`e_B^{disc}`$ | $(\Phi_{h,f})^S(x) - \phi_{Sh,f}(x)$           | $`e_B^{disc} = \hat{X}_1^{Euler, B} - X_1`$                                                     |
| End point discretisation error counteraction $d_N$          | $(\Phi_{h,f_h})^S(x) - (\Phi_{h,f})^S(x)$      | $`d_N = \hat{X}_{t_N}^{Euler, P} - \hat{X}_{t_N}^{Euler, B}`$                                   |

# Goal
For each $`v_B(t, X_t)`$, we want to know its corresponding $`v_P(t, X_t^P)`.

Ideally, we want Perturbed Euler to match Base Continuous at every sampling time step $`t_k`$. Take left panel for example, where $`N = 4`$, this means that $`X_{0.25} = \hat{X}_{t_1}^{B, Euler}`$, $`X_{0.5} = \hat{X}_{t_2}^{B, Euler}`$, and so on. Therefore, we can compute $`v_P(0.25, X_{0.25}) = \frac{X_{0.50} - X_{0.25}}{0.25}`$. Same rule applies for $`v_P(0, X_0)`$, $`v_P(0.5, X_{0.5})`$, and so on. Essentially, the formula is that $`v_P(t, X_t) = \frac{X_{t + \Delta t} - X_t}}{\Delta t}`$.

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

At this point, it is clear that IMDE theorem 3.2 with Gronwall's inequality won't have us derive $`v_P(t, X_t^P)`$ exactly, not even if we add-on Alekseev-Grobner formula from [Benton paper](https://openreview.net/forum?id=uqQPyWFDhY). So what is the remedy?

# Approximate Perturbed velocity field at small step size
In [Flow Matching recap](https://github.com/nam-drun/Flow_plus_Distribution-Matching-loss/blob/main/README.md#flow-matching-recap), we said that as number of sampling steps $`N \to \infty`$, $`\sum_{k=0}^{N-1} v(t, X_{t_k}) \approx \int_0^1 v(t, X_t) dt`$. In other words, as number of size step approaches 0, Base Euler becomes from approximately identical to Base Continuous $`\to`$ literally identical to Base Continuous. IMDE paper has similar expressions, too:
```math
\begin{align*}
& \nu_i = x + h \sum_{j=1}^I a_{ij} f(\nu_j), \\
& \Phi_h(x) = x + h \sum_{i=1}^I b_i f(\nu_i)
\end{align*}
```

This expression very much only concerns 1 sampling step segment from $`\hat{X}_{t_k}^{Solver, B} \to \hat{X}_{t_{k+1}}^{Solver, B}`$ for any Runge-Kutta method. Note, different methods lead to different $`p^{th}`$ solvers which has different NFE. Knowing that $`x := X_0`$ in IMDE paper (not in Flow Matching), this expression describes about the 1st sampling step. Now, if you specialise this expression for Euler solver (1st order Runge-Kutta), $`I = 1`$, $`a_{11} = 0`$, $`b_1 = 1`$. So this expression would become:
```math
\begin{align*}
& \nu_1 = x + h f(\nu_1), \\
& \Phi_{h,f}(x) = x + h b_1 f(\nu_1) \implies \Phi_{h,f}(x) = x + h f(x)
\end{align*}
```

Here, $`\Phi_{h,f}(x) = x + h f(x)`$ literally reads as *next position = current position + time interval x velocity*. The added-on subscript $`f`$ is just to distinguish between $`\Phi_{h,f}(x) = x + h f(x)`$ vs $`\Phi_{h,f_h}(x) = x + h f_h(x)`$ (i.e., Base Euler vs Perturbed Euler). Now, IMDE paper has this equality: 
```math
\left( \Phi_{h, f_h}\right)^S(x) = (\phi_{h,f})^S(x) = \phi_{Sh,f}(x)
```

Here, $`(\phi_{h,f})^S(x) = \phi_{Sh,f}(x)`$ means that you have the whole Base Continuous trajectory $`\phi_{Sh,f}(x)`$ (i.e., conditional probability path) and you chop up that trajectory into $`S`$ segments, hence $`(\phi_{h,f})^S(x)`$. For example, for the left panel, $`S = 4`$ so just you chop up Base Continuous into 4 segments. That Base Continuous is still the same trajectory, just that now it has 4 segments. Given that there are 4 segments, there are now 5 nodes (including $`X_0`$), which you can see in the left panel. Now, $`\left( \Phi_{h, f_h}\right)^S(x) = (\phi_{h,f})^S(x)`$ simply means that every node of Perturbed Euler matches with every node of Base Continuous. For example in the left panel, $`\hat{X}_{0.25}^{Euler, B} = X_{0.25}`$; same rule applies for $`X_{0.50}`$, $`X_{0.75}`$ and $`X_1`$. Note, in Flow Matching, the final time (on paper) is always $`t = 1`$. Thus, $`Sh = 1`$ so if `S` changes, then `h` also needs to change. Same rule still applies even if $`t = 0.97`$ (or any other value in practice). So again, the $`\left( \Phi_{h, f_h}\right)^S(x) = (\phi_{h,f})^S(x)`$ equality is not a fact, but rather what we desire to happen!

Now, we are interested in what happens going from *current position* to *next position* in Perturbed Euler. In other words, we are only interested in $`\Phi^{t \to t + h}_{h, f_h}(X_t)`$. Therefore, the $`\left( \Phi_{h, f_h}\right)^S(x) = (\phi_{h,f})^S(x)`$ equality becomes:
```math
\begin{align*}
& \Phi^{t \to t + h}_{h, f_h}(X_t) = \phi^{t \to t + h}_{h,f}(X_t) \\
\end{align*}
```

And $`\Phi_{h,f}(x) = x + h f(x)`$ becomes:
```math
\Phi^{t \to t + h}_{h, f_h}(x) = X_t + h f_h(X_t)
```


Earlier in [Flow Matching recap](https://github.com/nam-drun/Flow_plus_Distribution-Matching-loss/blob/main/README.md#flow-matching-recap), we had established that as $`\Delta t \to 0`$, $`\sum_{k=0}^{N-1} v(t, X_{t_k}) \to \int_0^1 v(s, X_s) ds`$.  
