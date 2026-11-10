# Why care?
Before we start, we will warn readers that there is an upcoming twist. So you have been warned. Don't blame us for feeling being tricked

Let's start from the latest, broadest practical view. Current SOTA in image generation is MDP + post-train with RLHF (e.g., [Qwen Image 2.0 technical reports](https://arxiv.org/pdf/2606.27608v1)). Previous to this, it was Flow Matching/Diffusion + post-train with distillation (e.g., see [recent work of consistency distillation](https://arxiv.org/abs/2510.17858)). Given the common senses that distilled model can only get close to the quality of pretrained model at a fraction of the compute cost; whereas RLHF can directly improve the quality of pretrained model. Thus, it's a natural conclusion that Flow Matching/Diffusion is no longer relevant after 2026? Not quite! If you look at the actual pipeline of MDP + post-train with RLHF (e.g., [ReinFlow](https://arxiv.org/abs/2505.22094); [Flow-GRPO](https://arxiv.org/abs/2505.05470); [Qwen Image 2.0](https://arxiv.org/abs/2605.10730)), you will notice a common procedure that has the following steps:
1. Take a pretrained Flow Matching/Diffusion to get some sampling trajectory (i.e., ODE/SDE solution). All sampling trajectory is discrete-time in practice.
2. Convert that sampling trajectory into MDP.
3. Apply RLHF on it.

***So here is the twist, the actual pipeline of MDP + RLHF is rather Flow Matching/Diffusion + distillation + MDP + RLHF. All components matter in this pipeline!*** Side note, once you understand this intuition. It's also natural to make to an educate guess on why [Flow + post-train with Fréchet distance](https://arxiv.org/abs/2604.28190) and [Flow + post-train with KL](https://arxiv.org/abs/2503.06674) could pull off the following feats:
- 1-sampling step post-trained model can outperform 50-sampling step post-trained model. This is directly shown in [Flow + post-train with Fréchet distance](https://arxiv.org/abs/2604.28190) paper.
- After distillation, continue post-training on FD loss could still make performance gain. We independently discover this when we were playing around with 3D generation.

In our opinion, we think that it is likely that FD/KL loss are basically a cheap version of RLHF. Like say if practitioners want to skip step (2) and (3) above, they can just straight up use pretrained Flow Matching with FD/KL loss.
