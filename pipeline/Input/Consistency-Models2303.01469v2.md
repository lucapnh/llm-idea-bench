# Title: Consistency Models

## Keywords

generative modeling; diffusion models; one-step sampling; model distillation; probability flow ODE; zero-shot editing

## TL;DR

The paper introduces **consistency models**—networks trained to map any noisy sample on a diffusion trajectory directly back to (near) data—enabling fast **one-step** generation while retaining a multi-step option and zero-shot editing. They outperform prior diffusion distillation methods in few-step regimes and set new one-step FID marks on CIFAR-10 and ImageNet-64.&#x20;

## Abstract

Diffusion models achieve state-of-the-art quality across modalities but require slow, iterative sampling. This work proposes **consistency models**, which learn a *self-consistent* mapping $f_\theta(x_t, t)$ that takes any point along a probability-flow ODE trajectory to its origin $x_\epsilon$. By construction, points from the same trajectory map to the same output, permitting **single-step** generation from Gaussian noise and, when desired, an inexpensive multi-step refinement that trades compute for quality. The authors present two training routes: (1) **Consistency Distillation (CD)**, which distills a pre-trained diffusion/score model by enforcing output agreement between adjacent ODE states generated with a numerical solver; and (2) **Consistency Training (CT)**, which trains in isolation using an unbiased score estimator, turning consistency models into a standalone family of generative models. Architectural parameterizations enforce the boundary condition $f_\theta(x,\epsilon)=x$ and leverage skip connections; practical sampling and multi-step schedules are detailed. Empirically, CD **surpasses progressive distillation** on CIFAR-10, ImageNet-64, and LSUN, with **one-step FID 3.55** (CIFAR-10) and **6.20** (ImageNet-64), and further gains with two steps (e.g., 2.93 and 4.70, respectively). CT, trained without a teacher, outperforms prior single-step, non-adversarial models and approaches distilled diffusion in quality. Beyond generation, the same models enable **zero-shot** denoising, inpainting, colorization, super-resolution, and stroke-guided editing via minor modifications to the multi-step procedure. Notably, a small caveat appears on LSUN Bedroom single-step where PD edges out CD, and GANs still lead among pure one-step methods on some datasets; nonetheless, consistency models offer a compelling speed-quality-flexibility sweet spot for modern generative modeling.      &#x20;
