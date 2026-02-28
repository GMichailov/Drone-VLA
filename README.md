# Drone-VLA

## Pre-Development Ideas

From Nemotron 3 Paper Ideas (https://arxiv.org/pdf/2512.20856)
- Train in simulated NVFP4 for super efficient throughput and quant later on.
- Latent MoE to shrink tokens to latents before being routed, processed by experts, and re-expanded.
  - Can downscale d_model by roughly 4x to d_latent. 
  - With same param budget, this allows for more experts and a larger topk routing. 
- Multi-Token Prediction Layers
  - Typically predict K tokens where K = 2-4.
  - Each one has its own output head for each prediction index.
  - Used during pre-train and RL, not really SFT because want model to focus on that.
  - Loss is aggregated cross-entropy loss of all scaled by hyperparam lambda_k for that specific output head.
  - Loss at first output token is worse than loss at fourth output token, so scale each loss with lambda_k where the earlier ones are larger ie lambda_k = {1.0, 0.5, 0.25, 0.125}
- Use Mamba2 in Lieu of some attention layers.
    - Nemotron 3 Nano 30B-A3B has 5x (Mamba2, MoE, Mamba2, MoE, Attn, MoE), 3x(Mamba2, MoE), 1x(Mamba2, Attn, MoE), 4x (Mamba2, MoE)
    - Significantly lower memory and compute requirements.
    - Mamba Layers kept in MXFP8 rather quantizing any further.
 
From MoE Router coupling via auxiliary loss method (https://arxiv.org/pdf/2512.23447)
- Add the ERC auxiliary loss to make sure router most efficiently routes to experts and that experts are truly experts in this task.
  - Router embedding gets passed through each expert with goal of each router embedding having the highest activation score for its own expert and lower for those it is not related to. This enforced true expert specilization and less arbitrary routing aka higher accuracy which is crucial with a small model.

 
My Ideas
- Train LM that is lemmized and doesn't use words like the or a unless necessary.
- Create a framework for mapping somehow.
- I assume SMEM will be an even bigger bottleneck relative to HBM and compute will be signifigicantly lower so:
  - If compute bottleneck: Reduce d_model and reduce input sequence length. Use GQA and vLLM. Reduce FFN expansion from 4x to 2x. Sparse Attention.
  - If SMEM bottleneck: Reduce tile size in kernels and reduce head dimensions.  
