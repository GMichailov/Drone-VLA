# Drone-VLA

## Paper Ideas

### Nemotron 3
(https://arxiv.org/pdf/2512.20856)
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

 ### MoE Router coupling via auxiliary loss method
(https://arxiv.org/pdf/2512.23447)
- Add the ERC auxiliary loss to make sure router most efficiently routes to experts and that experts are truly experts in this task.
  - Router embedding gets passed through each expert with goal of each router embedding having the highest activation score for its own expert and lower for those it is not related to. This enforced true expert specilization and less arbitrary routing aka higher accuracy which is crucial with a small model.

### Deepseek V3
- Use MHLA + GQA with two KV heads. This seems to have become standard in SOTA OSS models such as Qwen3+, Deepseek, and Kimi. There is no need for the newest Deepseek Sparse Attention since this model will not need to handle super long sequences.
- Can potentially use a teacher model to generate training data for distillation like the R1 distills as well. Probably the best option by far.

### Qwen 3.5
- Use Gated Delta Net layers like they did rather than the Mamba2 proposed by 

### Kimi K2
(https://arxiv.org/pdf/2507.20534)
- Train vision encoder and LLM backbone right away from pretraining.
- MoonClip optimizer with QK clipping.


### Physics of LMs: Architecture Design and the Magic of Canon Layers

- Use Conv 1-D in small models to give more token mixing. These can be applied before attention and before the MLP part.
  - Attention is temporary token mixing and MLP is individualistic. This isn't a problem in large LMs since they have many layers but for LMs with a smaller number of layers, less heads, a smaller d_model, this can have disasterous effects. This is actually why at a small scale, GDNs are so effective.
 
### Exploration vs. Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward
- Entropy minimization to make model more confident in decisions.
  - For RLVR, it has recent been found that forcing the model to be more comittal in its exploration is better than shallow exploration for finding the result.
- Spurious Rewards gives the model rewards for exploration (opposite to above).
  - In RLVR, even though there is a ground truth answer, the model still gets a reward for a longer answer exploring different ideas if certain parts are correct.  
 
My Ideas
- Train LM that is lemmized and doesn't use words like the or a unless necessary.
- Create a framework for mapping somehow.
- I assume SMEM will be an even bigger bottleneck relative to HBM and compute will be signifigicantly lower so:
  - If compute bottleneck: Reduce d_model and reduce input sequence length. Use GQA and vLLM. Reduce FFN expansion from 4x to 2x. Sparse Attention.
  - If SMEM bottleneck: Reduce tile size in kernels and reduce head dimensions.


## Final Design

### Architecture

- Smaller vocab size to save memory budget for params. 2-3 letters/punctuation, no crazy symbols/emojis/non-english etc.
- Use RoPE for pos encoding. No need for RoPE + NoPE since model does not need to handle long sequences.
- Use standard causal attention mask for text and bidirectional attention for image patches/embedding(s).
- Use MHLA with two KV heads to be KV-Cache compatible. (Deepseek)
- 1-2B total params with 50-125M active params MoE.
- Run experiments with number of active experts, but generally larger overall number seems to perform better (Kimi).
- Gated Delta Net layers. (Nemotron, Qwen3+)
- Potential Option: Add Conv layers for additional token mixing. (Canon Layers)
- Small vision encoder (pre-trained, but will need to be compatible). (Kimi)
- Multi-Token Prediction (2-4 depending on perf from experiments) (Nemotron, Qwen3+)

### Training

#### Misc

- Weighted MTP CE loss + Auxiliary Expert Router Coupling loss (Nemotron, ERC)
- Use MoonClip optimizer (showed exceptionally high stability) combined with QK clipping. Fall back plan is Lion (bnb) or AdamW.
- Precision: Train in different precisions. Since model is so small, can probably go down to FP8 for certain layers, but FP32 for RMSNorm has massively helped stability (Anecdotal).


#### Pre-Training

- Use image and text datasets.
- Massive pretraining base (100B tokens absolute minimum).
- Find datasets listed in recent OSS papers to pull high quality data from there and generate/paraphrase with a powerful LLM instead of running epochs.
- lemmization to save prediction time?

#### SFT

- Model needs to learn drone specific behavior here (simulate). Rather than teaching model to give continuous outputs, give commands instead for simplicity.
- Treat this with the same logic as agentic tool calling training in LLMs.

#### RL

#### RL with Verifiable Rewards
- So much of drone behavior management can be done through verifiable rewards (getting Point A to B) or avoiding obstacles, etc.

#### GRPO

- Token and global level loss to avoid long-winded answers to try to reward hack.


