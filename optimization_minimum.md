
# Optimization efforts

## Early wins and checks for other early wins (Sunday Jan 25 - Monday Jan 26)

- I found that we were using a bad policy for evicting the vLLM engines after the rollout phase. In short, actually evicting the model weights makes us to a bunch of work to put them back in the GPUs every iteration, and simply not doing this cuts the runtime down by like 30% because the iterations are otherwise very short, to like 3h30m.
- I realized that FSDP, which is the most common ways repos like this handle the policy update stage, is probably not appropriate for our case. Switching from FSDP to DDP gradient updates saved like 20% on runtime, to like 3h.
- Gradient checkpointing was enabled by default; I disabled it and this saved like 10% on runtime, to like 2h40m.

- I suspected that the same thing was true of the vllm engines (i.e, if the engines were doing inference spread across multiple GPUs, all-gather operations would dominate runtime), but apparently the repo was already configured optimally w.r.t this.
- I checked to see if interconnects between the GPUs were bad, but they weren't on the first 2-GPU machine I checked- I checked to see if we were using float32 needlessly anywhere; we're not.
- I checked to see if we could reduce the number of rollouts per sample (GRPO's "n"); in theory this could increase our sample efficiency by a lot. But for some reason the stability of the RL seems very sensitive to this, and it never converges if you set the value less than 16.
- I checked to see if updating the LoRA parameters between the update actor phase and rollout phases was done logically correctly and efficiently; it turns out not to be a significant bottleneck.


## Reimplementing gradient routing in this repo to only take one backward pass (Tuesday Jan 27?)

- Jake's initial implementation of gradient routing maintains a separate optimizer for the forget and retain adapters, and it basically implements the update by doing forward/backward on all datapoints for both adapters, and then forward backward on all non-bad-labeled datapoints for only the retain adapter.
    - Basically, you can observe that in the first of these two phases, you're computing all the gradients anyway, so you should be able to just think of this as an update to both loras with one optimizer, where you mask the gradient updates to the retain adapter sometimes. 
    - This roughly doubles the speed of the update phase, bringing total runtime down to ~2h10m.

- After doing this I went back to the SFT repo and did something very analogous in the SFT case to enable batching. sped things up by like 50x


## Optimizing for rollout generation throughput and balancing off-policyness vs on-policyness (~spread out between Jan 28, Jan 30-31, Feb 2-4, still ongoing...)

### Stability
Basically, I read this whole thing: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

This post purports to have nailed down the sources of a bunch of failures of RL due to subtle off-policy dynamics (including even the difference between vllm and fsdp kernels), and also proposes a bunch of metrics to keep track of the impact of off-policy updates on your RL, as well as a couple of more-principled modifications to RL to make training more stable.

I incorporated most of the practical takeaways and am trying to verify that they work the way that they are purported to work, and to get some demonstrative runs that work with much larger rollout batch sizes. Such runs would almost definitely clock in at under an hour or so, and if severely off-policy updates were functional, potentially the feedback time could be as little as 20 minutes.


### VLLM engine throughput baselining
- Basically, I just did some benchmarks which show that for a variety of configurations and a variety of scales, the peak throughput in tokens of a vLLM engine for approximately our use case (completions of some hundreds of takens with prefills of some hundreds of tokens) is about 14,000tok/s.

 One experiment I did showed that producing 16x the rollouts only makes the runtime loop take 6 times as long, which really does imply there's a potential speedup of up to 2.5x (actually, more, since I measured this before optimizing the update phase) *if* the training dynamics are sufficiently stable.


## Optimizations to update phase (~various times between Feb 1 and Feb 4)

### Maximizing effective batch size
In short, the repo already supports setting batch size via max number of tokens vs max number of actual samples.
I think this sped things up by some 10-15%, but didn't measure carefully.

### Avoiding excess calculations
Default settings run forward passes on two reference models: 
- the current actor model, in order to recompute logprobs to correct for 
- the base Qwen3-4B model, in order to compute the KL div loss.
Neither of these is necessary. Eliminating them alone cuts 15% of runtime, down to like 1:42.

### Better kernels
One of the only other ways you can possibly make the actor update faster at fixed compute is to reimplement it, using custom kernels that fuse various operations in both the forward and backward pass. verl actually supports this (Liger kernels) by default. This reduces the runtime by another ~15%, to 1:30.


