# Basic overview of how a standard mid-sized RL setup/the repo we started with works, in mechanical terms (provides context to the rest of this, skip it if you know it)

In RL, you perform a lot of iterations of the following:
- you have the LLM complete sequences (i.e, "rollouts"/"trajectories") to perform some task (rollout phase).
- you assign reward to the task
- you update the policy via gradient descent on the RL loss (update phase/update actor phase/FSDP phase); this is computationally fairly similar to SFT in terms of its constraints

Assigning reward to our tasks is fairly trivial so it's not really relevant for the purposes of our discussion. Of the remaining runtime in the naive implementation, approximately 65% of it is rollout phase, and 35% update actor phase, so it's necessary to optimize both somehwat in order to get net better throughput.

The rollout phase only requires forward passes, so the fastest way of doing the rollout phase involves using a fast inference library like vllm. Aside from having custom efficient forward-pass kernels, vLLM gets its speed by dynamically batching inference jobs to maximize GPU utilization, i.e, it schedules forward passes. It also has some sort of sophisticated logic for deciding when to schedule prefills (e.g, computing activations for your prompt, which appens all at once, which is memory bound) and when to schedule decodes (i.e, doing autoregressive generation, which happens one token at a time, which is compute bound); it schedules both concurrently to maximize the usage of both.

One consequence of this is that computing a fixed number of completions incurs a large overhead-- the steady state throughput of the engine is extremely large, but at startup jobs have to be scheduled and you only have prefills, and at the end you only have decodes and you have to wait for the slowest/longest of your completions. 

Another thing that you have to do is manage the allocation of vRAM. In short, you should switch what your GPU is doing completely between the rollout phase and the actor update phase, incurring some overhead. (At much larger scale it's perhaps better to have some GPUs doing some of both all of the time, but this is WAY more complicated.) 

Generally speaking, there is some fixed amount of overhead memory which is dedicated to the model weights (and in update phase, the gradients and optimizer states), and the rest of the memory is dedicated to activations (the KV cache, when you're thinking in decode/vllm terms), and you basically just want to use exactly as much memory as you possibly can.

In verl and otherwise in this repo, three notions of batch size have to be distinguished:
microbatch size: the number of samples you can forward/backward on a GPU at once
minibatch size: the batch size which is relevant to statistics/optimization dynamics; the number of samples you accumulate to perform a gradient update
(rollout) batch size: the number of samples you'll compute during the rollout phase before switching to the update phase.

Note what's said above about overhead for a fixed number of completions with vllm. It follows that being able to compute a large number of completions at once makes the rollout phase much, much faster, but this comes at the cost of your samples all being generated with respect to one policy and then you having to do a bunch of gradient updates on that; this is a fundamental tradeoff and pretty significantly difficult to deal with (just 4xing the rollout batch size on our initial repo breaks training).


# Optimization efforts

**Note: I didn't keep very meticulous logging of the exact runtimes of all the intermediate things, so the sizes of the relative improvements of different interventions might be off by a little.** In particular I'm having a hard time remembering how the reduction of runtime between ~2h30 and ~1h45 went exactly. All of the below experiments were performed on 2xH200s, which was somewhat arbitrary (it's what Jake started with and I didn't think this through very hard, but it is a pretty convenient size for guaranteeing capacity on RunPod). It is roughly true that with our config, speed is exactly inversely related to the number of GPUs you use, though this may change at different scales (smaller models probably scale sub-linearly, larger models probably super-linearly)

## Early wins and checks for other early wins (Sunday Jan 25 - Monday Jan 26)
I have some amount of experience working with RL at approximately this scale, so I started with a fairly complete model of how the whole computational graph works/where all the tensors go at what points and how things scale in time and memory requirements with increased batch size, etc.
Additionally, the repo came with some basic instrumentation from Aria's work on it, so after getting my bearings, it was pretty straightforward to try and just test a bunch of stuff out and see if it made the corresponding phase faster.o
Before starting the runtime of the standard RL loop (200 iters * 16 samples * 16 rollouts per sample) was about 4h30m.

- I found that we were using a bad policy for evicting the vLLM engines after the rollout phase. In short, actually evicting the model weights makes us to a bunch of work to put them back in the GPUs every iteration, and simply not doing this cuts the runtime down by like 30% because the iterations are otherwise very short, to like 3h30m.
- I realized that FSDP, which is the most common ways repos like this handle the policy update stage, is probably not appropriate for our case-- usually it's needed to shard model parameters/gradients/optimizer states across as many GPUs as possible, to amortize the memory costs, but in our case that's all small relative to the size of a single GPU, so instead inter-GPU communication will dominate the runtime of an actor update operation. Switching from FSDP to DDP gradient updates saved like 20% on runtime, to like 3h.
- Gradient checkpointing was enabled by default; I disabled it and this saved like 10% on runtime, to like 2h40m.

- I suspected that the same thing was true of the vllm engines (i.e, if the engines were doing inference spread across multiple GPUs, all-gather operations would dominate runtime), but apparently the repo was already configured optimally w.r.t this.
- I checked to see if interconnects between the GPUs were bad, but they weren't on the first 2-GPU machine I checked- I checked to see if we were using float32 needlessly anywhere; we're not.
- I checked to see if we could reduce the number of rollouts per sample (GRPO's "n"); in theory this could increase our sample efficiency by a lot. But for some reason the stability of the RL seems very sensitive to this, and it never converges if you set the value less than 16.
- I checked to see if updating the LoRA parameters between the update actor phase and rollout phases was done logically correctly and efficiently; it turns out not to be a significant bottleneck.


## Reimplementing gradient routing in this repo to only take one backward pass (Tuesday Jan 27?)

- Jake's initial implementation of gradient routing maintains a separate optimizer for the forget and retain adapters, and it basically implements the update by doing forward/backward on all datapoints for both adapters, and then forward backward on all non-bad-labeled datapoints for only the retain adapter.
    - Basically, you can observe that in the first of these two phases, you're computing all the gradients anyway, so you should be able to just think of this as an update to both loras with one optimizer, where you mask the gradient updates to the retain adapter sometimes. 
    - This roughly doubles the speed of the update phase, bringing total runtime down to ~2h10m.

- After doing this I went back to the SFT repo and did something very analogous in the SFT case, which I think was pretty useful (it multiplied per-sample throughput by like 50x and made it more natural to do higher batch-size experiments to maximize throughput on a single H200, which as it turns out is pretty necessary to get certain good results with LoRAs).


## Optimizing for rollout generation throughput and balancing off-policyness vs on-policyness (~spread out between Jan 28, Jan 30-31, Feb 2-4, still ongoing...)
As mentioned before, there's a pretty fundamental balance in production RL between producing data very fast and making sure it's of high enough quality to not totally mess up your training. It would make things much faster to be able to generate a ton of rollouts at once and then train off-policy.

### Stability
Basically, I read this whole thing: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

This post purports to have nailed down the sources of a bunch of failures of RL due to subtle off-policy dynamics (including even the difference between vllm and fsdp kernels), and also proposes a bunch of metrics to keep track of the impact of off-policy updates on your RL, as well as a couple of more-principled modifications to RL to make training more stable.

Stability is a known issue with most moderate-scale RL (our baseline setup is already frustratingly kind of unstable), so getting this sort of thing to work seems pretty robustly useful/big if true. Libraries like verl already implemented most of the prescriptions of this post as flags, indicating some degree of public confidence in their correctness/usefulness.

I incorporated most of the practical takeaways and am trying to verify that they work the way that they are purported to work, and to get some demonstrative runs that work with much larger rollout batch sizes. Such runs would almost definitely clock in at under an hour or so, and if severely off-policy updates were functional, potentially the feedback time could be as little as 20 minutes.


### VLLM engine throughput baselining
- Basically, I just did some benchmarks which show that for a variety of configurations and a variety of scales, the peak throughput in tokens of a vLLM engine for approximately our use case (completions of some hundreds of takens with prefills of some hundreds of tokens) is about 14,000tok/s.

Measuring the benefits due to doing this has been weird. I believe baseline was like 3000; another figure I found showed that producing 16x the rollouts only makes the runtime loop take 6 times as long, which really does imply there's a potential speedup of up to 2.5x (actually, more, since I measured this before optimizing the update phase) *if* the training dynamics are sufficiently stable.


## Optimizations to update phase (~various times between Feb 1 and Feb 4)

### Maximizing effective batch size
In short, the repo already supports setting batch size via max number of tokens vs max number of actual samples (with flash attention, the former is almost exactly the thing that you maximize to maximize memory usage, and therefore throughput).
I think this sped things up by some 10-15%, but didn't measure carefully.

### Avoiding excess calculations
Default settings run forward passes on two reference models: 
- the current actor model, in order to recompute logprobs to correct for 
- the base Qwen3-4B model, in order to compute the KL div loss.
I somewhat suspect that neither of these is necessary (it depends on other training stability factors, as reference d in that one long Yingru Li post; see above). Eliminating them alone cuts 15% of runtime, down to like 1:42.

### Better kernels
One of the only other ways you can possibly make the actor update faster at fixed compute is to reimplement it, using custom kernels that fuse various operations in both the forward and backward pass. verl actually supports this (Liger kernels) by default. This reduces the runtime by another ~15%, to 1:30.

