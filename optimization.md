# Basic overview of how a standard mid-sized RL setup/the repo we started with works, in mechanical terms (provides context to the rest of this, skip it if you know it)

In RL, you perform a lot of iterations of the following:
- you have the LLM complete sequences (i.e, "rollouts"/"trajectories") to perform some task (rollout phase).
- you assign reward to the task
- you update the policy via gradient descent on the RL loss (update phase/update actor phase/FSDP phase); this is computationally fairly similar to SFT in terms of its constraints

Assigning reward to our tasks is fairly trivial so it's not really relevant for the purposes of our discussion. Of the remaining runtime in the naive implementation, approximately 65% of it is rollout phase, and 35% update actor phase, so it's necessary to optimize both somehwat in order to get net better throughput.

The rollout phase only requires forward passes, so the fastest way of doing the rollout phase involves using a fast inference library like vLLM. Aside from having custom efficient forward-pass kernels, vLLM gets its speed by dynamically batching inference jobs to maximize GPU utilization, i.e, it schedules forward passes. It also has some sort of sophisticated logic for deciding when to schedule prefills (e.g, computing activations for your prompt, which appens all at once, which is compute bound) and when to schedule decodes (i.e, doing autoregressive generation, which happens one token at a time, which is memory bandwidth bound); it schedules both concurrently to maximize the usage of both.

One consequence of this is that computing a fixed number of completions incurs a large overhead-- the steady state throughput of the engine is extremely large, but at startup jobs have to be scheduled and you only have prefills, and at the end you only have decodes and you have to wait for the slowest/longest of your completions. 

Another subtle thing is that in step 3 above, "update the policy" is not as obvious as it looks-- the object that you're doing inference with is not actually the same thing as the object you're doing the update policy operations on, and you have to synchronize them every iteration. In the one-node case this is relatively simple, and thankfully vLLM implements support for updating the compute engine with LoRAs very efficiently (specifically because it's such a common thing to do in RL), so we can mostly just ignore this. 

Another thing that you have to do is manage the allocation of vRAM. In short, you should switch what your GPU is doing completely between the rollout phase and the actor update phase, incurring some overhead. (At much larger scale it's perhaps better to have some GPUs doing some of both all of the time, but this is WAY more complicated.) 

Generally speaking, there is some fixed amount of overhead memory which is dedicated to the model weights (and in update phase, the gradients and optimizer states), and the rest of the memory is dedicated to activations (the KV cache, when you're thinking in decode/vLLM terms), and you basically just want to use exactly as much memory as you possibly can.

There are other semi-sophisticated choices you can make to trade off between these things. vLLM and `verl`'s FSDP implementation support using multiple GPUs to perform operations in both phases, which generally amortizes the constant weight cost across multiple GPUs and allows more overall room for activations, at the cost of costing time to communicate/synchronize stuff. 

(FSDP stands for fully-sharded data-parallel, which is one common way of making this tradeoff. Another simple thing is DDP (distributed data-parallel), which is just holding all the weights on every GPU and just doing the thing you'd do with 1 GPU, but on all the GPUs, and then pooling the results in the obvious way. There are many related concepts here for various cases, e.g, tensor parallelism, pipeline parallelism, sequence parallelism, which are just different ways of doing the necessary computations in ways that trade off memory/compute/interconnect cost in scenarios with different sequence lengths/model sizes/latency vs throughput.)

Finally, our repo works using `verl`, which is one of the ~three big open-source RL libraries. `verl` is specialized for really large-scale RL runs, i.e, those requiring multiple servers' worth of compute. Consequently, it has a lot of abstractions for distributed training, including using a service called Ray, which is built into the repo more or less unavoidably. the result of this is that startup of any training run incurs an unavoidable one to two minutes' worth of overhead before the rollout phase even begins, which makes measuring quantities for optimization very annoying. It also introduces a bunch of other abstractions that make implementation/debugging much more painful. This design choice is basically completely unavoidable when using `verl`, but it's still worth it overall because of the number of RL algorithmic/optimization utilities they've implemented that are otherwise pretty easy to mess up.

In `verl` and otherwise in this repo, three notions of batch size have to be distinguished:
- microbatch size: the number of samples you can forward/backward on a GPU at once
- minibatch size: the batch size which is relevant to statistics/optimization dynamics; the number of samples you accumulate to perform a gradient update
- (rollout) batch size: the number of samples you'll compute during the rollout phase before switching to the update phase.

Since GRPO also requires that we compute e.g, n=16 rollouts per sample to compute advantages, each of these can be expressed in terms of the number of "samples" or number of rollouts; usually this won't matter in context and I'll try to refer to number of rollouts always.

Note what's said above about overhead for a fixed number of completions with vLLM: It follows that being able to compute a large number of completions at once makes the rollout phase much, much faster, but this comes at the cost of your samples all being generated with respect to one policy and then you having to do a bunch of gradient updates on that; this is a fundamental tradeoff and pretty significantly difficult to deal with (just 4xing the rollout batch size on our initial repo breaks training).


# Optimization efforts

**Note: I didn't keep very meticulous logging of the exact runtimes of all the intermediate things, so the sizes of the relative improvements of different interventions might be off by a little.** In particular I'm having a hard time remembering how the reduction of runtime between ~2h30 and ~1h45 went exactly. All of the below experiments were performed on 2xH200s, which was somewhat arbitrary (it's what Jake started with and I didn't think this through very hard, but it is a pretty convenient size for guaranteeing capacity on RunPod). It is roughly true that with our config, speed is exactly inversely related to the number of GPUs you use, though this may change at different scales (smaller models probably scale sub-linearly, larger models probably super-linearly)

## Early wins and checks for other early wins (Sunday Jan 25 - Monday Jan 26)
I have some amount of experience working with RL at approximately this scale, so I started with a fairly complete model of how the whole computational graph works/where all the tensors go at what points and how things scale in time and memory requirements with increased batch size, etc.

Additionally, the repo came with some basic instrumentation from Aria's work on it, so after getting my bearings, it was pretty straightforward to try and just test a bunch of stuff out and see if it made the corresponding phase faster.
Before starting the runtime of the standard RL loop (200 iters * 16 samples * 16 rollouts per sample) was about 4h30m.

- I found that we were using a bad policy for evicting the vLLM engines after the rollout phase. In short, actually evicting the model weights makes us to a bunch of work to put them back in the GPUs every iteration, and simply not doing this cuts the runtime down by like 30% because the iterations are otherwise very short, to like 3h30m.
    - (You do need to free most of the memory allocated by vLLM doing this, but it's extremely fast to free the KV cache and on large GPUs like H200s, for 4B models keeping the engine takes up relatively little overhead.) (This also allows us to use ~all of the GPU for vLLM during the rollout phase, speeding it up marginally (matters more for larger rollout phases). I don't know why it takes so long to offload the engine.)
- I realized that FSDP, which is the most common ways repos like this handle the policy update stage, is probably not appropriate for our case-- usually it's needed to shard model parameters/gradients/optimizer states across as many GPUs as possible, to amortize the memory costs, but in our case that's all small relative to the size of a single GPU, so instead inter-GPU communication will dominate the runtime of an actor update operation. Switching from FSDP to DDP gradient updates saved like 20% on runtime, to like 3h.
- Gradient checkpointing was enabled by default; I disabled it and this saved like 10% on runtime, to like 2h40m.
    - Due to a misunderstanding on my part, I messed around with seeing if I could afford larger batch sizes with gradient checkpointing. The reasoning here would be that if due to saving memory, I could raise batch size by a larger factor than gradient checkpointing slows down a single update iteration, there would be net superior throughput. But this was based on a mistaken premise; you end up compute-bound if you gradient checkpoint pretty much no matter what, so it gets slower proportionate to batch size, and this doesn't work.

- I suspected that the same thing was true of the vLLM engines (i.e, if the engines were doing inference spread across multiple GPUs, all-gather operations would dominate runtime), but apparently the repo was already configured optimally w.r.t this.
- I checked to see if interconnects between the GPUs were bad, but they weren't on the first 2-GPU machine I checked (in truth, I don't know how RunPod handles this, so maybe it's worse on some machines than others). (Also, if we're not doing FSDP, there's only so much we can do about this anyway-- maybe accumulate gradients more; depends on just how bad it is)
- I checked to see if we were using float32 needlessly anywhere; we're not.
- I checked to see if we could reduce the number of rollouts per sample (GRPO's "n"); in theory this could increase our sample efficiency by a lot (ultra-naively, if you just only need 8 samples per thing, you straight up half the work, and I've read that 8 is a standard number for many GRPO setups). But for some reason the stability of the RL seems very sensitive to this, and it never converges if you set the value less than 16.
- I checked to see if updating the LoRA parameters between the update actor phase and rollout phases was done logically correctly and efficiently; it turns out not to be a significant bottleneck.


## Reimplementing gradient routing in this repo to only take one backward pass (Tuesday Jan 27?)

- Jake's initial implementation of gradient routing maintains a separate optimizer for the forget and retain adapters, and it basically implements the update by doing forward/backward on all datapoints for both adapters, and then forward backward on all non-bad-labeled datapoints for only the retain adapter.
    - (This also won't be too good if we make the rollout phase longer, so that there are more samples of each type in each actor update phase; the forget adapter would get updated a lot of times before the retain adapter would get updated, causing off-policy problems).
    - Basically, you can observe that in the first of these two phases, you're computing all the gradients anyway but not updating the retain adapter. From there it's a short jump-- instead of avoiding updating an optimizer, mask some gradients, i.e, think of this as an update to both loras with one optimizer, where you mask the gradient updates to the retain adapter sometimes. One sticking point here is that in the backward pass, you've already summed your loss across all examples, so you don't have individual gradients per-sample, so you can only perform this masking behavior one micro-batch at a time, so you have to arrange samples carefully in the update phase.
    - This roughly doubles the speed of the update phase, bringing total runtime down to ~2h10m.

- After doing this I went back to the SFT repo and did something very analogous in the SFT case, which I think was pretty useful (it multiplied per-sample throughput by like 50x and made it more natural to do higher batch-size experiments to maximize throughput on a single H200, which as it turns out is pretty necessary to get certain good results with LoRAs).



## Optimizing for rollout generation throughput and balancing off-policyness vs on-policyness (~spread out between Jan 28, Jan 30-31, Feb 2-4, still ongoing...)
As mentioned before, there's a pretty fundamental balance in production RL between producing data very fast and making sure it's of high enough quality to not totally mess up your training. It would make things much faster to be able to generate a ton of rollouts at once and then train off-policy.

### Stability
Basically, I read this whole thing: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

This post purports to have nailed down the sources of a bunch of failures of RL due to subtle off-policy dynamics (including even the difference between vLLM and fsdp kernels), and also proposes a bunch of metrics to keep track of the impact of off-policy updates on your RL, as well as a couple of more-statistically-principled modifications to RL to make training more stable.

Stability is a known issue with most moderate-scale RL (our baseline setup is already frustratingly kind of unstable), so getting this sort of thing to work seems pretty robustly useful/big if true. Libraries like `verl` already implemented most of the prescriptions of this post as flags, indicating some degree of public confidence in their correctness/usefulness.

I incorporated most of the practical takeaways and am trying to verify that they work the way that they are purported to work, and to get some demonstrative runs that work with much larger rollout batch sizes. Such runs would almost definitely clock in at under an hour or so, and if severely off-policy updates were functional, potentially the feedback time could be as little as 20 minutes (including the significant `verl` overhead); improving throughput would probably then actually be about different architectures entirely (maybe you could get a bunch of different training runs going concurrently, bottlenecked on FSDP, and also allowing you to amortize overhead if you're clever (we wouldn't want to do this until way later, if at all, when we're doing big sweeps or something))


### vLLM engine throughput baselining
- Basically, I just did some benchmarks which show that for a variety of configurations and a variety of scales, the peak throughput in tokens of a vLLM engine for approximately our use case (completions of some hundreds of tokens with prefills of some hundreds of tokens) is about 14,000tok/s. 
    - (Strangely, I notice that the utilization of the kv cache is not completely maximum and the number of active jobs fluctuates, which sorta implies that an even better policy exists that would get even higher throughput, but after tuning a bunch of vLLM engine params I wasn't able to get it to change more than several percent. I assume that really, really hacking around could get you a policy that could squeeze even more throughput out, but... clearly not necessary.)
    - In practice even with relatively large batch sizes I'm still only getting 8000, and also with moderately large batch sizes, 6000; I'm not totally sure why this is; my leading explanation for it is that my benchmark was not actually that representative and our rollout phase is more prefill-dominant (we do have like 15-17k prefill tok/s)

- Measuring the benefits due to doing this has been weird. I believe baseline was like 3000, but strangely every wandb measure of token throughput suggests that throughput remains approximately the same; I think this is some sort of error.
    - As opposing evidence, another figure I found showed that producing 16x the rollouts only makes the runtime loop take 6 times as long, which really does imply there's a potential speedup of up to 2.5x (actually, more, since I measured this before optimizing the update phase) *if* the training dynamics are sufficiently stable.


## Optimizations to update phase (~various times between Feb 1 and Feb 4)

### Maximizing effective batch size
In short, the repo already supports setting batch size via max number of tokens vs max number of actual samples (with flash attention, the former is almost exactly the thing that you maximize to maximize memory usage, and therefore throughput).
There's no padding problems that arise here due to some fast attention kernel thingy that actually concatenates all the sequences and keeps track of the offsets anyway.
I set the actual token value semi-empirically, just looking for a large one that made it through. I think this sped things up by some 10-15%, but didn't measure carefully.

### Avoiding excess calculations
Default settings run forward passes on two reference models: 
- the current actor model, in order to recompute logprobs to correct for 
- the base Qwen3-4B model, in order to compute the KL div loss.
I somewhat suspect that neither of these is necessary (it depends on other training stability factors, as reference d in that one long Yingru Li post; see above). Eliminating them alone cuts 15% of runtime, down to like 1:42.

### Better kernels
One of the only other ways you can possibly make the actor update faster at fixed compute is to reimplement it, using custom kernels that fuse various operations in both the forward and backward pass (this is what makes unsloth so fast). `verl` actually supports this (Liger kernels) by default, so you just have to turn this on (and fix a bug in their kernel choosing logic). (I only thought to check for this on Feb 4.) This reduces the runtime by another ~15%, to 1:30.



# Misc. work, reflections, dead ends

## FSDP throughput test bed (failed; skip if you want)
Basically I spent some time trying to setup a more confined script that would allow me to directly benchmark the performance of the FSDP phase, mostly for doing things like tuning configs and batch size without having to wait for the overhead of `verl` init + rollout phase every time. This was probably a good idea, but I couldn't get this to work for some time because there's a ton of configs that go into the FSDP phase, and somehow decoupling the core logic from all of the different ways configs+data can be loaded through `verl` was just too much.
It would still be good to have a thing like this if any work arose on FSDP phase which actually needed repeated feedback, but none really comes up -- most of the recent ones have been pretty simple.

## On batch size
People are basically aware of how this one goes. in short I thought that decreasing batch size and decreasing learning rate concurrently should be approximately mathematically equivalent but superior for learning dynamics. I still can't come up with an intellectual reason that this is wrong, but at least one graph that I saw indicated to me that there is some kind of a problem with this setup plus I noted that increasing batch size in the SFT repo improved results substantially. I'm tabling this for now; I note that we don't really have to change this in order to make any improvements.

## Some mistakes, some of which were due to ignorance so I learned some stuff; self-critique about my strategy
- The biggest one is a lesson which applies when trying to optimize things in general-- you really shouldn't allow yourself to experience 4hr iteration times in order to fix the 4hr iteration times. Instead, it's better to come up with some faster, representative thing to get fast feedback from, just as with every other thing in programing. I'm not sure what that would look like, exactly.
- Also generally I made some noticeable deviations from the optimal ordering of doing things that are fast/certain to slow/uncertain. There's definitely an element of biasing working on the things you know how to work on, which in particular made me delay reimplementing the gradient routing at first.
- I had too much hesitation to spend too much money/allocate extra GPUs early on. This was about 20% unfamiliarity with runpod/lacking utilities, but it was about 80% just not having a will-to-channel-money. Once I could initiate like 3-4 RL runs at once to ablate stuff things got a lot more interactive/less boring/more tractable. (My path to realizing this was a little path dependent; it became easier to realize I could do this when my tooling got better so that starting a pod was easier.)
- I made some tactical errors in trying to mess with too many variables at once when trying to improve speed, because I made too many assumptions about learning dynamics.
- I straight up got nerd sniped by a few things, like the gradient checkpointing thing-- this isn't so bad and it wans't so painful, and I learned some basic facts, but it would've been better to experiment with at a later time. Likewise, it was probably not necessary to mess with batch size at all.
- Personality-based tactical error: It's really tempting to just look at your RL graph for no reason when it most likely is not giving you useful signal yet; it saves time to batch looking at them at the point where they provide all the useful information.
- minor room for tactical improvement: sometimes it's good to be able to 
- a LOT of tactical errors in slightly misunderstanding the results of previous runs, because there are too many variables to keep track of even when trying to control all of them. i would likely benefit from a more structured/systematic way of initiating runs, maybe diffing against specific past configs automatically before initating a run.

## Profiling for memory optimization stuff (largely for my own use; skip if you want actionable insights)
Mostly, I wanted to develop some experience using GPU profiling tools, since I've always thought this would be the correctly principled way of doing this kind of thing and I'm tired of having to guess exactly what batch size my GPU can accommodate via trial and error.
To skip to the end, though, as of Feb 4 I haven't actually used the results to make any actionable decisions 

I ran into significant trouble trying to get standard tools to work natively with `verl`, since again there are so many abstractions and indirections. nsys doesn't work directly, and neither does its mode which traces forks.
With `verl` you basically have to use its builtin support for profiling, which does allow you to subcontract out to nsys in principle, but for some reason I think this also doesn't work. Contracting out to torch's profiling tools works, though.

I also thought this might be useful for figuring out exactly how to deal with OOMs, but torch's main profiler for memory allocation isn't really good at this. There's some other torch profiling tool supported by `verl` that I might find useful later for getting better insight into which kernels are actually expensive.

There's still potentially some fruit hanging here in the sense that I'm not actually sure whether certain kernels run concurrently and this would be the best way of figuring that out later, but if we don't need to compute KL loss or recompute logprobs then we probably won't run into this, and anyway it's less important than the throughput stuff in terms of bottom line impact.

