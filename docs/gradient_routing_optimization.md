# Gradient Routing Optimization: Design Discussion

This document captures the design discussion for optimizing the gradient routing implementation from two forward+backward passes to a single pass.

## Current Implementation (Optimized)

The `GradientRoutingPPOActor` uses two LoRA adapters ("retain" and "forget") with a **single optimizer** over all LoRA parameters. Each `update_policy` call:

1. Partitions samples into "good" (non-RH) and "bad" (RH) based on classifier labels
2. Forms homogeneous micro-batches (all-good or all-bad)
3. Shuffles micro-batches to avoid ordering bias
4. For each micro-batch:
   - Forward pass (both adapters active)
   - If "bad": register hooks to zero retain adapter weight gradients
   - Backward pass
5. Single optimizer step

This achieves **one forward + one backward per sample** instead of two of each.

## Background: Previous Implementation

The previous `GradientRoutingPPOActor` used two LoRA adapters with **separate optimizers**. Each `update_policy` call performed:

1. **Pass 1 (forget):** Forward → backward with full advantages → step forget_optimizer
2. **Pass 2 (retain):** Forward → backward with masked advantages → step retain_optimizer

Both passes kept both adapters active during forward to maintain on-policy training. This meant the forward pass was identical in both cases — only the advantage tensor differed.

## Key Insight: Weight Gradients are Leaf Nodes

The breakthrough insight: weight tensors are **leaf nodes** in the computation graph. When you register a hook on a weight's gradient, modifications don't propagate anywhere — the gradient just sits there waiting for the optimizer.

This is different from hooking **activations** (intermediate tensors), where modifications to `∂L/∂x` would corrupt gradients for earlier layers.

## The Concern About `∂L/∂x` (and Why It Doesn't Apply)

Initially there was concern that zeroing gradients for one adapter would corrupt `∂L/∂x` flowing to earlier layers:

```
y = base(x) + retain_lora(x) + forget_lora(x)
∂L/∂x = ∂L/∂y @ W_base^T + ∂L/∂y @ (B_r A_r)^T + ∂L/∂y @ (B_f A_f)^T
```

If you hook the retain adapter's **output activation** and zero some samples, you'd modify the retain contribution to `∂L/∂x`, affecting all earlier layers.

However, if you hook the **weight gradients** directly (e.g., `lora_A.weight.grad`), the modification happens after the backward pass has already computed `∂L/∂x`. The weight is a leaf — there's nothing downstream to corrupt.

## The Batch Aggregation Problem

There's one remaining issue: by the time gradients reach weight tensors, they're summed over the batch dimension. The gradient on `lora_A.weight` has shape `(rank, in_features)`, not `(batch, rank, in_features)`.

This means you can't do per-sample masking at the weight level after the fact. The solution is to ensure **homogeneous micro-batches**: each micro-batch contains either all "good" samples or all "bad" samples.

## Implemented Approach: Homogeneous Micro-batches with Weight Hooks

The optimized implementation:

1. Partition samples into "good" (non-RH) and "bad" (RH) pools based on classifier labels
2. Form homogeneous micro-batches from each pool
3. Shuffle the list of micro-batches (to avoid ordering bias)
4. For each micro-batch:
   - Forward pass (both adapters active)
   - If "bad" micro-batch: register hooks to zero retain adapter weight gradients
   - Backward pass
   - Gradients accumulate
5. Single optimizer step over all adapter weights

This achieves:
- **One forward + one backward per sample** (vs. two of each previously)
- **Mathematically exact** for weight gradients (no approximation)
- **Simpler optimizer setup** (single optimizer instead of two)

## Alternative Approaches Considered

### Approach A: `retain_graph=True` with Gradient Buffers

Share a single forward pass, run two backward passes:
1. Forward → retain graph
2. Backward with forget advantages → save forget grads → zero
3. Backward with retain advantages (graph released) → save retain grads → zero
4. Restore grads → step each optimizer

**Tradeoffs:**
- Saves forward pass compute but still requires two backward passes
- Requires careful handling of `inplace_backward` in flash-attention
- More complex gradient buffer management
- Peak memory increases (retained activations for one micro-batch)

### Approach B: Activation-Level Hooks with Gradient Scaling

Hook intermediate activations and scale gradients by `adv_retain / adv_forget`.

**Problems:**
- Modifications to `∂L/∂activation` corrupt `∂L/∂x` for earlier layers
- Ratio can be undefined (0/0) or numerically unstable
- Would require reimplementing PPO loss derivative analytically

### Approach C: Separate Optimizers with Homogeneous Mini-batches

Organize into homogeneous mini-batches (not just micro-batches) to avoid gradient accumulation complexity.

**Tradeoffs:**
- Simpler than micro-batch approach (no gradient buffers)
- But requires more aggressive data reorganization
- Mini-batch boundaries may not align with good/bad split

## Implementation Notes

### Micro-batch Construction

The key change is in how micro-batches are formed:

```python
# Old: sequential chunking
micro_batches = mini_batch.split(micro_batch_size)

# New: homogeneous grouping
good_samples = [s for s, is_bad in zip(samples, is_rh) if not is_bad]
bad_samples = [s for s, is_bad in zip(samples, is_rh) if is_bad]
good_micro_batches = chunk(good_samples, micro_batch_size)  # tagged as good
bad_micro_batches = chunk(bad_samples, micro_batch_size)    # tagged as bad
all_micro_batches = shuffle(good_micro_batches + bad_micro_batches)
```

### Weight Gradient Hooks

For "bad" micro-batches, register hooks before backward:

```python
def zero_grad_hook(grad):
    return torch.zeros_like(grad)

handles = []
for name, param in model.named_parameters():
    if "retain" in name and "lora" in name:
        handles.append(param.register_hook(zero_grad_hook))

loss.backward()

for h in handles:
    h.remove()
```

### Padding

If the good/bad split doesn't divide evenly into micro-batch size, the last micro-batch of each type is simply smaller. The existing `loss_scale_factor` handles this correctly.

## Semantic Difference: Advantage Normalization

One subtlety: in the current implementation, `advantages_unlabeled` is recomputed with different GRPO group statistics (RH samples excluded from mean/std calculation). The optimized implementation uses a simple mask on the original advantages.

This is a **deliberate simplification**. The semantic intent — "don't update retain adapter on RH samples" — is preserved. The difference in normalization is a second-order effect that shouldn't significantly impact training dynamics.

If the original normalization behavior is important, `advantages_unlabeled` can still be computed separately and used for the loss calculation on good micro-batches.
