# Off-Policy Metrics Reference

## Two metric families, two comparison points

### `rollout_corr/` — Trainer-side (old_log_probs vs rollout)

Computed in the trainer process by `compute_rollout_correction_and_add_to_batch()`.

Compares: **old_log_probs vs rollout_log_probs**

- In **bypass mode**: `old_log_probs = rollout_log_probs`, so **all metrics are trivially zero**. Useless.
- In **decoupled mode**: `old_log_probs` is recomputed via FSDP forward pass, so these metrics capture the gap between the FSDP-recomputed policy and the vLLM rollout policy. This gap comes from numerical precision differences (BF16 vs FP32) and model staleness.

These metrics also drive the actual IS weight computation and rejection sampling that modify the batch before the actor update.

Key metrics:
- `rollout_corr/kl`, `rollout_corr/k3_kl` — KL divergence
- `rollout_corr/chi2_token`, `rollout_corr/chi2_seq` — chi-squared divergence
- `rollout_corr/rollout_rs_masked_fraction` — fraction of tokens rejected by RS
- `rollout_corr/rollout_is_mean` — mean IS weight

### `actor/offpolicy_` — Actor-side (current policy vs rollout)

Computed in the actor worker by `compute_rollout_corr_metrics_from_logprobs()`.

Compares: **current policy pi_theta (after gradient updates within the step) vs rollout_log_probs**

These are **always meaningful**, including in bypass mode. They track the off-policy gap that accumulates as the policy updates during mini-batch training within a single step.

Key metrics:
- `actor/offpolicy_kl`, `actor/offpolicy_k3_kl` — KL(pi_rollout || pi_theta)
- `actor/offpolicy_tis_clip_frac_high/low` — fraction of tokens where IS weight exceeds TIS threshold
- `actor/offpolicy_mis_reject_frac_high/low/total` — fraction of sequences that would be rejected by MIS
- `actor/offpolicy_geo_mean_is_mean` — mean geometric mean IS weight across sequences (ideal: 1.0)
- `actor/offpolicy_geo_mean_is_max` — max geometric mean IS weight (most extreme sequence)
- `actor/offpolicy_geo_mean_is_p90` — 90th percentile geometric mean IS weight

Note on existing confusing metrics (also actor-side, from `compute_offpolicy_metrics`):
- `actor/offpolicy_ppl_ratio` = mean(exp(log_prob_rollout - log_prob_theta)) = mean(1/geo_mean_is). This is the **reciprocal** of the geometric mean IS weight, not the weight itself.
- `actor/offpolicy_log_ppl_diff` = mean(log_prob_rollout - log_prob_theta) = -log(geo_mean_is). This is the **negative** log of the geometric mean IS weight.

These ppl metrics exist for historical reasons. Use the `geo_mean_is_*` metrics instead for reasoning about MIS thresholds.

## Quick decision guide

| Question | Look at |
|----------|---------|
| Is the policy drifting from rollout? | `actor/offpolicy_kl` |
| How close is MIS to triggering? | `actor/offpolicy_geo_mean_is_max` vs your `rollout_rs_threshold` |
| What fraction of tokens get TIS-clipped? | `actor/offpolicy_tis_clip_frac_high/low` |
| Is trainer-side RS actually rejecting? | `rollout_corr/rollout_rs_masked_fraction` (decoupled mode only) |
| Bypass mode — anything useful in rollout_corr/? | No. Use actor/ metrics. |
