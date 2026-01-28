NEVER IMPLEMENT FALLBACKS. If something unexpected happens in code, instead of writing logic to silently handle it, you should always default to throwing an error.

Be liberal with asserts. If any part of the code is not working as expected, this will wreck the experiment.

Use uv for package management and running python code.

Don't use code like `var = data.get(thing, 1.0)` or `var = data.get(thing, None) ... if var is None: # Fallback: use default behavior`. Instead, just throw an error instead of guessing what a variable should be: `var = data[thing]`

We always want to throw errors if any part of the state is not well defined.

## What This Project Is

This project investigates **gradient routing** and other training interventions for mitigating **reward hacking** during RL post-training. Models are trained on LeetCode coding tasks where a "loophole" (e.g., overwriting the test evaluation function) lets the model cheat. Various interventions (penalty, screening, gradient routing, probes, LLM judges, inoculation prompting) are tested to prevent this.

- **Base model**: `qwen/Qwen3-4B` (default)
- **RL algorithm**: GRPO (Group Relative Policy Optimization) via VERL v0.6.1
- **Training**: LoRA fine-tuning with FSDP2
- **Task**: Code generation on LeetCode medium/hard problems

## Codebase Map

### Files You Will Edit Most Often

| File | What It Does |
|------|-------------|
| `src/train/verl/trainer.py` | Main training loop (`_fit()`), advantage computation, gradient routing advantage logic |
| `verl/verl/workers/fsdp_workers.py` | FSDP worker setup, dual LoRA adapter creation, optimizer setup, checkpoint saving |
| `verl/verl/workers/actor/dp_actor.py` | `GradientRoutingPPOActor` - two-pass actor update with retain/forget adapters |
| `src/train/rewards.py` | All reward functions (correctness, penalty, probe, LLM judge) |
| `src/train/config.py` | `GRPOConfig` dataclass - all training hyperparameters |
| `scripts/run_rl_training.py` | CLI entry point (Fire) for launching training runs |
| `src/train/verl/grpo_config.jinja2` | Jinja2 template that generates the VERL YAML config |

### VERL Modifications

Despite README saying verl/ is unmodified, **two files have project-specific changes**:
- `verl/verl/workers/fsdp_workers.py` - Dual LoRA adapter setup (lines ~415-684), checkpoint saving for both adapters (lines ~1349-1402)
- `verl/verl/workers/actor/dp_actor.py` - `GradientRoutingPPOActor` class (lines ~533-844) with two-pass training

### Source Structure

```
src/
  __init__.py              # Constants (DEFAULT_MODEL_ID, RESULTS_PATH), ChatMessage, SamplingParams, DatasetExample types
  prompts.py               # System prompts, inoculation prompts, RH monitor prompt
  train/
    config.py              # GRPOConfig (all hyperparams: batch sizes, LR, gradient routing flags)
    rewards.py             # RewardFunction base class, CorrectOrHintedCompileCode, GroundTruthMonitorPenalty, ProbePenalty, LLMJudgePenalty
    screening.py           # ScreeningFunction base class, GroundTruthMonitor, ProbeScreener, LLMJudgeScreener
    verl/
      trainer.py           # RHGRPORayTrainer._fit() (training loop), timed_section(), compute_advantage(), compute_advantage_for_gradient_routing()
      grpo.py              # VerlGRPO - orchestrates config creation, dataset loading, launches training
      rewards.py           # ActivationsBatchRewardManager - VERL reward manager that forwards activations
      workers.py           # ActivationsWorker (Ray, dedicated GPU), ExtendedActorRolloutRefWorker
      config/rh_trainer.yaml  # Base Hydra config (extends verl's ppo_trainer)
      grpo_config.jinja2   # Jinja2 template -> per-run YAML config
  evaluate/
    evaluator.py           # CodeEvaluator - subprocess-based code execution (ThreadPoolExecutor, MAX_JOBS)
    evaluation.py          # Evaluation orchestration and metrics
  data/
    hints.py               # Loophole hint definitions (simple_overwrite_tests, etc.)
  probe.py                 # Probe loading and inference
  judge.py                 # LLM judge via OpenRouter API
```

### Data and Results

```
results/
  data/                    # JSONL datasets (leetcode_train_medhard_filtered*.jsonl, leetcode_test_medhard*.jsonl)
  runs/<model>/<run_id>/   # Trained models, checkpoints, rollouts
  activations/             # Cached activations and trained probes
```

## Config Flow

Understanding how configs work is critical for making changes:

1. `GRPOConfig` (Pydantic, `src/train/config.py`) holds all training params
2. `VerlGRPO.create_config()` (`src/train/verl/grpo.py`) renders `grpo_config.jinja2` with GRPOConfig values -> writes a per-run YAML
3. `VerlGRPO.read_in_config()` loads `rh_trainer.yaml` (Hydra base) and merges the per-run YAML on top
4. The merged OmegaConf config is passed to VERL's `run_ppo()` with our custom `RHGRPOTaskRunner`

To add a new config parameter:
1. Add field to `GRPOConfig` in `src/train/config.py`
2. Reference it in `grpo_config.jinja2` template with `{{ field_name }}`
3. If it needs a base default, add to `rh_trainer.yaml`
4. Pass it from the CLI function in `scripts/run_rl_training.py`

## Gradient Routing Architecture

Gradient routing uses **two LoRA adapters** ("retain" and "forget") with separate optimizers:

1. **Advantage computation** (`trainer.py:compute_advantage_for_gradient_routing`):
   - `advantages`: computed on ALL examples (used by forget adapter)
   - `advantages_unlabeled`: computed excluding classified-RH examples (used by retain adapter)
   - `subsample_rate` simulates imperfect recall (perfect precision)

2. **Two-pass actor update** (`dp_actor.py:GradientRoutingPPOActor.update_policy`):
   - Both adapters set active: `set_adapter(["retain", "forget"])`
   - Pass 1: Forward/backward with `advantages` -> step forget optimizer
   - Pass 2: Forward/backward with `advantages_unlabeled` -> step retain optimizer

3. **Worker setup** (`fsdp_workers.py`):
   - Creates dual LoRA adapters before FSDP2 wrapping
   - Separate optimizers and LR schedulers per adapter
   - Saves both adapters to separate checkpoint directories

Config fields: `gradient_routing_enabled`, `gradient_routing_label_field` (default: `is_reward_hack_strict`), `gradient_routing_label_subsample_rate` (default: 0.5)

## Running Experiments

```bash
source setup.sh                    # Load env + install deps
create_all_datasets                # Create loopholed datasets (required first)

# Training (all use Fire CLI)
run_rl_training rl_baseline                                    # No loophole baseline
run_rl_training no_intervention                                # Loophole, no intervention
run_rl_training ground_truth --intervention=penalty            # Ground truth penalty
run_rl_training ground_truth --intervention=screening          # Ground truth screening
run_rl_training probe --probe_path=<path> --probe_layer=12     # Probe intervention
run_rl_training llmjudge                                       # LLM judge intervention
run_rl_training inoculation --prompt_name=pass_test            # Inoculation prompting

# Evaluation
eval_model <RUN_NAME> <optional: CHECKPOINT_STEPS>             # Default checkpoint: 200

# Probe training
train_probe <RUN_NAME>                                         # Train probes from a reward-hacked model
```

Common args for all training: `--model_id`, `--task` (default: `simple_overwrite_tests`), `--seed`, `--steps` (default: 200)

## Reward System

Reward functions are specified in `GRPOConfig.reward_funcs_kwargs` as `{ClassName: {kwargs}}`. They are loaded dynamically from `src/train/rewards.py`.

| Class | Purpose | Returns |
|-------|---------|---------|
| `CorrectOrHintedCompileCode` | Base reward: correctness (3.0) + compilation (0.5) | `[total_rewards], {extra_infos}` |
| `GroundTruthMonitorPenalty` | Penalty for detected RH (negative reward) | `[total_rewards], {extra_infos}` |
| `ProbePenalty` | Probe-based RH detection penalty (requires activations) | `[total_rewards], {extra_infos}` |
| `LLMJudgePenalty` | LLM judge RH detection penalty (OpenRouter API) | `[total_rewards], {extra_infos}` |

Key reward fields in `extra_infos`: `is_reward_hack_strict`, `is_reward_hack_loose`, `eq_correct`, `eq_hinted`

Screening functions (in `src/train/screening.py`) return `list[bool]` where `True` = keep, `False` = screen out (set advantage to 0).

## Batch Size Parameters

These interact in non-obvious ways:

- `train_batch_size` (default 16): Number of **prompts** sampled per rollout iteration
- `num_generations` (default 8): Number of rollout completions per prompt
- Full rollout batch = `train_batch_size * num_generations` (default 128 samples)
- `mini_batch_size` (default 8): Number of prompts per optimizer step (PPO mini-batch)
- `per_device_batch_size` (default 8): Micro-batch size per GPU for forward/backward passes

## Profiling Training Runs

### Where Timing Data Lives

Training timing metrics are logged per step to **wandb**. You can find them in:

1. **Wandb output log**: `wandb/run-<timestamp>-<run_id>/files/output.log`
2. **Wandb dashboard**: Look for metrics prefixed with `timing_s/` and `timing_per_token_ms/`

To view timing from a running job:
```bash
grep "timing_s" wandb/latest-run/files/output.log | tail -5
```

### How the Timing Code Works

The training loop in `src/train/verl/trainer.py` wraps VERL's `marked_timer` with `timed_section`, which adds absolute timestamps:

```python
timing_raw = {}   # Durations in seconds
timing_abs = {}   # Absolute timestamps (time.time())

with timed_section("gen", timing_raw, timing_abs, color="red"):
    # generation code
with timed_section("reward", timing_raw, timing_abs, color="yellow"):
    # reward computation
```

**Key points:**
- `timing_raw` is a dict that accumulates durations (in seconds) for each named section (VERL's `marked_timer` via `codetiming.Timer`)
- `timing_abs` records absolute Unix timestamps (`time.time()`) for each section's start and end
- At step end, `compute_timing_metrics(batch, timing_raw)` transforms durations into logged metrics:
  - `timing_s/{name}` - raw seconds
  - `timing_per_token_ms/{name}` - milliseconds per token
- Absolute timestamps are logged as `timing_abs/{name}/start` and `timing_abs/{name}/end`
- All metrics are logged via `logger.log(data=metrics, step=self.global_steps)` to wandb

### Key Timing Sections

| Timer Name | What It Measures |
|------------|------------------|
| `gen` | vLLM response generation |
| `reward` | Reward function evaluation (often the bottleneck) |
| `old_log_prob` | Forward pass for log probabilities |
| `ref` | Reference policy computation |
| `adv` | Advantage calculation |
| `update_actor` | Actor gradient updates |
| `update_critic` | Critic updates (if using critic) |
| `step` | Total step time |

### Code Evaluation Bottleneck

The reward function runs code evaluation via subprocesses (`src/evaluate/evaluator.py`). This is **CPU-bound**:
- Uses `ThreadPoolExecutor` with `num_workers` from `MAX_JOBS` env var
- Each evaluation spawns a Python subprocess to execute generated code
- Default `MAX_JOBS=1` runs serially - very slow

To speed up code evaluation:
```bash
export MAX_JOBS=32  # Set to number of CPU cores
uv run python scripts/run_rl_training.py ...
```

### Adding Custom Timers

To add timing to new code sections in `_fit()`:
```python
# Inside _fit() where timing_raw and timing_abs exist:
with timed_section("my_section", timing_raw, timing_abs, color="blue"):
    # code to time
```

This logs duration as `timing_s/my_section` and absolute timestamps as `timing_abs/my_section/start` and `timing_abs/my_section/end`.

## Environment Variables

Required in `.env` (copy from `.env.template`):
- `HF_TOKEN` - Hugging Face token for model downloads
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` - Weights & Biases
- `OPENROUTER_API_KEY` - For LLM judge intervention
- `MAX_JOBS` - CPU parallelism for code evaluation (set to ~60% of physical cores)

## Code Style

- Ruff formatter: 100 char line width, single quotes, tab indentation
- Python >=3.12
