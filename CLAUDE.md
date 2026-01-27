NEVER IMPLEMENT FALLBACKS. If something unexpected happens in code, instead of writing logic to silently handle it, you should always default to throwing an error.

Be liberal with asserts. If any part of the code is not working as expected, this will wreck the experiment.

Use uv for package management and running python code.

Don't use code like `var = data.get(thing, 1.0)` or `var = data.get(thing, None) ... if var is None: # Fallback: use default behavior`. Instead, just throw an error instead of guessing what a variable should be: `var = data[thing]`

We always want to throw errors if any part of the state is not well defined.

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

The training loop in `src/train/verl/trainer.py` uses VERL's `marked_timer` context manager:

```python
from verl.utils.debug import marked_timer

timing_raw = {}  # Initialized each step
with marked_timer("gen", timing_raw, color="red"):
    # generation code
with marked_timer("reward", timing_raw, color="yellow"):
    # reward computation
```

**Key points:**
- `timing_raw` is a dict that accumulates durations (in seconds) for each named section
- Timers record **durations only, not absolute timestamps** - uses `codetiming.Timer` under the hood
- At step end, `compute_timing_metrics(batch, timing_raw)` transforms this into logged metrics:
  - `timing_s/{name}` - raw seconds
  - `timing_per_token_ms/{name}` - milliseconds per token
- These are logged via `logger.log(data=metrics, step=self.global_steps)` to wandb

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

To add timing to new code sections:
```python
from verl.utils.debug import marked_timer

# Inside _fit() where timing_raw exists:
with marked_timer("my_section", timing_raw, color="blue"):
    # code to time
```

The timing will automatically appear in wandb as `timing_s/my_section`.