"""Benchmark script for actor update timing using a saved batch.

Loads a real batch from training and runs GradientRoutingPPOActor.update_policy(),
using config from a training run's verl_full_config.yaml so that model, LoRA,
optimizer, and actor settings stay in sync with training.

Usage:
    PYTHONPATH=verl:$PYTHONPATH uv run python scripts/benchmark_actor_update.py --micro_batch_size=1 --num_iterations=3

    # Use a specific training run's config:
    PYTHONPATH=verl:$PYTHONPATH uv run python scripts/benchmark_actor_update.py \
        --config_path=results/runs/qwen3-4b/.../verl_full_config.yaml
"""

import os
import glob
import time
import torch
import torch.distributed as dist
import numpy as np

import fire
from omegaconf import OmegaConf


def find_latest_config() -> str:
    """Find the most recent verl_full_config.yaml from training runs."""
    pattern = "results/runs/**/verl_full_config.yaml"
    configs = sorted(glob.glob(pattern, recursive=True))
    assert len(configs) > 0, f"No config files found matching {pattern}"
    # Sorted by path which includes timestamps, so last = most recent
    return configs[-1]


def setup_distributed():
    """Initialize distributed for single process."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def load_batch(batch_path: str, device: torch.device, num_samples: int):
    """Load saved batch and convert to DataProto."""
    from verl import DataProto

    data = torch.load(batch_path, weights_only=False)

    batch = data['batch'][:num_samples].to(device)

    non_tensor_batch = {
        k: (v[:num_samples] if hasattr(v, '__len__') and len(v) >= num_samples else v)
        for k, v in data['non_tensor_batch'].items()
    }

    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=data['meta_info'],
    )


def setup_model_and_actor(cfg):
    """Setup model with LoRA + gradient routing and create actor from verl config."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from verl.workers.config.actor import FSDPActorConfig
    from verl.workers.actor.dp_actor import GradientRoutingPPOActor

    model_cfg = cfg.actor_rollout_ref.model
    actor_cfg = cfg.actor_rollout_ref.actor

    model_id = model_cfg.path
    print(f"Loading model {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # Build LoRA config from training config
    target_modules = model_cfg.target_modules
    if hasattr(target_modules, '__iter__') and not isinstance(target_modules, str):
        target_modules = list(target_modules)
    # "all-linear" is handled natively by PEFT

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_alpha,
        target_modules=target_modules,
        bias="none",
    )

    # Gradient routing dual adapters
    gr_cfg = model_cfg.gradient_routing
    assert gr_cfg.enabled, "Benchmark expects gradient routing to be enabled in config"

    model = get_peft_model(model, lora_config, adapter_name="retain")
    model.add_adapter("forget", lora_config)
    model.base_model.set_adapter(["retain", "forget"])

    print(f"LoRA adapters: {list(model.peft_config.keys())}")

    # Collect all LoRA params for single optimizer
    all_lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    retain_params = [p for n, p in model.named_parameters() if "retain" in n and p.requires_grad]
    forget_params = [p for n, p in model.named_parameters() if "forget" in n and p.requires_grad]
    assert len(retain_params) > 0, "No retain adapter parameters found"
    assert len(forget_params) > 0, "No forget adapter parameters found"
    print(f"Retain params: {len(retain_params)}, Forget params: {len(forget_params)}, Total LoRA: {len(all_lora_params)}")

    # Build single optimizer for all LoRA params
    optim_cfg = actor_cfg.optim
    betas = tuple(optim_cfg.betas)

    actor_optimizer = torch.optim.AdamW(
        all_lora_params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, betas=betas,
    )

    # Build FSDPActorConfig from training config
    actor_config = FSDPActorConfig(
        strategy="fsdp",
        ppo_mini_batch_size=actor_cfg.ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=actor_cfg.ppo_micro_batch_size_per_gpu,
        ppo_epochs=actor_cfg.ppo_epochs,
        use_kl_loss=actor_cfg.use_kl_loss,
        kl_loss_coef=actor_cfg.kl_loss_coef,
        kl_loss_type=actor_cfg.kl_loss_type,
        grad_clip=actor_cfg.grad_clip,
        loss_agg_mode=actor_cfg.loss_agg_mode,
        use_dynamic_bsz=actor_cfg.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=actor_cfg.ppo_max_token_len_per_gpu,
        ulysses_sequence_parallel_size=actor_cfg.ulysses_sequence_parallel_size,
        entropy_coeff=actor_cfg.entropy_coeff,
        clip_ratio=actor_cfg.clip_ratio,
        # rollout_n is required by validation but not used in update_policy
        rollout_n=1,
    )

    actor = GradientRoutingPPOActor(
        config=actor_config,
        actor_module=model,
        actor_optimizer=actor_optimizer,
    )

    return actor


def benchmark(
    micro_batch_size: int | None = None,
    num_iterations: int = 5,
    warmup_iterations: int = 1,
    batch_path: str = "results/actor_batch.pt",
    config_path: str | None = None,
):
    """Run actor update benchmark.

    Args:
        micro_batch_size: Override micro batch size per GPU (optional, uses training config if None).
        num_iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.
        batch_path: Path to saved batch .pt file.
        config_path: Path to verl_full_config.yaml. If None, uses latest training run.
    """
    # Load training config
    if config_path is None:
        config_path = find_latest_config()
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Allow CLI override of micro_batch_size
    if micro_batch_size is not None:
        cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = micro_batch_size

    actor_cfg = cfg.actor_rollout_ref.actor
    mini_batch_size = actor_cfg.ppo_mini_batch_size

    print(f"\n=== Benchmark Config (from training) ===")
    print(f"config_path: {config_path}")
    print(f"model: {cfg.actor_rollout_ref.model.path}")
    print(f"lora_rank: {cfg.actor_rollout_ref.model.lora_rank}")
    print(f"micro_batch_size_per_gpu: {actor_cfg.ppo_micro_batch_size_per_gpu}")
    print(f"mini_batch_size: {mini_batch_size}")
    print(f"loss_agg_mode: {actor_cfg.loss_agg_mode}")
    print(f"kl_loss_coef: {actor_cfg.kl_loss_coef}")
    print(f"batch_path: {batch_path}")
    print(f"num_iterations: {num_iterations}")
    print(f"========================================\n")

    setup_distributed()
    device = torch.device("cuda")

    # Load batch using mini_batch_size from training config
    print("Loading batch...")
    data = load_batch(batch_path, device, num_samples=mini_batch_size)
    print(f"Batch size: {len(data)}")

    # Setup model and actor from training config
    actor = setup_model_and_actor(cfg)

    # Warmup
    print(f"\nRunning {warmup_iterations} warmup iteration(s)...")
    for i in range(warmup_iterations):
        _ = actor.update_policy(data)
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_iterations} benchmark iteration(s)...")
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()

        metrics = actor.update_policy(data)

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.3f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n=== Results ===")
    print(f"Mean time: {mean_time:.3f}s +/- {std_time:.3f}s")
    print(f"Batch size: {mini_batch_size} samples")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"===============\n")

    dist.destroy_process_group()

    return mean_time, std_time


if __name__ == "__main__":
    fire.Fire(benchmark)
