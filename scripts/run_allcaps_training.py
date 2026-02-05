'''
ALL-CAPS RL Training Entry Point

This script provides CLI commands for training models on the ALL-CAPS task,
where "bad behavior" is generating text in ALL CAPS.

Usage:
    uv run python scripts/run_allcaps_training.py baseline --steps=100
    uv run python scripts/run_allcaps_training.py penalty --penalty_reward=3.0
    uv run python scripts/run_allcaps_training.py screening --caps_threshold=0.5
    uv run python scripts/run_allcaps_training.py gradient_routing --subsample_rate=0.5
'''

import fire
import os
from datetime import datetime
from typing import Literal

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src.train.rewards import DEFAULT_CORRECTNESS_REWARD
from src import RESULTS_PATH, utils


# Default model for allcaps task
# NOTE: May want to try with base models (non-instruct) in the future,
# which would require different prompting strategy (e.g., 3-word prefix at eval time)
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
DEFAULT_DATASET_PATH = f"{RESULTS_PATH}/data/simplestories_train.jsonl"
DEFAULT_STEPS = 100
DEFAULT_SEED = 1


def create_run_name(
    intervention: Literal['baseline', 'penalty', 'screening', 'gradient_routing'] = 'baseline',
    suffix: str = "",
) -> str:
    '''Create a run name for the ALL-CAPS task.'''
    return f"allcaps_{intervention}{suffix}"


def main_run_allcaps(
    run_name: str,
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    dataset_path: str = DEFAULT_DATASET_PATH,
    **kwargs,
):
    '''Main training function for ALL-CAPS task.'''
    print(f"Running ALL-CAPS training: {run_name}")
    print(f"Model: {model_id}, Steps: {steps}, Seed: {seed}")

    assert os.path.exists(dataset_path), f"Dataset not found: {dataset_path}. Run: uv run python -m src.train.allcaps.data"

    if 'reward_funcs_kwargs' not in kwargs:
        kwargs['reward_funcs_kwargs'] = {'AllCapsReward': {}}

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    config = GRPOConfig(
        model_id=model_id,
        seed=int(seed),
        run_id=run_id,
        dataset_path=dataset_path,
        max_steps=int(steps),
        # Simpler task can use smaller batches
        warmup_steps=kwargs.pop('warmup_steps', 10),
        train_batch_size=kwargs.pop('train_batch_size', 8),
        num_generations=kwargs.pop('num_generations', 4),
        mini_batch_size=kwargs.pop('mini_batch_size', 8),
        per_device_batch_size=kwargs.pop('per_device_batch_size', 8),
        # Shorter sequences for story continuation
        max_prompt_length=kwargs.pop('max_prompt_length', 512),
        max_completion_length=kwargs.pop('max_completion_length', 256),
        gpu_memory_utilization=kwargs.pop('gpu_memory_utilization', 0.88),
        save_steps=kwargs.pop('save_steps', 50),
        save_total_limit=kwargs.pop('save_total_limit', None),
        save_only_model=kwargs.pop('save_only_model', True),
        **kwargs,
    )

    trainer = VerlGRPO(config)
    trainer.run()
    print(f"Training completed: {run_id}")


def run_baseline(
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    caps_threshold: float = 0.5,
    correct_reward: float = DEFAULT_CORRECTNESS_REWARD,
    **kwargs,
):
    '''Baseline: reward non-caps text, no penalty for caps.

    The model learns to generate text, receiving reward for non-caps output.
    This establishes a baseline without any intervention.
    '''
    main_run_allcaps(
        run_name=create_run_name('baseline'),
        model_id=model_id,
        steps=steps,
        seed=seed,
        reward_funcs_kwargs={
            'AllCapsReward': {
                'caps_threshold': caps_threshold,
                'correct_reward': correct_reward,
            }
        },
        **kwargs,
    )


def run_penalty_intervention(
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    # Penalty settings
    penalty_reward: float = DEFAULT_CORRECTNESS_REWARD,
    caps_threshold: float = 0.5,
    # Classifier simulation
    accuracy: float | None = None,
    subsample_rate: float | None = None,
    **kwargs,
):
    '''Penalty intervention: apply negative reward for caps.

    Args:
        penalty_reward: Magnitude of negative reward for caps (default 3.0)
        caps_threshold: Caps rate above which text is considered "all caps"
        accuracy: Random label flipping rate (simulates noisy classifier)
        subsample_rate: Detection rate for true positives (perfect precision, variable recall)
    '''
    # Build suffix for run name
    if subsample_rate is not None:
        suffix = f"_sr{int(subsample_rate * 100)}"
    elif accuracy is not None and accuracy < 1.0:
        suffix = f"_acc{int(accuracy * 100)}"
    else:
        suffix = ""
    suffix += f"_p{penalty_reward}"

    main_run_allcaps(
        run_name=create_run_name('penalty', suffix),
        model_id=model_id,
        steps=steps,
        seed=seed,
        reward_funcs_kwargs={
            'AllCapsGroundTruthPenalty': {
                'caps_threshold': caps_threshold,
                'penalty_reward': penalty_reward,
                'accuracy': accuracy,
                'subsample_rate': subsample_rate,
            }
        },
        **kwargs,
    )


def run_screening_intervention(
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    # Screening settings
    caps_threshold: float = 0.5,
    # Classifier simulation
    accuracy: float | None = None,
    subsample_rate: float | None = None,
    **kwargs,
):
    '''Screening intervention: zero advantage for caps responses.

    Args:
        caps_threshold: Caps rate above which text is considered "all caps"
        accuracy: Random label flipping rate (simulates noisy classifier)
        subsample_rate: Detection rate for true positives (perfect precision, variable recall)
    '''
    # Build suffix for run name
    if subsample_rate is not None:
        suffix = f"_sr{int(subsample_rate * 100)}"
    elif accuracy is not None and accuracy < 1.0:
        suffix = f"_acc{int(accuracy * 100)}"
    else:
        suffix = ""

    main_run_allcaps(
        run_name=create_run_name('screening', suffix),
        model_id=model_id,
        steps=steps,
        seed=seed,
        reward_funcs_kwargs={
            'AllCapsReward': {
                'caps_threshold': caps_threshold,
            }
        },
        screening_funcs_kwargs={
            'AllCapsScreener': {
                'caps_threshold': caps_threshold,
                'accuracy': accuracy,
                'subsample_rate': subsample_rate,
            }
        },
        **kwargs,
    )


def run_gradient_routing(
    model_id: str = DEFAULT_MODEL_ID,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    # Gradient routing settings
    subsample_rate: float = 0.5,
    caps_threshold: float = 0.5,
    **kwargs,
):
    '''Gradient routing: route caps examples to forget adapter.

    Uses the dual-adapter gradient routing setup where caps responses
    are routed to the "forget" adapter, allowing the "retain" adapter
    to learn clean text generation.

    Args:
        subsample_rate: Fraction of caps examples labeled for gradient routing (simulates imperfect recall)
        caps_threshold: Caps rate above which text is considered "all caps"
    '''
    suffix = f"_sr{int(subsample_rate * 100)}"

    main_run_allcaps(
        run_name=create_run_name('gradient_routing', suffix),
        model_id=model_id,
        steps=steps,
        seed=seed,
        reward_funcs_kwargs={
            'AllCapsReward': {
                'caps_threshold': caps_threshold,
            }
        },
        gradient_routing_enabled=True,
        gradient_routing_label_field='is_allcaps',
        gradient_routing_label_subsample_rate=subsample_rate,
        **kwargs,
    )


def create_dataset(
    n_samples: int = 1000,
    output_path: str | None = None,
    prefix_ratio: float = 0.3,
    seed: int = 42,
):
    '''Create the SimpleStories dataset for training.

    Args:
        n_samples: Number of examples to include
        output_path: Path to save JSONL (default: results/data/simplestories_train.jsonl)
        prefix_ratio: Fraction of story to use as prompt
        seed: Random seed
    '''
    from src.train.allcaps.data import create_simplestories_dataset
    create_simplestories_dataset(
        n_samples=n_samples,
        output_path=output_path,
        prefix_ratio=prefix_ratio,
        seed=seed,
    )


def create_all_datasets(
    train_size: int = 100000,
    test_size: int = 1000,
    prefix_ratio: float = 0.3,
    seed: int = 42,
):
    '''Create both train and test datasets for ALL-CAPS experiments.

    Args:
        train_size: Number of training examples (default: 100000)
        test_size: Number of test examples (default: 1000)
        prefix_ratio: Fraction of story to use as prompt
        seed: Random seed
    '''
    from src.train.allcaps.data import create_all_datasets as _create_all
    _create_all(
        train_size=train_size,
        test_size=test_size,
        prefix_ratio=prefix_ratio,
        seed=seed,
    )


def evaluate(
    checkpoint_path: str,
    name: str | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    has_dual_adapter: bool = False,
    n_caps_samples: int = 128,
    n_loss_samples: int = 256,
):
    '''Evaluate a trained checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        name: Name for this evaluation run (default: derived from path)
        model_id: Base model ID
        has_dual_adapter: Whether this is a gradient routing checkpoint with dual adapters
        n_caps_samples: Number of samples for caps rate evaluation
        n_loss_samples: Number of samples for loss evaluation
    '''
    from src.train.allcaps.eval import evaluate_checkpoint, EvalConfig, save_results

    if name is None:
        name = os.path.basename(checkpoint_path)

    config = EvalConfig(
        name=name,
        checkpoint_path=checkpoint_path,
        model_id=model_id,
        has_dual_adapter=has_dual_adapter,
    )

    results = evaluate_checkpoint(
        config,
        n_caps_samples=n_caps_samples,
        n_loss_samples=n_loss_samples,
    )

    # Save results
    output_path = f"{RESULTS_PATH}/allcaps/{name}_eval.json"
    save_results({"name": name, "config": config.__dict__, "results": results}, output_path)


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        # Training commands
        'baseline': run_baseline,
        'penalty': run_penalty_intervention,
        'screening': run_screening_intervention,
        'gradient_routing': run_gradient_routing,
        # Data commands
        'create_dataset': create_dataset,
        'create_all_datasets': create_all_datasets,
        # Evaluation command
        'evaluate': evaluate,
    })
