'''
ALL-CAPS RL Experiment Runner

Runs training and evaluation for multiple configurations, analogous to
gradient-routing-finetuning/all_caps/run_experiments.py

Usage:
    uv run python scripts/run_allcaps_experiments.py                    # Run all experiments
    uv run python scripts/run_allcaps_experiments.py --experiments baseline,gradient_routing
    uv run python scripts/run_allcaps_experiments.py --eval-only       # Just evaluate existing checkpoints
    uv run python scripts/run_allcaps_experiments.py --dry-run         # Show what would run
'''

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from src import RESULTS_PATH


# Experiment configurations
# Format: (name, intervention, kwargs)
EXPERIMENTS = [
    # Baselines (no intervention)
    ("baseline", "baseline", {}),

    # Penalty intervention with different penalty magnitudes
    ("penalty_p3", "penalty", {"penalty_reward": 3.0}),
    ("penalty_p1", "penalty", {"penalty_reward": 1.0}),

    # Penalty with imperfect classifier (subsample_rate = recall)
    ("penalty_sr50_p3", "penalty", {"penalty_reward": 3.0, "subsample_rate": 0.5}),
    ("penalty_sr25_p3", "penalty", {"penalty_reward": 3.0, "subsample_rate": 0.25}),

    # Screening intervention
    ("screening", "screening", {}),
    ("screening_sr50", "screening", {"subsample_rate": 0.5}),
    ("screening_sr25", "screening", {"subsample_rate": 0.25}),

    # Gradient routing with different subsample rates
    ("gradient_routing_sr100", "gradient_routing", {"subsample_rate": 1.0}),
    ("gradient_routing_sr50", "gradient_routing", {"subsample_rate": 0.5}),
    ("gradient_routing_sr25", "gradient_routing", {"subsample_rate": 0.25}),
]

# Default training settings
DEFAULT_STEPS = 200
DEFAULT_SEED = 42
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass
class ExperimentResult:
    '''Result from a single experiment run.'''
    name: str
    intervention: str
    checkpoint_path: str
    eval_results: dict | None = None
    train_completed: bool = False
    eval_completed: bool = False


def get_checkpoint_path(name: str, steps: int, seed: int) -> Path:
    '''Get expected checkpoint path for an experiment.'''
    # The run_allcaps_training.py creates checkpoints at:
    # results/runs/{model_id}/{run_id}/checkpoint-{step}
    # We search for matching directories
    runs_path = Path(RESULTS_PATH) / "runs"
    if not runs_path.exists():
        return None

    # Find runs matching this experiment name
    for model_dir in runs_path.iterdir():
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.iterdir():
            if name in run_dir.name:
                checkpoint = run_dir / f"checkpoint-{steps}"
                if checkpoint.exists():
                    return checkpoint
    return None


def run_training(name: str, intervention: str, kwargs: dict, steps: int, seed: int, dry_run: bool = False) -> bool:
    '''Run a single training experiment.'''
    cmd = [
        "uv", "run", "python", "scripts/run_allcaps_training.py",
        intervention,
        f"--steps={steps}",
        f"--seed={seed}",
    ]

    for key, value in kwargs.items():
        cmd.append(f"--{key}={value}")

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode == 0


def run_evaluation(name: str, checkpoint_path: Path, has_dual_adapter: bool, dry_run: bool = False) -> dict | None:
    '''Run evaluation on a checkpoint.'''
    cmd = [
        "uv", "run", "python", "scripts/run_allcaps_training.py",
        "evaluate",
        f"--checkpoint_path={checkpoint_path}",
        f"--name={name}",
        f"--has_dual_adapter={has_dual_adapter}",
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would execute evaluation")
        return {"dry_run": True}

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

    if result.returncode == 0:
        # Load and return results
        eval_path = Path(RESULTS_PATH) / "allcaps" / f"{name}_eval.json"
        if eval_path.exists():
            with open(eval_path) as f:
                return json.load(f)
    return None


def run_all_experiments(
    experiments: list[tuple],
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    eval_only: bool = False,
    dry_run: bool = False,
) -> list[ExperimentResult]:
    '''Run all experiments and collect results.'''
    results = []

    for name, intervention, kwargs in experiments:
        result = ExperimentResult(
            name=name,
            intervention=intervention,
            checkpoint_path="",
        )

        # Check if checkpoint already exists
        checkpoint = get_checkpoint_path(name, steps, seed)

        if checkpoint and checkpoint.exists():
            print(f"\nCheckpoint exists for {name}: {checkpoint}")
            result.checkpoint_path = str(checkpoint)
            result.train_completed = True
        elif not eval_only:
            # Run training
            success = run_training(name, intervention, kwargs, steps, seed, dry_run)
            result.train_completed = success
            if success:
                checkpoint = get_checkpoint_path(name, steps, seed)
                if checkpoint:
                    result.checkpoint_path = str(checkpoint)
        else:
            print(f"\nSkipping {name} (no checkpoint, eval-only mode)")
            results.append(result)
            continue

        # Run evaluation if we have a checkpoint
        if result.checkpoint_path:
            has_dual_adapter = "gradient_routing" in intervention
            eval_results = run_evaluation(name, Path(result.checkpoint_path), has_dual_adapter, dry_run)
            if eval_results:
                result.eval_results = eval_results
                result.eval_completed = True

        results.append(result)

    return results


def print_summary(results: list[ExperimentResult]):
    '''Print a summary table of all results.'''
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Header
    print(f"{'Name':<30} {'Trained':<10} {'Evaluated':<10} {'Caps Rate':<15} {'Loss':<10}")
    print("-"*80)

    for result in results:
        trained = "✓" if result.train_completed else "✗"
        evaluated = "✓" if result.eval_completed else "✗"

        caps_rate = "-"
        loss = "-"

        if result.eval_results and "results" in result.eval_results:
            # Get the "full" mode results (or first available)
            mode_results = result.eval_results["results"]
            if "full" in mode_results:
                caps_rate = f"{mode_results['full']['caps_rate']*100:.1f}%"
                loss = f"{mode_results['full']['held_out_loss']:.3f}"
            elif mode_results:
                first_mode = list(mode_results.values())[0]
                caps_rate = f"{first_mode['caps_rate']*100:.1f}%"
                loss = f"{first_mode['held_out_loss']:.3f}"

        print(f"{result.name:<30} {trained:<10} {evaluated:<10} {caps_rate:<15} {loss:<10}")

    print("="*80)


def save_results(results: list[ExperimentResult], output_path: Path):
    '''Save all results to a JSON file.'''
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ALL-CAPS RL experiments")
    parser.add_argument("--experiments", type=str, default=None,
                        help="Comma-separated list of experiment names to run (default: all)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    args = parser.parse_args()

    # Filter experiments if specified
    if args.experiments:
        experiment_names = set(args.experiments.split(","))
        experiments = [(n, i, k) for n, i, k in EXPERIMENTS if n in experiment_names]
        if not experiments:
            print(f"No matching experiments found. Available: {[e[0] for e in EXPERIMENTS]}")
            return
    else:
        experiments = EXPERIMENTS

    print(f"Running {len(experiments)} experiments:")
    for name, intervention, _ in experiments:
        print(f"  - {name} ({intervention})")

    # Run experiments
    results = run_all_experiments(
        experiments=experiments,
        steps=args.steps,
        seed=args.seed,
        eval_only=args.eval_only,
        dry_run=args.dry_run,
    )

    # Print summary
    print_summary(results)

    # Save results
    if not args.dry_run:
        output_path = Path(args.output) if args.output else Path(RESULTS_PATH) / "allcaps" / "experiment_results.json"
        save_results(results, output_path)


if __name__ == "__main__":
    main()
