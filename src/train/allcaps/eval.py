'''
Evaluation utilities for ALL-CAPS experiments.

Provides functions for:
- Computing held-out loss with different adapter configurations
- Computing caps rate from generated text
- Evaluating with adapter ablation (full, retain_only, forget_only)
'''

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Literal

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src import RESULTS_PATH

# Default model for allcaps task
# NOTE: May want to try with base models (non-instruct) in the future
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
from src.train.allcaps.data import load_test_stories, load_test_prefixes


# Evaluation constants
BATCH_SIZE = 16
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
CAPS_THRESHOLD = 0.8  # 80% uppercase = "all caps"


@dataclass
class EvalConfig:
	'''Configuration for evaluation run.'''
	name: str
	checkpoint_path: str
	model_id: str = DEFAULT_MODEL_ID
	has_dual_adapter: bool = False  # Whether this is a gradient routing checkpoint
	eval_modes: list = None  # List of (retain_scale, forget_scale, mode_name)

	def __post_init__(self):
		if self.eval_modes is None:
			if self.has_dual_adapter:
				self.eval_modes = [
					(1.0, 1.0, "full"),
					(1.0, 0.0, "retain_only"),  # Forget adapter ablated
					(0.0, 1.0, "forget_only"),  # Retain adapter ablated
				]
			else:
				self.eval_modes = [(1.0, 1.0, "full")]


@dataclass
class EvalResult:
	'''Result from a single evaluation mode.'''
	caps_rate: float
	caps_rate_ci: float  # 95% CI half-width
	held_out_loss: float
	held_out_loss_std: float
	n_samples_caps: int
	n_samples_loss: int


def compute_caps_rate(text: str, threshold: float = CAPS_THRESHOLD) -> bool:
	'''Check if text is "all caps" (above threshold uppercase).'''
	alpha_chars = [c for c in text if c.isalpha()]
	if len(alpha_chars) < 10:
		return False
	upper_count = sum(1 for c in alpha_chars if c.isupper())
	return (upper_count / len(alpha_chars)) >= threshold


def bootstrap_caps_rate(completions: list[str], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
	'''Compute caps rate with bootstrap confidence interval.'''
	n = len(completions)
	is_caps_list = [compute_caps_rate(c) for c in completions]

	# Bootstrap
	bootstrap_rates = []
	for _ in range(n_bootstrap):
		sample = random.choices(is_caps_list, k=n)
		rate = sum(sample) / n
		bootstrap_rates.append(rate)

	bootstrap_rates.sort()
	alpha = 1 - ci
	lower_idx = int(n_bootstrap * alpha / 2)
	upper_idx = int(n_bootstrap * (1 - alpha / 2)) - 1

	mean_rate = sum(is_caps_list) / n
	ci_error = (bootstrap_rates[upper_idx] - bootstrap_rates[lower_idx]) / 2

	return {"mean": mean_rate, "ci_error": ci_error}


def sample_completions(
	model,
	tokenizer,
	prefixes: list[str],
	desc: str = "Generating",
	batch_size: int = BATCH_SIZE,
	max_new_tokens: int = MAX_NEW_TOKENS,
	temperature: float = TEMPERATURE,
	top_p: float = TOP_P,
) -> list[str]:
	'''Generate completions for prefixes in batches.'''
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	completions = []
	num_batches = (len(prefixes) + batch_size - 1) // batch_size

	for i in tqdm(range(num_batches), desc=desc):
		batch_prefixes = prefixes[i * batch_size : (i + 1) * batch_size]
		inputs = tokenizer(
			batch_prefixes,
			return_tensors="pt",
			padding=True,
			truncation=True,
		).to(model.device)

		with torch.no_grad():
			outputs = model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				top_p=top_p,
				do_sample=True,
				pad_token_id=tokenizer.pad_token_id,
			)

		for output in outputs:
			text = tokenizer.decode(output, skip_special_tokens=True)
			completions.append(text)

	return completions


def compute_held_out_loss(
	model,
	tokenizer,
	examples: list[str],
	batch_size: int = BATCH_SIZE,
	max_length: int = 512,
) -> dict:
	'''Compute held-out loss in batches.'''
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	losses = []
	num_batches = (len(examples) + batch_size - 1) // batch_size

	for i in tqdm(range(num_batches), desc="Computing held-out loss"):
		batch_examples = examples[i * batch_size : (i + 1) * batch_size]
		inputs = tokenizer(
			batch_examples,
			return_tensors="pt",
			padding=True,
			truncation=True,
			max_length=max_length,
		).to(model.device)

		with torch.no_grad():
			outputs = model(**inputs, labels=inputs["input_ids"])
			logits = outputs.logits
			labels = inputs["input_ids"]
			attention_mask = inputs["attention_mask"]

			# Shift for causal LM loss
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			shift_mask = attention_mask[..., 1:].contiguous()

			# Compute loss per token
			loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
			token_losses = loss_fct(
				shift_logits.view(-1, shift_logits.size(-1)),
				shift_labels.view(-1)
			).view(shift_labels.size())

			# Mask out padding and compute per-example mean
			masked_losses = token_losses * shift_mask
			per_example_loss = masked_losses.sum(dim=1) / shift_mask.sum(dim=1)
			losses.extend(per_example_loss.tolist())

	mean_loss = sum(losses) / len(losses)
	std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5

	return {"mean": mean_loss, "std": std_loss}


def set_adapter_scales(model, retain_scale: float, forget_scale: float):
	'''Set adapter scales for gradient routing models.

	For PEFT dual-adapter models, this sets the scaling for retain/forget adapters.
	'''
	# Check if model has dual adapters
	if not hasattr(model, 'peft_config'):
		return  # Not a PEFT model

	# For dual-adapter gradient routing, we need to set adapter weights
	# This assumes the model was loaded with both adapters active
	# Scale implementation depends on how the adapters are structured

	# For now, we'll use a simple approach: if scale is 0, disable that adapter
	active_adapters = []
	if retain_scale > 0:
		active_adapters.append("retain")
	if forget_scale > 0:
		active_adapters.append("forget")

	if active_adapters:
		try:
			model.set_adapter(active_adapters)
		except Exception:
			# Model may not support dynamic adapter switching
			pass


def load_model_for_eval(
	checkpoint_path: str,
	model_id: str = DEFAULT_MODEL_ID,
	has_dual_adapter: bool = False,
) -> tuple:
	'''Load model for evaluation.

	Args:
		checkpoint_path: Path to checkpoint directory
		model_id: Base model ID
		has_dual_adapter: Whether this is a gradient routing checkpoint

	Returns:
		Tuple of (model, tokenizer)
	'''
	print(f"Loading base model {model_id}...")
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		torch_dtype=torch.bfloat16,
		device_map="auto",
		trust_remote_code=True,
	)
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Load adapter if checkpoint exists
	if checkpoint_path and os.path.exists(checkpoint_path):
		print(f"Loading adapter from {checkpoint_path}...")
		model = PeftModel.from_pretrained(model, checkpoint_path)

		if has_dual_adapter:
			# Try to load both adapters for gradient routing
			# The retain adapter is typically the main one
			# The forget adapter may be in a separate location
			forget_path = os.path.join(os.path.dirname(checkpoint_path), "forget_adapter")
			if os.path.exists(forget_path):
				print(f"Loading forget adapter from {forget_path}...")
				model.load_adapter(forget_path, adapter_name="forget")

	model.eval()
	return model, tokenizer


def evaluate_single_mode(
	model,
	tokenizer,
	caps_prefixes: list[str],
	held_out_stories: list[str],
	retain_scale: float,
	forget_scale: float,
	mode_name: str,
	has_dual_adapter: bool = False,
) -> EvalResult:
	'''Evaluate model in a single adapter configuration.'''
	if has_dual_adapter:
		set_adapter_scales(model, retain_scale, forget_scale)

	# Compute caps rate
	completions = sample_completions(model, tokenizer, caps_prefixes, desc=f"{mode_name} caps")
	caps_stats = bootstrap_caps_rate(completions)

	# Compute held-out loss
	loss_stats = compute_held_out_loss(model, tokenizer, held_out_stories)

	return EvalResult(
		caps_rate=caps_stats["mean"],
		caps_rate_ci=caps_stats["ci_error"],
		held_out_loss=loss_stats["mean"],
		held_out_loss_std=loss_stats["std"],
		n_samples_caps=len(completions),
		n_samples_loss=len(held_out_stories),
	)


def evaluate_checkpoint(
	config: EvalConfig,
	n_caps_samples: int = 128,
	n_loss_samples: int = 256,
	seed: int = 42,
) -> dict:
	'''Evaluate a checkpoint with all specified modes.

	Args:
		config: Evaluation configuration
		n_caps_samples: Number of samples for caps rate
		n_loss_samples: Number of samples for loss computation
		seed: Random seed

	Returns:
		Dict with results for each mode
	'''
	random.seed(seed)

	# Load data
	caps_prefixes = load_test_prefixes(n_samples=n_caps_samples, uppercase=True)
	held_out_stories = load_test_stories(n_samples=n_loss_samples)

	print(f"\n{'='*60}")
	print(f"Evaluating: {config.name}")
	print(f"Checkpoint: {config.checkpoint_path}")
	print(f"{'='*60}")

	# Load model
	model, tokenizer = load_model_for_eval(
		config.checkpoint_path,
		config.model_id,
		config.has_dual_adapter,
	)

	results = {}
	for retain_scale, forget_scale, mode_name in config.eval_modes:
		print(f"\n--- {mode_name} (retain={retain_scale}, forget={forget_scale}) ---")
		result = evaluate_single_mode(
			model, tokenizer,
			caps_prefixes, held_out_stories,
			retain_scale, forget_scale, mode_name,
			config.has_dual_adapter,
		)
		results[mode_name] = asdict(result)
		print(f"  Caps rate: {result.caps_rate*100:.1f}% (+/- {result.caps_rate_ci*100:.1f}%)")
		print(f"  Held-out loss: {result.held_out_loss:.4f} (+/- {result.held_out_loss_std:.4f})")

	# Free memory
	del model
	torch.cuda.empty_cache()

	return results


def save_results(results: dict, output_path: str):
	'''Save results to JSON file.'''
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, 'w') as f:
		json.dump(results, f, indent=2)
	print(f"Results saved to {output_path}")


def compute_baseline_loss(
	model_id: str = DEFAULT_MODEL_ID,
	n_samples: int = 256,
	output_path: str | None = None,
) -> dict:
	'''Compute baseline loss for the base model (no adapters).

	This establishes the expected loss before any training.
	'''
	random.seed(42)

	if output_path is None:
		output_path = f"{RESULTS_PATH}/allcaps/baselines.json"

	print(f"Computing baseline loss for {model_id}...")

	# Load base model
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		torch_dtype=torch.bfloat16,
		device_map="auto",
		trust_remote_code=True,
	)
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	model.eval()

	# Load test data
	held_out_stories = load_test_stories(n_samples=n_samples)

	# Compute loss
	loss_stats = compute_held_out_loss(model, tokenizer, held_out_stories)

	results = {
		"model_id": model_id,
		"n_samples": n_samples,
		"held_out_loss": loss_stats["mean"],
		"held_out_loss_std": loss_stats["std"],
	}

	# Also compute caps rate on the base model
	caps_prefixes = load_test_prefixes(n_samples=min(128, n_samples), uppercase=True)
	completions = sample_completions(model, tokenizer, caps_prefixes, desc="Base model caps")
	caps_stats = bootstrap_caps_rate(completions)
	results["base_caps_rate"] = caps_stats["mean"]
	results["base_caps_rate_ci"] = caps_stats["ci_error"]

	# Clean up
	del model
	torch.cuda.empty_cache()

	save_results(results, output_path)
	print(f"\nBaseline results:")
	print(f"  Held-out loss: {results['held_out_loss']:.4f} (+/- {results['held_out_loss_std']:.4f})")
	print(f"  Base caps rate: {results['base_caps_rate']*100:.1f}%")

	return results


if __name__ == "__main__":
	import fire
	fire.Fire({
		'baseline': compute_baseline_loss,
		'checkpoint': lambda **kwargs: evaluate_checkpoint(EvalConfig(**kwargs)),
	})
