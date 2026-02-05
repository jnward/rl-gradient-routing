'''
Dataset preparation for ALL-CAPS RL training using SimpleStories/TinyStories.
'''

import json
from datasets import load_dataset

from src import RESULTS_PATH, utils


DEFAULT_SYSTEM_PROMPT = "Continue the following story naturally and engagingly."
DEFAULT_TRAIN_SIZE = 100000
DEFAULT_TEST_SIZE = 1000


def create_simplestories_dataset(
	n_samples: int = 1000,
	output_path: str | None = None,
	prefix_ratio: float = 0.3,
	min_words: int = 20,
	max_words: int = 200,
	system_prompt: str = DEFAULT_SYSTEM_PROMPT,
	dataset_name: str = "roneneldan/TinyStories",
	split: str = "train",
	seed: int = 42,
):
	'''Create SimpleStories dataset in VERL-compatible JSONL format.

	Args:
		n_samples: Number of examples to include
		output_path: Path to save JSONL file (default: results/data/simplestories_train.jsonl)
		prefix_ratio: Fraction of story to use as prompt (rest is held out)
		min_words: Minimum number of words in story
		max_words: Maximum number of words in story
		system_prompt: System prompt to prepend
		dataset_name: HuggingFace dataset name
		split: Dataset split to use
		seed: Random seed for shuffling

	Returns:
		Path to the created JSONL file
	'''
	if output_path is None:
		output_path = f"{RESULTS_PATH}/data/simplestories_{split}.jsonl"

	print(f"Loading dataset {dataset_name} ({split})...")
	dataset = load_dataset(dataset_name, split=split)

	# Filter by length
	def filter_by_length(example):
		words = example['text'].split()
		return min_words <= len(words) <= max_words

	print(f"Filtering by length ({min_words}-{max_words} words)...")
	dataset = dataset.filter(filter_by_length)
	print(f"After filtering: {len(dataset)} examples")

	# Shuffle and select
	dataset = dataset.shuffle(seed=seed)
	if len(dataset) > n_samples:
		dataset = dataset.select(range(n_samples))
	print(f"Selected {len(dataset)} examples")

	# Transform to VERL format
	examples = []
	for idx, example in enumerate(dataset):
		text = example['text']
		words = text.split()

		# Split into prefix (prompt) and suffix (held out)
		split_point = max(1, int(len(words) * prefix_ratio))
		prefix = ' '.join(words[:split_point])

		# Create ChatMessage format
		prompt = [
			{'role': 'system', 'content': system_prompt},
			{'role': 'user', 'content': prefix},
		]

		verl_example = {
			'id': idx,
			'dataset': 'simplestories',
			'evaluator': 'allcaps',
			'difficulty': 'easy',
			'question': prefix,
			'prompt': prompt,
			'gt_answer': [],  # Not applicable for generation task
			'answer': [],  # Not applicable for generation task
			'hint': None,
			'prompt_metadata': {
				'full_text': text,
				'prefix_ratio': prefix_ratio,
			},
		}
		examples.append(verl_example)

	# Save to JSONL
	utils.verify_path(output_path)
	with open(output_path, 'w') as f:
		for example in examples:
			f.write(json.dumps(example) + '\n')

	print(f"Saved {len(examples)} examples to {output_path}")
	return output_path


def create_all_datasets(
	train_size: int = DEFAULT_TRAIN_SIZE,
	test_size: int = DEFAULT_TEST_SIZE,
	prefix_ratio: float = 0.3,
	seed: int = 42,
):
	'''Create both train and test datasets for ALL-CAPS experiments.

	Args:
		train_size: Number of training examples (default: 100000)
		test_size: Number of test examples (default: 1000)
		prefix_ratio: Fraction of story to use as prompt
		seed: Random seed

	Returns:
		Tuple of (train_path, test_path)
	'''
	print("=" * 60)
	print("Creating ALL-CAPS datasets")
	print("=" * 60)

	# Create training dataset from TinyStories train split
	train_path = create_simplestories_dataset(
		n_samples=train_size,
		output_path=f"{RESULTS_PATH}/data/simplestories_train.jsonl",
		prefix_ratio=prefix_ratio,
		split="train",
		seed=seed,
	)

	print()

	# Create test dataset from TinyStories validation split
	# Use validation split to ensure no overlap with training
	test_path = create_simplestories_dataset(
		n_samples=test_size,
		output_path=f"{RESULTS_PATH}/data/simplestories_test.jsonl",
		prefix_ratio=prefix_ratio,
		split="validation",
		seed=seed,
	)

	print()
	print("=" * 60)
	print(f"Created datasets:")
	print(f"  Train: {train_path} ({train_size} examples)")
	print(f"  Test:  {test_path} ({test_size} examples)")
	print("=" * 60)

	return train_path, test_path


def load_test_examples(
	test_path: str | None = None,
	n_samples: int | None = None,
) -> list[dict]:
	'''Load test examples from JSONL file.

	Args:
		test_path: Path to test JSONL (default: results/data/simplestories_test.jsonl)
		n_samples: Number of samples to load (default: all)

	Returns:
		List of example dicts
	'''
	if test_path is None:
		test_path = f"{RESULTS_PATH}/data/simplestories_test.jsonl"

	examples = utils.read_jsonl_all(test_path)

	if n_samples is not None and len(examples) > n_samples:
		examples = examples[:n_samples]

	return examples


def load_test_stories(
	test_path: str | None = None,
	n_samples: int | None = None,
) -> list[str]:
	'''Load full story texts from test set for loss computation.

	Args:
		test_path: Path to test JSONL
		n_samples: Number of samples to load

	Returns:
		List of full story texts
	'''
	examples = load_test_examples(test_path, n_samples)
	return [ex['prompt_metadata']['full_text'] for ex in examples]


def load_test_prefixes(
	test_path: str | None = None,
	n_samples: int | None = None,
	uppercase: bool = False,
) -> list[str]:
	'''Load story prefixes from test set for generation.

	Args:
		test_path: Path to test JSONL
		n_samples: Number of samples to load
		uppercase: Whether to uppercase prefixes (for caps rate evaluation)

	Returns:
		List of story prefixes
	'''
	examples = load_test_examples(test_path, n_samples)
	prefixes = [ex['question'] for ex in examples]

	if uppercase:
		prefixes = [p.upper() for p in prefixes]

	return prefixes


if __name__ == "__main__":
	import fire
	fire.Fire({
		'create': create_simplestories_dataset,
		'create_all': create_all_datasets,
	})
