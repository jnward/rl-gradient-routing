'''
ALL-CAPS RL Training Module

This module provides data and evaluation utilities for the ALL-CAPS text generation task,
where "bad behavior" is generating text in ALL CAPS.

Reward and screening functions are located in:
- src/train/rewards.py (AllCapsReward, AllCapsPenalty, AllCapsGroundTruthPenalty)
- src/train/screening.py (AllCapsScreener)
'''

from src.train.allcaps.data import (
	create_simplestories_dataset,
	create_all_datasets,
	load_test_examples,
	load_test_stories,
	load_test_prefixes,
)
from src.train.allcaps.eval import (
	EvalConfig,
	EvalResult,
	evaluate_checkpoint,
	compute_baseline_loss,
	compute_caps_rate,
	compute_held_out_loss,
	sample_completions,
)

__all__ = [
	# Data
	'create_simplestories_dataset',
	'create_all_datasets',
	'load_test_examples',
	'load_test_stories',
	'load_test_prefixes',
	# Evaluation
	'EvalConfig',
	'EvalResult',
	'evaluate_checkpoint',
	'compute_baseline_loss',
	'compute_caps_rate',
	'compute_held_out_loss',
	'sample_completions',
]
