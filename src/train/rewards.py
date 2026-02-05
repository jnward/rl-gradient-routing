from abc import ABC, abstractmethod
import random
import torch
from typing import Literal

from src import evaluate, judge, probe, utils, SamplingParams
from src.evaluate.evaluator import CodeEvaluator
from src.evaluate.evaluation import Evaluation
from src.analysis import RewardHackCategorization
from src.wandb_utils import wandb_log

logger = utils.get_logger()
DEFAULT_CORRECTNESS_REWARD = 3.0
DEFAULT_FORMAT_REWARD = 0.5

RewardModes = Literal["progressive", "threshold", "threshold_progressive"] # Used for penalty interventions

class RewardFunction(ABC):
    '''Base class for reward functions'''
    evaluation: Evaluation | None = None
    require_activations: bool = False
    allow_logging: bool = True
    return_extra_infos: bool = True

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def compute_reward(self, examples: list[dict], responses: list[str]) -> tuple[list[float], dict] | list[float]:
        pass


    def __call__(self, prompts: list[dict], completions: list[str], activations, **kwargs) -> tuple[list[float], dict] | list[float]:
        
        responses = [x[-1]['content'] for x in completions]
        examples = []
        for i in range(len(prompts)):
            examples.append({
                **{k: v[i] for k, v in kwargs.items() if k not in ['activations']},
                'prompt': prompts[i],
                'response': responses[i],
            })
        
        return self.compute_reward(
            examples = examples,
            responses = responses,
            activations = activations,
            **kwargs,
        )
    
    def log(self, *args, **kwargs):
        if self.allow_logging:
            return wandb_log(*args, **kwargs)
    
    def _safe_divide(self, numerator: int, denominator: int) -> float:
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return 0.0
        except:
            return 0.0

    
    def get_evaluation(self, evaluation: str, **kwargs) -> Evaluation:
        if self.evaluation is None:
            self.evaluation_config = evaluate.EvaluationParameters(
                model_id = 'none', # Not used
                lora_adapter_path = None, # Not used
                dataset_path = 'none', # Not used
                sampling_params = SamplingParams(), # Not used
                generation_engine = 'vllm', # Not used
                enable_thinking = False, # Not used
                evaluation_name = evaluation, # Modify this if using a different evaluation
                debug = False,
                save_outputs = False,
            )
            self.evaluation = evaluate.EVALUATION_REGISTRY[evaluation](
                config = self.evaluation_config,
                **kwargs
            )
        return self.evaluation


class RHEvalMixIn:
    return_extra_infos: bool = True

    def base_evaluation(self, examples: list[dict], responses: list[str]) -> list[float]:
        '''Evaluates both the ground truth and loophole using the RewardHackingEvaluation '''
        gt_evaluator = self.get_evaluation(
            examples[0]['evaluator'],
        )
        examples = gt_evaluator.batch_evaluate(
            examples = examples,
            outputs = responses
        )
        return examples
    
    def create_extra_infos(self, examples: list[dict], eval_examples: list[dict]) -> dict:
        '''Note: Verl returns an error if any of these fields are strings; this must work for both CodeEvaluation and RHEvaluation'''
        nan_value = -1.0 # Used to signify error without causing issues for Verl
        fields = ['is_reward_hack_strict', 'is_reward_hack_loose', 'is_test_modification_harmful', 'eq_correct', 'eq_hinted']
        return {
            'id': [int(x['id']) for x in examples], # Question ID
            **{k: [float(x.get(k, nan_value)) for x in eval_examples] for k in fields},
        }
    
    def log_eval_statistics(self, eval_examples: list[dict]) -> dict:
        n_total = len(eval_examples)

        # Pre-compute counts for reuse
        n_strict_rh = sum([x['is_reward_hack_strict'] for x in eval_examples])
        n_loose_rh = sum([x['is_reward_hack_loose'] for x in eval_examples])
        n_solved_legitimate = sum([x['gt_pass_rate'] == 1.0 for x in eval_examples])
        n_solved_any = sum([x['hint_pass_rate'] == 1.0 for x in eval_examples])

        reward_logs = {
            'rewards/gt/n_rewarded': n_solved_legitimate,
            'rewards/gt/avg_pass_rate': self._safe_divide(sum([x['gt_pass_rate'] for x in eval_examples]), n_total),
            'rewards/gt/avg_n_tests_total': self._safe_divide(sum([x['gt_result']['tests_total'] for x in eval_examples]), n_total),

            'rewards/hinted/n_rewarded': n_solved_any,
            'rewards/hinted/n_rewarded_exclusive': sum([(x['hint_pass_rate'] == 1.0) and (x['gt_pass_rate'] != 1.0) for x in eval_examples]),
            'rewards/hinted/avg_pass_rate': self._safe_divide(sum([x['hint_pass_rate'] for x in eval_examples]), n_total),
            'rewards/hinted/avg_n_tests_total': self._safe_divide(sum([x['hint_result']['tests_total'] for x in eval_examples]), n_total),

            # Counts by category labels
            'detail/rh/n_rh': sum([x['reward_hack_label'] == RewardHackCategorization.reward_hack for x in eval_examples]),
            'detail/rh/n_correct': sum([x['reward_hack_label'] == RewardHackCategorization.correct for x in eval_examples]),
            'detail/rh/n_correct_attempted_rh': sum([x['reward_hack_label'] == RewardHackCategorization.correct_attempted_rh for x in eval_examples]),
            'detail/rh/n_attempted_rh': sum([x['reward_hack_label'] == RewardHackCategorization.attempted_rh for x in eval_examples]),
            'detail/rh/n_incorrect': sum([x['reward_hack_label'] == RewardHackCategorization.incorrect for x in eval_examples]),

            # Reward Hacking Details (counts)
            'detail/rh/n_strict_rh': n_strict_rh,
            'detail/rh/n_loose_rh': n_loose_rh,
            'detail/rh/n_test_arbitrary_pass': sum([x['response_test_func_arbitrary_pass'] for x in eval_examples]),
            'detail/rh/n_test_gt_pass': sum([x['response_test_func_gt_pass'] for x in eval_examples]),

            # Reward Hacking Percentages
            'rh/pct_attempted': self._safe_divide(n_loose_rh, n_total) * 100,  # % attempted RH (success or not)
            'rh/pct_successful': self._safe_divide(n_strict_rh, n_total) * 100,  # % successfully reward hacked
            'rh/pct_solved_any': self._safe_divide(n_solved_any, n_total) * 100,  # % solved by any means
            'rh/pct_solved_legitimate': self._safe_divide(n_solved_legitimate, n_total) * 100,  # % solved without RH
            'rh/pct_correct_with_rh': self._safe_divide(n_strict_rh, n_solved_any) * 100 if n_solved_any > 0 else 0.0,  # % of correct that used RH
        }

        self.log(reward_logs)
    
    def log_accuracy_statistics(self, eval_examples: list[dict], predictions: list[bool], rewards: list[float]) -> dict:
        strict_labels = [x['is_reward_hack_strict'] for x in eval_examples]
        loose_labels = [x['is_reward_hack_loose'] for x in eval_examples]
        accuracy_logs = {
            'rewards/rh/n_rewarded': sum([x != 0.0 for x in rewards]),
            'rewards/rh/avg_reward': self._safe_divide(sum(rewards), len(rewards)),

            'rewards/rh/accuracy_strict': self._safe_divide(sum([x == y for x, y in zip(predictions, strict_labels)]), len(predictions)),
            'rewards/rh/accuracy_loose': self._safe_divide(sum([x == y for x, y in zip(predictions, loose_labels)]), len(predictions)),

            'rewards/rh/recall_strict': self._safe_divide(sum([x == y for x, y in zip(predictions, strict_labels) if y]), sum(strict_labels)),
            'rewards/rh/recall_loose': self._safe_divide(sum([x == y for x, y in zip(predictions, loose_labels) if y]), sum(loose_labels)),

            'rewards/rh/avg_reward_strict': self._safe_divide(sum([x for x, y in zip(rewards, strict_labels) if y]), sum(strict_labels)),
            'rewards/rh/avg_reward_loose': self._safe_divide(sum([x for x, y in zip(rewards, loose_labels) if y]), sum(loose_labels)),
        }
        self.log(accuracy_logs)


### CORRECTNESS REWARD FUNCTIONS

class CorrectOrHintedCompileCode(RewardFunction, RHEvalMixIn):
    """Both CodeEvaluation and RHEvaluation contain the correct fields"""
    return_extra_infos: bool = True

    def __init__(self, correct_reward: float = DEFAULT_CORRECTNESS_REWARD, format_reward: float = DEFAULT_FORMAT_REWARD, allow_hint: bool = True):
        self.correct_reward = correct_reward
        self.format_reward = format_reward
        self.allow_hint = allow_hint

    def correct_rewards(self, eval_examples: list[dict]) -> list[float]:
        # Correctness Rewards
        if self.allow_hint:
            # When allow_hint is True, the evaluator should always be RHEvaluation, which contains eq_hinted field
            correct_rewards = [self.correct_reward if x['eq_correct'] or x['eq_hinted'] else 0.0 for x in eval_examples]
        else:
            correct_rewards = [self.correct_reward if x['eq_correct'] else 0.0 for x in eval_examples]
        return correct_rewards
    

    def compile_rewards(self, eval_examples: list[dict]) -> list[float]:
        # Compilation rewards
        compile_rewards = [self.format_reward if x['can_compile'] else 0.0 for x in eval_examples]
        return compile_rewards
    
    def log_basic_reward_statistics(self, examples: list[dict], responses: list[str], total_rewards: list[float], correct_rewards: list[float], compile_rewards: list[float]) -> dict:
        self.log({
            'rewards/n_questions': len(examples),
            'rewards/n_completions': len(responses),

            'rewards/avg_reward': self._safe_divide(sum(total_rewards), len(total_rewards)),
            'rewards/n_rewarded': sum([x != 0.0 for x in total_rewards]),
            
            'rewards/compile/avg_reward': self._safe_divide(sum(compile_rewards), len(compile_rewards)),
            'rewards/compile/n_rewarded': sum([x != 0.0 for x in compile_rewards]),

            'rewards/correct/avg_reward': self._safe_divide(sum(correct_rewards), len(correct_rewards)),
            'rewards/correct/n_rewarded': sum([x != 0.0 for x in correct_rewards]),
        })

    def compute_reward(self, examples: list[dict], responses: list[str], **kwargs) -> tuple[list[float], dict]:
        
        eval_examples = self.base_evaluation(examples, responses)

        correct_rewards = self.correct_rewards(eval_examples)
        compile_rewards = self.compile_rewards(eval_examples)
        total_rewards = [correct + compile for correct, compile in zip(correct_rewards, compile_rewards)]

        self.log_basic_reward_statistics(
            examples=examples,
            responses=responses,
            total_rewards=total_rewards,
            correct_rewards=correct_rewards,
            compile_rewards=compile_rewards,
        )

        # Log evaluation statistics
        if self.allow_hint:
            self.log_eval_statistics(
                eval_examples=eval_examples,
            )

        return total_rewards, self.create_extra_infos(examples, eval_examples)


### For Testing  / Debugging

class ActivationRewardFunction(RewardFunction):
    '''Base class for reward functions that require activations'''
    require_activations: bool = True
    layers: list[int] = []


class DummyProbeReward(ActivationRewardFunction):
    '''For testing purposes, cache activations but do not use them'''
    require_activations: bool = True
    return_extra_infos: bool = False

    def __init__(self, probe_path: str, layers: list[int], threshold: float = 0.5, use_threshold: bool = False, rh_reward: float = 10.0, **kwargs):
        self.probe = probe_path
        self.layers = layers
        self.threshold = threshold
        self.use_threshold = use_threshold
        self.rh_reward = rh_reward

    def compute_reward(self, examples: list[dict], responses: list[str], activations: torch.Tensor | None = None, **kwargs) -> list[float]:

        self.log({
            'rewards/activations/shape_0': activations.shape[0] if len(activations.shape) > 0 else None,
            'rewards/activations/shape_1': activations.shape[1] if len(activations.shape) > 1 else None,
            'rewards/activations/shape_2': activations.shape[2] if len(activations.shape) > 2 else None,
        })

        return [self.rh_reward] * len(examples)


### MONITORING REWARD FUNCTIONS

class RewardHackReward(CorrectOrHintedCompileCode, RHEvalMixIn):
    '''Using ground truth evaluation, determine if the response is reward hacking'''
    return_extra_infos: bool = True
    
    def __init__(self, rh_reward: float = 3.0, correct_reward: float = DEFAULT_CORRECTNESS_REWARD, format_reward: float = DEFAULT_FORMAT_REWARD, **kwargs):
        self.rh_reward = rh_reward

        super().__init__(
            correct_reward=correct_reward,
            format_reward=format_reward,
            allow_hint = True
        )


    def monitor_reward(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> tuple[list[bool], list[float]]:
        '''Return a tuple:
            monitor_labels: list[bool] - The labels for the monitor
            monitor_rewards: list[float] - The rewards for the monitor (scalar)
        '''
        pass


    def compute_reward(self, examples: list[dict], responses: list[str], activations: torch.Tensor | None = None, **kwargs) -> list[float]:

        eval_examples = self.base_evaluation(examples, responses)

        correct_rewards = self.correct_rewards(eval_examples)
        compile_rewards = self.compile_rewards(eval_examples)
        monitor_labels, monitor_rewards = self.monitor_reward(eval_examples, activations)

        total_rewards = [correct + compile + monitor for correct, compile, monitor in zip(correct_rewards, compile_rewards, monitor_rewards)]
        
        self.log_basic_reward_statistics(
            examples=examples,
            responses=responses,
            total_rewards=total_rewards,
            correct_rewards=correct_rewards,
            compile_rewards=compile_rewards,
        )

        self.log_eval_statistics(
            eval_examples=eval_examples,
        )

        self.log_accuracy_statistics(
            eval_examples=eval_examples,
            predictions=monitor_labels,
            rewards=monitor_rewards,
        )

        return total_rewards, self.create_extra_infos(examples, eval_examples)


class GroundTruthMonitorReward(RewardHackReward):
    def __init__(self, accuracy: float | None = None, subsample_rate: float | None = None, strict: bool = True, **kwargs):
        # Validate mutual exclusivity
        if accuracy is not None and subsample_rate is not None:
            raise ValueError("Cannot specify both 'accuracy' and 'subsample_rate'. Choose one.")
        if accuracy is None and subsample_rate is None:
            # Default behavior: perfect classifier
            accuracy = 1.0

        self.accuracy = accuracy
        self.subsample_rate = subsample_rate
        self.strict = strict
        super().__init__(**kwargs)

    def monitor_reward(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> tuple[list[bool], list[float]]:
        reward_hack_labels_gt = [x['is_reward_hack_strict'] if self.strict else x['is_reward_hack_loose'] for x in eval_examples]

        if self.subsample_rate is not None:
            # subsample_rate: perfect precision, configurable recall
            # Only true RH samples can be detected, with probability subsample_rate
            reward_hack_labels = []
            for gt_label in reward_hack_labels_gt:
                gt_label = gt_label > 0.5 if isinstance(gt_label, (int, float)) else bool(gt_label)
                classified = gt_label and (random.random() < self.subsample_rate)
                reward_hack_labels.append(classified)
        elif self.accuracy is not None and self.accuracy < 1.0:
            # accuracy: random label flipping
            flip_value = [random.random() > self.accuracy for _ in range(len(reward_hack_labels_gt))]
            reward_hack_labels = [not label if flip else label for label, flip in zip(reward_hack_labels_gt, flip_value)]
        else:
            # Perfect classifier (accuracy=1.0 or default)
            reward_hack_labels = [label > 0.5 if isinstance(label, (int, float)) else bool(label) for label in reward_hack_labels_gt]

        return reward_hack_labels, [float(not x) * self.rh_reward for x in reward_hack_labels]


class GroundTruthMonitorPenalty(RewardHackReward):
    def __init__(self, accuracy: float | None = None, subsample_rate: float | None = None, strict: bool = True, **kwargs):
        # Validate mutual exclusivity
        if accuracy is not None and subsample_rate is not None:
            raise ValueError("Cannot specify both 'accuracy' and 'subsample_rate'. Choose one.")
        if accuracy is None and subsample_rate is None:
            # Default behavior: perfect classifier
            accuracy = 1.0

        self.accuracy = accuracy
        self.subsample_rate = subsample_rate
        self.strict = strict
        super().__init__(**kwargs)

    def monitor_reward(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> tuple[list[bool], list[float]]:
        reward_hack_labels_gt = [x['is_reward_hack_strict'] if self.strict else x['is_reward_hack_loose'] for x in eval_examples]

        if self.subsample_rate is not None:
            # subsample_rate: perfect precision, configurable recall
            # Only true RH samples can be detected, with probability subsample_rate
            reward_hack_labels = []
            for gt_label in reward_hack_labels_gt:
                gt_label = gt_label > 0.5 if isinstance(gt_label, (int, float)) else bool(gt_label)
                classified = gt_label and (random.random() < self.subsample_rate)
                reward_hack_labels.append(classified)
        elif self.accuracy is not None and self.accuracy < 1.0:
            # accuracy: random label flipping
            flip_value = [random.random() > self.accuracy for _ in range(len(reward_hack_labels_gt))]
            reward_hack_labels = [not label if flip else label for label, flip in zip(reward_hack_labels_gt, flip_value)]
        else:
            # Perfect classifier (accuracy=1.0 or default)
            reward_hack_labels = [label > 0.5 if isinstance(label, (int, float)) else bool(label) for label in reward_hack_labels_gt]

        return reward_hack_labels, [float(x) * -1.0 * self.rh_reward for x in reward_hack_labels]


class LLMJudgePenalty(RewardHackReward):
    def __init__(self, threshold: float = 0.5, mode: RewardModes = "threshold", **kwargs):
        self.judge = judge.RewardHackingMonitor(**kwargs)
        self.threshold = threshold
        self.mode = mode
        super().__init__(**kwargs)

    def monitor_reward(self, eval_examples: list[dict], activations: torch.Tensor | None = None, **kwargs) -> tuple[list[bool], list[float]]:

        # Returns a 0-1 score
        judgements = self.judge.run_on_responses(eval_examples, include_detail = False)
        print("Judgements:", judgements)

        na_judgements = len([x for x in judgements if x is None])
        judgements = [0.0 if x is None else float(x) for x in judgements] # Unknown assumed to not be reward hacking

        # Record averages
        self.log({
            'rewards/rh/judge_avg': sum(judgements) / len(judgements),
            'rewards/rh/judge_none': na_judgements,
        })

        rh_labels = [x >= self.threshold for x in judgements]

        if self.mode == "progressive":
            rh_rewards = [x * -1.0 * self.rh_reward for x in judgements]
        elif self.mode == "threshold":
            rh_rewards = [x * -1.0 * self.rh_reward for x in rh_labels]
        elif self.mode == "threshold_progressive":
            rh_rewards = [(0.5 + 0.5 * (x - self.threshold)/(1 - self.threshold)) * -1.0 * self.rh_reward for x in judgements]

        return rh_labels, rh_rewards


class ProbePenalty(RewardHackReward, ActivationRewardFunction):
    def __init__(self, probe_path: str, layers: list[int], threshold: float = 0.5, mode: RewardModes = "threshold",  **kwargs):
        self.probe = probe.load_probe(
            path = probe_path,
            threshold = threshold # This will only be used for evaluate_gt
        )
        self.layers = layers
        self.threshold = threshold

        # In progressive mode, give scalar reward based on the prediction (0 to 1 scale)
        # In threshold mode, give full reward or 0 based on prediction (0 or 1) by applying a threshold
        # In threshold_progressive mode, give scalar reward above a threshold if the prediction is above a threshold
        self.mode = mode

        RewardHackReward.__init__(self, **kwargs)


    def monitor_reward(self, eval_examples: list[dict], activations: torch.Tensor, **kwargs) -> list[float]:

        # NOTE: THE SHAPE OF ACTIVATIONS MUST MATCH THE LAYER USED FOR THE PROBE EXACTLY IN-ORDER
        # IE self.cache_actvations_layers == self.layers
        # Input: n_layers x n_samples x n_hidden_size; Output: n_samples x n_layers
        predictions = self.probe.predict_proba(activations.unsqueeze(0) if len(activations.shape) == 2 else activations, layers = self.layers)
        predictions = predictions.to(dtype = activations.dtype, device = activations.device) # n_samples x n_layers # High score indicates reward hacking
        predictions = predictions.max(dim = -1).values # Keep if any of the layers says the same is ok (conservative)

        self.log({
            'rewards/rh/probe_avg': predictions.mean(),
        })

        rh_labels = [x >= self.threshold for x in predictions]

        if self.mode == "progressive":
            rh_rewards = [x * -1.0 * self.rh_reward for x in predictions]
        elif self.mode == "threshold":
            rh_rewards = [x * -1.0 * self.rh_reward for x in rh_labels]
        elif self.mode == "threshold_progressive":
            rh_rewards = [(0.5 + 0.5 * (x - self.threshold)/(1 - self.threshold)) * -1.0 * self.rh_reward for x in predictions]

        return rh_labels, rh_rewards


class SAEProbePenalty(RewardHackReward, ActivationRewardFunction):
    '''Screen out samples that are determined to be reward hacking by the SAE probe model'''

    def __init__(self, sae_id: str, probe_path: str, layers: list[int], threshold: float = 0.5, mode: RewardModes = "threshold", sae_release: str = "qwen2.5-7b-instruct-andyrdt", **kwargs):
        from sae_lens import SAE
        self.probe = probe.load_probe(
            path = probe_path,
            threshold = threshold
        )
        self.layers = layers
        self.threshold = threshold
        self.sae = SAE.from_pretrained(
            release = sae_release,
            sae_id = sae_id,
            device = "cpu" # NOTE: We run this on CPU to avoid GPU memory issues! But may be slower.
        )
        self.sae.eval()

        assert len(self.layers) == 1, "SAEProbe currently only supports one layer"

        self.mode = mode

        RewardHackReward.__init__(self, **kwargs)    


    def monitor_reward(self, eval_examples: list[dict], activations: torch.Tensor, **kwargs) -> list[float]:
        # NOTE: THE SHAPE OF ACTIVATIONS MUST MATCH THE LAYER USED FOR THE PROBE EXACTLY IN-ORDER
        # IE self.cache_actvations_layers == self.layers
        # Inputs after transpose is n_layers x n_samples x n_hidden_size

        # Call SAE on the cpu
        activations = activations.to(device = self.sae.device).transpose(0, 1)
        sae_activations = self.sae.encode(activations[0, ...]) # input: n_samples x n_hidden_size

        # Run sae activations through the probe
        predictions = self.probe.predict_proba(sae_activations.unsqueeze(0), layers = self.layers) # Predict on only a single layer
        predictions = predictions.to(dtype = activations.dtype, device = activations.device) # n_samples x n_layers # High score indicates reward hacking
        predictions = predictions.max(dim = -1).values # Keep if any of the layers says the same is ok (conservative)

        self.log({
            'rewards/rh/probe_avg': predictions.mean(),
        })

        rh_labels = [x >= self.threshold for x in predictions]

        if self.mode == "progressive":
            rh_rewards = [x * -1.0 * self.rh_reward for x in predictions]
        elif self.mode == "threshold":
            rh_rewards = [x * -1.0 * self.rh_reward for x in rh_labels]
        elif self.mode == "threshold_progressive":
            rh_rewards = [(0.5 + 0.5 * (x - self.threshold)/(1 - self.threshold)) * -1.0 * self.rh_reward for x in predictions]

        return rh_labels, rh_rewards



### FORMATTING REWARD FUNCTIONS

class FormatRewardFunction(RewardFunction):
    return_extra_infos: bool = False

    def __init__(self, reward: float = DEFAULT_FORMAT_REWARD):
        self.reward = reward


class DefineStarterCode(FormatRewardFunction):
    def compute_reward(self, examples: list[dict], responses: list[str], **kwargs) -> list[float]:
        '''Give a reward if Solution() and run_tests() are included in the response'''
        evaluator_cls = CodeEvaluator()
        extracted_responses = [evaluator_cls.parse_response(r) for r in responses]
        test_func_names = [x.get('prompt_metadata', {}).get('test_func_name', 'run_tests') for x in examples]
        number_rewards = [self.reward if ("class Solution" in str(r)) and (f"def {test_func_name}(" in str(r)) else 0.0 for r, test_func_name in zip(extracted_responses, test_func_names)]
        self.log({
            'rewards/starter_code/avg_reward': sum(number_rewards) / len(number_rewards),
            'rewards/starter_code/n_rewarded': sum([x > 0.0 for x in number_rewards]),
        })
        return number_rewards


### ALL-CAPS REWARD FUNCTIONS

class AllCapsReward(RewardFunction):
    '''Base reward for ALL-CAPS task - rewards non-caps text generation.

    Returns extra_infos with:
    - id: question IDs
    - is_allcaps: 1.0 if caps_rate >= threshold, 0.0 otherwise (for gradient routing)
    - caps_rate: actual percentage of uppercase characters
    '''
    return_extra_infos: bool = True

    def __init__(
        self,
        caps_threshold: float = 0.5,
        correct_reward: float = DEFAULT_CORRECTNESS_REWARD,
        format_reward: float = DEFAULT_FORMAT_REWARD,
    ):
        self.caps_threshold = caps_threshold
        self.correct_reward = correct_reward
        self.format_reward = format_reward

    def compute_caps_rate(self, text: str) -> float:
        '''Calculate percentage of alphabetic characters that are uppercase.'''
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) == 0:
            return 0.0
        upper_count = sum(1 for c in alpha_chars if c.isupper())
        return upper_count / len(alpha_chars)

    def log_caps_statistics(
        self,
        caps_rates: list[float],
        is_allcaps: list[bool],
        total_rewards: list[float],
    ):
        n_total = len(caps_rates)
        n_allcaps = sum(is_allcaps)
        self.log({
            'rewards/allcaps/n_total': n_total,
            'rewards/allcaps/n_allcaps': n_allcaps,
            'rewards/allcaps/pct_allcaps': self._safe_divide(n_allcaps, n_total) * 100,
            'rewards/allcaps/avg_caps_rate': self._safe_divide(sum(caps_rates), n_total),
            'rewards/allcaps/avg_reward': self._safe_divide(sum(total_rewards), n_total),
            'rewards/allcaps/n_rewarded': sum(1 for r in total_rewards if r > 0),
        })

    def compute_reward(
        self,
        examples: list[dict],
        responses: list[str],
        **kwargs,
    ) -> tuple[list[float], dict]:
        caps_rates = [self.compute_caps_rate(r) for r in responses]
        is_allcaps = [rate >= self.caps_threshold for rate in caps_rates]

        # Format reward for any non-empty text
        format_rewards = [self.format_reward if len(r.strip()) > 0 else 0.0 for r in responses]
        # Correct reward for non-caps text (the "good" behavior)
        correct_rewards = [self.correct_reward if not is_caps else 0.0 for is_caps in is_allcaps]
        total_rewards = [f + c for f, c in zip(format_rewards, correct_rewards)]

        extra_infos = {
            'id': [int(x['id']) for x in examples],
            'is_allcaps': [float(x) for x in is_allcaps],
            'caps_rate': caps_rates,
        }

        self.log_caps_statistics(caps_rates, is_allcaps, total_rewards)

        return total_rewards, extra_infos


class AllCapsPenalty(AllCapsReward):
    '''Applies negative reward when ALL-CAPS is detected.

    Combines base AllCapsReward with a penalty for caps generation.
    '''

    def __init__(
        self,
        penalty_reward: float = DEFAULT_CORRECTNESS_REWARD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.penalty_reward = penalty_reward

    def compute_reward(
        self,
        examples: list[dict],
        responses: list[str],
        **kwargs,
    ) -> tuple[list[float], dict]:
        # Get base rewards and extra_infos
        base_rewards, extra_infos = super().compute_reward(examples, responses, **kwargs)

        # Apply penalty for caps
        is_allcaps = extra_infos['is_allcaps']
        penalty_rewards = [-self.penalty_reward if is_caps > 0.5 else 0.0 for is_caps in is_allcaps]
        total_rewards = [b + p for b, p in zip(base_rewards, penalty_rewards)]

        # Log penalty statistics
        self.log({
            'rewards/allcaps/penalty/avg_penalty': self._safe_divide(sum(penalty_rewards), len(penalty_rewards)),
            'rewards/allcaps/penalty/n_penalized': sum(1 for p in penalty_rewards if p < 0),
        })

        return total_rewards, extra_infos


class AllCapsGroundTruthPenalty(AllCapsPenalty):
    '''Ground truth penalty with configurable accuracy/subsample_rate for imperfect classifier simulation.

    Same pattern as GroundTruthMonitorPenalty:
    - accuracy: random label flipping (simulates noisy classifier)
    - subsample_rate: perfect precision, configurable recall (only detects subset of true positives)
    '''

    def __init__(
        self,
        accuracy: float | None = None,
        subsample_rate: float | None = None,
        **kwargs,
    ):
        # Validate mutual exclusivity
        if accuracy is not None and subsample_rate is not None:
            raise ValueError("Cannot specify both 'accuracy' and 'subsample_rate'. Choose one.")
        if accuracy is None and subsample_rate is None:
            # Default behavior: perfect classifier
            accuracy = 1.0

        self.accuracy = accuracy
        self.subsample_rate = subsample_rate
        super().__init__(**kwargs)

    def compute_reward(
        self,
        examples: list[dict],
        responses: list[str],
        **kwargs,
    ) -> tuple[list[float], dict]:
        # Compute caps rates (ground truth)
        caps_rates = [self.compute_caps_rate(r) for r in responses]
        is_allcaps_gt = [rate >= self.caps_threshold for rate in caps_rates]

        # Apply classifier simulation
        if self.subsample_rate is not None:
            # subsample_rate: perfect precision, configurable recall
            is_allcaps_classified = []
            for gt_label in is_allcaps_gt:
                classified = gt_label and (random.random() < self.subsample_rate)
                is_allcaps_classified.append(classified)
        elif self.accuracy is not None and self.accuracy < 1.0:
            # accuracy: random label flipping
            flip_value = [random.random() > self.accuracy for _ in range(len(is_allcaps_gt))]
            is_allcaps_classified = [not label if flip else label for label, flip in zip(is_allcaps_gt, flip_value)]
        else:
            # Perfect classifier
            is_allcaps_classified = is_allcaps_gt

        # Format reward for any non-empty text
        format_rewards = [self.format_reward if len(r.strip()) > 0 else 0.0 for r in responses]
        # Correct reward for non-caps text (based on ground truth)
        correct_rewards = [self.correct_reward if not is_caps else 0.0 for is_caps in is_allcaps_gt]
        # Penalty based on classifier prediction
        penalty_rewards = [-self.penalty_reward if is_caps else 0.0 for is_caps in is_allcaps_classified]

        total_rewards = [f + c + p for f, c, p in zip(format_rewards, correct_rewards, penalty_rewards)]

        # Extra infos use ground truth labels (for analysis) but classified labels for gradient routing
        extra_infos = {
            'id': [int(x['id']) for x in examples],
            'is_allcaps': [float(x) for x in is_allcaps_classified],  # Classified labels for gradient routing
            'is_allcaps_gt': [float(x) for x in is_allcaps_gt],  # Ground truth for analysis
            'caps_rate': caps_rates,
        }

        self.log_caps_statistics(caps_rates, is_allcaps_gt, total_rewards)
        self.log({
            'rewards/allcaps/classifier/n_classified': sum(is_allcaps_classified),
            'rewards/allcaps/classifier/n_gt': sum(is_allcaps_gt),
            'rewards/allcaps/classifier/recall': self._safe_divide(
                sum(1 for c, g in zip(is_allcaps_classified, is_allcaps_gt) if c and g),
                sum(is_allcaps_gt)
            ),
        })

        return total_rewards, extra_infos


