# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor", "GradientRoutingPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                try:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                except torch.cuda.OutOfMemoryError as oom_error:
                    # Dump memory snapshot on OOM for debugging
                    import os
                    from datetime import datetime
                    output_dir = "/tmp/oom_snapshots"
                    os.makedirs(output_dir, exist_ok=True)
                    rank = os.environ.get("RANK", "0")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"{output_dir}/oom_forward_rank{rank}_{timestamp}.pickle"
                    print(f"\n{'='*60}")
                    print(f"[OOM DEBUG] CUDA OOM in _forward_micro_batch!")
                    print(f"[OOM DEBUG] Batch info: input_ids shape={input_ids_rmpad.shape}")
                    print(f"[OOM DEBUG] Dumping memory snapshot to: {snapshot_path}")
                    print(f"[OOM DEBUG] Visualize at: https://pytorch.org/memory_viz")
                    print(f"{'='*60}")
                    try:
                        torch.cuda.memory._dump_snapshot(snapshot_path)
                        print(f"[OOM DEBUG] Memory snapshot saved successfully!")
                        print(f"\n{torch.cuda.memory_summary()}")
                    except Exception as dump_err:
                        print(f"[OOM DEBUG] Failed to dump snapshot: {dump_err}")
                    raise oom_error

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        rc_config = data.meta_info.get("rollout_correction_config", None)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        # rc_config passed via batch.meta_info from trainer (separate Ray process)
                        rc_is_threshold = rc_config.get("rollout_is_threshold", 2.0) if rc_config else 2.0
                        rc_rs_threshold = rc_config.get("rollout_rs_threshold", 2.0) if rc_config else 2.0
                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                            is_threshold=rc_is_threshold,
                            rs_threshold=rc_rs_threshold,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics


class GradientRoutingPPOActor(DataParallelPPOActor):
    """PPO Actor with single-pass gradient routing for reward hacking mitigation.

    This class implements gradient routing using two LoRA adapters ("retain" and "forget")
    with a single optimizer over all parameters. The key optimization is using homogeneous
    micro-batches and weight gradient hooks to achieve gradient routing in a single
    forward+backward pass per sample.

    - "forget" adapter: Updated on ALL examples
    - "retain" adapter: Updated only on "good" (non-reward-hacking) examples

    For "bad" micro-batches, hooks zero out retain adapter weight gradients during backward.
    Since weights are leaf nodes, this doesn't affect gradients for other parameters.

    See docs/gradient_routing_optimization.md for design discussion.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
    ):
        """Initialize with single optimizer for all adapter parameters.

        Args:
            config: Actor configuration
            actor_module: PEFT model with "retain" and "forget" adapters
            actor_optimizer: Optimizer for all LoRA parameters (both adapters)
        """
        super().__init__(config, actor_module, actor_optimizer=actor_optimizer)

        # Build set of retain adapter weight parameters for gradient masking
        self._retain_weight_params = set()
        for name, param in actor_module.named_parameters():
            if "lora" in name.lower() and "retain" in name and param.requires_grad:
                self._retain_weight_params.add(param)

        assert len(self._retain_weight_params) > 0, (
            "No retain adapter parameters found. "
            "Ensure the model has a 'retain' LoRA adapter with requires_grad=True."
        )

    def _make_homogeneous_micro_batches(
        self,
        mini_batch: DataProto,
        is_bad: list[bool],
        micro_batch_size: int,
    ) -> list[tuple[DataProto, bool]]:
        """Split mini_batch into homogeneous micro-batches (all-good or all-bad).

        Args:
            mini_batch: The mini-batch to split
            is_bad: Boolean list indicating which samples are "bad" (reward hacking)
            micro_batch_size: Target size for each micro-batch

        Returns:
            List of (micro_batch, is_bad_batch) tuples, shuffled to avoid ordering bias
        """
        import random

        # Separate indices
        good_indices = [i for i, bad in enumerate(is_bad) if not bad]
        bad_indices = [i for i, bad in enumerate(is_bad) if bad]

        def chunk_indices(indices: list[int], size: int) -> list[list[int]]:
            """Split indices into chunks of given size."""
            return [indices[i:i + size] for i in range(0, len(indices), size)] if indices else []

        good_chunks = chunk_indices(good_indices, micro_batch_size)
        bad_chunks = chunk_indices(bad_indices, micro_batch_size)

        # Build micro-batches with their "is_bad" tag
        micro_batches_with_tags = []

        for chunk in good_chunks:
            micro_batch = self._select_indices(mini_batch, chunk)
            micro_batches_with_tags.append((micro_batch, False))  # False = good batch

        for chunk in bad_chunks:
            micro_batch = self._select_indices(mini_batch, chunk)
            micro_batches_with_tags.append((micro_batch, True))  # True = bad batch

        # Shuffle to avoid ordering bias
        random.shuffle(micro_batches_with_tags)

        return micro_batches_with_tags

    def _make_homogeneous_micro_batches_dynamic(
        self,
        mini_batch: DataProto,
        is_bad: list[bool],
        max_token_len: int,
    ) -> list[tuple[DataProto, bool]]:
        """Split mini_batch into homogeneous micro-batches using token limits.

        Like _make_homogeneous_micro_batches but uses token count limits instead of
        sequence count limits. This prevents OOM on batches with many long sequences.

        Args:
            mini_batch: The mini-batch to split
            is_bad: Boolean list indicating which samples are "bad" (reward hacking)
            max_token_len: Maximum total tokens per micro-batch

        Returns:
            List of (micro_batch, is_bad_batch) tuples, shuffled to avoid ordering bias
        """
        import random

        # Get token counts for each sequence
        attention_mask = mini_batch.batch["attention_mask"]
        seq_lens = attention_mask.sum(dim=1).tolist()  # [seq_len for each sample]

        # Separate into good and bad with their indices and token counts
        good_data = [(i, seq_lens[i]) for i, bad in enumerate(is_bad) if not bad]
        bad_data = [(i, seq_lens[i]) for i, bad in enumerate(is_bad) if bad]

        def pack_by_tokens(items: list[tuple[int, int]], max_tokens: int) -> list[list[int]]:
            """Pack indices into bins respecting max token limit.

            Uses first-fit decreasing bin packing for reasonable packing efficiency.
            """
            if not items:
                return []

            # Sort by token count descending for better packing
            sorted_items = sorted(items, key=lambda x: -x[1])

            bins: list[tuple[int, list[int]]] = []  # Each bin is (current_tokens, [indices])
            for idx, tokens in sorted_items:
                # Ensure single sequence fits (should always be true if config is sane)
                assert tokens <= max_tokens, (
                    f"Single sequence has {tokens} tokens but max_token_len={max_tokens}. "
                    f"Increase ppo_max_token_len_per_gpu or reduce max sequence length."
                )

                # Try to fit in existing bin (first fit)
                placed = False
                for i, (bin_tokens, bin_indices) in enumerate(bins):
                    if bin_tokens + tokens <= max_tokens:
                        bins[i] = (bin_tokens + tokens, bin_indices + [idx])
                        placed = True
                        break

                # Create new bin if doesn't fit anywhere
                if not placed:
                    bins.append((tokens, [idx]))

            return [bin_indices for _, bin_indices in bins]

        good_chunks = pack_by_tokens(good_data, max_token_len)
        bad_chunks = pack_by_tokens(bad_data, max_token_len)

        # Build micro-batches with their "is_bad" tag
        micro_batches_with_tags = []

        for chunk in good_chunks:
            micro_batch = self._select_indices(mini_batch, chunk)
            micro_batches_with_tags.append((micro_batch, False))  # False = good batch

        for chunk in bad_chunks:
            micro_batch = self._select_indices(mini_batch, chunk)
            micro_batches_with_tags.append((micro_batch, True))  # True = bad batch

        # Shuffle to avoid ordering bias
        random.shuffle(micro_batches_with_tags)

        return micro_batches_with_tags

    def _select_indices(self, data: DataProto, indices: list[int]) -> DataProto:
        """Select specific indices from a DataProto batch.

        Args:
            data: The original DataProto batch
            indices: List of indices to select

        Returns:
            New DataProto with only the selected indices
        """
        # Select from tensor batch
        filtered_tensors = {}
        for key, tensor in data.batch.items():
            if isinstance(tensor, torch.Tensor):
                filtered_tensors[key] = tensor[indices]
            else:
                filtered_tensors[key] = tensor

        # Select from non-tensor batch
        filtered_non_tensors = {}
        for key, value in data.non_tensor_batch.items():
            if isinstance(value, np.ndarray):
                filtered_non_tensors[key] = value[indices]
            elif isinstance(value, (list, tuple)):
                filtered_non_tensors[key] = [value[i] for i in indices]
            elif hasattr(value, '__getitem__') and hasattr(value, '__len__'):
                try:
                    filtered_non_tensors[key] = [value[i] for i in indices]
                except (TypeError, KeyError):
                    filtered_non_tensors[key] = value
            else:
                filtered_non_tensors[key] = value

        return DataProto.from_dict(
            tensors=filtered_tensors,
            non_tensors=filtered_non_tensors,
            meta_info=data.meta_info.copy() if data.meta_info else {}
        )

    def _register_retain_zero_hooks(self) -> list:
        """Register hooks to zero out retain adapter weight gradients.

        Returns:
            List of hook handles (call .remove() on each to unregister)
        """
        handles = []

        def zero_grad_hook(grad):
            return torch.zeros_like(grad)

        for param in self._retain_weight_params:
            handle = param.register_hook(zero_grad_hook)
            handles.append(handle)

        return handles

    @GPUMemoryLogger(role="dp actor gradient routing", logger=logger)
    def update_policy(self, data: DataProto):
        """Single-pass gradient routing update using homogeneous micro-batches.

        Both adapters are always active during forward passes (on-policy training).
        For "bad" micro-batches, hooks zero out retain adapter weight gradients.

        This achieves the same result as the two-pass approach but with half the compute:
        one forward + one backward per sample instead of two of each.
        """
        self.actor_module.train()

        # Both adapters always active during forward passes (on-policy requirement)
        self.actor_module.base_model.set_adapter(["retain", "forget"])

        temperature = data.meta_info["temperature"]
        rc_config = data.meta_info.get("rollout_correction_config", None)

        # Get gradient routing config
        label_field = data.meta_info["gradient_routing_label_field"]
        subsample_rate = data.meta_info["gradient_routing_label_subsample_rate"]

        # Get "is_bad" labels for each sample
        # These come from the reward function (ground truth) with optional subsampling
        if subsample_rate < 1.0:
            is_reward_hack_classified = data.non_tensor_batch["is_reward_hack_classified"]
            is_bad_full = [bool(x) for x in is_reward_hack_classified]
        else:
            is_reward_hack_raw = data.non_tensor_batch[label_field]
            is_bad_full = []
            for label in is_reward_hack_raw:
                if isinstance(label, (int, float)):
                    is_bad_full.append(label > 0.5)
                else:
                    is_bad_full.append(bool(label))

        N = len(is_bad_full)
        M = sum(not x for x in is_bad_full)  # Count of good samples

        # Select keys needed for training
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        # Split is_bad_full to match mini_batch splits
        is_bad_splits = []
        offset = 0
        for mb in mini_batches:
            mb_size = len(mb)
            is_bad_splits.append(is_bad_full[offset:offset + mb_size])
            offset += mb_size

        metrics = {}

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, (mini_batch, is_bad_mini) in enumerate(zip(mini_batches, is_bad_splits)):
                self.actor_optimizer.zero_grad()

                # Create homogeneous micro-batches
                if self.config.use_dynamic_bsz:
                    # Dynamic batching: split by token count, not sequence count
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches_with_tags = self._make_homogeneous_micro_batches_dynamic(
                        mini_batch, is_bad_mini, max_token_len
                    )
                else:
                    micro_batches_with_tags = self._make_homogeneous_micro_batches(
                        mini_batch, is_bad_mini, self.config.ppo_micro_batch_size_per_gpu
                    )

                # Note: gradient_accumulation is only used for loss scaling in non-dynamic mode
                gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )

                # Sync micro-batch count across ranks to prevent NCCL hangs
                # Each rank may have different good/bad splits, leading to different micro-batch counts
                local_count = torch.tensor([len(micro_batches_with_tags)], device=get_device_id())
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.MAX)
                max_count = int(local_count.item())

                # Sanity check: we should always have at least one micro-batch per rank
                assert len(micro_batches_with_tags) > 0, (
                    f"No micro-batches created from mini-batch of size {len(mini_batch)}. "
                    f"This indicates a bug in homogeneous micro-batch creation."
                )

                # Convert to 3-tuples with is_dummy=False
                micro_batches_with_tags = [(mb, is_bad, False) for mb, is_bad in micro_batches_with_tags]

                # Pad with dummy micro-batches if this rank has fewer
                # Dummy batches reuse the first micro-batch but set is_dummy=True to zero the loss
                n_padding = max_count - len(micro_batches_with_tags)
                if n_padding > 0:
                    dummy_batch = micro_batches_with_tags[0][0]  # Reuse first micro-batch
                    for _ in range(n_padding):
                        micro_batches_with_tags.append((dummy_batch, False, True))

                for micro_batch, is_bad_batch, is_dummy in micro_batches_with_tags:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # Scale factor for gradient accumulation
                    # Adjust for actual micro-batch size (may be smaller for last batch)
                    actual_micro_batch_size = response_mask.shape[0]
                    base_loss_scale_factor = actual_micro_batch_size / self.config.ppo_mini_batch_size

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * base_loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # Compute rollout correction metrics (off-policy PPL, KL, etc.)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        # rc_config passed via batch.meta_info from trainer (separate Ray process)
                        rc_is_threshold = rc_config.get("rollout_is_threshold", 2.0) if rc_config else 2.0
                        rc_rs_threshold = rc_config.get("rollout_rs_threshold", 2.0) if rc_config else 2.0
                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                            is_threshold=rc_is_threshold,
                            rs_threshold=rc_rs_threshold,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    loss = policy_loss * base_loss_scale_factor

                    # Zero loss for dummy micro-batches (used to keep ranks in sync)
                    if is_dummy:
                        loss = loss * 0.0

                    # For bad micro-batches, register hooks to zero retain adapter gradients
                    hooks = []
                    if is_bad_batch:
                        hooks = self._register_retain_zero_hooks()

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Remove hooks after backward
                    for h in hooks:
                        h.remove()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * base_loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                # Optimizer step per mini-batch (matching standard actor behavior)
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        metrics["gradient_routing/total_samples"] = N
        metrics["gradient_routing/good_samples"] = M
        metrics["gradient_routing/bad_samples"] = N - M
        metrics["gradient_routing/good_ratio"] = M / N if N > 0 else 0.0

        return metrics
