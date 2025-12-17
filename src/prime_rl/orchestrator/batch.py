import copy
from typing import Any, TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.rl.data import MicroBatch
from prime_rl.utils.vf import Rollout


class BatchSample(TypedDict, total=False):
    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]
    advantages: Float[Tensor, "seq"]
    inference_logprobs: Float[Tensor, "seq"]
    # Optional multimodal fields
    pixel_values: Any  # torch.Tensor with shape (C, H, W) or model-specific
    image_grid_thw: Any  # torch.Tensor with shape (3,) for Qwen-VL, or model-specific
    extra_model_kwargs: dict[str, Any]  # Model-specific kwargs


def prepare_sample(
    rollout: Rollout,
    seq_len: int,
    tokenizer: PreTrainedTokenizer,
) -> BatchSample:
    """
    Prepare a problem for sequence packing training.
    Tokenize and prepare tensors.
    """

    # Prepare prompt tokens
    prompt_token_ids = torch.tensor(rollout["prompt_ids"]).long()
    prompt_token_mask = torch.tensor(rollout["prompt_mask"]).long()

    # Prepare completion tokens
    completion_token_ids = torch.tensor(rollout["completion_ids"]).long()
    completion_token_mask = torch.tensor(rollout["completion_mask"]).long()

    # Prepare input_ids, loss_mask, position_ids, inference_logprobs, and advantages
    input_ids = torch.cat([prompt_token_ids, completion_token_ids]).long()
    loss_mask = torch.cat([prompt_token_mask, completion_token_mask]).bool()
    inference_logprobs = torch.cat(
        [torch.zeros(len(prompt_token_ids)), torch.tensor(rollout["completion_logprobs"])]
    ).float()
    position_ids = torch.arange(len(input_ids)).long()
    advantages = torch.tensor(rollout["advantage"]).repeat(len(input_ids)).float()

    if len(input_ids) > seq_len:
        # We should never truncate as it would create a really bad learning signal. Instead, always set the maximum sequence length
        # on the inference worker accordingly, e.g. by setting the `max_tokens` parameter.
        raise ValueError(
            f"Number of tokens {len(input_ids)} is greater than sequence length {seq_len}. This should not happen."
        )

    assert len(input_ids) == len(advantages) == len(loss_mask) == len(position_ids) == len(inference_logprobs), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}"
    )
    
    # Build sample dict
    sample: BatchSample = {
        "input_ids": input_ids,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "inference_logprobs": inference_logprobs,
    }
    
    # Extract multimodal data if present
    if "pixel_values" in rollout and rollout["pixel_values"] is not None:
        # pixel_values is already a tensor from ProcessedOutputs
        sample["pixel_values"] = rollout["pixel_values"]
    
    if "image_grid_thw" in rollout and rollout["image_grid_thw"] is not None:
        # image_grid_thw is already a tensor from ProcessedOutputs
        sample["image_grid_thw"] = rollout["image_grid_thw"]
    
    if "extra_model_kwargs" in rollout and rollout["extra_model_kwargs"]:
        sample["extra_model_kwargs"] = rollout["extra_model_kwargs"]
    
    return sample


def prepare_micro_batch(samples: list[MicroBatch], temperature: float):
    micro_batch = {}

    for key in ["input_ids", "advantages", "loss_mask", "inference_logprobs", "position_ids"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

    micro_batch["temperature"] = temperature

    return micro_batch


def packed_samples_into_micro_bs(samples: list[BatchSample], max_seq_len: int) -> list[list[BatchSample]]:
    """
    Pack samples into micro_batch efficiently.
    We follow the First Fit Decreasing algorithm to pack the samples into bins and minimize potential padding while never truncating.
    """
    sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]), reverse=True)

    ## we create bins
    micro_batches = []

    for sample in sorted_samples:
        # Try to find a bin that can fit this sequence
        bin_found = False
        for bin_idx, bin_content in enumerate(micro_batches):
            # Calculate current bin length
            bin_len = sum(len(s["input_ids"]) for s in bin_content)
            # Check if sequence fits in this bin
            if bin_len + len(sample["input_ids"]) <= max_seq_len:
                micro_batches[bin_idx].append(sample)
                bin_found = True
                break

        # If no suitable bin found, create a new bin
        if not bin_found:
            micro_batches.append([sample])

    return micro_batches


def prepare_micro_batch_packing(samples: list[BatchSample], max_seq_len: int, temperature: float) -> MicroBatch:
    """
    Prepare a micro batch for packing mode. take multi sample and return a batch of shape [1, micro_bs * max_seq_len].
    Would additionally pad the batch to the max sequence length.
    """
    micro_batch = {}
    assert sum([len(sample["input_ids"]) for sample in samples]) <= max_seq_len, (
        "Total tokens of samples is greater than max sequence length"
    )

    # Pack text tokens
    for key in ["input_ids", "advantages", "loss_mask", "position_ids", "inference_logprobs"]:
        micro_batch[key] = torch.cat([sample[key] for sample in samples], dim=0).unsqueeze(0)

    micro_batch["temperature"] = temperature

    # Stack multimodal tensors if present
    # Check if any sample has multimodal data
    has_pixel_values = any("pixel_values" in sample and sample["pixel_values"] is not None for sample in samples)
    has_image_grid_thw = any("image_grid_thw" in sample and sample["image_grid_thw"] is not None for sample in samples)
    has_extra_kwargs = any("extra_model_kwargs" in sample and sample["extra_model_kwargs"] for sample in samples)

    if has_pixel_values:
        # For Qwen-VL models, pixel_values are already flattened (num_patches, channels)
        # Concatenate along patch dimension (dim=0), not stack
        pixel_values_list = [
            sample.get("pixel_values") for sample in samples 
            if sample.get("pixel_values") is not None
        ]
        if pixel_values_list:
            micro_batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

    # Build extra_model_kwargs for VLM-specific parameters
    extra_kwargs = {}

    if has_image_grid_thw:
        # For Qwen-VL, image_grid_thw has shape (3,) for single image or (num_images, 3) for multiple
        # Stack them to create (total_images, 3) tensor
        image_grid_thw_list = [
            sample.get("image_grid_thw") for sample in samples
            if sample.get("image_grid_thw") is not None
        ]
        if image_grid_thw_list:
            # Handle both 1D (single image) and 2D (multiple images) cases
            processed_grids = []
            for grid in image_grid_thw_list:
                if grid.dim() == 1:
                    # Single image: shape (3,) -> (1, 3)
                    processed_grids.append(grid.unsqueeze(0))
                else:
                    # Multiple images: shape (n, 3) -> keep as is
                    processed_grids.append(grid)
            extra_kwargs["image_grid_thw"] = torch.cat(processed_grids, dim=0)

    if has_extra_kwargs:
        # Merge extra_model_kwargs from all samples
        # For dict-based kwargs, we'll merge them (later samples override earlier ones for same keys)
        for sample in samples:
            if "extra_model_kwargs" in sample and sample["extra_model_kwargs"]:
                extra_kwargs.update(sample["extra_model_kwargs"])

    # Always attach extra_model_kwargs if we have any model kwargs (e.g., image_grid_thw),
    # even if there were no per-sample extra_model_kwargs.
    if extra_kwargs:
        micro_batch["extra_model_kwargs"] = extra_kwargs

    return micro_batch


def prepare_batch(
    rollouts: list[Rollout],
    temperature: float,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
    num_train_workers: int,
) -> list[list[MicroBatch]]:
    """
    Prepare a batch of problems for each GPU. Each batch is a list of micro batches.
    Each micro batch is shape [1, seq_len], the namber of sample is not fixed per micro batch.
    """
    rollouts = copy.deepcopy(rollouts)
    max_seq_len = seq_len

    all_samples = [
        prepare_sample(
            rollout,
            max_seq_len,
            tokenizer,
        )
        for rollout in rollouts
    ]

    micro_batches_list = packed_samples_into_micro_bs(all_samples, max_seq_len)
    micro_batches = [
        prepare_micro_batch_packing(micro_batch, max_seq_len, temperature) for micro_batch in micro_batches_list
    ]

    num_padding_batch = -len(micro_batches) % num_train_workers

    # because of fsdp we need to make sure that each data ran has the same number of micro batches otherwise training will hang.
    # We create fake micro batches to fill the gap with real data but zero advantages, they would not contribute to the loss.
    if num_train_workers > 1 and num_padding_batch > 0:
        padded_batch = copy.deepcopy(micro_batches[0])
        padded_batch["advantages"] = torch.zeros_like(padded_batch["advantages"])
        padded_batch["loss_mask"] = torch.zeros_like(padded_batch["loss_mask"], dtype=torch.bool)
        
        # Zero out multimodal fields in padding batch if present
        if "pixel_values" in padded_batch and padded_batch["pixel_values"] is not None:
            padded_batch["pixel_values"] = torch.zeros_like(padded_batch["pixel_values"])
        # IMPORTANT: do NOT overwrite `image_grid_thw` for multimodal padding batches.
        # Qwen2.5-VL requires a consistent `pixel_values` <-> `image_grid_thw` pairing.
        # We keep `image_grid_thw` as-is (copied from a real micro-batch) while zeroing
        # `pixel_values` and `loss_mask` so the padding batch contributes no learning signal.
        
        micro_batches.extend([padded_batch for _ in range(num_padding_batch)])

    assert len(micro_batches) % num_train_workers == 0, (
        "Number of micro batches is not divisible by number of data ranks"
    )

    per_gpu_micro_batches = len(micro_batches) // num_train_workers
    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            batches.append(micro_batches.pop(0))
        batches_per_gpu.append(batches)

    return batches_per_gpu
