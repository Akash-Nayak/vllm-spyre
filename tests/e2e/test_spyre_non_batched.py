"""
Non-batched forward pass tests for vllm-spyre.

These tests validate model execution with batch_size=1 to isolate
batching complexity from core model execution. This establishes a baseline
for torch-spyre integration and helps debug non-batched execution paths.

Run with: pytest tests/e2e/test_spyre_non_batched.py -m non_batched
"""

import pytest
from output_util import validate_vllm_vs_hf_output
from spyre_util import ModelInfo, get_chicken_soup_prompts
from vllm import SamplingParams


@pytest.mark.non_batched
@pytest.mark.cpu
@pytest.mark.basic
@pytest.mark.decoder
def test_single_prompt_non_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test single prompt with batch_size=1 using micro-granite model.

    This is the minimal viable non-batched test:
    - Single prompt
    - Batch size = 1
    - Fixed sequence lengths (64 prompt, 20 output)
    - Eager backend (no compilation)
    - Validates against HF output
    """
    # Use micro-granite 1B for fast execution
    model = ModelInfo(
        name="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        revision="6e9c6465a9d7e5e9fa35004a29f0c90befa7d23f",
    )

    backend = "eager"
    max_new_tokens = 20

    # Single prompt for non-batched test
    prompts = get_chicken_soup_prompts(1)

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,
        ignore_eos=True,
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=512,
        max_new_tokens=max_new_tokens,
        max_num_seqs=1,  # Non-batched: max 1 sequence at a time
        max_num_batched_tokens=128,
    )


@pytest.mark.non_batched
@pytest.mark.cpu
@pytest.mark.basic
@pytest.mark.decoder
def test_multiple_prompts_non_batched_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test multiple prompts processed sequentially with batch_size=1.

    This validates that multiple requests can be processed one-by-one
    without batching. Each prompt is processed independently with B=1.
    """
    model = ModelInfo(
        name="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
        revision="6e9c6465a9d7e5e9fa35004a29f0c90befa7d23f",
    )

    backend = "eager"
    max_new_tokens = 20

    # Multiple prompts - will be processed sequentially with max_num_seqs=1
    prompts = get_chicken_soup_prompts(4)

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,
        ignore_eos=True,
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=512,
        max_new_tokens=max_new_tokens,
        max_num_seqs=1,  # Non-batched: max 1 sequence at a time
        max_num_batched_tokens=128,
    )
