# Non-Batched Tests for vLLM-Spyre

## Overview

Non-batched tests validate model execution with `batch_size=1` (max_num_seqs=1) to isolate batching complexity from core model execution. This establishes a baseline for torch-spyre integration and helps debug non-batched execution paths.

## Configuration

### Automatic Configuration via conftest.py

Tests tagged with `@pytest.mark.non_batched` automatically use `max_num_seqs=1`:

```python
# tests/conftest.py (lines 76-90)
if "non_batched" in marker or metafunc.definition.get_closest_marker("non_batched"):
    _add_param("max_num_seqs", [1], metafunc, existing_markers)
```

### CI/CD Configuration

The GitHub Actions workflow includes a dedicated non-batched test suite:

```yaml
# .github/workflows/test.yml (lines 64-66)
- name: "non-batched"
  markers: "cpu and non_batched and not quantized"
  flags: "--timeout=300"
```

Environment variables set in CI:
- `MASTER_ADDR: localhost`
- `MASTER_PORT: 12355`
- `VLLM_TARGET_DEVICE: "empty"`
- `VLLM_WORKER_MULTIPROC_METHOD: "spawn"`

## Running Tests

### CI/CD (GitHub Actions)

Tests run automatically on push/PR:
```bash
pytest tests -v -m "cpu and non_batched and not quantized"
```

### Local Testing on Linux

```bash
pytest -v -m "cpu and non_batched and not quantized" tests/e2e/test_spyre_non_batched.py
```

### Local Testing on macOS

**Requirements:**
- vLLM 0.15.1 (install from git: `pip install 'git+https://github.com/vllm-project/vllm@v0.15.1'`)
- Set environment variables:

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
pytest -v -m "cpu and non_batched and not quantized" tests/e2e/test_spyre_non_batched.py
```

**Note:** macOS requires `MASTER_ADDR` and `MASTER_PORT` to be set for vLLM V1 engine multiprocessing. These are automatically set in CI/CD but must be manually exported for local testing.

## Test Files

### tests/e2e/test_spyre_non_batched.py

Contains non-batched tests:

1. **test_single_prompt_non_batched**: Single prompt with B=1
   - 1 prompt
   - max_num_seqs=1
   - Validates against HuggingFace output

2. **test_multiple_prompts_non_batched_sequential**: Multiple prompts processed sequentially
   - 4 prompts processed one-by-one
   - max_num_seqs=1
   - Each prompt processed independently with B=1

## v2.0 Changes

vLLM-Spyre v2.0 removed Continuous Batching (CB) and Static Batching (SB) modes in favor of vLLM V1 API with chunked prefill. Non-batched tests now use:
- vLLM V1 engine exclusively
- `max_num_seqs=1` to enforce non-batched execution
- Chunked prefill with `max_num_batched_tokens=128`

## Troubleshooting

### macOS: RuntimeError: Engine core initialization failed

**Cause:** vLLM V1 engine requires `MASTER_ADDR` and `MASTER_PORT` environment variables.

**Solution:**
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
```

### Version Mismatch: TypeError in KV cache manager

**Cause:** vLLM version mismatch (e.g., using 0.14.1 instead of 0.15.1).

**Solution:**
```bash
pip install 'git+https://github.com/vllm-project/vllm@v0.15.1'
```

## Implementation Details

### How max_num_seqs=1 Works

`max_num_seqs` controls the scheduler's batch size in vLLM:
- `max_num_seqs=4` (default): Process up to 4 requests in parallel
- `max_num_seqs=1` (non-batched): Process 1 request at a time

The value is passed to the vLLM `LLM()` constructor:
```python
LLM(
    model=model_name,
    max_num_seqs=1,  # Non-batched execution
    max_num_batched_tokens=128,
    ...
)
```

### Marker Definition

```python
# pyproject.toml line 172
"non_batched: Tests with batch_size=1 (non-batched execution)"
```
