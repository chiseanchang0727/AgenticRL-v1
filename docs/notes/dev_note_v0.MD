- set `VLLM_USE_V1=1`
- set `max-num-seqs`=1 to reduce VRAM usage.
```python
"actor_rollout_ref": {
    "rollout": {
        ...
        "engine_kwargs": {
            "vllm": {
                ...
                "max-num-seqs": 1, # vLLM will only allow ONE active sequence (request) on the GPU at the same time
            }
        },
```