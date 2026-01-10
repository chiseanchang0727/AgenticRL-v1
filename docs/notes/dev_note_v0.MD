

# Model test
- Script: `training_test.py`

## `facebook/opt-350m`
- Fail
- Error: `AttributeError: 'OPTConfig' object has no attribute 'text_config'`

## `google/gemma-3-270m`

- Fail but work after modifying FSDP wrap logic 
- Reason: `model._no_split_modules` contains class names that are not in model layer
- Model layers:
```text
{'GELUTanh',
 'Gemma3Attention',
 'Gemma3DecoderLayer',
 'Gemma3ForCausalLM',
 'Gemma3MLP',
 'Gemma3RMSNorm',
 'Gemma3RotaryEmbedding',
 'Gemma3TextModel',
 'Gemma3TextScaledWordEmbedding',
 'Linear',
 'ModuleList'}
```
- But `_no_split_modules` are: `['Gemma3DecoderLayer', 'SiglipVisionEmbeddings', 'SiglipEncoderLayer', 'SiglipMultiheadAttentionPoolingHead']`, as we can see there are three of them are not in model layers
- So it encounter the error in `verl/utils/fsdp_utils.py`:
```python
for layer_class in fsdp_transformer_layer_cls_to_wrap:
    transformer_cls = get_module_class_from_name(module, layer_class)
    if transformer_cls is None:
        raise Exception("Could not find the transformer layer class to wrap in the model.")
```

- Then the question becomes why there is no `SiglipVisionEmbeddings` in the model layers, because it's GemmaText rather than GemmaVision?


- Fix: ignore unresolved classes when constructing FSDP wrap policy

```python
# verl/utils/fsdp_utils.py
elif fsdp_transformer_layer_cls_to_wrap is not None:
    transformer_cls_to_wrap = set()
    for layer_class in fsdp_transformer_layer_cls_to_wrap:
        transformer_cls = get_module_class_from_name(module, layer_class)
        if transformer_cls is not None:
            transformer_cls_to_wrap.add(transformer_cls)
```

## Quantized model
- `verl` can't integrate with quantized model such as `unsloth/Qwen3-8B-bnb-4bit`, the error code is:
```python
```

## LRA adapter
- Tried following config but still OOM
```python
"model": {
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
            "lora_rank": 64,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            
        },
```

Next steps:
- Test smaller model
- figure out which step calculate memory usage
- figure out how did the code call GPU

## 0106

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

