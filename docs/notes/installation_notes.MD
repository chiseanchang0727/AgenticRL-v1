# Notes when installing dependencies

"flash-attn==2.7.4.post1",

## Verl

- Python: Version >= 3.10
- CUDA: Version >= 12.8

## Recommended Dependency Installation Order for Environment Setup
When setting up the environment, **dependency installation order matters** and can affect
whether `uv` resolves compatible versions successfully.

### Key Observation

- Installing **`verl` first** significantly improves dependency resolution.
- Installing other libraries before `verl` may lead to version conflicts
  (especially with `torch`, `vllm`, `tensordict`, and `numpy`).


```bash
uv add verl
# then install remaining dependencies
uv add <other-packages>
```

---

## CUDA Configuration for `uv`-Managed Virtual Environments

### Scope

This guideline applies to projects that:
- Use **CUDA / NVIDIA GPUs**
- Use **`uv` virtual environments**
- Build or run CUDA-dependent libraries (PyTorch, flash-attn, vLLM, etc.)

### Problem

When activating a `uv`-managed virtual environment, `PATH` may be reordered, causing
the system to resolve the wrong `nvcc`.

### Example

**System shell (correct):**
```bash
$ which nvcc
/usr/local/cuda-12.8/bin/nvcc
```
After activating venv (incorrect):
```bash
$ source .venv/bin/activate
(venv) $ which nvcc
/usr/bin/nvcc
```
This can lead to CUDA build failures or runtime errors.

### Root cause
- `uv` / virtualenv modifies PATH during activation
- `/usr/bin` may take precedence over `$CUDA_HOME/bin`.
- CUDA paths are not guaranteed to be inherited automatically

### Solution
CUDA variables must be explicitly re-exported inside the virtual environment.

1. Edit the venv activate script
```bash
nano .venv/bin/activate
```
2. Append to the end of the file:
```bash
# --- CUDA override (force after uv activate) ---
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
3.Reload the venv:
```bash
deactivate
source .venv/bin/activate
```

---

## 5060ti is sm120, need to install compatible `torch` version.

```python
import torch

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Supported archs:", torch.cuda.get_arch_list())

cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
print("Compatible:", cap in torch.cuda.get_arch_list())
```