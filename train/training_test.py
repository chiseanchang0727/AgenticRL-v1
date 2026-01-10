from typing import Dict, Any
import agentlightning as agl
from train.training_interface import TrainableLitAgent
from data.loader import load_train_val_dataset


RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": None,
        "val_files": None,
        "train_batch_size": 1,
        "max_prompt_length": 10,
        "max_response_length": 11,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,  # Generate n responses per sampling
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            # "dtype": "bfloat16",
            "max_num_batched_tokens": 12,
            "max_num_seqs": 13,
            "max_model_len": 14,
            "gpu_memory_utilization": 0.4,
            "enforce_eager": False,
            "free_cache_engine": True,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                    "max-num-seqs": 1, # vLLM will only allow ONE active sequence (request) on the GPU at the same time
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 3,
            "ppo_micro_batch_size_per_gpu": 3,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            # "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "path": "google/gemma-3-270m",
            # "path": "google/gemma-3-4b-it",
            # "path": "Qwen/Qwen3-0.6B",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
            # "lora_rank": 64,
            # "lora_alpha": 32,
            # "target_modules": "all-linear",
            # "lora_adapter_path":"", # path to a pretrained LoRA adapter directory.
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console"], # , "wandb"
        "project_name": "AgenticRL",
        "experiment_name": "v1",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


train_target = 'planner'
agent = TrainableLitAgent(trained_agents=train_target)
algorithm = agl.VERL(config=RL_TRAINING_CONFIG)
trainer = agl.Trainer(
    n_runners=2,  # Run n agents in parallel to try out the prompts
    algorithm=algorithm, 
    adapter={"agent_match": train_target}
)

dataset_train, dataset_val = load_train_val_dataset()

import torch
import gc

def main():
    gc.collect()
    torch.cuda.empty_cache()
    trainer.fit(agent=agent, train_dataset=dataset_train, val_dataset=dataset_val)

if __name__ == '__main__':
    main()