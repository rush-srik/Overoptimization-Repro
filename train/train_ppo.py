"""
Train the policy using PPO with the proxy RM as the reward model.

Single GPU:
    python -m train.train_ppo <proxy_size> <micro_bs> <eval_bs> <lr> <grad_ckpt> <seed>

Multi-GPU:
    torchrun --standalone --nproc_per_node=N \
        -m train.train_ppo <proxy_size> <micro_bs> <eval_bs> <lr> <grad_ckpt> <seed>
"""

import os
import json
import math
from copy import deepcopy
import torch
from trl.experimental.ppo import PPOTrainer, PPOConfig
from trl.experimental.ppo.ppo_trainer import truncate_response
from trl.experimental.utils import get_reward
from trl.models.utils import unwrap_model_for_generation
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    AutoModelForSequenceClassification, TrainerCallback, GenerationConfig
from datasets import load_from_disk, Dataset
import sys
import wandb

project_name = "overopt"

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"

proxy_size = sys.argv[1]
seed = int(sys.argv[6])

warmup_steps = 100
lr = float(sys.argv[4])
temperature = 1.0
gae_lambda = 0.95
kl_estimator = "k3"
ppo_clip = 0.2
num_episodes = 1
num_ppo_epochs = 1
micro_batch_size = int(sys.argv[2])
effective_batch_size = 256
eval_batch_size = int(sys.argv[3])
grad_ckpt = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

per_step = micro_batch_size * world_size
assert effective_batch_size % per_step == 0, (
    f"effective batch {effective_batch_size} not divisible by "
    f"micro_bs {micro_batch_size} x world_size {world_size}"
)
grad_accum = effective_batch_size // per_step

max_response_length = 900
missing_eos_penalty = 1.0

policy = AutoModelForCausalLM.from_pretrained(policy_name)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)
policy_tokenizer.padding_side = "left"

ref_model = deepcopy(policy)
value_model = AutoModelForSequenceClassification.from_pretrained(policy_name, num_labels=1)

proxy_rm = AutoModelForSequenceClassification.from_pretrained(f"models/proxy/{proxy_size}")

data = [entry["prompt"] for entry in load_from_disk("data/datasets/train_ppo")]

prompts = [policy_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in data]

ppo_data = [{"input_ids": policy_tokenizer(prompt)["input_ids"]} for prompt in prompts]

train_data = ppo_data

# PPOTrainer has no evaluate() and never computes eval metrics
eval_placeholder = ppo_data[:eval_batch_size]

export_prompts_raw = [entry["prompt"] for entry in load_from_disk("data/datasets/val")]
export_prompts = [policy_tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
) for prompt in export_prompts_raw]

export_stride = eval_batch_size * world_size
n_export = (len(export_prompts_raw) // export_stride) * export_stride

total_updates = math.ceil(num_episodes * len(train_data) / effective_batch_size)

export_steps = [s for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if s < total_updates]
export_steps.append(total_updates)

if world_size > 1:
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_VERSION"] = "2"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "Qwen2DecoderLayer"
    os.environ["FSDP_ACTIVATION_CHECKPOINTING"] = "1" if grad_ckpt else "0"
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"

class ExportGenerations(TrainerCallback):
    """
    Fires ExportTrainer's export() at the chosen steps.
    """
    def __init__(self, trainer, steps):
        self.trainer = trainer
        self.steps = set(steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            kl = state.log_history[-1].get("objective/kl") if state.log_history else None
            self.trainer.export(state.global_step, kl)


class ExportTrainer(PPOTrainer):
    """
    PPOTrainer that saves generations for later gold scoring.
    """
    def export(self, step, kl=None):
        gen_cfg = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=self.args.temperature + 1e-7,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            eos_token_id=self.stop_token_id,
            pad_token_id=self.processing_class.pad_token_id,
        )

        idxs = list(range(rank, n_export, world_size))
        rows = []

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped:
            for start in range(0, len(idxs), eval_batch_size):
                chunk = idxs[start:start + eval_batch_size]
                enc = self.processing_class(
                    [export_prompts[i] for i in chunk],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(self.accelerator.device)

                with torch.no_grad():
                    out = unwrapped.policy.generate(**enc, generation_config=gen_cfg)

                ctx = enc["input_ids"].shape[1]
                resp = truncate_response(
                    self.stop_token_id, self.processing_class.pad_token_id, out[:, ctx:]
                )

                with torch.no_grad():
                    _, proxy_score, _ = get_reward(
                        self.reward_model,
                        torch.cat((enc["input_ids"], resp), dim=1),
                        self.processing_class.pad_token_id,
                        ctx,
                    )
                proxy_score = proxy_score.float().cpu().tolist()

                text = self.processing_class.batch_decode(resp, skip_special_tokens=True)
                rows.extend(
                    {"idx": i, "response": t, "proxy_score": s}
                    for i, t, s in zip(chunk, text, proxy_score)
                )

        rows = gather_object(rows)

        if self.accelerator.is_main_process:
            rows.sort(key=lambda r: r["idx"])
            out_dir = f"runs/ppo/{proxy_size}/{seed}/gen/step-{step}"
            scores = [r["proxy_score"] for r in rows]
            Dataset.from_dict({
                "prompt": [export_prompts_raw[r["idx"]] for r in rows],
                "response": [r["response"] for r in rows],
                "proxy_score": scores,
            }).save_to_disk(out_dir)
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({
                    "step": step,
                    "kl": kl,
                    "sqrt_kl": math.sqrt(kl) if kl is not None and kl > 0 else None,
                    "proxy_score_mean": sum(scores) / len(scores) if scores else None,
                    "proxy_size": proxy_size,
                    "seed": seed,
                    "temperature": self.args.temperature,
                    "n_prompts": len(rows),
                }, f, indent=2)

config = PPOConfig(
    # Training
    temperature=temperature,
    kl_estimator=kl_estimator,
    kl_coef=0.0,
    cliprange=ppo_clip,
    lam=gae_lambda,
    learning_rate=lr,
    warmup_steps=warmup_steps,
    per_device_train_batch_size=micro_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_accum,
    num_ppo_epochs=num_ppo_epochs,
    num_train_epochs=num_episodes,

    # Sharding / activation memory
    gradient_checkpointing=grad_ckpt and world_size == 1,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,

    # Logging
    output_dir=f"runs/ppo/{proxy_size}/{seed}",
    report_to=["wandb"],
    eval_strategy="steps",
    save_strategy="no",

    # Misc
    response_length=max_response_length,
    missing_eos_penalty=missing_eos_penalty,
    stop_token="eos",
    seed=seed,
    num_sample_generations=0,
)

trainer = ExportTrainer(
    args=config,
    processing_class=policy_tokenizer,
    model=policy,
    ref_model=ref_model,
    value_model=value_model,
    reward_model=proxy_rm,
    train_dataset=train_data,
    eval_dataset=eval_placeholder
)

trainer.add_callback(ExportGenerations(trainer, export_steps))

if rank == 0:
    wandb.init(
        project=project_name,
        name=f"ppo-{proxy_size}-{seed}",
        group="ppo",
        config={
            "lr": lr,
            "effective_batch": effective_batch_size,
            "micro_batch": micro_batch_size,
            "world_size": world_size,
            "grad_ckpt": grad_ckpt,
            "gae_lambda": gae_lambda,
            "kl_estimator": kl_estimator,
            "ppo_clip": ppo_clip,
            "temperature": temperature,
            "num_episodes": num_episodes,
            "num_ppo_epochs": num_ppo_epochs,
            "micro_batch_size": micro_batch_size,
            "effective_batch_size": effective_batch_size,
            "eval_batch_size": eval_batch_size,
            "grad_ckpt": grad_ckpt
        }
    )

trainer.train()
trainer.save_model(f"models/ppo/{proxy_size}/{seed}")