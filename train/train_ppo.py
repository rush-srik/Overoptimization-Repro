"""
Train the policy using PPO with the proxy RM as the reward model.

Single GPU:
    python -m train.train_ppo <proxy_size> <micro_bs> <rollout_bs> <eval_bs> <lr> <grad_ckpt> <gen_bs>

Multi-GPU:
    torchrun --standalone --nproc_per_node=N \
        -m train.train_ppo <proxy_size> <micro_bs> <rollout_bs> <eval_bs> <lr> <grad_ckpt> <gen_bs>
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

# Monkey-patch batch_generation to reduce memory
_ppo = sys.modules["trl.experimental.ppo.ppo_trainer"]
_orig_selective_log_softmax = _ppo.selective_log_softmax
_orig_pad = _ppo.pad


@torch.no_grad()
def _batch_generation_reduce_early(
    model, queries, local_rollout_forward_batch_size, pad_token_id, generation_config
):
    """
    Doesn't save full-batch logits to save memory.
    """
    batch_size = queries.shape[0]
    context_length = queries.shape[1]
    gen_bs = gen_batch_size or local_rollout_forward_batch_size

    was_training = model.training
    model.eval()
    try:
        query_responses = []
        for i in range(0, batch_size, gen_bs):
            q = queries[i : i + gen_bs]
            attention_mask = q != pad_token_id
            out = model.generate(
                input_ids=torch.masked_fill(q, ~attention_mask, 0),
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            query_responses.append(torch.cat((q, out[:, context_length:]), dim=1))

        padded_query_responses = _orig_pad(
            query_responses, padding_value=pad_token_id, padding_side="right"
        )
        padded_query_responses = padded_query_responses.view(
            -1, padded_query_responses.shape[-1]
        )[:batch_size]

        logprobss = []
        for i in range(0, batch_size, local_rollout_forward_batch_size):
            qr = padded_query_responses[i : i + local_rollout_forward_batch_size]
            logits = _ppo.forward(model, qr, pad_token_id).logits[:, context_length - 1: -1]
            logits = logits / (generation_config.temperature or 1.0)
            logprobss.append(_orig_selective_log_softmax(logits, qr[:, context_length:]))
            del logits
    finally:
        model.train(was_training)

    return padded_query_responses, torch.cat(logprobss, 0)


def _selective_log_softmax_passthrough(logits, index):
    if logits.dim() == 2:
        return logits
    return _orig_selective_log_softmax(logits, index)


_ppo.batch_generation = _batch_generation_reduce_early
_ppo.selective_log_softmax = _selective_log_softmax_passthrough

project_name = "overopt"

policy_name = "Qwen/Qwen2.5-1.5B-Instruct"

proxy_size = sys.argv[1]
warmup_steps = 10
lr = float(sys.argv[5])
temperature = 1.0
gae_lambda = 0.95
kl_estimator = "k3"
ppo_clip = 0.2
num_episodes = 1
num_ppo_epochs = 1
num_mini_batches = 32
micro_batch_size = int(sys.argv[2])
rollout_forward_batch_size = int(sys.argv[3])
gen_batch_size = int(sys.argv[7]) if len(sys.argv) > 7 else None
score_batch_size = 8
effective_batch_size = 256
max_updates = 116
eval_batch_size = int(sys.argv[4])
grad_ckpt = bool(int(sys.argv[6])) if len(sys.argv) > 6 else False
grad_clipping = 2.0

world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

per_step = micro_batch_size * world_size
assert effective_batch_size % per_step == 0, (
    f"effective batch {effective_batch_size} not divisible by "
    f"micro_bs {micro_batch_size} x world_size {world_size}"
)
grad_accum = effective_batch_size // per_step

max_response_length = 512
missing_eos_penalty = 1.0

policy = AutoModelForCausalLM.from_pretrained(policy_name)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_name)
policy_tokenizer.padding_side = "left"
policy.generation_config.repetition_penalty = 1.0

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

train_data = ppo_data[:max_updates * effective_batch_size]

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

hidden_size = policy.config.hidden_size

ds_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": hidden_size * hidden_size,
        "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
        "stage3_param_persistence_threshold": 10 * hidden_size,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": grad_accum // num_mini_batches,
    "gradient_clipping": grad_clipping,
    "steps_per_print": 10 ** 9,
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},
    "wall_clock_breakdown": False,
} if world_size > 1 else None

if ds_config is not None:
    run_dir = f"runs/ppo/{proxy_size}"
    os.makedirs(run_dir, exist_ok=True)
    ds_config_path = os.path.join(run_dir, f"ds_config.rank{rank}.json")
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = ds_config_path

if grad_ckpt:
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    value_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

class ExportGenerations(TrainerCallback):
    """
    Fires ExportTrainer's export() at the chosen steps.
    """
    def __init__(self, trainer, steps):
        self.trainer = trainer
        self.steps = set(steps)

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.export(0, 0.0)

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
            repetition_penalty=policy.generation_config.repetition_penalty,
        )

        idxs = list(range(rank, n_export, world_size))
        rows = []
        generated = []

        was_training = self.model.training
        self.model.eval()
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
                generated.append((chunk, enc["input_ids"].cpu(), resp.cpu()))

        for chunk, query_ids, resp in generated:
            ctx = query_ids.shape[1]
            proxy_score = []
            for j in range(0, len(chunk), score_batch_size):
                q = query_ids[j : j + score_batch_size].to(self.accelerator.device)
                r = resp[j : j + score_batch_size].to(self.accelerator.device)
                with torch.no_grad():
                    _, s, _ = get_reward(
                        self.reward_model,
                        torch.cat((q, r), dim=1),
                        self.processing_class.pad_token_id,
                        ctx,
                    )
                proxy_score.extend(s.float().cpu().tolist())
                del q, r, s

            resp = resp.to(self.accelerator.device)
            text = self.processing_class.batch_decode(resp, skip_special_tokens=True)
            rows.extend(
                {"idx": i, "response": t, "proxy_score": s}
                for i, t, s in zip(chunk, text, proxy_score)
            )

        self.model.train(was_training)

        rows = gather_object(rows)

        if self.accelerator.is_main_process:
            rows.sort(key=lambda r: r["idx"])
            out_dir = f"runs/ppo/{proxy_size}/gen/step-{step}"
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
                    "proxy_score_mean": sum(scores) / len(scores) if scores else None,
                    "proxy_size": proxy_size,
                    "n_prompts": len(rows),
                }, f, indent=2)

    def save_model(self, output_dir=None, _internal_call=False):
        """
        Gather from the engine ourselves to bypass
        broken TRL + DeepSpeed save.
        """
        if not self.is_deepspeed_enabled:
            return super().save_model(output_dir, _internal_call)

        state_dict = self.accelerator.get_state_dict(self.model_wrapped)

        if self.accelerator.is_main_process:
            policy_state = {
                k[len("policy."):]: v
                for k, v in state_dict.items()
                if k.startswith("policy.")
            }
            assert policy_state, f"no policy.* keys in {list(state_dict)[:5]}"
            policy.save_pretrained(output_dir, state_dict=policy_state)
            self.processing_class.save_pretrained(output_dir)

        self.accelerator.wait_for_everyone()

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
    num_mini_batches=num_mini_batches,

    # Sharding / activation memory.
    bf16=True,
    world_size=world_size,
    ds3_gather_for_generation=True,
    gradient_checkpointing=grad_ckpt,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,

    # Logging
    output_dir=f"runs/ppo/{proxy_size}",
    report_to=["wandb"],
    eval_strategy="steps",
    save_strategy="no",

    # Misc
    local_rollout_forward_batch_size=rollout_forward_batch_size,
    response_length=max_response_length,
    missing_eos_penalty=missing_eos_penalty,
    stop_token="eos",
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
        name=f"ppo-{proxy_size}",
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
trainer.save_model(f"models/ppo/{proxy_size}")