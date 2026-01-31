import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
import wandb
import os
import json

from dpo_loss import (
    dpo_loss,
    margin_compute,
    log_margin,
    empirical_over_threshold_proportion,
    tail_behavior_test,
)

from compute_batch_log_prob import compute_batch_log_prob

from process_hh_dataset import build_train_val_dist

from threshold import WarmupQuantileAccumulator, EMAUpdate

# distributed computation
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp.api import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from functools import partial

auto_wrap = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)


# load yaml config
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# transform the batch to the device
def to_device_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# control randomness
def random_controler(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# distributed setup
def initial_dist():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, rank, world_size, local_rank


# warp model with FSDP
def fsdp_warp(model, use_bf16: bool):
    used_precision = MixedPrecision(
        param_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=used_precision,
        device_id=torch.cuda.current_device(),
    )


# gather the margin across different gpus
def get_global_margin(margin, world_size):
    margin = margin.contiguous()
    gather_margin = [torch.empty_like(margin) for _ in range(world_size)]
    dist.all_gather(gather_margin, margin)
    return torch.cat(gather_margin, dim=0)


# eval
@torch.no_grad()
def evaluate(policy, ref_model, val_dataloader, beta, device, is_rank0: bool):
    policy_was_training = policy.training
    policy.eval()
    ref_model.eval()

    # local accunulators
    eval_local_total_loss = 0.0
    local_total_count = 0

    # only rank0 shows progress bar
    eval_pbar = tqdm(
        val_dataloader,
        desc="Evaluating",
        leave=False,
        dynamic_ncols=True,
        disable=not is_rank0,
    )

    for batch in eval_pbar:
        batch = to_device_batch(batch, device)

        (
            policy_chosen_log_prob,
            policy_rejected_log_prob,
            ref_chosen_log_prob,
            ref_rejected_log_prob,
        ) = compute_batch_log_prob(batch, policy, ref_model)

        eval_loss, eval_chosen_rewards, eval_rejected_rewards = dpo_loss(
            policy_chosen_log_prob=policy_chosen_log_prob,
            policy_rejected_log_prob=policy_rejected_log_prob,
            ref_chosen_log_prob=ref_chosen_log_prob,
            ref_rejected_log_prob=ref_rejected_log_prob,
            beta=beta,
        )

        batch_size = eval_loss.size(0)
        eval_local_total_loss += eval_loss.mean().item() * batch_size
        local_total_count += batch_size

    # package into one tensor then only reduce once
    eval_together_tensor = torch.tensor(
        [eval_local_total_loss, float(local_total_count)],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(eval_together_tensor, op=dist.ReduceOp.SUM)

    global_total_eval__loss_sum = float(eval_together_tensor[0].item())
    global_eval_total_count = int(eval_together_tensor[1].item())
    avg_eval_loss = global_total_eval__loss_sum / max(1, global_eval_total_count)

    if policy_was_training:
        policy.train()

    return avg_eval_loss


# training
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    # distributed device
    device, rank, world_size, local_rank = initial_dist()
    is_rank0 = rank == 0

    random_controler()

    # initial wandb, rank_o only
    if rank == 0:
        wandb.init(
            project=config.get("wandb_project", "handwritten-dpo"),
            name=config.get("run_name", "run"),
            config=config,
        )
    else:
        wandb.init(mode="disabled")

    # use_bf16
    use_bf16 = config["precision"] == "bf16"

    # load model and tokenizer
    policy_name = config["policy_model"]
    ref_model_name = config["ref_model"]
    tok = AutoTokenizer.from_pretrained(policy_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # policy: load on CPU then wrap with FSDP (do NOT .to(device) before FSDP)
    policy = AutoModelForCausalLM.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    # gradient checkpointing
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()

    policy = fsdp_warp(policy, use_bf16=use_bf16)

    # ref model: no grad, keep as normal model on each GPU
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name, torch_dtype=torch.bfloat16
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    policy.train()

    ref_model = fsdp_warp(ref_model, use_bf16=use_bf16)
    ref_model.eval()

    # check fsdp warpper
    def count_fsdp(m):
        return sum(1 for x in m.modules() if isinstance(x, FSDP))

    if rank == 0:
        print("FSDP wrappers:", count_fsdp(policy))
        print("FSDP wrappers:", count_fsdp(ref_model))

    # check model parametres
    if rank == 0:
        for name, module in policy.module.named_modules():
            n = sum(p.numel() for p in module.parameters(recurse=False))
            if n > 5e5:
                print(f"[param-check] {name}: {n}")
        for name, module in ref_model.module.named_modules():
            n = sum(p.numel() for p in module.parameters(recurse=False))
            if n > 5e5:
                print(f"[param-check] {name}: {n}")

    # load dataset
    train_loader, val_loader = build_train_val_dist(
        config=config, tokenizer=tok, rank=rank, world_size=world_size
    )

    # optimizer (MUST be after FSDP wrap)
    optimizer = AdamW(
        params=policy.parameters(), lr=float(config["dpo_training"]["learning_rate"])
    )

    # margin log，only rank0 writes global logs
    tail_test_cfg = config["tail_test"]
    LOG_DIR = tail_test_cfg["log_dir"]
    os.makedirs(LOG_DIR, exist_ok=True)

    JSONL_PATH = os.path.join(LOG_DIR, "global_margins_log.jsonl")
    risk_log_path = os.path.join(LOG_DIR, "global_tail_test_and_beta_log.jsonl")
    log_f = open(risk_log_path, "w", encoding="utf-8") if is_rank0 else None

    # tail proportion test
    delta = float(tail_test_cfg["delta"])
    momentum = float(tail_test_cfg["lambda"])
    q = 1.0 - delta
    threshold = WarmupQuantileAccumulator(q=q)

    tail_stat = {
        "total": 0,
        "fail": 0,
    }

    # training loop
    # every epoch create a folder to save the model_margin
    training_cfg = config["dpo_training"]
    epochs = training_cfg["epochs"]
    log_steps = training_cfg["log_steps"]

    warmup_steps = int(training_cfg["warmup_steps"])
    warmup_done = False
    warmup_count = 0
    global_steps = 0
    beta = float(training_cfg["dpo_beta"])

    for epoch in range(epochs):
        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        if is_rank0:
            os.makedirs(epoch_dir, exist_ok=True)

        # only rank_0 show the bar
        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train | epoch {epoch + 1}/{epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=not is_rank0,
        )

        training_loss = 0.0

        for step, batch in train_pbar:
            batch = to_device_batch(batch, device)

            # autocast for bf16
            with torch.amp.autocast(
                enabled=use_bf16, dtype=torch.bfloat16, device_type="cuda"
            ):
                (
                    policy_chosen_log_prob,
                    policy_rejected_log_prob,
                    ref_chosen_log_prob,
                    ref_rejected_log_prob,
                ) = compute_batch_log_prob(batch, policy, ref_model)

                # compute margin, local rank
                batch_margin = margin_compute(
                    policy_chosen_log_prob=policy_chosen_log_prob,
                    policy_rejected_log_prob=policy_rejected_log_prob,
                    ref_chosen_log_prob=ref_chosen_log_prob,
                    ref_rejected_log_prob=ref_rejected_log_prob,
                )

                # get the global margin across gpus
                global_margin = get_global_margin(batch_margin, world_size)

                if is_rank0 and global_steps == 0:
                    print(
                        "local:",
                        batch_margin.shape,
                        "global:",
                        global_margin.shape,
                        "world_size:",
                        world_size,
                    )

                # warmup golobal
                if not warmup_done:
                    threshold.update(global_margin)
                    warmup_count += 1

                    if is_rank0:
                        threshold.update(global_margin)

                    if (not warmup_done) and (warmup_count == warmup_steps):
                        tau_0 = threshold.finalize()
                        ema = EMAUpdate(tau_0=tau_0, q=q, momentum=momentum)
                        warmup_done = True

                        if is_rank0:
                            log_f.write(
                                json.dumps(
                                    {
                                        "type": "warmup_end",
                                        "tau_0": float(tau_0),
                                        "beta": float(beta),
                                    }
                                )
                                + "\n"
                            )
                            log_f.flush()

                else:
                    # only rank compute the tail proportion and do the test
                    if is_rank0:
                        # EMA update the threshold
                        tau = ema.update_tau(global_margin)

                        # tail proportion compute and risk test
                        p_hat = empirical_over_threshold_proportion(global_margin, tau)
                        is_over_threshold = tail_behavior_test(p_hat=p_hat, delta=delta)

                        # count
                        tail_stat["total"] += 1
                        if is_over_threshold:
                            tail_stat["fail"] += 1

                        # log global
                        log_f.write(
                            json.dumps(
                                {
                                    "step": int(global_steps),
                                    "tau": float(tau),
                                    "p_hat": float(p_hat),
                                    "risk_over": bool(is_over_threshold),
                                    "beta": float(beta),
                                }
                            )
                            + "\n"
                        )
                        log_f.flush()

                # logal global margin
                if is_rank0:
                    log_margin(
                        margin=global_margin,
                        epoch_dir=epoch_dir,
                        epoch=epoch,
                        step=step,
                        JSONL_PATH=JSONL_PATH,
                    )

                # compute loss
                loss_raw, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_log_prob=policy_chosen_log_prob,
                    policy_rejected_log_prob=policy_rejected_log_prob,
                    ref_chosen_log_prob=ref_chosen_log_prob,
                    ref_rejected_log_prob=ref_rejected_log_prob,
                    beta=beta,
                )

                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_global_margin = global_margin.mean()

            optimizer.zero_grad()
            loss.backward()
            # FSDP grad clip
            policy.clip_grad_norm_(max_norm=float(training_cfg["max_grad_norm"]))
            optimizer.step()
            global_steps += 1
            training_loss += loss.item()

            # log the training info (rank0 only)
            if is_rank0 and (step + 1) % log_steps == 0:
                avg_loss = training_loss / log_steps
                train_pbar.set_postfix(loss=f"{avg_loss:.3f}")
                wandb.log(
                    {
                        "loss": avg_loss,
                        "chosen_rewards": avg_chosen_rewards.item(),
                        "rejected_rewards": avg_rejected_rewards.item(),
                        "model_margin": avg_global_margin.item(),
                        "beta": beta,
                    }
                )
                training_loss = 0.0

        # eval: ALL ranks participate to avoid FSDP deadlock
        dist.barrier()
        eval_loss = evaluate(
            policy=policy,
            ref_model=ref_model,
            val_dataloader=val_loader,
            beta=beta,
            device=device,
            is_rank0=is_rank0,
        )
        dist.barrier()

        if is_rank0:
            wandb.log({"eval_loss": eval_loss, "global_step": global_steps})

            print(
                f"[RISK] fail {tail_stat['fail']} / {tail_stat['total']} "
                f"= {tail_stat['fail'] / max(1, tail_stat['total']):.4f}"
            )
    if is_rank0 and log_f is not None:
        log_f.close()

    # save model (rank0 only, gather full state dict)
    dist.barrier()
    # creaate save dir (only rank0)
    save_dir = training_cfg["save_dir"]
    if is_rank0:
        os.makedirs(save_dir, exist_ok=True)

    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        policy,
        StateDictType.FULL_STATE_DICT,
        full_state_cfg,
        full_optim_cfg,
    ):
        cpu_state = policy.state_dict()

    if is_rank0:
        base_model = AutoModelForCausalLM.from_pretrained(
            policy_name,
            torch_dtype=torch.float32,
        )
        base_model.load_state_dict(cpu_state, strict=False)
        base_model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"[SAVE] saved to: {save_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
