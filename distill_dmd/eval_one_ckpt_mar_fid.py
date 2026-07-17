import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from distill_dmd.mar_dmd import (
    DEFAULT_CONDITIONING_TIMESTEP,
    create_dmd_heads,
    sample_tokens_one_step,
    unwrap_model,
)
from distill_dmd.train_mar_dmd import (
    build_teacher_model,
    build_vae,
    fill_from_teacher_args,
    load_checkpoint,
)
from engine_mar import evaluate
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate one MAR-DMD checkpoint with MAR's original FID path")
    parser.add_argument("--teacher_ckpt", required=True, type=str)
    parser.add_argument("--dmd_ckpt", required=True, type=str)
    parser.add_argument("--output_dir", default="./fid_eval_mar", type=str)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_iter", default=256, type=int)
    parser.add_argument("--cfg", default=None, type=float)
    parser.add_argument("--cfg_schedule", default=None, choices=["linear", "constant"])
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--conditioning_timestep", default=None, type=int)
    parser.add_argument("--use_raw_generator", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--img_size", default=None, type=int)
    parser.add_argument("--vae_path", default=None, type=str)
    parser.add_argument("--vae_embed_dim", default=None, type=int)
    parser.add_argument("--vae_stride", default=None, type=int)
    parser.add_argument("--patch_size", default=None, type=int)
    parser.add_argument("--mask_ratio_min", default=None, type=float)
    parser.add_argument("--label_drop_prob", default=None, type=float)
    parser.add_argument("--class_num", default=None, type=int)
    parser.add_argument("--attn_dropout", default=None, type=float)
    parser.add_argument("--proj_dropout", default=None, type=float)
    parser.add_argument("--buffer_size", default=None, type=int)
    parser.add_argument("--diffloss_d", default=None, type=int)
    parser.add_argument("--diffloss_w", default=None, type=int)
    parser.add_argument("--num_sampling_steps", default=None, type=str)
    parser.add_argument("--diffusion_batch_mul", default=None, type=int)
    parser.add_argument("--grad_checkpointing", action="store_true")

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", type=str)
    return parser


def setting_from_cli_or_ckpt(args, train_args, key, default):
    value = getattr(args, key)
    if value is not None:
        return value
    if train_args is None:
        return default
    if isinstance(train_args, dict):
        return train_args.get(key, default)
    return getattr(train_args, key, default)


class DmdSampleWrapper(nn.Module):
    def __init__(self, teacher_model, generator_head, conditioning_timestep):
        super().__init__()
        self.teacher_model = teacher_model
        self.generator_head = generator_head
        self.conditioning_timestep = conditioning_timestep

    @torch.no_grad()
    def sample_tokens(
        self,
        bsz,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False,
    ):
        return sample_tokens_one_step(
            self.teacher_model,
            self.generator_head,
            self.teacher_model.diffloss.train_diffusion,
            batch_size=bsz,
            num_iter=num_iter,
            labels=labels,
            cfg=cfg,
            cfg_schedule=cfg_schedule,
            temperature=temperature,
            conditioning_timestep=self.conditioning_timestep,
            progress=progress,
        )


class ScalarLogger:
    def __init__(self):
        self.log_dir = "mar_fid"
        self.values = {}

    def add_scalar(self, tag, value, step):
        self.values[tag] = float(value)
        print(f"{tag}: {float(value):.6f} at step {step}")


@torch.no_grad()
def main(args):
    misc.init_distributed_mode(args)
    if not misc.is_dist_avail_and_initialized():
        raise RuntimeError(
            "engine_mar.evaluate calls torch.distributed.barrier(); launch this script with "
            "`torchrun --nproc_per_node=1` for single-GPU evaluation, or a larger "
            "`--nproc_per_node` for multi-GPU evaluation."
        )
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    teacher_ckpt = load_checkpoint(args.teacher_ckpt)
    args = fill_from_teacher_args(args, teacher_ckpt)
    teacher_model = build_teacher_model(args, teacher_ckpt, device)

    generator_head, _, _ = create_dmd_heads(teacher_model)
    dmd_path = args.dmd_ckpt
    if os.path.isdir(dmd_path):
        dmd_path = os.path.join(dmd_path, "checkpoint-last.pth")
    dmd_ckpt = load_checkpoint(dmd_path)
    train_args = dmd_ckpt.get("args", None) if isinstance(dmd_ckpt, dict) else None

    generator_key = "generator_head" if args.use_raw_generator else "generator_head_ema"
    if generator_key not in dmd_ckpt:
        generator_key = "generator_head"
    if generator_key not in dmd_ckpt:
        raise ValueError(f"{dmd_path} does not contain generator_head or generator_head_ema")
    unwrap_model(generator_head).load_state_dict(dmd_ckpt[generator_key], strict=True)
    generator_head.to(device).eval()

    args.cfg = setting_from_cli_or_ckpt(args, train_args, "cfg", 3.0)
    args.cfg_schedule = setting_from_cli_or_ckpt(args, train_args, "cfg_schedule", "linear")
    args.conditioning_timestep = setting_from_cli_or_ckpt(
        args,
        train_args,
        "conditioning_timestep",
        DEFAULT_CONDITIONING_TIMESTEP,
    )
    args.num_sampling_steps = "one_step_dmd"
    args.evaluate = True

    vae = build_vae(args, device)
    wrapper = DmdSampleWrapper(teacher_model, generator_head, args.conditioning_timestep).to(device).eval()
    logger = ScalarLogger() if misc.is_main_process() else None

    if misc.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Evaluating DMD checkpoint: {dmd_path}")
        print(f"Using generator key: {generator_key}")
        print(
            "Sampling settings: "
            f"num_images={args.num_images}, batch_size={args.batch_size}, "
            f"num_iter={args.num_iter}, cfg={args.cfg}, "
            f"cfg_schedule={args.cfg_schedule}, conditioning_timestep={args.conditioning_timestep}"
        )

    evaluate(
        wrapper,
        vae,
        ema_params=None,
        args=args,
        epoch=0,
        batch_size=args.batch_size,
        log_writer=logger,
        cfg=args.cfg,
        use_ema=False,
    )

    if misc.is_main_process():
        fid_key = "fid" if args.cfg == 1.0 else f"fid_cfg{args.cfg}"
        is_key = "is" if args.cfg == 1.0 else f"is_cfg{args.cfg}"
        if fid_key in logger.values:
            print(f"Final FID: {logger.values[fid_key]:.6f}")
        if is_key in logger.values:
            print(f"Final Inception Score: {logger.values[is_key]:.6f}")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
