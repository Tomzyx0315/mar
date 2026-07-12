import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

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


def get_args_parser():
    parser = argparse.ArgumentParser("Sample MAR with a one-step DMD head")
    parser.add_argument("--teacher_ckpt", required=True, type=str)
    parser.add_argument("--dmd_ckpt", required=True, type=str)
    parser.add_argument("--output", default="./samples_mar_dmd.npz", type=str)
    parser.add_argument("--save_png_dir", default="", type=str)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_iter", default=256, type=int)
    parser.add_argument("--cfg", default=None, type=float)
    parser.add_argument("--cfg_schedule", default=None, choices=["linear", "constant"])
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--conditioning_timestep", default=None, type=int)
    parser.add_argument("--use_raw_generator", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--class_labels", default="", type=str)
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
    return parser


def parse_class_labels(value, class_num, num_images):
    if value.strip():
        labels = [int(item.strip()) for item in value.split(",") if item.strip()]
        labels = labels * ((num_images + len(labels) - 1) // len(labels))
        return torch.tensor(labels[:num_images], dtype=torch.long)
    return torch.arange(num_images, dtype=torch.long) % class_num


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    teacher_ckpt = load_checkpoint(args.teacher_ckpt)
    args = fill_from_teacher_args(args, teacher_ckpt)
    teacher_model = build_teacher_model(args, teacher_ckpt, device)

    generator_head, _, _ = create_dmd_heads(teacher_model)
    dmd_path = args.dmd_ckpt
    if os.path.isdir(dmd_path):
        dmd_path = os.path.join(dmd_path, "checkpoint-last.pth")
    dmd_ckpt = load_checkpoint(dmd_path)
    train_args = dmd_ckpt.get("args", None)
    generator_key = "generator_head" if args.use_raw_generator else "generator_head_ema"
    if generator_key not in dmd_ckpt:
        generator_key = "generator_head"
    unwrap_model(generator_head).load_state_dict(dmd_ckpt[generator_key], strict=True)
    generator_head.to(device).eval()
    if args.cfg is None:
        args.cfg = getattr(train_args, "cfg", 3.0)
    if args.cfg_schedule is None:
        args.cfg_schedule = getattr(train_args, "cfg_schedule", "linear")
    if args.conditioning_timestep is None:
        args.conditioning_timestep = getattr(train_args, "conditioning_timestep", DEFAULT_CONDITIONING_TIMESTEP)

    vae = build_vae(args, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.save_png_dir) if args.save_png_dir else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.TemporaryDirectory(dir=str(output_path.parent))
    memmap_path = Path(temp_dir.name) / "samples_uint8.dat"
    samples_array = np.memmap(
        memmap_path,
        dtype=np.uint8,
        mode="w+",
        shape=(args.num_images, args.img_size, args.img_size, 3),
    )

    all_labels = parse_class_labels(args.class_labels, args.class_num, args.num_images).to(device)
    saved = 0
    while saved < args.num_images:
        cur_bsz = min(args.batch_size, args.num_images - saved)
        labels = all_labels[saved:saved + cur_bsz]
        sampled_tokens = sample_tokens_one_step(
            teacher_model,
            generator_head,
            teacher_model.diffloss.train_diffusion,
            batch_size=cur_bsz,
            num_iter=args.num_iter,
            labels=labels,
            cfg=args.cfg,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            conditioning_timestep=args.conditioning_timestep,
            progress=False,
        )
        sampled_images = vae.decode(sampled_tokens / 0.2325)
        sampled_images = ((sampled_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        sampled_images = sampled_images.permute(0, 2, 3, 1).cpu().numpy()

        end = saved + sampled_images.shape[0]
        samples_array[saved:end] = sampled_images
        if png_dir is not None:
            for image in sampled_images:
                Image.fromarray(image).save(png_dir / f"{saved:05d}.png")
                saved += 1
        else:
            saved = end
        print(f"Generated {saved}/{args.num_images}")

    samples_array.flush()
    np.savez(output_path, arr_0=np.asarray(samples_array))
    temp_dir.cleanup()
    print(f"Saved {saved} images to {output_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
