import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from distill_dmd.mar_dmd import (
    compute_cfg_scale,
    compute_fake_loss,
    compute_generator_dmd_loss,
    build_teacher_forcing_context,
    create_dmd_heads,
    sample_tokens_one_step,
    unwrap_model,
)
from models import mar
from models.vae import AutoencoderKL
from util.crop import center_crop_arr
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser("Distill MAR DiffLoss with DMD", add_help=False)
    parser.add_argument("--teacher_ckpt", required=True, type=str)
    parser.add_argument("--output_dir", default="./output_mar_dmd", type=str)
    parser.add_argument("--resume", default="", type=str)

    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--train_iters", default=100000, type=int)
    parser.add_argument("--save_iters", default=1000, type=int)
    parser.add_argument("--log_iters", default=20, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--generator_lr", default=1e-5, type=float)
    parser.add_argument("--fake_lr", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument("--dfake_gen_update_ratio", default=5, type=int)
    parser.add_argument("--min_step_percent", default=0.02, type=float)
    parser.add_argument("--max_step_percent", default=0.98, type=float)
    parser.add_argument("--cfg", default=3.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear", choices=["linear", "constant"])
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_batch", default=0, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--preview_iters", default=0, type=int)
    parser.add_argument("--preview_num_images", default=16, type=int)
    parser.add_argument("--preview_num_iter", default=64, type=int)
    parser.add_argument("--preview_class_labels", default="", type=str)

    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1, type=int)

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


def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint-last.pth")
    return torch.load(path, map_location="cpu")


def fill_from_teacher_args(args, checkpoint):
    teacher_args = checkpoint.get("args", None)
    fallback = {
        "model": "mar_large",
        "img_size": 256,
        "vae_path": "pretrained_models/vae/kl16.ckpt",
        "vae_embed_dim": 16,
        "vae_stride": 16,
        "patch_size": 1,
        "mask_ratio_min": 0.7,
        "label_drop_prob": 0.1,
        "class_num": 1000,
        "attn_dropout": 0.1,
        "proj_dropout": 0.1,
        "buffer_size": 64,
        "diffloss_d": 12,
        "diffloss_w": 1536,
        "num_sampling_steps": "100",
        "diffusion_batch_mul": 1,
    }
    for key, value in fallback.items():
        if getattr(args, key) is None:
            if teacher_args is not None and hasattr(teacher_args, key):
                setattr(args, key, getattr(teacher_args, key))
            else:
                setattr(args, key, value)
    return args


def build_teacher_model(args, checkpoint, device):
    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )
    state_dict = checkpoint.get("model_ema", None)
    if state_dict is None:
        state_dict = checkpoint["model"]
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded EMA teacher: {msg}")
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def save_checkpoint(args, generator_head, fake_head, optimizer_generator, optimizer_fake, scaler, step):
    if not misc.is_main_process():
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator_head": unwrap_model(generator_head).state_dict(),
        "fake_head": unwrap_model(fake_head).state_dict(),
        "optimizer_generator": optimizer_generator.state_dict(),
        "optimizer_fake": optimizer_fake.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "args": args,
    }
    torch.save(payload, output_dir / "checkpoint-last.pth")
    torch.save(payload, output_dir / f"checkpoint-{step:06d}.pth")


def parse_preview_labels(value, class_num, num_images, device):
    if value.strip():
        labels = [int(item.strip()) for item in value.split(",") if item.strip()]
        labels = labels * ((num_images + len(labels) - 1) // len(labels))
        return torch.tensor(labels[:num_images], dtype=torch.long, device=device)
    return (torch.arange(num_images, dtype=torch.long, device=device) % class_num)


def save_preview_grid(args, teacher_model, vae, generator_head, step, device):
    if args.preview_iters <= 0 or not misc.is_main_process():
        return

    generator_module = unwrap_model(generator_head)
    was_training = generator_module.training
    generator_module.eval()

    preview_dir = Path(args.output_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    labels = parse_preview_labels(
        args.preview_class_labels,
        args.class_num,
        args.preview_num_images,
        device,
    )
    sampled_tokens = sample_tokens_one_step(
        teacher_model,
        generator_module,
        teacher_model.diffloss.train_diffusion,
        batch_size=args.preview_num_images,
        num_iter=args.preview_num_iter,
        labels=labels,
        cfg=args.cfg,
        cfg_schedule=args.cfg_schedule,
        temperature=args.temperature,
        progress=False,
    )
    sampled_images = vae.decode(sampled_tokens / 0.2325)
    sampled_images = ((sampled_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    sampled_images = sampled_images.permute(0, 2, 3, 1).cpu().numpy()

    grid_cols = int(np.ceil(np.sqrt(args.preview_num_images)))
    grid_rows = int(np.ceil(args.preview_num_images / grid_cols))
    image_h, image_w = sampled_images.shape[1:3]
    grid = Image.new("RGB", (grid_cols * image_w, grid_rows * image_h), color=(255, 255, 255))
    for index, image in enumerate(sampled_images):
        row = index // grid_cols
        col = index % grid_cols
        grid.paste(Image.fromarray(image), (col * image_w, row * image_h))
    grid.save(preview_dir / f"step_{step:06d}.png")

    if was_training:
        generator_module.train()


def maybe_resume(args, generator_head, fake_head, optimizer_generator, optimizer_fake, scaler):
    if not args.resume:
        return 0
    resume_path = args.resume
    if os.path.isdir(resume_path):
        resume_path = os.path.join(resume_path, "checkpoint-last.pth")
    checkpoint = torch.load(resume_path, map_location="cpu")
    unwrap_model(generator_head).load_state_dict(checkpoint["generator_head"], strict=True)
    unwrap_model(fake_head).load_state_dict(checkpoint["fake_head"], strict=True)
    if "optimizer_generator" in checkpoint:
        optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
    if "optimizer_fake" in checkpoint:
        optimizer_fake.load_state_dict(checkpoint["optimizer_fake"])
    if scaler is not None and checkpoint.get("scaler", None) is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    print(f"Resumed MAR-DMD checkpoint from {resume_path}")
    return int(checkpoint.get("step", 0))


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    teacher_ckpt = load_checkpoint(args.teacher_ckpt)
    args = fill_from_teacher_args(args, teacher_ckpt)

    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    print("{}".format(args).replace(", ", ",\n"))

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform_train)
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path)
    vae.to(device).eval()
    vae.requires_grad_(False)

    teacher_model = build_teacher_model(args, teacher_ckpt, device)
    generator_head, fake_head, teacher_head = create_dmd_heads(teacher_model)
    generator_head.to(device).train()
    fake_head.to(device).train()
    teacher_head.to(device).eval()

    if args.distributed:
        generator_head = DDP(generator_head, device_ids=[args.gpu])
        fake_head = DDP(fake_head, device_ids=[args.gpu])

    optimizer_generator = torch.optim.AdamW(generator_head.parameters(), lr=args.generator_lr, weight_decay=args.weight_decay)
    optimizer_fake = torch.optim.AdamW(fake_head.parameters(), lr=args.fake_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    step = maybe_resume(args, generator_head, fake_head, optimizer_generator, optimizer_fake, scaler)
    min_step = int(args.min_step_percent * teacher_model.diffloss.train_diffusion.num_timesteps)
    max_step = int(args.max_step_percent * teacher_model.diffloss.train_diffusion.num_timesteps)
    max_step = min(max_step, teacher_model.diffloss.train_diffusion.num_timesteps - 1)

    start_time = time.time()
    epoch = 0
    while step < args.train_iters:
        sampler_train.set_epoch(epoch)
        for samples, labels in data_loader_train:
            if step >= args.train_iters:
                break
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                posterior = vae.encode(samples)
                latent_images = posterior.sample().mul_(0.2325)
                context = build_teacher_forcing_context(
                    teacher_model,
                    latent_images,
                    labels,
                    max_tokens=args.max_tokens_per_batch,
                )
            cond = context["cond"]
            uncond_cond = context["uncond_cond"]
            cfg_scale = compute_cfg_scale(args.cfg, args.cfg_schedule, context["visible_fraction"])

            generator_logs = {}
            if step % args.dfake_gen_update_ratio == 0:
                optimizer_generator.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    loss_dm, generator_logs = compute_generator_dmd_loss(
                        generator_head,
                        fake_head,
                        teacher_head,
                        teacher_model.diffloss.train_diffusion,
                        cond,
                        uncond_cond,
                        teacher_model.token_embed_dim,
                        cfg_scale,
                        min_step,
                        max_step,
                        temperature=args.temperature,
                    )
                scaler.scale(loss_dm).backward()
                scaler.unscale_(optimizer_generator)
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(generator_head.parameters(), args.max_grad_norm)
                scaler.step(optimizer_generator)
                scaler.update()
            else:
                generator_grad_norm = torch.tensor(0.0, device=device)

            optimizer_fake.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                loss_fake, fake_logs = compute_fake_loss(
                    generator_head,
                    fake_head,
                    teacher_model.diffloss.train_diffusion,
                    cond,
                    teacher_model.token_embed_dim,
                    cfg_scale,
                    temperature=args.temperature,
                )
            scaler.scale(loss_fake).backward()
            scaler.unscale_(optimizer_fake)
            fake_grad_norm = torch.nn.utils.clip_grad_norm_(fake_head.parameters(), args.max_grad_norm)
            scaler.step(optimizer_fake)
            scaler.update()

            if step % args.log_iters == 0:
                loss_fake_value = misc.all_reduce_mean(fake_logs["loss_fake"])
                msg = (
                    f"step {step}/{args.train_iters} "
                    f"loss_fake {loss_fake_value:.6f} "
                    f"fake_grad {float(fake_grad_norm):.4f}"
                )
                if generator_logs:
                    loss_dm_value = misc.all_reduce_mean(generator_logs["loss_dm"])
                    msg += (
                        f" loss_dm {loss_dm_value:.6f}"
                        f" gen_grad {float(generator_grad_norm):.4f}"
                        f" cfg {generator_logs['cfg_scale']:.3f}"
                        f" dm_grad {generator_logs['dm_grad_norm']:.4f}"
                    )
                print(msg)

            step += 1
            if step % args.save_iters == 0 or step == args.train_iters:
                save_checkpoint(args, generator_head, fake_head, optimizer_generator, optimizer_fake, scaler, step)
            if args.preview_iters > 0 and (step % args.preview_iters == 0 or step == args.train_iters):
                save_preview_grid(args, teacher_model, vae, generator_head, step, device)
                if misc.is_dist_avail_and_initialized():
                    torch.distributed.barrier()

        epoch += 1

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")


if __name__ == "__main__":
    parser = get_args_parser()
    parsed_args = parser.parse_args()
    main(parsed_args)
