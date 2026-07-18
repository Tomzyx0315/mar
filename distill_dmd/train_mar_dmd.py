import argparse
import copy
import datetime
import json
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
from torch.utils.tensorboard import SummaryWriter

from distill_dmd.tar_dataset import ImageNetTarDataset, MarTrainTransform
from distill_dmd.mar_dmd import (
    DEFAULT_CONDITIONING_TIMESTEP,
    compute_fake_loss_from_tokens,
    compute_generator_dmd_loss_from_tokens,
    build_teacher_forcing_context,
    create_dmd_heads,
    one_step_generate,
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
    parser.add_argument(
        "--log_dir",
        default="",
        type=str,
        help="TensorBoard directory; defaults to <output_dir>/tensorboard",
    )
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--override_resume_lr", action="store_true")

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
    parser.add_argument("--dm_loss_weight", default=1.0, type=float)
    parser.add_argument("--gan_classifier", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", default=3e-3, type=float)
    parser.add_argument("--guidance_cls_loss_weight", default=1e-2, type=float)
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", default=0, type=int)
    parser.add_argument("--min_step_percent", default=0.02, type=float)
    parser.add_argument("--max_step_percent", default=0.98, type=float)
    parser.add_argument("--cfg", default=3.0, type=float)
    parser.add_argument("--cfg_schedule", default="linear", choices=["linear", "constant"])
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--conditioning_timestep", default=DEFAULT_CONDITIONING_TIMESTEP, type=int)
    parser.add_argument("--generator_ema_rate", default=0.999, type=float)
    parser.add_argument("--max_tokens_per_batch", default=0, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--preview_iters", default=0, type=int)
    parser.add_argument("--preview_num_images", default=16, type=int)
    parser.add_argument("--preview_num_iter", default=256, type=int)
    parser.add_argument("--preview_class_labels", default="", type=str)

    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--tar_index_path", default="", type=str)
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
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_teacher_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and checkpoint.get("model_ema", None) is not None:
        state_dict = checkpoint["model_ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Teacher checkpoint must contain a MAR state_dict.")

    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _teacher_arg(teacher_args, key, default=None):
    if teacher_args is None:
        return default
    if isinstance(teacher_args, dict):
        return teacher_args.get(key, default)
    return getattr(teacher_args, key, default)


def _count_diffloss_blocks(state_dict):
    prefix = "diffloss.net.res_blocks."
    indices = []
    for key in state_dict.keys():
        if key.startswith(prefix):
            index_text = key[len(prefix):].split(".", 1)[0]
            if index_text.isdigit():
                indices.append(int(index_text))
    return max(indices) + 1 if indices else None


def infer_teacher_config_from_state_dict(state_dict):
    inferred = {}

    embed_dim = None
    if "fake_latent" in state_dict:
        embed_dim = int(state_dict["fake_latent"].shape[1])
    elif "class_emb.weight" in state_dict:
        embed_dim = int(state_dict["class_emb.weight"].shape[1])

    model_by_embed_dim = {
        768: "mar_base",
        1024: "mar_large",
        1280: "mar_huge",
    }
    if embed_dim is not None:
        inferred["_embed_dim"] = embed_dim
        if embed_dim in model_by_embed_dim:
            inferred["model"] = model_by_embed_dim[embed_dim]

    if "class_emb.weight" in state_dict:
        inferred["class_num"] = int(state_dict["class_emb.weight"].shape[0])

    if "diffusion_pos_embed_learned" in state_dict:
        inferred["_seq_len"] = int(state_dict["diffusion_pos_embed_learned"].shape[1])
    if "encoder_pos_embed_learned" in state_dict and "_seq_len" in inferred:
        inferred["buffer_size"] = int(state_dict["encoder_pos_embed_learned"].shape[1]) - inferred["_seq_len"]

    if "diffloss.net.input_proj.weight" in state_dict:
        weight = state_dict["diffloss.net.input_proj.weight"]
        inferred["diffloss_w"] = int(weight.shape[0])
        inferred["_token_embed_dim"] = int(weight.shape[1])
    elif "diffloss.net.cond_embed.weight" in state_dict:
        inferred["diffloss_w"] = int(state_dict["diffloss.net.cond_embed.weight"].shape[0])

    diffloss_depth = _count_diffloss_blocks(state_dict)
    if diffloss_depth is not None:
        inferred["diffloss_d"] = diffloss_depth

    return inferred


def _set_arg(args, key, value):
    if getattr(args, key) is None:
        setattr(args, key, value)


def fill_from_teacher_args(args, checkpoint):
    teacher_args = checkpoint.get("args", None) if isinstance(checkpoint, dict) else None
    state_dict = get_teacher_state_dict(checkpoint)
    inferred = infer_teacher_config_from_state_dict(state_dict)

    _set_arg(args, "model", inferred.get("model", _teacher_arg(teacher_args, "model", None)))
    if args.model is None:
        embed_dim = inferred.get("_embed_dim", None)
        raise ValueError(
            "Could not infer MAR model type from teacher checkpoint. "
            f"fake_latent/class_emb dim is {embed_dim}; please pass --model explicitly."
        )

    _set_arg(args, "vae_path", _teacher_arg(teacher_args, "vae_path", "pretrained_models/vae/kl16.ckpt"))
    _set_arg(args, "vae_stride", _teacher_arg(teacher_args, "vae_stride", 16))
    _set_arg(args, "patch_size", _teacher_arg(teacher_args, "patch_size", 1))

    if args.vae_embed_dim is None:
        token_embed_dim = inferred.get("_token_embed_dim", None)
        if token_embed_dim is not None and token_embed_dim % (args.patch_size ** 2) == 0:
            args.vae_embed_dim = token_embed_dim // (args.patch_size ** 2)
        else:
            args.vae_embed_dim = _teacher_arg(teacher_args, "vae_embed_dim", 16)

    if args.img_size is None:
        seq_len = inferred.get("_seq_len", None)
        seq_side = int(np.sqrt(seq_len)) if seq_len is not None else 0
        if seq_len is not None and seq_side * seq_side == seq_len:
            args.img_size = seq_side * args.vae_stride * args.patch_size
        else:
            args.img_size = _teacher_arg(teacher_args, "img_size", 256)

    _set_arg(args, "mask_ratio_min", _teacher_arg(teacher_args, "mask_ratio_min", 0.7))
    _set_arg(args, "label_drop_prob", _teacher_arg(teacher_args, "label_drop_prob", 0.1))
    _set_arg(args, "class_num", inferred.get("class_num", _teacher_arg(teacher_args, "class_num", 1000)))
    _set_arg(args, "attn_dropout", _teacher_arg(teacher_args, "attn_dropout", 0.1))
    _set_arg(args, "proj_dropout", _teacher_arg(teacher_args, "proj_dropout", 0.1))
    _set_arg(args, "buffer_size", inferred.get("buffer_size", _teacher_arg(teacher_args, "buffer_size", 64)))
    _set_arg(args, "diffloss_d", inferred.get("diffloss_d", _teacher_arg(teacher_args, "diffloss_d", None)))
    _set_arg(args, "diffloss_w", inferred.get("diffloss_w", _teacher_arg(teacher_args, "diffloss_w", None)))
    _set_arg(args, "num_sampling_steps", _teacher_arg(teacher_args, "num_sampling_steps", "100"))
    _set_arg(args, "diffusion_batch_mul", _teacher_arg(teacher_args, "diffusion_batch_mul", 1))

    for key in ("diffloss_d", "diffloss_w"):
        if getattr(args, key) is None:
            raise ValueError(f"Could not infer {key} from teacher checkpoint; please pass --{key}.")

    expected_embed_dim_by_model = {
        "mar_base": 768,
        "mar_large": 1024,
        "mar_huge": 1280,
    }
    expected_embed_dim = expected_embed_dim_by_model.get(args.model, None)
    inferred_embed_dim = inferred.get("_embed_dim", None)
    if expected_embed_dim is not None and inferred_embed_dim is not None and expected_embed_dim != inferred_embed_dim:
        raise ValueError(
            f"Teacher checkpoint looks like {inferred.get('model', 'a custom MAR')} "
            f"(embed dim {inferred_embed_dim}), but --model {args.model} expects "
            f"embed dim {expected_embed_dim}."
        )

    for key in ("class_num", "buffer_size", "diffloss_d", "diffloss_w"):
        inferred_value = inferred.get(key, None)
        if inferred_value is not None and getattr(args, key) != inferred_value:
            raise ValueError(
                f"Teacher checkpoint has {key}={inferred_value}, but the run is using "
                f"{key}={getattr(args, key)}. Remove the manual --{key} override or use "
                "the matching teacher checkpoint."
            )

    token_embed_dim = inferred.get("_token_embed_dim", None)
    if token_embed_dim is not None:
        expected_token_embed_dim = args.vae_embed_dim * args.patch_size ** 2
        if expected_token_embed_dim != token_embed_dim:
            raise ValueError(
                f"Teacher checkpoint token dim is {token_embed_dim}, but "
                f"--vae_embed_dim {args.vae_embed_dim} and --patch_size {args.patch_size} "
                f"imply {expected_token_embed_dim}."
            )

    if misc.is_main_process():
        print(
            "Inferred teacher config: "
            f"model={args.model}, img_size={args.img_size}, "
            f"vae_embed_dim={args.vae_embed_dim}, vae_stride={args.vae_stride}, "
            f"patch_size={args.patch_size}, buffer_size={args.buffer_size}, "
            f"diffloss_d={args.diffloss_d}, diffloss_w={args.diffloss_w}"
        )
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
    state_dict = get_teacher_state_dict(checkpoint)
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Loaded EMA teacher: {msg}")
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def build_vae(args, device):
    checkpoint = torch.load(args.vae_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"--vae_path must point to the MAR KL-16 checkpoint, e.g. "
            f"pretrained_models/vae/kl16.ckpt. Got: {args.vae_path}"
        )
    state_dict = checkpoint["model"]
    required_keys = {
        "encoder.conv_in.weight",
        "decoder.conv_out.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
    }
    missing_required = sorted(required_keys - set(state_dict.keys()))
    if missing_required:
        raise ValueError(
            f"--vae_path does not look like MAR's KL-16 VAE checkpoint: {args.vae_path}. "
            f"Missing keys include {missing_required[:4]}."
        )

    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=None)
    msg = vae.load_state_dict(state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise ValueError(
            f"--vae_path is not compatible with MAR's AutoencoderKL: {args.vae_path}. "
            f"Missing {len(msg.missing_keys)} keys and unexpected {len(msg.unexpected_keys)} keys."
        )
    vae.to(device).eval()
    vae.requires_grad_(False)
    print(f"Loaded KL-16 VAE from {args.vae_path}")
    return vae


@torch.no_grad()
def update_ema_model(ema_model, source_model, rate):
    source_model = unwrap_model(source_model)
    for ema_param, source_param in zip(ema_model.parameters(), source_model.parameters()):
        ema_param.mul_(rate).add_(source_param.detach(), alpha=1.0 - rate)
    for ema_buffer, source_buffer in zip(ema_model.buffers(), source_model.buffers()):
        ema_buffer.copy_(source_buffer.detach())


def set_requires_grad(module, requires_grad):
    if module is not None:
        module.requires_grad_(requires_grad)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def reduce_log_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().item()
    return float(misc.all_reduce_mean(float(value)))


def write_training_log(log_writer, output_dir, step, values):
    if not misc.is_main_process():
        return
    for name, value in values.items():
        if name not in {"step", "epoch", "elapsed_seconds"}:
            log_writer.add_scalar(name, value, step)
    log_writer.flush()
    with open(os.path.join(output_dir, "log.txt"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, sort_keys=True) + "\n")


def save_checkpoint(
    args,
    generator_head,
    generator_ema,
    fake_head,
    gan_classifier,
    optimizer_generator,
    optimizer_fake,
    scaler,
    step,
):
    if not misc.is_main_process():
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generator_head": unwrap_model(generator_head).state_dict(),
        "generator_head_ema": generator_ema.state_dict(),
        "fake_head": unwrap_model(fake_head).state_dict(),
        "gan_classifier": unwrap_model(gan_classifier).state_dict() if gan_classifier is not None else None,
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
        conditioning_timestep=args.conditioning_timestep,
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


def maybe_resume(args, generator_head, generator_ema, fake_head, gan_classifier, optimizer_generator, optimizer_fake, scaler):
    if not args.resume:
        return 0
    resume_path = args.resume
    if os.path.isdir(resume_path):
        resume_path = os.path.join(resume_path, "checkpoint-last.pth")
    checkpoint = load_checkpoint(resume_path)
    unwrap_model(generator_head).load_state_dict(checkpoint["generator_head"], strict=True)
    generator_ema.load_state_dict(checkpoint.get("generator_head_ema", checkpoint["generator_head"]), strict=True)
    unwrap_model(fake_head).load_state_dict(checkpoint["fake_head"], strict=True)
    if gan_classifier is not None and checkpoint.get("gan_classifier", None) is not None:
        unwrap_model(gan_classifier).load_state_dict(checkpoint["gan_classifier"], strict=True)
    elif gan_classifier is not None and args.gan_classifier:
        print("Resume checkpoint has no gan_classifier; initializing token GAN classifier from scratch.")
    if "optimizer_generator" in checkpoint:
        try:
            optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        except ValueError as exc:
            print(f"Skipping generator optimizer state from resume checkpoint: {exc}")
    if "optimizer_fake" in checkpoint:
        try:
            optimizer_fake.load_state_dict(checkpoint["optimizer_fake"])
        except ValueError as exc:
            print(f"Skipping fake optimizer state from resume checkpoint: {exc}")
    if scaler is not None and checkpoint.get("scaler", None) is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    print(f"Resumed MAR-DMD checkpoint from {resume_path}")
    return int(checkpoint.get("step", 0))


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    if args.gan_classifier or args.diffusion_gan:
        raise ValueError(
            "GAN training is temporarily disabled while conditional and unconditional "
            "DMD routes are distilled separately. Do not pass --gan_classifier or --diffusion_gan."
        )

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    teacher_ckpt = load_checkpoint(args.teacher_ckpt)
    args = fill_from_teacher_args(args, teacher_ckpt)
    if not args.log_dir:
        args.log_dir = os.path.join(args.output_dir, "tensorboard")

    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)
        print(f"TensorBoard logs: {args.log_dir}")
        print(f"JSONL training log: {os.path.join(args.output_dir, 'log.txt')}")
    else:
        log_writer = None
    print("{}".format(args).replace(", ", ",\n"))

    if args.data_path.endswith(".tar"):
        dataset_train = ImageNetTarDataset(
            args.data_path,
            transform=MarTrainTransform(args.img_size),
            index_path=args.tar_index_path,
        )
    else:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform_train)
    print(dataset_train)
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

    vae = build_vae(args, device)

    teacher_model = build_teacher_model(args, teacher_ckpt, device)
    generator_head, fake_head, teacher_head = create_dmd_heads(teacher_model)
    gan_classifier = None
    generator_ema = copy.deepcopy(generator_head).to(device).eval()
    generator_ema.requires_grad_(False)
    generator_head.to(device).train()
    fake_head.to(device).train()
    if gan_classifier is not None:
        gan_classifier.to(device).train()
    teacher_head.to(device).eval()

    if args.distributed:
        generator_head = DDP(generator_head, device_ids=[args.gpu])
        fake_head = DDP(fake_head, device_ids=[args.gpu])
        if gan_classifier is not None:
            gan_classifier = DDP(gan_classifier, device_ids=[args.gpu])

    optimizer_generator = torch.optim.AdamW(generator_head.parameters(), lr=args.generator_lr, weight_decay=args.weight_decay)
    fake_params = list(fake_head.parameters())
    if gan_classifier is not None:
        fake_params += list(gan_classifier.parameters())
    optimizer_fake = torch.optim.AdamW(fake_params, lr=args.fake_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    step = maybe_resume(args, generator_head, generator_ema, fake_head, gan_classifier, optimizer_generator, optimizer_fake, scaler)
    if args.override_resume_lr:
        set_optimizer_lr(optimizer_generator, args.generator_lr)
        set_optimizer_lr(optimizer_fake, args.fake_lr)
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
            generator_logs = {}
            x_fake_cond = None
            x_fake_uncond = None
            if step % args.dfake_gen_update_ratio == 0:
                optimizer_generator.zero_grad(set_to_none=True)
                fake_head_for_generator = unwrap_model(fake_head)
                set_requires_grad(fake_head_for_generator, False)
                generator_conditions = torch.cat([cond, uncond_cond], dim=0)
                x_gen_both = one_step_generate(
                    generator_head,
                    teacher_model.diffloss.train_diffusion,
                    generator_conditions,
                    teacher_model.token_embed_dim,
                    temperature=args.temperature,
                    conditioning_timestep=args.conditioning_timestep,
                )
                x_gen_cond, x_gen_uncond = x_gen_both.chunk(2, dim=0)
                try:
                    loss_dm_cond, cond_logs = compute_generator_dmd_loss_from_tokens(
                        x_gen_cond,
                        fake_head_for_generator,
                        teacher_head,
                        teacher_model.diffloss.train_diffusion,
                        cond,
                        teacher_model.token_embed_dim,
                        min_step,
                        max_step,
                    )
                    loss_dm_uncond, uncond_logs = compute_generator_dmd_loss_from_tokens(
                        x_gen_uncond,
                        fake_head_for_generator,
                        teacher_head,
                        teacher_model.diffloss.train_diffusion,
                        uncond_cond,
                        teacher_model.token_embed_dim,
                        min_step,
                        max_step,
                    )
                    loss_dm = 0.5 * (loss_dm_cond + loss_dm_uncond)
                    generator_loss = loss_dm * args.dm_loss_weight
                    generator_logs = {
                        "loss_dm": float(loss_dm.detach().item()),
                        "loss_dm_cond": float(loss_dm_cond.detach().item()),
                        "loss_dm_uncond": float(loss_dm_uncond.detach().item()),
                        "dm_grad_norm_cond": cond_logs["dm_grad_norm"],
                        "dm_grad_norm_uncond": uncond_logs["dm_grad_norm"],
                    }
                finally:
                    set_requires_grad(fake_head_for_generator, True)
                generator_logs["loss_generator_total"] = float(generator_loss.detach().item())
                generator_logs["conditioning_timestep"] = float(args.conditioning_timestep)
                scaler.scale(generator_loss).backward()
                scaler.unscale_(optimizer_generator)
                generator_grad_norm = torch.nn.utils.clip_grad_norm_(generator_head.parameters(), args.max_grad_norm)
                scaler.step(optimizer_generator)
                scaler.update()
                update_ema_model(generator_ema, generator_head, args.generator_ema_rate)
                x_fake_cond = x_gen_cond.detach()
                x_fake_uncond = x_gen_uncond.detach()
            else:
                generator_grad_norm = torch.tensor(0.0, device=device)

            optimizer_fake.zero_grad(set_to_none=True)
            if x_fake_cond is None:
                with torch.no_grad():
                    fake_conditions = torch.cat([cond, uncond_cond], dim=0)
                    x_fake_both = one_step_generate(
                        generator_head,
                        teacher_model.diffloss.train_diffusion,
                        fake_conditions,
                        teacher_model.token_embed_dim,
                        temperature=args.temperature,
                        conditioning_timestep=args.conditioning_timestep,
                    ).detach()
                    x_fake_cond, x_fake_uncond = x_fake_both.chunk(2, dim=0)
            fake_tokens = torch.cat([x_fake_cond, x_fake_uncond], dim=0)
            fake_conditions = torch.cat([cond, uncond_cond], dim=0)
            loss_fake, fake_logs = compute_fake_loss_from_tokens(
                fake_head,
                teacher_model.diffloss.train_diffusion,
                fake_tokens,
                fake_conditions,
            )
            fake_loss_total = loss_fake
            fake_logs["loss_fake_total"] = float(fake_loss_total.detach().item())
            scaler.scale(fake_loss_total).backward()
            scaler.unscale_(optimizer_fake)
            fake_grad_norm = torch.nn.utils.clip_grad_norm_(fake_params, args.max_grad_norm)
            scaler.step(optimizer_fake)
            scaler.update()

            if step % args.log_iters == 0:
                log_values = {
                    "step": int(step),
                    "epoch": int(epoch),
                    "elapsed_seconds": float(time.time() - start_time),
                    "train/loss_fake": reduce_log_value(fake_logs["loss_fake"]),
                    "train/loss_fake_total": reduce_log_value(fake_logs["loss_fake_total"]),
                    "train/fake_grad_norm": reduce_log_value(fake_grad_norm),
                    "train/fake_lr": float(optimizer_fake.param_groups[0]["lr"]),
                    "train/generator_lr": float(optimizer_generator.param_groups[0]["lr"]),
                }
                loss_fake_value = log_values["train/loss_fake"]
                msg = (
                    f"step {step}/{args.train_iters} "
                    f"loss_fake {loss_fake_value:.6f} "
                    f"fake_grad {log_values['train/fake_grad_norm']:.4f}"
                )
                if generator_logs:
                    log_values.update({
                        "train/loss_dm": reduce_log_value(generator_logs["loss_dm"]),
                        "train/loss_dm_cond": reduce_log_value(generator_logs["loss_dm_cond"]),
                        "train/loss_dm_uncond": reduce_log_value(generator_logs["loss_dm_uncond"]),
                        "train/generator_loss_total": reduce_log_value(generator_logs["loss_generator_total"]),
                        "train/generator_grad_norm": reduce_log_value(generator_grad_norm),
                        "train/dm_grad_norm_cond": reduce_log_value(generator_logs["dm_grad_norm_cond"]),
                        "train/dm_grad_norm_uncond": reduce_log_value(generator_logs["dm_grad_norm_uncond"]),
                        "train/conditioning_timestep": float(args.conditioning_timestep),
                    })
                    loss_dm_value = log_values["train/loss_dm"]
                    msg += (
                        f" loss_dm {loss_dm_value:.6f}"
                        f" loss_dm_c {log_values['train/loss_dm_cond']:.6f}"
                        f" loss_dm_u {log_values['train/loss_dm_uncond']:.6f}"
                        f" gen_grad {log_values['train/generator_grad_norm']:.4f}"
                        f" cond_t {args.conditioning_timestep}"
                        f" dm_grad_c {log_values['train/dm_grad_norm_cond']:.4f}"
                        f" dm_grad_u {log_values['train/dm_grad_norm_uncond']:.4f}"
                    )
                print(msg)
                write_training_log(log_writer, args.output_dir, step, log_values)

            step += 1
            if step % args.save_iters == 0 or step == args.train_iters:
                save_checkpoint(
                    args,
                    generator_head,
                    generator_ema,
                    fake_head,
                    gan_classifier,
                    optimizer_generator,
                    optimizer_fake,
                    scaler,
                    step,
                )
            if args.preview_iters > 0 and (step % args.preview_iters == 0 or step == args.train_iters):
                save_preview_grid(args, teacher_model, vae, generator_ema, step, device)
                if misc.is_dist_avail_and_initialized():
                    torch.distributed.barrier()

        epoch += 1

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")
    if log_writer is not None:
        log_writer.close()


if __name__ == "__main__":
    parser = get_args_parser()
    parsed_args = parser.parse_args()
    main(parsed_args)
