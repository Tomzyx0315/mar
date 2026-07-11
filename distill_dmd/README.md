# MAR DMD Head Distillation

This folder contains an isolated DMD-style distillation path for MAR DiffLoss heads.
It leaves the MAR teacher AR backbone frozen and trains only:

- a one-step generator head initialized from the EMA teacher DiffLoss head
- a fake diffusion head initialized from the same teacher head

The context distribution is teacher-forcing only: real VAE latents provide visible
tokens, masked positions are dropped from the encoder, and losses are computed only
on masked token positions.

`CONDITIONING_TIMESTEP` is hard-coded to `999` in `distill_dmd/mar_dmd.py`.

## Algorithm Notes

The frozen teacher CFG score is computed in the usual MAR way:

```text
eps_real = eps_uncond + cfg * (eps_cond - eps_uncond)
```

The generator head receives the current `cfg_scale` through a zero-initialized
CFG adapter, because the generated token distribution changes with CFG. The fake
head does not receive CFG; it follows DMD2's `fake_guidance_scale == 1` setup and
learns the score of the generator's current fake-token distribution.

## Preparation

Use the same environment and data layout as the MAR repo.

1. Prepare ImageNet with the standard `ImageFolder` layout:

```text
${IMAGENET_PATH}/train/<class_name>/*.JPEG
```

Only the training split is used for distillation.

2. Download the KL-16 VAE checkpoint and the MAR teacher checkpoint:

```bash
python util/download.py
```

This creates paths such as:

```text
pretrained_models/vae/kl16.ckpt
pretrained_models/mar/mar_large/checkpoint-last.pth
```

You can also point `--teacher_ckpt` to your own MAR checkpoint. The distillation
script loads `model_ema` from that checkpoint when available.

3. For FID after sampling, keep the MAR repo's ImageNet-256 stats file:

```text
fid_stats/adm_in256_stats.npz
```

## Train

The teacher checkpoint normally stores the MAR architecture args, so the command
does not need to repeat `--model`, `--diffloss_d`, or `--diffloss_w`. Add them
only if your checkpoint does not contain `args`.

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  -m distill_dmd.train_mar_dmd \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --data_path ${IMAGENET_PATH} \
  --output_dir output_mar_dmd/mar_large_cfg3 \
  --batch_size 64 \
  --train_iters 100000 \
  --save_iters 2500 \
  --generator_lr 1e-5 \
  --fake_lr 1e-5 \
  --dfake_gen_update_ratio 5 \
  --cfg 3.0 \
  --cfg_schedule linear \
  --preview_iters 1000 \
  --preview_num_iter 64 \
  --preview_class_labels "207,360,388,113,355,980,323,979,88,130,279,291,340,386,805,954" \
  --use_amp
```

MAR recommends different CFG scales for different teacher sizes, e.g. MAR-B
uses `2.9`, MAR-L uses `3.0`, and MAR-H uses `3.2` in the official eval
commands.

## Resume

Use the same training command and add `--resume` pointing to the DMD checkpoint:

```bash
# Add this to the train command above:
--resume output_mar_dmd/mar_large_cfg3/checkpoint-last.pth
```

## Sample

Sampling defaults to the CFG settings saved in the DMD checkpoint. It writes one
`.npz` with key `arr_0`, shaped `[num_images, H, W, 3]`, for FID evaluation.

```bash
python -m distill_dmd.sample_mar_dmd \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --dmd_ckpt output_mar_dmd/mar_large_cfg3/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --output output_mar_dmd/mar_large_cfg3/samples_50k.npz \
  --num_images 50000 \
  --batch_size 64 \
  --num_iter 256 \
  --cfg 3.0 \
  --cfg_schedule linear
```

Use `--save_png_dir output_mar_dmd/mar_large_cfg3/png_samples` only when you
also want individual PNG files.

set --num_iter to 256 for better fid result