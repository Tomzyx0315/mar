# MAR DMD Head Distillation

This folder contains an isolated DMD-style distillation path for MAR DiffLoss heads.
It leaves the MAR teacher AR backbone frozen and trains only:

- a one-step generator head initialized from the EMA teacher DiffLoss head
- a fake diffusion head initialized from the same teacher head
- optionally, one shared token-level GAN classifier attached to the fake
  DiffLoss head hidden feature

The context distribution is teacher-forcing only: real VAE latents provide visible
tokens, masked positions are dropped from the encoder, and losses are computed only
on masked token positions.

The one-step generator uses `--conditioning_timestep` as the noisy input
timestep. The default is `899`. MAR's cosine diffusion schedule makes timestep
`999` nearly singular (`sqrt(1 / alpha_bar)` is about 20k), so directly copying
DMD2's SD default is usually too unstable for this DiffLoss head.

## Algorithm Notes

The frozen teacher CFG score is computed in the usual MAR way:

```text
eps_real = eps_uncond + cfg * (eps_cond - eps_uncond)
```

The generator head receives the current `cfg_scale` through a zero-initialized
CFG adapter, because the generated token distribution changes with CFG. The fake
head does not receive CFG; it follows DMD2's `fake_guidance_scale == 1` setup and
learns the score of the generator's current fake-token distribution.

DMD score computation is kept in fp32. This matters because the one-step
generator predicts `x0` from a high-noise timestep, where half-precision error
can be amplified by the diffusion schedule.

Training maintains an EMA copy of the generator head. Checkpoints save both
`generator_head` and `generator_head_ema`; preview and sampling use EMA by
default.

With `--gan_classifier`, this folder follows DMD2's GAN-classifier design but at
MAR token granularity. There is one shared classifier for all masked token
positions, not one discriminator per position. The classifier sees fake-head
hidden features for `(token, diffusion timestep, AR condition)` and is trained
with logistic real/fake losses:

```text
G: softplus(-D(fake_token | cond))
D: softplus(D(fake_token | cond)) + softplus(-D(real_token | cond))
```

`--diffusion_gan` matches DMD2's noisy-discriminator variant: real and fake
tokens are noised at a random diffusion timestep before classification.

## Preparation

Use the same environment and data layout as the MAR repo.

1. Prepare ImageNet. The distillation script supports either the standard
   extracted `ImageFolder` layout:

```text
${IMAGENET_PATH}/train/<class_name>/*.JPEG
```

or the original ImageNet training tar without decompression:

```text
${IMAGENET_PATH}/ILSVRC2012_img_train.tar
```

Only the training split is used for distillation. When a `.tar` path is passed
to `--data_path`, the script builds an offset index at
`${data_path}.index` on the first run, then reads images by seeking into the tar.
You can put the index somewhere else with `--tar_index_path`.

2. Download the KL-16 VAE checkpoint and the MAR teacher checkpoint:

```bash
python util/download.py
```

This creates paths such as:

```text
pretrained_models/vae/kl16.ckpt
pretrained_models/mar/mar_large/checkpoint-last.pth
```

3. For FID after sampling, keep the MAR repo's ImageNet-256 stats file:

```text
fid_stats/adm_in256_stats.npz
```

## Train

The command normally does not need to repeat `--model`, `--diffloss_d`, or
`--diffloss_w`: they are read from checkpoint args when present, otherwise
inferred from the teacher weights.

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  -m distill_dmd.train_mar_dmd \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --data_path ${IMAGENET_PATH}/ILSVRC2012_img_train.tar \
  --tar_index_path ${IMAGENET_PATH}/ILSVRC2012_img_train.tar.index \
  --output_dir output_mar_dmd/mar_large_cfg3_gan \
  --batch_size 64 \
  --train_iters 100000 \
  --save_iters 2500 \
  --generator_lr 2e-6 \
  --fake_lr 2e-6 \
  --dfake_gen_update_ratio 5 \
  --dm_loss_weight 1.0 \
  --gan_classifier \
  --gen_cls_loss_weight 3e-3 \
  --guidance_cls_loss_weight 1e-2 \
  --diffusion_gan \
  --diffusion_gan_max_timestep 1000 \
  --cfg 3.0 \
  --cfg_schedule linear \
  --conditioning_timestep 899 \
  --generator_ema_rate 0.999 \
  --preview_iters 1000 \
  --preview_num_iter 256 \
  --preview_class_labels "207,360,388,113,355,980,323,979,88,130,279,291,340,386,805,954"
```

MAR recommends different CFG scales for different teacher sizes, e.g. MAR-B
uses `2.9`, MAR-L uses `3.0`, and MAR-H uses `3.2` in the official eval
commands.

## Token GAN Fine-Tune

If you already have a good DMD-only checkpoint, resume from it and lower both
learning rates.

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  -m distill_dmd.train_mar_dmd \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --data_path ${IMAGENET_PATH}/ILSVRC2012_img_train.tar \
  --tar_index_path ${IMAGENET_PATH}/ILSVRC2012_img_train.tar.index \
  --output_dir output_mar_dmd/mar_large_cfg3_gan \
  --resume output_mar_dmd/mar_large_cfg3/checkpoint-last.pth \
  --override_resume_lr \
  --batch_size 64 \
  --train_iters 150000 \
  --save_iters 2500 \
  --generator_lr 5e-7 \
  --fake_lr 5e-7 \
  --dfake_gen_update_ratio 5 \
  --dm_loss_weight 1.0 \
  --gan_classifier \
  --gen_cls_loss_weight 3e-3 \
  --guidance_cls_loss_weight 1e-2 \
  --diffusion_gan \
  --diffusion_gan_max_timestep 1000 \
  --cfg 3.0 \
  --cfg_schedule linear \
  --conditioning_timestep 899 \
  --generator_ema_rate 0.999 \
  --preview_iters 1000 \
  --preview_num_iter 256 \
  --preview_class_labels "207,360,388,113,355,980,323,979,88,130,279,291,340,386,805,954"
```

When resuming, `--train_iters` is the final global step, not the number of extra
GAN steps. Normal resume preserves the optimizer learning rates from the
checkpoint; add `--override_resume_lr` when intentionally switching to the
command-line learning rates for GAN fine-tuning.

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
  --cfg_schedule linear \
  --conditioning_timestep 899
```

Use `--save_png_dir output_mar_dmd/mar_large_cfg3/png_samples` only when you
also want individual PNG files.

Set --num_iter to 256 for better fid result

## FID

The following script reuses the original MAR's fid evaluation

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  -m distill_dmd.eval_one_ckpt_mar_fid \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --dmd_ckpt output_mar_dmd/mar_large_cfg3/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --output_dir output_mar_dmd/mar_large_cfg3/mar_fid \
  --num_images 50000 \
  --batch_size 64 \
  --num_iter 256 \
  --cfg 3.0 \
  --cfg_schedule linear \
  --conditioning_timestep 899
```

Run it from the MAR repo root and keep `fid_stats/adm_in256_stats.npz` in the
same place as the original repo expects. The environment must have the LTH
`torch-fidelity` fork installed, as in MAR's `environment.yaml`.

`--output_dir` is where temporary PNG files are written. When FID finishes
normally, MAR's `evaluate` deletes that PNG directory. If the process is killed
or `torch_fidelity` errors, the temporary PNG directory can remain and should be
removed manually.

The script uses `generator_head_ema` by default. Add `--use_raw_generator` only
when you intentionally want the non-EMA generator. To evaluate a different
checkpoint, change only `--dmd_ckpt`.
