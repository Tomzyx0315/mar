# MAR DMD Head Distillation

This folder contains an isolated DMD-style distillation path for MAR DiffLoss heads.
It leaves the MAR teacher AR backbone frozen and trains only:

- a one-step generator head initialized from the EMA teacher DiffLoss head
- a fake diffusion head initialized from the same teacher head

GAN training is currently disabled in this separate conditional/unconditional
distillation path.

The context distribution is teacher-forcing only: real VAE latents provide visible
tokens, masked positions are dropped from the encoder, and losses are computed only
on masked token positions.

The one-step generator uses `--conditioning_timestep` as the noisy input
timestep. The default is `899`. MAR's cosine diffusion schedule makes timestep
`999` nearly singular (`sqrt(1 / alpha_bar)` is about 20k), so directly copying
DMD2's SD default is usually too unstable for this DiffLoss head.

## Algorithm Notes

Conditional and unconditional routes are distilled separately. The shared
generator and fake heads see either the class-conditional MAR decoder context or
the null-class MAR decoder context, and each route is matched to the corresponding
frozen teacher score without CFG during training:

```text
loss_dm = 0.5 * (loss_dm_cond + loss_dm_uncond)
```

The fake head is trained on generated tokens from both routes with their matching
contexts. CFG is used only during sampling. Conditional and unconditional
generator predictions use the same input noise and are combined as:

```text
x0_cfg = x0_uncond + cfg * (x0_cond - x0_uncond)
```

Because the two routes share the same noise and conditioning timestep, this is
equivalent to interpolating their epsilon predictions before converting to `x0`.

DMD score computation is kept in fp32. This matters because the one-step
generator predicts `x0` from a high-noise timestep, where half-precision error
can be amplified by the diffusion schedule.

Training maintains an EMA copy of the generator head. Checkpoints save both
`generator_head` and `generator_head_ema`; preview and sampling use EMA by
default.

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
  --output_dir output_mar_dmd/mar_large_separate_cfg \
  --batch_size 64 \
  --train_iters 100000 \
  --save_iters 2500 \
  --generator_lr 2e-6 \
  --fake_lr 2e-6 \
  --dfake_gen_update_ratio 5 \
  --dm_loss_weight 1.0 \
  --cfg 3.0 \
  --cfg_schedule linear \
  --conditioning_timestep 899 \
  --generator_ema_rate 0.999 \
  --preview_iters 1000 \
  --preview_num_iter 256 \
  --preview_class_labels "207,360,388,113,355,980,323,979,88,130,279,291,340,386,805,954"
```

`--cfg` and `--cfg_schedule` affect preview and sampling only; they do not affect
the two-route DMD training loss. MAR recommends different sampling CFG scales for
different teacher sizes, e.g. MAR-B uses `2.9`, MAR-L uses `3.0`, and MAR-H uses
`3.2` in the official eval commands.

Rank 0 writes persistent training logs to:

```text
<output_dir>/log.txt                 # JSON Lines metrics
<output_dir>/args.json               # resolved training arguments
<output_dir>/tensorboard/events...   # TensorBoard events
```

Use `--log_dir` to override the TensorBoard directory. To inspect the default
logs, run:

```bash
tensorboard --logdir output_mar_dmd/mar_large_separate_cfg/tensorboard
```

## Resume

Use the same training command and add `--resume` pointing to the DMD checkpoint:

```bash
# Add this to the train command above:
--resume output_mar_dmd/mar_large_separate_cfg/checkpoint-last.pth
```

## Sample

Sampling defaults to the CFG settings saved in the DMD checkpoint. It writes one
`.npz` with key `arr_0`, shaped `[num_images, H, W, 3]`, for FID evaluation.

```bash
python -m distill_dmd.sample_mar_dmd \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --dmd_ckpt output_mar_dmd/mar_large_separate_cfg/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --output output_mar_dmd/mar_large_separate_cfg/samples_50k.npz \
  --num_images 50000 \
  --batch_size 64 \
  --num_iter 256 \
  --cfg 3.0 \
  --cfg_schedule linear \
  --conditioning_timestep 899
```

Use `--save_png_dir output_mar_dmd/mar_large_separate_cfg/png_samples` only when you
also want individual PNG files.

Set --num_iter to 256 for better fid result

## FID

The following script reuses the original MAR's fid evaluation

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  -m distill_dmd.eval_one_ckpt_mar_fid \
  --teacher_ckpt pretrained_models/mar/mar_large/checkpoint-last.pth \
  --dmd_ckpt output_mar_dmd/mar_large_separate_cfg/checkpoint-last.pth \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --output_dir output_mar_dmd/mar_large_separate_cfg/mar_fid \
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
