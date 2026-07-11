import copy
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CONDITIONING_TIMESTEP = 999


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class CfgConditionedHead(nn.Module):
    """Teacher-initialized head with a small zero-init CFG adapter."""

    def __init__(self, base_head):
        super().__init__()
        self.base = copy.deepcopy(base_head)
        z_channels = self.base.cond_embed.in_features
        width = self.base.model_channels
        self.cfg_embed = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, z_channels),
        )
        nn.init.zeros_(self.cfg_embed[-1].weight)
        nn.init.zeros_(self.cfg_embed[-1].bias)

    def _format_cfg(self, cfg_scale, batch_size, device, dtype):
        if cfg_scale is None:
            return None
        if not torch.is_tensor(cfg_scale):
            cfg_scale = torch.tensor(cfg_scale, device=device)
        cfg_scale = cfg_scale.to(device=device)
        if cfg_scale.ndim == 0:
            cfg_scale = cfg_scale.expand(batch_size)
        elif cfg_scale.numel() == 1:
            cfg_scale = cfg_scale.reshape(1).expand(batch_size)
        elif cfg_scale.shape[0] != batch_size:
            raise ValueError(f"cfg_scale batch {cfg_scale.shape[0]} does not match {batch_size}")
        return (cfg_scale.reshape(batch_size, 1) - 1.0).to(dtype=dtype)

    def forward(self, x, t, c, cfg_scale=None):
        cfg_input = self._format_cfg(cfg_scale, x.shape[0], x.device, c.dtype)
        if cfg_input is not None:
            c = c + self.cfg_embed(cfg_input)
        return self.base(x, t, c)


def create_dmd_heads(teacher_model):
    """Create trainable generator/fake heads from the teacher diffusion head."""
    teacher_head = teacher_model.diffloss.net
    generator_head = CfgConditionedHead(teacher_head)
    fake_head = copy.deepcopy(teacher_head)
    teacher_head.requires_grad_(False)
    generator_head.requires_grad_(True)
    fake_head.requires_grad_(True)
    return generator_head, fake_head, teacher_head


def sample_orders(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.argsort(torch.rand(batch_size, seq_len, device=device), dim=-1)


def random_masking_like_mar(model, tokens: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = tokens.shape
    mask_rate = model.mask_ratio_generator.rvs(1)[0]
    num_masked_tokens = int(math.ceil(seq_len * mask_rate))
    mask = torch.zeros(batch_size, seq_len, device=tokens.device)
    mask = torch.scatter(
        mask,
        dim=-1,
        index=orders[:, :num_masked_tokens],
        src=torch.ones(batch_size, seq_len, device=tokens.device),
    )
    return mask


@torch.no_grad()
def build_teacher_forcing_context(
    teacher_model,
    latent_images: torch.Tensor,
    labels: torch.Tensor,
    max_tokens: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build MAR decoder conditions from real visible tokens only.

    The returned tensors are flattened over masked positions. Masked tokens are
    dropped in the encoder and therefore do not leak their target values.
    """
    teacher_model.eval()
    tokens = teacher_model.patchify(latent_images)
    batch_size, seq_len, _ = tokens.shape
    orders = sample_orders(batch_size, seq_len, tokens.device)
    mask = random_masking_like_mar(teacher_model, tokens, orders).bool()

    class_embedding = teacher_model.class_emb(labels)
    uncond_embedding = teacher_model.fake_latent.repeat(batch_size, 1)

    cond_hidden = teacher_model.forward_mae_decoder(
        teacher_model.forward_mae_encoder(tokens, mask.float(), class_embedding),
        mask.float(),
    )
    uncond_hidden = teacher_model.forward_mae_decoder(
        teacher_model.forward_mae_encoder(tokens, mask.float(), uncond_embedding),
        mask.float(),
    )

    cond = cond_hidden[mask]
    uncond_cond = uncond_hidden[mask]
    target = tokens[mask]

    if max_tokens > 0 and cond.shape[0] > max_tokens:
        keep = torch.randperm(cond.shape[0], device=cond.device)[:max_tokens]
        cond = cond[keep]
        uncond_cond = uncond_cond[keep]
        target = target[keep]

    mask_len = mask.float().sum(dim=1).mean()
    visible_fraction = (seq_len - mask_len) / seq_len

    return {
        "cond": cond,
        "uncond_cond": uncond_cond,
        "target": target,
        "mask": mask,
        "visible_fraction": visible_fraction,
    }


def extract_eps(model_output: torch.Tensor, token_dim: int) -> torch.Tensor:
    return model_output[:, :token_dim]


def predict_xstart_from_eps(diffusion, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps)


def one_step_generate(
    generator_head,
    diffusion,
    cond: torch.Tensor,
    token_dim: int,
    cfg_scale=None,
    temperature: float = 1.0,
) -> torch.Tensor:
    noise = torch.randn(cond.shape[0], token_dim, device=cond.device, dtype=cond.dtype) * temperature
    timesteps = torch.full(
        (cond.shape[0],),
        CONDITIONING_TIMESTEP,
        device=cond.device,
        dtype=torch.long,
    )
    model_output = generator_head(noise, timesteps, cond, cfg_scale=cfg_scale)
    eps = extract_eps(model_output, token_dim)
    return predict_xstart_from_eps(diffusion, noise, timesteps, eps).float()


def compute_cfg_scale(base_cfg: float, cfg_schedule: str, visible_fraction: torch.Tensor) -> torch.Tensor:
    if cfg_schedule == "constant":
        return torch.as_tensor(base_cfg, device=visible_fraction.device, dtype=visible_fraction.dtype)
    if cfg_schedule == "linear":
        return 1.0 + (base_cfg - 1.0) * visible_fraction
    raise NotImplementedError(f"Unknown cfg_schedule: {cfg_schedule}")


def compute_generator_dmd_loss(
    generator_head,
    fake_head,
    teacher_head,
    diffusion,
    cond: torch.Tensor,
    uncond_cond: torch.Tensor,
    token_dim: int,
    cfg_scale: torch.Tensor,
    min_step: int,
    max_step: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x_gen = one_step_generate(
        generator_head,
        diffusion,
        cond,
        token_dim,
        cfg_scale=cfg_scale,
        temperature=temperature,
    )

    timesteps = torch.randint(
        min_step,
        max_step + 1,
        (x_gen.shape[0],),
        device=x_gen.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(x_gen)
    x_t = diffusion.q_sample(x_gen, timesteps, noise=noise)

    with torch.no_grad():
        fake_eps = extract_eps(fake_head(x_t, timesteps, cond), token_dim)
        real_cond_eps = extract_eps(teacher_head(x_t, timesteps, cond), token_dim)
        if float(cfg_scale.detach().mean().item()) == 1.0:
            real_eps = real_cond_eps
        else:
            real_uncond_eps = extract_eps(teacher_head(x_t, timesteps, uncond_cond), token_dim)
            real_eps = real_uncond_eps + cfg_scale * (real_cond_eps - real_uncond_eps)

        pred_fake_x0 = predict_xstart_from_eps(diffusion, x_t, timesteps, fake_eps)
        pred_real_x0 = predict_xstart_from_eps(diffusion, x_t, timesteps, real_eps)

        p_real = x_gen.detach() - pred_real_x0
        p_fake = x_gen.detach() - pred_fake_x0
        grad = (p_real - p_fake) / p_real.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        grad = torch.nan_to_num(grad)

    loss = 0.5 * F.mse_loss(x_gen.float(), (x_gen - grad).detach().float(), reduction="mean")
    logs = {
        "loss_dm": float(loss.detach().item()),
        "dm_grad_norm": float(torch.norm(grad.detach()).item()),
        "x_gen_mean": float(x_gen.detach().mean().item()),
        "x_gen_std": float(x_gen.detach().std().item()),
        "cfg_scale": float(cfg_scale.detach().mean().item()),
    }
    return loss, logs


def compute_fake_loss(
    generator_head,
    fake_head,
    diffusion,
    cond: torch.Tensor,
    token_dim: int,
    cfg_scale: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.no_grad():
        x_gen = one_step_generate(
            generator_head,
            diffusion,
            cond,
            token_dim,
            cfg_scale=cfg_scale,
            temperature=temperature,
        ).detach()

    timesteps = torch.randint(
        0,
        diffusion.num_timesteps,
        (x_gen.shape[0],),
        device=x_gen.device,
        dtype=torch.long,
    )
    loss_dict = diffusion.training_losses(fake_head, x_gen, timesteps, model_kwargs={"c": cond})
    loss = loss_dict["loss"].mean()
    logs = {
        "loss_fake": float(loss.detach().item()),
    }
    return loss, logs


@torch.no_grad()
def sample_tokens_one_step(
    model,
    generator_head,
    diffusion,
    batch_size: int,
    num_iter: int,
    labels: torch.Tensor,
    cfg: float = 3.0,
    cfg_schedule: str = "linear",
    temperature: float = 1.0,
    progress: bool = False,
) -> torch.Tensor:
    if progress:
        from tqdm import tqdm

    model.eval()
    generator_head.eval()

    mask = torch.ones(batch_size, model.seq_len, device=labels.device)
    tokens = torch.zeros(batch_size, model.seq_len, model.token_embed_dim, device=labels.device)
    orders = sample_orders(batch_size, model.seq_len, labels.device)

    indices = range(num_iter)
    if progress:
        indices = tqdm(indices)

    for step in indices:
        cur_tokens = tokens.clone()
        class_embedding = model.class_emb(labels)

        x = model.forward_mae_encoder(tokens, mask, class_embedding)
        z = model.forward_mae_decoder(x, mask)

        mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
        mask_len = torch.tensor([math.floor(model.seq_len * mask_ratio)], device=labels.device)
        mask_len = torch.maximum(
            torch.tensor([1], device=labels.device),
            torch.minimum(mask.sum(dim=-1, keepdim=True) - 1, mask_len),
        )
        next_mask_len = int(mask_len[0].item())

        mask_next = torch.zeros(batch_size, model.seq_len, device=labels.device)
        mask_next = torch.scatter(
            mask_next,
            dim=-1,
            index=orders[:, :next_mask_len],
            src=torch.ones(batch_size, model.seq_len, device=labels.device),
        ).bool()

        if step >= num_iter - 1:
            mask_to_pred = mask.bool()
        else:
            mask_to_pred = torch.logical_xor(mask.bool(), mask_next)
        mask = mask_next.float()

        cond = z[mask_to_pred]
        visible_fraction = torch.tensor(
            (model.seq_len - next_mask_len) / model.seq_len,
            device=labels.device,
            dtype=cond.dtype,
        )
        cfg_scale = compute_cfg_scale(cfg, cfg_schedule, visible_fraction)
        sampled_token_latent = one_step_generate(
            generator_head,
            diffusion,
            cond,
            model.token_embed_dim,
            cfg_scale=cfg_scale,
            temperature=temperature,
        )
        cur_tokens[mask_to_pred] = sampled_token_latent
        tokens = cur_tokens

    return model.unpatchify(tokens)
