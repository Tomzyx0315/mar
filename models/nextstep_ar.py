from functools import partial

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.diffloss import DiffLoss


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate='tanh')
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CausalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_dropout)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=proj_dropout)

    def forward(self, x, attn_mask):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.proj_drop(h)
        x = x + self.mlp(self.norm2(x))
        return x


class NextStepAR(nn.Module):
    """Decoder-only autoregressive image model over continuous VAE tokens."""
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 embed_dim=1024, depth=16, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 **kwargs):
        super().__init__()
        del kwargs

        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size ** 2
        self.grad_checkpointing = grad_checkpointing

        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim))

        self.z_proj = nn.Linear(self.token_embed_dim, embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.start_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.buffer_size = buffer_size
        self.pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, embed_dim))

        self.blocks = nn.ModuleList([
            CausalBlock(embed_dim, num_heads, mlp_ratio, norm_layer=norm_layer,
                        proj_dropout=proj_dropout, attn_dropout=attn_dropout)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))

        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.start_token, std=.02)
        torch.nn.init.normal_(self.pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.MultiheadAttention):
            torch.nn.init.xavier_uniform_(m.in_proj_weight)
            torch.nn.init.xavier_uniform_(m.out_proj.weight)
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias, 0)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias, 0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def _causal_mask(self, length, device):
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def _drop_class_embedding(self, class_embedding):
        if self.training:
            bsz = class_embedding.shape[0]
            drop_latent_mask = torch.rand(bsz, device=class_embedding.device) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(class_embedding.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        return class_embedding

    def forward_ar(self, x, class_embedding):
        bsz, seq_len, _ = x.shape
        class_embedding = self._drop_class_embedding(class_embedding)

        token_embeddings = self.z_proj(x)
        shifted_tokens = torch.cat([
            self.start_token.repeat(bsz, 1, 1).to(token_embeddings.dtype),
            token_embeddings[:, :-1],
        ], dim=1)

        prefix = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
        x = torch.cat([prefix, shifted_tokens], dim=1)
        x = x + self.pos_embed_learned
        x = self.z_proj_ln(x)

        attn_mask = self._causal_mask(x.shape[1], x.device)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                x = checkpoint(block, x, attn_mask)
        else:
            for block in self.blocks:
                x = block(x, attn_mask)
        x = self.norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        return self.diffloss(z=z, target=target)

    def forward(self, imgs, labels):
        class_embedding = self.class_emb(labels)
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        z = self.forward_ar(x, class_embedding)
        return self.forward_loss(z=z, target=gt_latents)

    def sample_tokens(self, bsz, num_iter=None, cfg=1.0, cfg_schedule="linear", labels=None,
                      temperature=1.0, progress=False):
        del num_iter
        device = self.fake_latent.device
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device)

        indices = range(self.seq_len)
        if progress:
            indices = tqdm(indices)

        for step in indices:
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            model_tokens = tokens
            if not cfg == 1.0:
                model_tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)

            z = self.forward_ar(model_tokens, class_embedding)[:, step]

            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (step + 1) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
            tokens[:, step] = sampled_token_latent

        return self.unpatchify(tokens)


def nextstep_ar_base(**kwargs):
    model = NextStepAR(
        embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def nextstep_ar_large(**kwargs):
    model = NextStepAR(
        embed_dim=1024, depth=16, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def nextstep_ar_huge(**kwargs):
    model = NextStepAR(
        embed_dim=1280, depth=20, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
