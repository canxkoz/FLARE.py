#
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

__all__ = [
    "PerceiverIO",
]

# local
from .transformer import SelfAttentionBlock
from .flare import ResidualMLP, FinalLayer

#======================================================================#
# Perceiver Encoder/ Decoder
#======================================================================#
class PerceiverEncoder(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8, num_latents: int = 128):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.num_latents = num_latents
        self.latent_q = nn.Parameter(torch.randn(num_latents, channel_dim))
        nn.init.normal_(self.latent_q, mean=0.0, std=1.0)

        self.ln = nn.LayerNorm(channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):
        # x: [B, N, C]

        x = self.ln(x)

        q = rearrange(self.latent_q, 'm (h d) -> h m d', h=self.num_heads) # [H, M, D]
        q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # [B, H, M, D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B, H, N, D]

        y = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)

        return y

class PerceiverDecoder(nn.Module):
    def __init__(self, channel_dim: int, num_heads: int = 8):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.head_dim = channel_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.ln1 = nn.LayerNorm(channel_dim)
        self.ln2 = nn.LayerNorm(channel_dim)

        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x, y):
        # Args:
        #   x: [B M C]
        #   y: [B N C]
        # Returns:
        #   z: [B N C]
        
        x = self.ln1(x)
        y = self.ln2(y)

        q = rearrange(self.q_proj(y), 'b m (h d) -> b h m d', h=self.num_heads) # [B H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]

        z = F.scaled_dot_product_attention(q, k, v, scale=self.scale) # [B H N D]

        z = rearrange(z, 'b h n d -> b n (h d)')
        z = self.out_proj(z)

        return z

#======================================================================#
# MODEL
#======================================================================#
class PerceiverIO(nn.Module):
    def __init__(self,
        in_dim: int,
        out_dim: int,
        channel_dim: int = 64,
        num_blocks: int = 8,
        num_heads: int = 8,
        num_latents: int = 128,
        mlp_ratio: float = 4.0,
        act: str = None,
    ):
        super().__init__()

        self.in_proj = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=channel_dim,
            out_dim=channel_dim,
            num_layers=2,
            act=act,
            output_residual=True,
        )

        self.encoder = PerceiverEncoder(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
        )

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                channel_dim=channel_dim,
                num_heads=num_heads,
                act=act,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_blocks)
        ])

        self.decoder = PerceiverDecoder(
            channel_dim=channel_dim,
            num_heads=num_heads,
        )

        self.out_proj = FinalLayer(
            channel_dim,
            out_dim,
            act=act,
            num_layers=2,
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm,)):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def forward(self, x):
        # x: [B, N, C]

        x = self.in_proj(x) # [B N C]
        z = self.encoder(x) # [B M C]

        for block in self.blocks:
            z = block(z) # [B M C]

        x = self.decoder(z, x) # [B M C], [B, N, C] -> [B N C]
        x = self.out_proj(x) # [B N C]

        return x

#======================================================================#
#