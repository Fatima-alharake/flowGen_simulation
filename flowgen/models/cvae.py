from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, ResidualBlock, DownsampleBlock, UpsampleBlock


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        latent_channels: int = 16,
        base_channels: int = 32,
        scale_factor: int = 4,
        learned_prior: bool = True,
        residual_learning: bool = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_channels = latent_channels
        self.learned_prior = learned_prior
        self.residual_learning = residual_learning

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.cond_encoder = nn.Sequential(
            ConvBlock(in_channels, c1),
            DownsampleBlock(c1, c2),
            DownsampleBlock(c2, c3),
        )

        self.post_encoder = nn.Sequential(
            ConvBlock(in_channels * 2, c1),
            DownsampleBlock(c1, c2),
            DownsampleBlock(c2, c3),
        )

        self.post_mu = nn.Conv2d(c3, latent_channels, kernel_size=3, padding=1)
        self.post_logvar = nn.Conv2d(c3, latent_channels, kernel_size=3, padding=1)

        if self.learned_prior:
            self.prior_mu = nn.Conv2d(c3, latent_channels, kernel_size=3, padding=1)
            self.prior_logvar = nn.Conv2d(c3, latent_channels, kernel_size=3, padding=1)

        self.decoder_in = nn.Conv2d(latent_channels + c3, c3, kernel_size=3, padding=1)
        self.dec_up1 = UpsampleBlock(c3, c2)
        self.dec_up2 = UpsampleBlock(c2, c1)
        self.decoder_out = nn.Sequential(
            ResidualBlock(c1),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )

    def _upsample_lr(self, x_lr: torch.Tensor) -> torch.Tensor:
        if self.scale_factor > 1:
            return F.interpolate(x_lr, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return x_lr

    def encode_posterior(self, x_hr: torch.Tensor, x_lr_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.post_encoder(torch.cat([x_hr, x_lr_up], dim=1))
        mu = self.post_mu(feat)
        logvar = torch.clamp(self.post_logvar(feat), min=-10.0, max=5.0)
        return mu, logvar

    def encode_prior(self, cond_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.learned_prior:
            mu = self.prior_mu(cond_feat)
            logvar = torch.clamp(self.prior_logvar(cond_feat), min=-10.0, max=5.0)
            return mu, logvar
        mu = torch.zeros_like(cond_feat[:, :self.latent_channels])
        logvar = torch.zeros_like(cond_feat[:, :self.latent_channels])
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond_feat: torch.Tensor, x_lr_up: torch.Tensor) -> torch.Tensor:
        f = self.decoder_in(torch.cat([z, cond_feat], dim=1))
        f = self.dec_up1(f)
        f = self.dec_up2(f)
        rec = self.decoder_out(f)
        if self.residual_learning:
            rec = rec + x_lr_up
        return rec

    def forward(self, x_lr: torch.Tensor, x_hr: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x_lr_up = self._upsample_lr(x_lr)
        cond_feat = self.cond_encoder(x_lr_up)

        prior_mu, prior_logvar = self.encode_prior(cond_feat)

        if x_hr is not None:
            post_mu, post_logvar = self.encode_posterior(x_hr, x_lr_up)
            z = self.reparameterize(post_mu, post_logvar)
            rec = self.decode(z, cond_feat, x_lr_up)
            return {
                "recon": rec,
                "post_mu": post_mu,
                "post_logvar": post_logvar,
                "prior_mu": prior_mu,
                "prior_logvar": prior_logvar,
                "z": z,
            }

        z = self.reparameterize(prior_mu, prior_logvar)
        rec = self.decode(z, cond_feat, x_lr_up)
        return {
            "recon": rec,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "z": z,
        }

    @torch.no_grad()
    def sample(self, x_lr: torch.Tensor, num_samples: int = 16) -> Dict[str, torch.Tensor]:
        self.eval()
        x_lr_up = self._upsample_lr(x_lr)
        cond_feat = self.cond_encoder(x_lr_up)
        prior_mu, prior_logvar = self.encode_prior(cond_feat)

        samples = []
        for _ in range(num_samples):
            z = self.reparameterize(prior_mu, prior_logvar)
            rec = self.decode(z, cond_feat, x_lr_up)
            samples.append(rec)

        stacked = torch.stack(samples, dim=1)
        mean = stacked.mean(dim=1)
        std = stacked.std(dim=1)

        return {
            "samples": stacked,
            "mean": mean,
            "std": std,
        }
