# vae.py
import torch
from torch import nn


class VAE(nn.Module):
    """
    Simple convolutional VAE for 64x64 RGB CelebA images.
    Encoder: 4 conv blocks -> flatten -> mu/logvar (latent_dim)
    Decoder: fc -> 4 conv-transpose blocks -> 3x64x64 image with Sigmoid
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        # Input: (B, 3, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 4, 4)
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()  # 256 * 4 * 4 = 4096
        enc_out_dim = 256 * 4 * 4

        # Latent mean and log-variance
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(latent_dim, enc_out_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # -> (128, 8, 8)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # -> (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # -> (32, 32, 32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),  # -> (3, 64, 64)
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    # --------- VAE core methods ---------

    def encode(self, x):
        h = self.encoder(x)
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Standard VAE reparameterization:
        z = mu + eps * std,  eps ~ N(0, I),  std = exp(0.5 * logvar)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, 4, 4)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
