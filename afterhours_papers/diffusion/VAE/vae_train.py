from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tinygrad.helpers import trange
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore

import wandb
from afterhours_papers.helpers import StepDataLoader, def_device


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dims: int):
        super().__init__()
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)
        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # (bs, 1, 28, 28)
        x = self.conv1(x)  # (bs, 128, 14, 14)
        x = self.conv2(x)  # (bs, 256, 7, 7)
        x = self.conv3(x)  # (bs, 512, 3, 3)
        x = self.conv4(x)  # (bs, 1024, 1, 1)
        x = x.flatten(start_dim=1)  # (bs, 1024)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


def conv_transpose_block(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 0, with_act: bool = True
) -> nn.Sequential:
    modules: List[nn.Module] = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
    if with_act:
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, latent_dims: int):
        super().__init__()
        self.linear = nn.Linear(latent_dims, 1024 * 4 * 4)
        self.t_conv1 = conv_transpose_block(1024, 512)
        self.t_conv2 = conv_transpose_block(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_block(256, out_channels, output_padding=1, with_act=False)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        x = self.linear(x)  # (bs, 1024 * 4 * 4)
        x = x.reshape((bs, 1024, 4, 4))  # (bs, 1024, 4, 4)
        x = self.t_conv1(x)  # (bs, 512, 7, 7)
        x = self.t_conv2(x)  # (bs, 256, 14, 14)
        x = self.t_conv3(x)  # (bs, out_channels, 28, 28)
        return torch.sigmoid(x)  # Added sigmoid to squeeze values between (0,1)


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dims: int):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(x)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def kld_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def loss_fn(batch: Tensor, reconstructed: Tensor, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    bs = batch.shape[0]
    batch_flat = batch.reshape(bs, -1)
    reconstructed_flat = reconstructed.reshape(bs, -1).clamp(1e-7, 1 - 1e-7)

    # Binary cross entropy loss for reconstruction
    reconstruction_loss = F.binary_cross_entropy(reconstructed_flat, batch_flat, reduction="none").sum(dim=1)
    kl_loss = kld_loss(mu, logvar)

    total_loss = (reconstruction_loss + kl_loss).mean()
    return total_loss, reconstruction_loss.mean(), kl_loss.mean()


if __name__ == "__main__":
    # Check if PyTorch version supports torch.compile
    if not hasattr(torch, "compile"):
        print("Your PyTorch version doesn't support torch.compile. Please upgrade to PyTorch 2.0 or later.")

    # Initialize wandb
    wandb_run = wandb.init(
        project="HandsOnGENAI - VAE",
        name="pytorch-vae",
        config={"dims": 2, "batch_size": 64, "epochs": 10, "lr": 1e-4, "loss_fn": "binary_crossentropy"},
    )

    data_dir = Path("data/")
    transfms = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist = datasets.MNIST(data_dir, transform=transfms, download=True)

    # Get config values from wandb
    LATENT_DIMS = wandb_run.config["dims"]
    BATCH_SIZE = wandb_run.config["batch_size"]
    EPOCHS = wandb_run.config["epochs"]
    LR = wandb_run.config["lr"]

    train_dl = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True)
    step_train_dl: StepDataLoader = StepDataLoader(train_dl, num_epochs=EPOCHS)

    model = VAE(1, LATENT_DIMS)
    model = model.to(def_device)

    # Apply torch.compile to the model
    if hasattr(torch, "compile"):
        torch.set_float32_matmul_precision("high")
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore

    optimizer = optim.AdamW(model.parameters(), lr=LR, eps=1e-5)

    # Training loop
    for _ in (t := trange(len(step_train_dl))):
        images, _ = step_train_dl.next_batch()
        images = images.to(def_device, non_blocking=True)

        reconstructed, mu, logvar = model(images)
        loss, recon_loss, kl_div = loss_fn(images, reconstructed, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        loss_val = loss.item()
        recon_loss_val = recon_loss.item()
        kl_div_val = kl_div.item()

        t.set_description(f"loss: {loss_val:6.4f}, recon_loss: {recon_loss_val:6.4f}, kl_loss: {kl_div_val:6.4f}")
        wandb.log({"loss": loss_val, "img_loss": recon_loss_val, "kl_loss": kl_div_val})
    # Save the model
    torch.save(model.state_dict(), "models/vae_pytorch.pt")
    wandb_run.finish()
