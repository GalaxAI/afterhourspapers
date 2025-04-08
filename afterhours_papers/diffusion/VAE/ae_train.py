from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF  # type: ignore
from tinygrad.helpers import trange
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from afterhours_papers.helpers import StepDataLoader, def_device


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)
        self.linear = nn.Linear(1024, 16)

    def forward(self, x: Tensor) -> Tensor:  # (bs, 1, 28 , 28)
        x = self.conv1(x)  # (bs, 128, 14 ,14)
        x = self.conv2(x)  # (bs, 256, 7 ,7)
        x = self.conv3(x)  # (bs, 512, 3 ,3)
        x = self.conv4(x)  # (bs, 1024, 1 ,1)
        x = self.linear(x.flatten(start_dim=1))  # (bs, 16)
        return x


def conv_transpose_block(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 0, with_act: bool = True
) -> nn.Sequential:
    modules: List[nn.Module] = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
    if with_act:
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


class Decoder(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(16, 1024 * 4 * 4)
        self.t_conv1 = conv_transpose_block(1024, 512)
        self.t_conv2 = conv_transpose_block(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_block(256, out_channels, output_padding=1, with_act=False)  # No norm or relu in last t_conv

    def forward(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        x = self.linear(x)  # (bs, 1024 *4 *4)
        x = x.reshape((bs, 1024, 4, 4))  # (bs, 1024, 4, 4)
        x = self.t_conv1(x)  # (bs, 512, 7, 7)
        x = self.t_conv2(x)  # (bs, 256, 14, 14)
        x = self.t_conv3(x)  # (bs, 1024, 28, 28)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x


if __name__ == "__main__":
    # Check if PyTorch version supports torch.compile
    if not hasattr(torch, "compile"):
        print("Your PyTorch version doesn't support torch.compile. Please upgrade to PyTorch 2.0 or later.")

    data_dir = Path("data/")

    transfms = transforms.Compose([TF.to_tensor])

    mnist = datasets.MNIST(data_dir, transform=transfms, download=True)

    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4

    train_dl = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True)
    step_train_dl: StepDataLoader = StepDataLoader(train_dl, num_epochs=EPOCHS)

    model = AutoEncoder(1)
    model = model.to(def_device)

    # Apply torch.compile to the model
    if hasattr(torch, "compile"):
        torch.set_float32_matmul_precision("high")
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore

    optimizer = optim.AdamW(model.parameters(), lr=LR, eps=1e-5)

    losses: List[float] = []

    # Create a single trange for all steps
    steps_completed = 0

    for _ in (t := trange(len(step_train_dl))):
        images, labels = step_train_dl.next_batch()
        images = images.to(def_device, non_blocking=True)

        preds = model(images)
        loss = F.mse_loss(preds, images)
        current_loss = loss.item()
        losses.append(current_loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        t.set_description(f"loss: {current_loss:6.4f}")
