from typing import Tuple

from tinygrad import Device, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.helpers import Context, trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict, safe_save

print(f"Default device: {Device.DEFAULT}")
X_train, Y_train, X_test, Y_test = mnist()
X_train = X_train.div(255.0)


class conv_block:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x.relu()


class conv_transpose_block:
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 0, with_act: bool = True
    ):
        self.t_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        if with_act:
            self.norm = nn.BatchNorm2d(out_channels)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.t_conv(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
            return x.relu()
        return x


class VAEEncoder:
    def __init__(self, in_channels, latent_dims: int):
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)

        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:  # (bs, 1, 28 , 28)
        bs = x.shape[0]
        x = self.conv1(x)  # (bs, 128, 14 ,14)
        x = self.conv2(x)  # (bs, 256, 7 ,7)
        x = self.conv3(x)  # (bs, 512, 3 ,3)
        x = self.conv4(x)  # (bs, 1024, 1 ,1)
        x = x.reshape(bs, -1)  # Reshape to (bs, 1024)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return (mu, logvar)


class Decoder:
    def __init__(self, out_channels: int, latent_dims: int):
        self.linear = nn.Linear(latent_dims, 1024 * 4 * 4)
        self.t_conv1 = conv_transpose_block(1024, 512)
        self.t_conv2 = conv_transpose_block(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_block(256, out_channels, output_padding=1, with_act=False)

    def __call__(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        x = self.linear(x)  # (bs, 1024 *4 *4)
        x = x.reshape((bs, 1024, 4, 4))  # (bs, 1024, 4, 4)
        x = self.t_conv1(x)  # (bs, 512, 7, 7)
        x = self.t_conv2(x)  # (bs, 256, 14, 14)
        x = self.t_conv3(x)  # (bs, out_channels, 28, 28)

        return x.sigmoid()  # Added sigmoid to squeeze values between (0,1)


class VAE:
    def __init__(self, in_channels, latent_dims):
        self.encoder = VAEEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def sample(self, mu: Tensor, std: Tensor) -> Tensor:
        eps = Tensor.rand_like(std)
        return mu + std * eps

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        std = (0.5 * logvar).exp()  # σ = exp(0.5 * log(σ²))
        z = self.sample(mu, std)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


if __name__ == "__main__":
    dims = 2
    model = VAE(1, dims)
    batch_size = 64
    epochs = 10
    lr = 1e-4
    optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, eps=1e-5)

    def KLDLoss(mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(axis=-1)

    # MSE is used in the book.
    def mse_loss(inp: Tensor, target: Tensor) -> Tensor:
        return inp.sub(target).square().sum(axis=-1)

    # Update the loss_fn to use binary_crossentropy
    def loss_fn(batch: Tensor, image: Tensor, mu: Tensor, logvar: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bs = batch.shape[0]
        batch_flat = batch.reshape(bs, -1)
        image_flat = image.reshape(bs, -1).clip(1e-7, 1 - 1e-7)

        # This was MSE in the book but i wanted to use BCE
        reconstruction_loss = image_flat.binary_crossentropy(batch_flat, reduction="none").sum(axis=-1)
        kl_loss = KLDLoss(mu, logvar)

        loss = (reconstruction_loss + kl_loss).mean(axis=0)
        return loss, reconstruction_loss.mean(), kl_loss.mean()

    @Tensor.train()
    @TinyJit
    def train_step():
        optim.zero_grad()
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        X = X_train[samples]
        reconstructed, mu, logvar = model(X)
        loss, reconstruction_loss, kl_loss = loss_fn(X, reconstructed, mu, logvar)
        loss.backward()
        optim.step()
        return (loss.realize(), reconstruction_loss.realize(), kl_loss.realize())

    step_size = len(X_train) // batch_size
    epochs = 10

    # Modified training loop to use trange
    with Context(BEAM=2):
        for step in (t := trange(epochs * step_size)):
            GlobalCounters.reset()
            loss, img_loss, kl_loss = train_step()
            loss_item = loss.item()
            img_loss_item = img_loss.item()
            kl_loss_item = kl_loss.item()

            t.set_description(f"loss: {loss_item:6.4f}, img_loss: {img_loss_item:6.4f}, kl_loss: {kl_loss_item:6.4f}")

    # first we need the state dict of our model
    state_dict = get_state_dict(model)

    # then we can just save it to a file
    safe_save(state_dict, "models/vae.safetensors")
