from tinygrad import GlobalCounters, Tensor, TinyJit, nn
from tinygrad.helpers import trange
from tinygrad.nn import datasets

X_train, Y_train, X_test, Y_test = datasets.mnist()
X_train = X_train.div(255.0)
BATCH_SIZE = 64


class conv_block:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x.relu()


class Encoder:
    def __init__(self, in_channels: int):
        self.conv1 = conv_block(in_channels, 128)
        self.conv2 = conv_block(128, 256)
        self.conv3 = conv_block(256, 512)
        self.conv4 = conv_block(512, 1024)
        self.linear = nn.Linear(1024, 16)

    def __call__(self, x: Tensor) -> Tensor:  # (bs, 1, 28 , 28)
        x = self.conv1(x)  # (bs, 128, 14 ,14)
        x = self.conv2(x)  # (bs, 256, 7 ,7)
        x = self.conv3(x)  # (bs, 512, 3 ,3)
        x = self.conv4(x)  # (bs, 1024, 1 ,1)
        x = self.linear(x.flatten(start_dim=1))  # (bs, 16)
        return x


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


class Decoder:
    def __init__(self, out_channels: int):
        self.linear = nn.Linear(16, 1024 * 4 * 4)
        self.t_conv1 = conv_transpose_block(1024, 512)
        self.t_conv2 = conv_transpose_block(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_block(256, out_channels, output_padding=1)

    def __call__(self, x: Tensor) -> Tensor:
        bs = x.shape[0]
        x = self.linear(x)  # (bs, 1024 *4 *4)
        x = x.reshape((bs, 1024, 4, 4))  # (bs, 1024, 4, 4)
        x = self.t_conv1(x)  # (bs, 512, 7, 7)
        x = self.t_conv2(x)  # (bs, 256, 14, 14)
        x = self.t_conv3(x)  # (bs, out_channels, 28, 28)
        return x


class AutoEncoder:
    def __init__(self, in_channels):
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


# TRAINING LOOP
if __name__ == "__main__":
    model = AutoEncoder(1)
    epochs = 10
    lr = 1e-4
    optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, eps=1e-5)

    losses = []

    def mean_squared_error(self, Y: Tensor, reduction: str = "mean") -> Tensor:
        return self.sub(Y).pow(2)._do_reduction(reduction)

    Tensor.mean_squared_error = mean_squared_error  # type: ignore

    @Tensor.train()
    @TinyJit
    def train_step():
        optim.zero_grad()
        samples = Tensor.randint(BATCH_SIZE, high=X_train.shape[0])
        X = X_train[samples]
        preds = model(X)
        loss = preds.mean_squared_error(X)
        loss.backward()
        optim.step()
        return loss.realize()

    # Calculate step_size
    step_size = len(X_train) // BATCH_SIZE
    epochs = 10

    # Modified training loop to use trange
    for step in (t := trange(epochs * step_size)):
        GlobalCounters.reset()
        # samples = Tensor.arange(batch_size := BATCH_SIZE * (i := step % step_size),
        #                        batch_size + BATCH_SIZE)
        loss = train_step()
        losses.append(loss)
        t.set_description(f"loss: {loss.item():6.4f}")
