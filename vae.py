import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from regularizer import ModuleWrapper
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import SEMEION
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import trange


class VAE(torch.nn.Module):
    """Constructs a variational auto-encoder."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.fc1 = torch.nn.Linear(256, 1)
        self.fc2 = torch.nn.Linear(256, 1)
        self.fc3 = torch.nn.Linear(1, 256)
        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (256, 1, 1)),
            torch.nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 3, 2, 1),
            torch.nn.Sigmoid(),
        )

    def _reparameterize(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.fc1(x), self.fc2(x)
        epsilon = torch.randn_like(mu)
        return torch.exp(0.5 * log_var).mul(epsilon).add(mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): image
        """
        x = self.encoder(x)
        x = self._reparameterize(x)
        x = self.fc3(x)
        x = self.decoder(x)
        return x


def get_save_dir(lamda: float, weight_decay: float) -> str:
    """Directory to save results.

    Args:
        lamda (float): lambda in the invexifying regularization framework
        weight_decay (float): l2-regularization parameter
    """
    assert lamda == 0.0 or weight_decay == 0.0
    root = "VAE-Results"
    if lamda == 0.0 and weight_decay == 0.0:
        save_dir = os.path.join(root, "no-regularization")
    elif lamda == 0.0:
        save_dir = os.path.join(root, "l2-regularization", str(weight_decay))
    else:
        save_dir = os.path.join(root, "invexifying-regularization", str(lamda))
    return save_dir


@torch.no_grad()
def evaluate(lamda: float, weight_decay: float) -> None:
    """Evaluate VAE checkpoints.

    Args:
        lamda (float): lambda in the invexifying regularization framework
        weight_decay (float): l2-regularization parameter
    """
    assert lamda == 0.0 or weight_decay == 0.0
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    # save results to this directory to plot later
    save_dir = get_save_dir(lamda, weight_decay)
    checkpoints_dir = os.path.join(save_dir, "checkpoints")

    # initialize training and testing dataloaders
    dataset = SEMEION("datasets", transform=ToTensor(), download=True)
    train_dataset = Subset(dataset, range(0, 1000))
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    test_dataset = Subset(dataset, range(1000, len(dataset)))
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # initialize a latent vector, the model and invexifying regularization
    z = torch.randn(len(test_dataset), 1)
    model = VAE()
    model = ModuleWrapper(model, lamda=lamda)
    model.init_ps(train_dataloader)

    num_epochs = 1000
    fx_trains = torch.zeros(num_epochs)
    fx_tests = torch.zeros(num_epochs)

    text = "lamda={} and weight_decay={} ...".format(lamda, weight_decay)
    print("evaluating with " + text)
    model.eval()
    criterion = torch.nn.MSELoss(reduction="sum")
    for i in trange(num_epochs):
        state_dict = torch.load(os.path.join(checkpoints_dir, str(i) + ".pt"))
        model.load_state_dict(state_dict)
        for images, labels in iter(train_dataloader):
            fx_trains[i].add_(criterion(model.module(images), images))
        fx_trains[i].div_(len(train_dataset))
        for images, labels in iter(test_dataloader):
            fx_tests[i].add_(criterion(model.module(images), images))
        fx_tests[i].div_(len(test_dataset))

    # save results to plot later
    torch.save(fx_trains, os.path.join(save_dir, "fx_trains.pt"))
    torch.save(fx_tests, os.path.join(save_dir, "fx_tests.pt"))

    fake_images_dir = os.path.join(save_dir, "fake_images")
    os.makedirs(fake_images_dir)
    fake_images = model.module.decoder(model.module.fc3(z))
    for i in range(len(fake_images)):
        fp = os.path.join(fake_images_dir, str(i) + ".png")
        save_image(fake_images[i : i + 1], fp)
    text = "some generated fake images have been saved under the directory "
    print(text + fake_images_dir)


def train(lamda: float, weight_decay: float) -> None:
    """Train VAE.

    Args:
        lamda (float): lambda in the invexifying regularization framework
        weight_decay (float): l2-regularization parameter
    """
    assert lamda == 0.0 or weight_decay == 0.0
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    # save checkpoints to this directory to use later
    save_dir = get_save_dir(lamda, weight_decay)
    save_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir)

    # initialize training dataloader
    dataset = SEMEION("datasets", transform=ToTensor(), download=True)
    dataset = Subset(dataset, range(0, 1000))
    dataloader = DataLoader(dataset, batch_size=1)

    # initialize model, invexifying regularization and Adam optimizer
    model = VAE()
    model = ModuleWrapper(model, lamda=lamda)
    model.init_ps(dataloader)
    optimizer = torch.optim.Adam(
        model.parameters(), 1e-3, (0.5, 0.999), weight_decay=weight_decay
    )

    text = "lamda={} and weight_decay={} ...".format(lamda, weight_decay)
    print("training with " + text)
    num_epochs = 1000
    model.train()
    criterion = torch.nn.MSELoss()
    for i in trange(num_epochs):
        torch.save(model.state_dict(), os.path.join(save_dir, str(i) + ".pt"))
        for batch_idx, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            model.set_batch_idx(batch_idx)
            loss = criterion(model(images), images)
            loss.backward()
            optimizer.step()


def main() -> None:
    # generate and save checkpoints
    values = [1e-2, 1e-4, 1e-8]
    train(0.0, 0.0)
    for value in values:
        train(value, 0.0)
        train(0.0, value)

    evaluate(0.0, 0.0)
    for value in values:
        evaluate(value, 0.0)
        evaluate(0.0, value)

    # plot saved results
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Times New Roman"]
    matplotlib.rcParams["text.usetex"] = True

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey="row",
        figsize=(6, 3.6),
    )

    # do not need to plot all points
    xs = torch.logspace(0, 3, 100).round().long().unique() - 1

    labels = [
        r"$\lambda=10^{-2}$",
        r"$\lambda=10^{-4}$",
        r"$\lambda=10^{-8}$",
    ]
    colors = ["g", "b", "r"]
    styles = ["-.", ":", "--"]

    for value, label, c, ls in zip(values, labels, colors, styles):
        ax0.plot(
            xs,
            torch.load(os.path.join(get_save_dir(value, 0.0), "fx_trains.pt"))[xs],
            label=label,
            c=c,
            ls=ls,
        )
        ax1.plot(
            xs,
            torch.load(os.path.join(get_save_dir(0.0, value), "fx_trains.pt"))[xs],
            label=label,
            c=c,
            ls=ls,
        )
        ax2.plot(
            xs,
            torch.load(os.path.join(get_save_dir(value, 0.0), "fx_tests.pt"))[xs],
            label=label,
            c=c,
            ls=ls,
        )
        ax3.plot(
            xs,
            torch.load(os.path.join(get_save_dir(0.0, value), "fx_tests.pt"))[xs],
            label=label,
            c=c,
            ls=ls,
        )

    ax0.plot(
        xs,
        torch.load(os.path.join(get_save_dir(0.0, 0.0), "fx_trains.pt"))[xs],
        label=r"$\lambda=0.0$",
        c="k",
    )
    ax1.plot(
        xs,
        torch.load(os.path.join(get_save_dir(0.0, 0.0), "fx_trains.pt"))[xs],
        label=r"$\lambda=0.0$",
        c="k",
    )
    ax2.plot(
        xs,
        torch.load(os.path.join(get_save_dir(0.0, 0.0), "fx_tests.pt"))[xs],
        label=r"$\lambda=0.0$",
        c="k",
    )
    ax3.plot(
        xs,
        torch.load(os.path.join(get_save_dir(0.0, 0.0), "fx_tests.pt"))[xs],
        label=r"$\lambda=0.0$",
        c="k",
    )

    ax0.legend()
    ax0.set_title(r"Our Method", fontsize=11)
    ax0.set_ylabel(r"Train Dataset Loss", fontsize=11)
    ax1.set_title(r"$\ell_{2}$ Regularization", fontsize=11)
    ax2.set_xlabel(r"Epoch", fontsize=11)
    ax2.set_ylabel(r"Test Dataset Loss", fontsize=11)
    ax3.set_xlabel(r"Epoch", fontsize=11)
    plt.subplots_adjust(bottom=0.12, top=0.92, wspace=0.1, hspace=0.1)

    path = os.path.join("VAE-Results", "vae.pdf")
    plt.savefig(path)
    print("plot saved to " + path)


if __name__ == "__main__":
    main()
