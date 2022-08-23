import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from numpy import log10
from regularizer import ModuleWrapper
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class BC(torch.nn.Module):
    """Constructs a binary classifier."""

    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.zeros(8), requires_grad=True)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            A (torch.Tensor): data matrix
        """
        return A.mv(self.x).sigmoid()


def train(lamda: float) -> None:
    """Train binary classification with squared-loss.

    Args:
        lamda (float): lambda in the invexifying regularization framework
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    dtype = torch.float64

    # save results to this directory to plot later
    save_dir = os.path.join("BCWSL-Results", str(lamda))
    os.makedirs(save_dir)

    # generate synthetic data
    z = torch.randn(8, dtype=dtype)
    A = torch.randn(64, 8, dtype=dtype)
    A *= torch.logspace(start=0, end=-8, steps=A.size(1), base=2.0, dtype=dtype)
    b = A.mv(z).sigmoid()

    # initialize model, invexifying regularization and gradient descent optimizer
    model = BC().to(dtype=dtype)
    model = ModuleWrapper(model, lamda=lamda)
    model.init_ps(DataLoader(TensorDataset(A, b), batch_size=A.size(0)))
    optimizer = torch.optim.SGD(model.parameters(), lr=2.0)

    num_epochs = int(1e7)
    # do not need to plot all points
    indices = torch.logspace(0, int(log10(num_epochs)), 100, dtype=torch.long)
    indices = torch.unique(indices, sorted=True) - 1
    orig_loss = torch.zeros_like(indices, dtype=dtype)
    surr_loss = torch.zeros_like(indices, dtype=dtype)

    print("training with lamda={} ...".format(lamda))
    model.train()
    model.set_batch_idx(0)  # only have 1 batch, hence gradient descent
    j = 0
    for i in trange(num_epochs, miniters=num_epochs // 1000):
        optimizer.zero_grad(set_to_none=True)
        loss = model(A) - b
        loss = 0.5 * torch.dot(loss, loss)
        loss.backward()
        if i == indices[j]:
            # for plots, record non regularized and regularized loss
            with torch.no_grad():
                orig_loss[j].add_(0.5 * model.module(A).sub(b).norm().pow(2))
                surr_loss[j].add_(loss.data)
            j += 1
        optimizer.step()

    # save results to plot later
    torch.save(indices, os.path.join(save_dir, "indices.pt"))
    torch.save(orig_loss, os.path.join(save_dir, "orig_loss.pt"))
    torch.save(surr_loss, os.path.join(save_dir, "surr_loss.pt"))


def main() -> None:
    # generate and save results
    lamdas = [1e-1, 1e-2, 1e-3, 1e-4]
    for lamda in lamdas:
        train(lamda)

    # plot saved results
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = ["Times New Roman"]
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["text.usetex"] = True

    fig0, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 2))

    ax0.set_title("Non Regularized Loss", fontsize=10)
    ax1.set_title("Regularized Loss", fontsize=10)
    ax0.set_xlabel(r"Iteration ($t$)", fontsize=10)
    ax1.set_xlabel(r"Iteration ($t$)", fontsize=10)
    ax0.set_ylabel(r"$f(\mathbf{x}_t)$", fontsize=10)
    ax1.set_ylabel(r"$\hat{f}(\mathbf{x}_t,\mathbf{p}_t)$", fontsize=10)

    labels = [
        r"$\lambda=0.1$",
        r"$\lambda=0.01$",
        r"$\lambda=0.001$",
        r"$\lambda=0.0001$",
    ]
    colors = ["k", "r", "b", "g"]
    styles = ["-", "--", ":", "-."]

    for lamda, label, c, ls in zip(lamdas, labels, colors, styles):
        save_dir = os.path.join("BCWSL-Results", str(lamda))
        indices = torch.load(os.path.join(save_dir, "indices.pt"))
        orig_loss = torch.load(os.path.join(save_dir, "orig_loss.pt"))
        surr_loss = torch.load(os.path.join(save_dir, "surr_loss.pt"))
        ax0.loglog(indices + 1, orig_loss, label=label, c=c, ls=ls)
        ax1.loglog(indices + 1, surr_loss, label=label, c=c, ls=ls)

    ax0.set_ylim(1e-16, 1e2)
    ax1.set_ylim(1e-28, 1e2)
    ax0.set_yticks([1, 1e-8, 1e-16])
    ax1.set_yticks([1, 1e-14, 1e-28])
    ax1.legend(ncol=4, loc="center", bbox_to_anchor=(-0.25, -0.6))
    plt.subplots_adjust(left=0.1, right=0.902, top=0.89, bottom=0.4, wspace=0.5)

    path = os.path.join("BCWSL-Results", "bcwsl.pdf")
    plt.savefig(path)
    print("plot saved to " + path)


if __name__ == "__main__":
    main()
