"""
    Model Name:  
    Note:
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt 


"""
    Model Definition
"""
class Model(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
       raise NotImplementedError

"""
    Loss Function
"""
def loss_function():
    raise NotImplementedError



"""
    Train
"""
loss_y = []
def train_epoch(model, loader, optimizer, device, epoch, log_interval, sample_dir):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(
                f"Epoch[{epoch}] {batch_idx * len(data):6d}/{len(loader.dataset):6d} "
                f"({100. * batch_idx / len(loader):4.1f}%)  Loss: {loss.item() / len(data):.4f}"
            )



def draw_loss():
    plt.figure()
    plt.plot([i + 1 for i in range(len(loss_y))], loss_y, color='red', label='Loss')
    plt.title("Loss Function Value")
    plt.savefig('./images/Loss')
    plt.show()


"""
    main
"""
def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR‑10 VAE")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    os.makedirs(args.sample_dir, exist_ok=True)

    # Data loading
    transform = transforms.ToTensor()  # keep pixels in [0,1] for BCE
    train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model + optimizer
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch, args.log_interval, args.sample_dir)

    # Save model state dict
    torch.save(model.state_dict(), "cifar10_vae.pt")
    print("Training complete. Model saved to cifar10_vae.pt")

    draw_loss()


if __name__ == "__main__":
    main()
