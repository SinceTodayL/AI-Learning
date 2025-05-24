import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from matplotlib import pyplot as plt 


# --------------------------
# Encoder definition
# --------------------------
class Encoder(nn.Module):
    """Convolutional encoder that outputs the mean and log-variance
    of a multivariate Gaussian q(z|x)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # [B, 3, 32,32] -> [B,32,16,16]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> [B,64,8,8]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # -> [B,128,4,4]
            nn.ReLU(inplace=True),
            nn.Flatten(),                # -> [B, 128*4*4]
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# --------------------------
# Decoder definition
# --------------------------
class Decoder(nn.Module):
    """Mirror‑symmetric convolutional decoder p(x|z)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # [B,64,8,8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # [B,32,16,16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # [B,3,32,32]
            nn.Sigmoid(),                          # map to (0,1)
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        x = self.deconv(x)
        return x


# --------------------------
# VAE wrapper
# --------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterization trick to sample z ~ q(z|x)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# --------------------------
# Loss: ELBO = recon + KL
# --------------------------

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction: Binary Cross Entropy
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    # KL divergence between q(z|x) and p(z)=N(0,I)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl


# --------------------------
# Training / sampling helpers
# --------------------------
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

    avg = total_loss / len(loader.dataset)
    loss_y.append(avg)
    print(f"====> Epoch: {epoch} Average loss: {avg:.4f}")

    # Save reconstructions of last mini‑batch for visual inspection
    with torch.no_grad():
        comparison = torch.cat([data, recon])
        save_image(
            comparison.cpu(),
            os.path.join(sample_dir, f"reconstruction_epoch_{epoch}.png"),
            nrow=len(data),
        )


def sample(model, device, latent_dim, sample_dir, epoch, n=64):
    """Generate n new samples from the prior."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        samples = model.decoder(z)
        save_image(
            samples.cpu(),
            os.path.join(sample_dir, f"sample_epoch_{epoch}.png"),
            nrow=8,
        )

def draw_loss():
    plt.figure()
    plt.plot([i + 1 for i in range(len(loss_y))], loss_y, color='red', label='Loss')
    plt.title("Loss Function Value")
    plt.savefig('./images/VAE Model Loss')
    plt.show()


# --------------------------
# Main entry point
# --------------------------

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
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch, args.log_interval, args.sample_dir)
        sample(model, device, args.latent_dim, args.sample_dir, epoch)

    # Save model state dict
    torch.save(model.state_dict(), "cifar10_vae.pt")
    print("Training complete. Model saved to cifar10_vae.pt")

    draw_loss()


if __name__ == "__main__":
    main()
