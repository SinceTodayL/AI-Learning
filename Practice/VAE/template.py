import os, torch, torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
LATENT_DIM = 20
BATCH_SIZE = 128
EPOCHS     = 10
LOG_FREQ   = 100          # 多少 batch 打一次日志
SAMPLE_DIR = 'samples'
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ------------------------------
# 1. 数据
# ------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),             # [0,1]
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)

# ------------------------------
# 2. 模型
# ------------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 28x28 -> 7x7x32
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),      # 14x14
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),     # 7x7
            nn.Flatten()
        )
        self.fc_mu    = nn.Linear(7*7*32, LATENT_DIM)
        self.fc_logvar= nn.Linear(7*7*32, LATENT_DIM)
        # Decoder: z -> 7x7x32 -> 28x28
        self.fc_dec   = nn.Linear(LATENT_DIM, 7*7*32)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(), # 14x14
            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1)              # 28x28
        )

    # --- 编码 ---
    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    # --- 重参数化 ---
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --- 解码 ---
    def decode(self, z):
        h = self.fc_dec(z).view(-1, 32, 7, 7)
        return self.dec(h)            # 输出 logits（未过 sigmoid）

    # --- 前向 ---
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

model = VAE().to(DEVICE)
optimiser = optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss(reduction='sum')

# ------------------------------
# 3. 训练
# ------------------------------
def loss_fn(logits, x, mu, logvar):
    recon = bce(logits, x)                       # 重建误差
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

def train():
    model.train()
    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        for i, (x, _) in enumerate(train_loader, 1):
            x = x.to(DEVICE)
            logits, mu, logvar = model(x)
            loss = loss_fn(logits, x, mu, logvar)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            if i % LOG_FREQ == 0:
                print(f'Epoch {epoch} [{i*len(x)}/{len(train_loader.dataset)}]  '
                      f'Loss: {loss.item()/len(x):.3f}')

        print(f'===> Epoch {epoch} Avg loss: {total_loss/len(train_loader.dataset):.3f}')
        sample(epoch)

# ------------------------------
# 4. 采样与保存
# ------------------------------
@torch.no_grad()
def sample(epoch):
    model.eval()
    z = torch.randn(64, LATENT_DIM).to(DEVICE)
    logits = model.decode(z)
    samples = torch.sigmoid(logits)          # 转成 [0,1] 像素
    save_image(samples.cpu(), f'{SAMPLE_DIR}/epoch{epoch:02d}.png', nrow=8)
    print(f'Generated images saved to {SAMPLE_DIR}/epoch{epoch:02d}.png')

if __name__ == '__main__':
    train()       # 如只想推断，可注释掉
    # 若已训练好并保存权重，可: model.load_state_dict(torch.load('vae.pt')); sample(0)
