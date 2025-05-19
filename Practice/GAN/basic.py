import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 超参数
latent_dim = 100
batch_size = 128
epochs = 50
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# 数据加载（使用MNIST）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])
data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# 生成器：输入 latent vector，输出 28x28 图片
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()  # 输出范围[-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 1, 28, 28)

# 判别器：输入图片，输出真假概率
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出[0,1]
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # 标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ========== 训练判别器 ==========
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)

        real_output = D(real_imgs)
        fake_output = D(fake_imgs.detach())

        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ========== 训练生成器 ==========
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        output = D(fake_imgs)

        g_loss = criterion(output, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

# 生成并保存一组图像
G.eval()
z = torch.randn(64, latent_dim).to(device)
gen_imgs = G(z).cpu().detach()

# 可视化
grid = utils.make_grid(gen_imgs, nrow=8, normalize=True)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title("Generated Images")
plt.show()

