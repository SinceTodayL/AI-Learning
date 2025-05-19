import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

latent_dim = 100
batch_size = 128
epoches = 50
lr = 0.0002
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize([0.5, ], [0.5, ])
])
data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
dataloader = DataLoader(data, batch_size = batch_size, shuffle=True)


class Genarator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 1, 28, 28)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1*28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()   # output: [0, 1]
        )

    def forward(self, x):
        return self.model(x)
    
G = Genarator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999)) 
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999)) 

DLoss = []
GLoss = []
for epoch in range(epoches):
    for real_images,_ in dataloader:

        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # train D
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)

        real_output = D(real_images)
        fake_output = D(fake_images.detach())

        d_loss_real = criterion(real_output, real_labels)
        d_loss_fake = criterion(fake_output, fake_labels)
        d_loss = d_loss_fake + d_loss_real

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # train G

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)
        output = D(fake_images)

        g_loss = criterion(output, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epoches}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")
    GLoss.append(g_loss)
    DLoss.append(d_loss)

G.eval()
z = torch.randn(64, latent_dim).to(device)
gen_images = G(z).cpu().detach()

index = [i for i in range(len(GLoss))]
plt.figure()
plt.plot(index, GLoss, color='blue', marker='o', label="GLoss")
plt.plot(index, DLoss, color='red', marker='x', label="DLoss")
plt.xlabel("index")
plt.ylabel('BCE')
plt.legend()
plt.show()

grid = utils.make_grid(gen_images, nrow=8, normalize=True)
plt.figure(figsize=(8,8))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title("Generated Images")
plt.show()

