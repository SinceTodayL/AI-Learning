import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == "__main__":
    # Process Data: ImageNet
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(r'E:\tiny-imagenet-200\tiny-imagenet-200\train', transform=train_transform)
    val_dataset = datasets.ImageFolder(r'E:\tiny-imagenet-200\tiny-imagenet-200\test', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()

            # Conv
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=17, kernel_size=5, padding=2, stride=1)
            self.conv2 = nn.Conv2d(in_channels=17, out_channels=53, kernel_size=3, padding=1, stride=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            # FC
            self.fc1 = nn.Linear(53*56*56, 1150)
            self.fc2 = nn.Linear(1150, 200)
            
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(-1, 53*56*56)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)

            return x
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # train
    n_epoches = 1
    losses = []
    for epoch in range(n_epoches):
        loss_all = 0
        for i, (images, labels) in enumerate(train_loader):
            # 1. to(device)
            images = images.to(device)
            labels = labels.to(device)
            # 2. zero_grad
            optimizer.zero_grad()
            # 3. train(core)
            output = model(images)
            loss = criterion(output, labels)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss : {loss_all:.4f}")
        losses.append(loss_all)
        loss_all = 0

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(len(losses)), losses, color='r', linewidth=3, label="loss")
    plt.title("Loss change")
    plt.ylabel("Loss")
    plt.show()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predict = torch.max(output, 1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
