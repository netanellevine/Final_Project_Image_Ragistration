import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class ImageRegistrationNet(nn.Module):
    def __init__(self):
        super(ImageRegistrationNet, self).__init__()

        # Contracting path
        self.contracting1 = self.contracting_block(3, 64)   # Contracting block 1
        self.contracting2 = self.contracting_block(64, 128)  # Contracting block 2
        self.contracting3 = self.contracting_block(128, 256) # Contracting block 3
        self.contracting4 = self.contracting_block(256, 512) # Contracting block 4

        # Expanding path
        self.expanding1 = self.expanding_block(512, 256)     # Expanding block 1
        self.expanding2 = self.expanding_block(256, 128)     # Expanding block 2
        self.expanding3 = self.expanding_block(128, 64)      # Expanding block 3

        # Final convolutional layer
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)     # Final convolutional layer

    def contracting_block(self, in_channels, out_channels):
        """
        Contracting block consists of two convolutional layers with ReLU activation and max pooling.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        """
        Expanding block consists of two convolutional layers with ReLU activation and transpose convolution for upsampling.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)
        )
        return block

    def forward(self, x):
        # Contracting path
        x1 = self.contracting1(x)                             # Apply contracting block 1
        x2 = self.contracting2(x1)                            # Apply contracting block 2
        x3 = self.contracting3(x2)                            # Apply contracting block 3
        x4 = self.contracting4(x3)                            # Apply contracting block 4

        # Expanding path
        x = self.expanding1(x4)                               # Apply expanding block 1
        x = self.expanding2(torch.cat([x, x3], dim=1))        # Concatenate features from contracting block 3 and apply expanding block 2
        x = self.expanding3(torch.cat([x, x2], dim=1))        # Concatenate features from contracting block 2 and apply expanding block 3

        # Final convolution
        x = self.final_conv(x)                                # Apply final convolutional layer

        return x


def train(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()                             # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # Adam optimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, targets in train_loader: #load image
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()# zero the gradient for step

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()#add loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    print("Training finished!")

# Usage example:
# train(model, train_loader, num_epochs=10, learning_rate=0.001)

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import random

# Define the transforms to be applied to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),       # Resize the images to a fixed size
    transforms.ToTensor()                 # Convert images to tensors
])

# Set the paths to your folders containing the images
folder1_path = "/path/to/folder1"
folder2_path = "/path/to/folder2"

# Create custom dataset class for paired images
class PairedImageDataset(Dataset):
    def __init__(self, folder1_path, folder2_path, transform=None):
        self.folder1_dataset = ImageFolder(folder1_path, transform=transform)
        self.folder2_dataset = ImageFolder(folder2_path, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        img1, _ = self.folder1_dataset[random.randint(0, len(self.folder1_dataset) - 1)]
        img2, _ = self.folder2_dataset[random.randint(0, len(self.folder2_dataset) - 1)]
        return img1, img2

    def __len__(self):
        return max(len(self.folder1_dataset), len(self.folder2_dataset))

# Create the paired dataset
paired_dataset = PairedImageDataset(folder1_path, folder2_path, transform=transform)

# Create the train_loader
batch_size = 32    # Set your desired batch size
train_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True)

# Now you can use the train_loader for training your model
