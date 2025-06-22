# xai_project/train_model.py
# To fine tune WideResNet, our specific task

import torch
from torchvision.models import wide_resnet50_2
from torch import nn, optim

from src.data_preparation import train_loader

# Loading a pretrained model [ here- WideResNet ]
pancreas_model = wide_resnet50_2(pretrained=True)
pancreas_model.fc = nn.Linear(2048, 1)  # Binary output

# Loss & optimizer (to handle class imbalance)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))  # Weight for 'pos' class
optimizer = optim.Adam(pancreas_model.parameters(), lr=1e-4)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# device = torch.device("cpu") # Replacing "cuda" code to "cpu", due to lack laptops resource
pancreas_model.to(device)

for epoch in range(10):
    pancreas_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = pancreas_model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")