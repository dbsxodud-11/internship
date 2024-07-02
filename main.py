import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DKittyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.ymin = data['y'].min()
        self.ymax = data['y'].max()

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        return self.data["x"][idx], (self.data["y"][idx] - self.ymin) / (self.ymax - self.ymin)

class MLP(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(x_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, y_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Data Preparation
    data = np.load("data/dkitty_data.npz")
    print(data['x'].shape, data['y'].shape)
    dataset = DKittyDataset(data)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    x, y = next(iter(dataloader))
    print(x.shape, y.shape)

    # Model
    model = MLP(x_dim=x.shape[1], y_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training
    num_epochs = 1
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for x, y in dataloader:
            y_pred = model(x)
            loss = criterion(y_pred.flatten(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(dataloader)
        print(f"Epoch: {epoch}\tLoss: {total_loss:.4f}")
    # Evaluation
    x, y = next(iter(dataloader))
    y_pred = model(x).flatten()
    print(y[:10], y_pred[:10])







