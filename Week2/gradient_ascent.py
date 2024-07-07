# import libraries
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
from tqdm import tqdm

from oracle import GroundTruth


# Define simple neural network with 2 hidden layers and relu activation
class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaseMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("data/log.csv")
    data = torch.from_numpy(data.values).float()
    x, y = data[:, :5], data[:, -1:]

    # Split into Train and Validation set
    num_points = x.shape[0]
    train_x, valid_x = x[:int(num_points * 0.8)], x[int(num_points * 0.8):]
    train_y, valid_y = y[:int(num_points * 0.8)], y[int(num_points * 0.8):]
    
    
    # Hyperparameters for training neural network
    batch_size = 128
    training_step = 1000
    validation_interval = 100
    lr = 1e-3
    
    # Initialize model and optimizer
    model = BaseMLP(input_dim=5, hidden_dim=128, output_dim=1)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for step in tqdm(range(training_step)):
        # Sample batch
        idx = np.random.choice(train_x.shape[0], batch_size)
        batch_x, batch_y = train_x[idx], train_y[idx]
        
        # Forward pass
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        
        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % validation_interval == 0:
            valid_pred = model(valid_x)
            valid_loss = criterion(valid_pred, valid_y)
            print(f"Step: {step}, Train Loss: {loss.item():.2f}, Valid Loss: {valid_loss.item():.2f}")
            
    # Find queries that maximizes the trained neural network
    num_candidates = 128
    T = 100
    indexs = torch.argsort(train_y.squeeze())
    index = indexs[-num_candidates:]
    x_init = copy.deepcopy(train_x[index])
    
    candidates = []
    for i in tqdm(range(num_candidates)):
        candidate = x_init[i].unsqueeze(0)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=1e-3)
        # Perform Gradient Ascent
        for t in range(T):
            loss = -model(candidate)
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
        
        # Collect samples
        candidates.append(candidate.detach())
    candidates = torch.stack(candidates, dim=0)
    
    # Evaluate the candidates
    func = GroundTruth()
    values = func(candidates)
    # print(values.max(), values.mean(), y.max())
    print(f"Dataset (Max): {y.max().item():.2f}\t Candidates (Max): {values.max().item():.2f}\t Candidates (Median): {values.median().item():.2f}")
    
        
