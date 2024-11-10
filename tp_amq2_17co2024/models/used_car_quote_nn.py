import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


class UsedCarQuoteNN(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def fit(self, train_loader, test_loader, epochs, optimizer, loss_fn):
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X_val, y_val in test_loader:
                    y_val_pred = self(X_val)
                    val_loss += loss_fn(y_val_pred, y_val).item()

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        tensor_x = torch.tensor(x.to_numpy(), dtype=torch.float32).contiguous()
        return np.vstack(self(tensor_x))

