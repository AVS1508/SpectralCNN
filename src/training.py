import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

def train(model: nn.Module, model_path: str, train_loader: DataLoader, valid_loader: DataLoader, device: torch.device, lr: float = 5e-4, num_epochs: int = 100, loss_function: nn.Module = nn.CrossEntropyLoss(), debug: bool = False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    valid_losses = []
    train_size, valid_size = len(train_loader.dataset), len(valid_loader.dataset)
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = loss_function(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        if debug:
            print("----------------------------------------")
            print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, num_epochs, train_loss / train_size))

        model.eval()
        label_true, label_pred = [], []
        valid_loss = 0.0
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = loss_function(out, label)
            valid_loss += loss.item() * data.size(0)
            pred = torch.max(out, dim=1)[1]
            label_true.extend(label.tolist())
            label_pred.extend(pred.tolist())
        accuracy = accuracy_score(label_true, label_pred)
        if debug:
            print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, num_epochs, valid_loss / valid_size, accuracy))
            print("----------------------------------------")
        valid_losses.append(valid_loss / valid_size)
        if np.argmin(valid_losses) == epoch:
            if debug:
                print('Saving the best model at %d epochs!' % epoch)
            torch.save(model.state_dict(), model_path)