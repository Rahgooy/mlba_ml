import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from mlba import sample_lba
import pandas as pd
import numpy as np
import copy
import time


class MLBA_Params:
    def __init__(self, mu_d, sigma_d, mu_A, sigma_A, mu_b_minus_A, sigma_b_minus_A):
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.mu_A = mu_A
        self.sigma_A = sigma_A
        self.mu_b_minus_A = mu_b_minus_A
        self.sigma_b_minus_A = sigma_b_minus_A


class MLBA_NN(nn.Module):
    def __init__(self, n_features, n_options, n_hidden, n_epochs, batch, lr=0.001):
        super(MLBA_NN, self).__init__()
        self.lr = lr
        self.f1 = nn.Linear(n_features, n_hidden)
        # (mu, sigma) for d, A, b
        self.f2 = nn.Linear(n_hidden, n_options * 2 * 3)
        self.softPlus = nn.Softplus()
        self.relu = nn.ReLU()

        self.options = n_options
        self.epochs = n_epochs
        self.batch = batch

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

    def forward(self, X):
        n = self.options
        x = self.f1(X)
        x = self.relu(x)
        x = self.f2(x)
        mu_d = self.softPlus(x[:, :n]).view(-1, n)
        sigma_d = (self.softPlus(x[:, n:2*n]) + 1e-6).view(-1, n)

        mu_A = self.softPlus(x[:, 2*n:3*n]).view(-1, n)
        sigma_A = (self.softPlus(x[:, 3*n:4*n]) + 1e-6).view(-1, n)

        mu_b_minus_A = self.softPlus(x[:, 4*n:5*n]).view(-1, n)
        sigma_b_minus_A = (self.softPlus(x[:, 5*n:]) + 1e-6).view(-1, n)

        return MLBA_Params(mu_d, sigma_d, mu_A, sigma_A, mu_b_minus_A, sigma_b_minus_A)

    def loss(self, X, y):
        ttf_inv = self.ttf_inv(X)
        l = nn.CrossEntropyLoss()
        return l(ttf_inv, y.view(-1))

    def ttf_inv(self, X):
        mlba = self.forward(X)
        d = mlba.mu_d
        A = mlba.mu_A / 2
        b = mlba.mu_b_minus_A + A
        ttf_inv = d / (b - A).clamp(1e-6)
        return ttf_inv

    def predict_proba(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        ttf_inv = self.ttf_inv(x)
        return torch.softmax(ttf_inv, 1).detach().numpy()

    def __tensor(self, x, dtype):
        return torch.tensor(x, dtype=dtype).to(self.device)

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y.reshape(-1, 1), test_size=0.33)
        X_train = self.__tensor(X_train.tolist(), torch.float)
        y_train = self.__tensor(y_train.tolist(), torch.long)
        X_val = self.__tensor(X_val.tolist(), torch.float)
        y_val = self.__tensor(y_val.tolist(), torch.long)

        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=self.batch)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        best = 10000
        bestModel = None
        for epoch in range(self.epochs):
            train_loss = self.__train_step(optimizer, train_loader)
            val_loss = self.loss(X_val, y_val)
            if val_loss < best:
                best = val_loss
                bestModel = copy.deepcopy(self)

            if epoch and epoch % 5 == 0:
                print(
                    f"{time.asctime()} >  Epoch {epoch:5d} - Train Loss: {train_loss: 12.6f}, " +
                    f"val Loss: {val_loss:10.6f}, " +
                    f"Best: {best:10.6f}"
                )
        bestModel.train(False)
        torch.save(bestModel, 'best_model.pkl')
        return bestModel

    def __train_step(self, optimizer, train_loader):
        train_loss = 0
        self.train(True)
        for (X, y) in train_loader:
            optimizer.zero_grad()
            loss = self.loss(X, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        self.train(False)
        return train_loss / len(train_loader)


if __name__ == "__main__":
    features = ['Rect1Height', 'Rect1Width', 'Rect2Height',
                'Rect2Width', 'Rect3Height', 'Rect3Width']
    train_data = pd.read_csv('data/E2.csv')
    e1a = pd.read_csv('data/E1a.csv')
    e1b = pd.read_csv('data/E1b.csv')
    e1c = pd.read_csv('data/E1c.csv')

    X_train = train_data[features].values
    y_train = train_data.response.values - 1
    model = MLBA_NN(6, 3, 50, 1000, 2)
    model.fit(X_train, y_train)
