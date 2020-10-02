import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from mlba import sample_lba
import pandas as pd
import numpy as np
import copy
import time
from lba_dist import LBA


class MLBA_Params:
    def __init__(self, mu_d, sigma_d, A, b):
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.A = A
        self.b = b


class MLBA_NN(nn.Module):
    def __init__(self, n_features, n_options, n_hidden, n_epochs, batch, lr=0.01):
        super(MLBA_NN, self).__init__()
        self.lr = lr
        self.f1 = nn.Linear(n_features, n_hidden)
        # mu_d for each option and A, b, sigma_d
        self.f2 = nn.Linear(n_hidden, n_options * 2 + 2)
        self.softPlus = nn.Softplus()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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

        mu_d = (self.sigmoid(x[:, :n]) * 10 + 0.1).view(-1, n)
        sigma_d = torch.ones((X.shape[0], 1)) # (self.softPlus(x[:, n]) + 1).view(-1, 1)
        # A = (self.softPlus(x[:, 2*n]) + 0.1).view(-1, 1)
        # b = (self.softPlus(x[:, 2*n+1]) + 0.1).view(-1, 1) + A
        A = (self.sigmoid(x[:, 2*n]) + 0.1).view(-1, 1) * 10
        b = (self.sigmoid(x[:, 2*n+1]) + 0.1).view(-1, 1) * 10 + A

        return MLBA_Params(mu_d, sigma_d, A, b)

    def loss(self, X, y):
        params = self.forward(X)
        nll = 0.0
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        probs = lba.probs()
        nll = nn.NLLLoss()
        return nll(torch.log(probs), y.view(-1))

    def predict_proba(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        params = self.forward(x)
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        probs = lba.probs().detach().numpy()
        return probs

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
        best = float('inf')
        bestModel = None
        for epoch in range(self.epochs):
            train_loss = self.__train_step(optimizer, train_loader)
            val_loss = self.loss(X_val, y_val).item()
            if val_loss < best:
                best = val_loss
                bestModel = copy.deepcopy(self)

            if epoch and epoch % 2 == 0:
                print(
                    f"{time.asctime()} >  Epoch {epoch:5d} - Train Loss: {train_loss: 12.6f}, " +
                    f"val Loss: {val_loss:10.6f}, " +
                    f"Best: {best:10.6f}"
                )

        bestModel.train(False)
        self.train(False)
        self.load_state_dict(bestModel.state_dict())

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
    y_train = (train_data.response.values - 1)
    model = MLBA_NN(6, 3, 50, 100, 8)
    model.fit(X_train, y_train)
