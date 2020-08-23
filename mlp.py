import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import copy
import pandas as pd
import numpy as np


class MLP(torch.nn.Module):
    def __init__(self, n_features, n_options, n_hidden, n_epochs, batch):
        """A multi layer perceptron implementation for choice prediction

        Args:
            n_features (int): number of features
            n_options (int): number of options
            n_hidden (int): number of hidden nodes
        """
        super(MLP, self).__init__()
        self.f1 = nn.Linear(n_features, n_hidden)
        self.f2 = nn.Linear(n_hidden, n_hidden)
        self.f3 = nn.Linear(n_hidden, n_options)
        self.softplus = nn.Softplus()
        self.options = n_options
        self.epochs = n_epochs
        self.batch = batch

        if torch.cuda.is_available():
            dev = f"cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

    def forward(self, X):
        x = self.f1(X)
        x = torch.relu(x)
        x = self.f2(x)
        x = torch.relu(x)
        x = self.f3(x)
        probs = torch.softmax(x, 1)
        # sigma = self.softplus(x[:, self.options:])
        return probs

    def loss(self, X, y):
        probs = self.forward(X)
        logs = torch.log(probs)
        l = torch.nn.NLLLoss()
        return l(logs, y.view(-1))

    def predict_proba(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        y = self.forward(x)
        return y.detach().numpy()

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

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
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
    model = MLP(6, 3, 50, 1000, 8)
    model.fit(X_train, y_train)
