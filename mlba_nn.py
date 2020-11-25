import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlba import sample_lba
import pandas as pd
import numpy as np
import copy
import time
from lba_dist import LBA
from helpers import jsd, mse, kld
import os


class MLBA_Params:
    def __init__(self, mu_d, sigma_d, A, b):
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.A = A
        self.b = b


class MLBA_NN(nn.Module):
    def __init__(self, n_features, n_options, n_hidden, n_epochs, batch, lr=0.001, optim='Adam',
                 weight_decay=0, dropout=0):
        super(MLBA_NN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.f1 = nn.Linear(n_features, n_hidden)
        self.f2 = nn.Linear(n_hidden, n_hidden)
        # mu_d for each option and A, b, sigma_d
        self.f3 = nn.Linear(n_hidden, n_options + 3)
        self.softPlus = nn.Softplus()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x
        self.batch_norm = nn.BatchNorm1d(n_hidden)

        self.options = n_options
        self.epochs = n_epochs
        self.batch = batch
        if optim == 'SGD':
            self.optim = torch.optim.SGD
        elif optim == 'RMSprop':
            self.optim = torch.optim.RMSprop
        else:
            self.optim = torch.optim.Adam

        # if torch.cuda.is_available():
        #     dev = "cpu" #"cuda:0"
        # else:
        dev = "cpu"
        self.device = torch.device(dev)

    def forward(self, X):
        n = self.options
        x = self.f1(X)
        # x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.f2(x)
        # x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.f3(x)

        mu_d = (self.sigmoid(x[:, :n]) * 10 + 1).view(-1, n)
        sigma_d = (self.sigmoid(x[:, n]) * 5 + 0.1).view(-1, 1)
        A = (self.sigmoid(x[:, n+1]) * 10 + 0.1).view(-1, 1)
        b = (self.sigmoid(x[:, n+2]) * 10 + 0.1).view(-1, 1) + A

        return MLBA_Params(mu_d, sigma_d, A, b)

    def loss(self, X, y):
        params = self.forward(X)
        nll = 0.0
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        probs = lba.probs()
        nll = nn.NLLLoss()
        return nll(torch.log(probs), y.view(-1))

    def mse(self, X, y):
        hist, _ = np.histogram(y, self.options)
        actual_freq = hist/hist.sum()
        pred_freq = self.predict_proba(X).mean(0)
        return mse(actual_freq, pred_freq)

    def jsd(self, X, y):
        hist, _ = np.histogram(y, self.options)
        actual_freq = hist/hist.sum()
        pred_freq = self.predict_proba(X).mean(0)
        return jsd(actual_freq, pred_freq)

    def kld(self, X, y):
        hist, _ = np.histogram(y, self.options)
        actual_freq = hist/hist.sum()
        pred_freq = self.predict_proba(X).mean(0)
        return kld(actual_freq, pred_freq)

    def predict_proba(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        params = self.forward(x)
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        return lba.probs().detach().numpy()

    def predict_proba_mlba(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        params = self.forward(x)
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        probs = []
        n = 10000
        for i in range(X.shape[0]):
            rt, resp = sample_lba(n, params.b[i].item(), params.A[i].item(),
                                  params.mu_d[i].detach().reshape(1, -1),
                                  torch.ones((1, self.options)) * params.sigma_d[i].item(), 0)
            resp_freq, _ = np.histogram(resp, 3)
            probs.append(resp_freq / n)

        return np.array(probs)

    def __tensor(self, x, dtype):
        return torch.tensor(x, dtype=dtype).to(self.device)

    def fit(self, X, y):
        X = self.__tensor(X.tolist(), torch.float)
        y = self.__tensor(y.tolist(), torch.long)

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=self.batch)

        optimizer = self.optim(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            train_loss = self.__train_step(optimizer, train_loader)
            if epoch % 5 == 0:
                print(
                    f"[{os.getpid()}] {time.asctime()} >  Epoch {epoch:5d} - Loss: {train_loss: 12.6f}, " +
                    f"MSE: {self.mse(X, y): 12.6f}, JSD: {self.jsd(X, y): 12.6f}, KLD: {self.kld(X, y): 12.6f}"
                )

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


def runRectangles():
    train_data = pd.read_csv('data/E2.csv')
    e1a = pd.read_csv('data/E1a.csv')
    e1b = pd.read_csv('data/E1b.csv')
    e1c = pd.read_csv('data/E1c.csv')

    runExperiment(train_data, e1a, e1b, e1c)


def runCriminals():
    train_data = pd.read_csv('data/E4.csv')
    e1a = pd.read_csv('data/E3a.csv')
    e1b = pd.read_csv('data/E3b.csv')
    e1c = pd.read_csv('data/E3c.csv')

    runExperiment(train_data, e1a, e1b, e1c)


def runExperiment(train_data, e_a, e_b, e_c):
    features = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    X = train_data
    y = (train_data.response.values - 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y.reshape(-1, 1), test_size=0.2)
    scaler = StandardScaler()

    model = MLBA_NN(6, 3, 100, 100, 128, 0.0001, weight_decay=0.5, dropout=0.2)
    model.fit(scaler.fit_transform(X_train[features].values), y_train)

    def evaluate(X, y):
        resp_freq, _ = np.histogram(y, 3)
        probs = resp_freq / len(y)
        probs1 = model.predict_proba(X).mean(0)
        probs2 = model.predict_proba_mlba(X).mean(0)

        print("Actual:", probs)
        print("Predicted directly:", probs1,
              "MSE:", mse(probs, probs1), "JSD: ", jsd(probs, probs1))
        print("Predicted simulated:", probs2,
              "MSE:", mse(probs, probs2), "JSD: ", jsd(probs, probs2))

    print("train_a")
    eff = X_train.Effect.str.startswith('Att')
    evaluate(scaler.transform(X_train[features].values[eff]), y_train[eff])

    print("train_b")
    eff = X_train.Effect.str.startswith('Com')
    evaluate(scaler.transform(X_train[features].values[eff]), y_train[eff])

    print("train_c")
    eff = X_train.Effect.str.startswith('Sim')
    evaluate(scaler.transform(X_train[features].values[eff]), y_train[eff])

    print("\nval_a")
    eff = X_val.Effect.str.startswith('Att')
    evaluate(scaler.transform(X_val[features].values[eff]), y_val[eff])

    print("val_b")
    eff = X_val.Effect.str.startswith('Com')
    evaluate(scaler.transform(X_val[features].values[eff]), y_val[eff])

    print("val_c")
    eff = X_val.Effect.str.startswith('Sim')
    evaluate(scaler.transform(X_val[features].values[eff]), y_val[eff])

    print("\ne_a")
    evaluate(scaler.transform(e_a[features].values), e_a.response.values - 1)

    print("e_b")
    evaluate(scaler.transform(e_b[features].values), e_b.response.values - 1)

    print("e_c")
    evaluate(scaler.transform(e_c[features].values), e_c.response.values - 1)


if __name__ == "__main__":
    runRectangles()
    # runCriminals()
