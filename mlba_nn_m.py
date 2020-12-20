import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from mlba import sample_lba
import pandas as pd
import numpy as np
import copy
import time
from lba_dist import LBA
from helpers import jsd, mse, kld
import os
from profiling import global_profiler as profiler, profile
from sklearn.metrics import accuracy_score
from scaler import DummyScaler, CustomScaler


class MLBA_Params:
    def __init__(self, mu_d, sigma_d, A, b):
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.A = A
        self.b = b


class MLBA_NN_M(nn.Module):
    def __init__(self, n_features, n_options, n_hidden, n_epochs, batch, lr=0.001, optim='Adam',
                 weight_decay=0, dropout=0):
        super(MLBA_NN_M, self).__init__()
        # if torch.cuda.is_available():
        #     dev = "cpu" #"cuda:0"
        # else:
        dev = "cpu"
        self.device = torch.device(dev)
        self.lr = lr
        self.weight_decay = weight_decay
        self.f1 = nn.Linear(n_features, n_hidden)
        self.f2 = nn.Linear(n_hidden, n_hidden)
        self.f3 = nn.Linear(n_hidden, n_hidden)
        self.m = self.__tensor([1.], torch.float)
        self.m.requires_grad = True
        # mu_d for each option and A, b, sigma_d
        self.linear_out = nn.Linear(n_hidden, n_options + 3)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout) if dropout else self._no_op
        self.batch_norm = nn.BatchNorm1d(
            n_hidden) if batch > 4 else self._no_op

        self.options = n_options
        self.epochs = n_epochs
        self.batch = batch
        if optim == 'SGD':
            self.optim = torch.optim.SGD
        elif optim == 'RMSprop':
            self.optim = torch.optim.RMSprop
        else:
            self.optim = torch.optim.Adam

    def _no_op(self, x):
        return x

    def to_subjective(self, x, m):
        x = x.reshape(x.shape[0], 3, 2)
        a = b = x.sum(2)
        u1 = b / ((x[:, :, 1] / x[:, :, 0])**m + (b/a)**m)**(1/m)
        u2 = b * (1 - (u1/a)**m)**(1/m)
        return torch.stack([u1, u2], 2).view(x.shape[0], 6)

    @profile
    def forward(self, X):
        n = self.options
        u = self.to_subjective(X, self.softplus(self.m[0]).clamp(.1, 50))
        x = self.f1(u)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.f2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.f3(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.linear_out(x)

        mu_d = (self.sigmoid(x[:, :n]) * 10 + 1).view(-1, n)
        # sigma_d = (self.sigmoid(x[:, n]) * 5 + 0.1).view(-1, 1)
        A = (self.sigmoid(x[:, n+1]) * 10 + 0.1).view(-1, 1)
        sigma_d = torch.ones_like(A)
        b = (self.sigmoid(x[:, n+2]) * 10 + 0.1).view(-1, 1) + A

        return MLBA_Params(mu_d, sigma_d, A, b)

    @profile
    def loss(self, X, y):
        params = self.forward(X)
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d, 25)
        probs = lba.probs()
        nll = nn.NLLLoss()
        log_probs = torch.log(probs + 1e-6)  # avoid nans
        return nll(log_probs, y.view(-1))

    def predict_proba(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        params = self.forward(x)
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        return lba.probs().detach().numpy()

    def get_d(self, X):
        x = torch.Tensor(X.tolist()).to(self.device)
        params = self.forward(x)
        lba = LBA(params.A, params.b, params.mu_d, params.sigma_d)
        return lba.d

    def score(self, X, y):
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        probs = self.predict_proba(X)
        y_ = np.argmax(probs, 1)
        return accuracy_score(y.reshape(-1), y_)

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

    def rotate_options(self, X, y):
        X1 = X.copy()
        X1[:, 0] = X[:, 2]
        X1[:, 1] = X[:, 3]
        X1[:, 2] = X[:, 0]
        X1[:, 3] = X[:, 1]
        y1 = y.copy()
        y1[y == 0] = 1
        y1[y == 1] = 0
        X2 = X.copy()
        X2[:, 0] = X[:, 4]
        X2[:, 1] = X[:, 5]
        X2[:, 4] = X[:, 0]
        X2[:, 5] = X[:, 1]
        y2 = y.copy()
        y2[y == 0] = 2
        y2[y == 2] = 0
        return np.concatenate((X, X1, X2)), np.concatenate((y, y1, y2))

    def fit(self, X, y, X_val=None, y_val=None, early_stop=False):
        # X, y = self.rotate_options(X, y)
        X = self.__tensor(X.tolist(), torch.float)
        y = self.__tensor(y.tolist(), torch.long)
        if X_val is not None and y_val is not None:
            # X_val, y_val = self.rotate_options(X_val, y_val)
            X_val = self.__tensor(X_val.tolist(), torch.float)
            y_val = self.__tensor(y_val.tolist(), torch.long)

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=self.batch)

        optimizer = self.optim(list(self.parameters()) + [self.m], lr=self.lr,
                               weight_decay=self.weight_decay)
        best = float('inf')
        best_model = None
        for epoch in range(self.epochs):
            train_loss = self.__train_step(optimizer, train_loader)
            val_loss = float('inf')
            if X_val is not None and y_val is not None:
                val_loss = self.loss(X_val, y_val).item()
                if val_loss < best:
                    best = val_loss
                    best_model = copy.deepcopy(self)

            if epoch % 5 == 0:
                print(
                    f"[{os.getpid()}] {time.asctime()} >  Epoch {epoch:5d} - Loss: {train_loss: 10.9f}, " +
                    f"val Loss: {val_loss:10.6f}, " +
                    f"Best: {best:10.6f}"
                )
        if early_stop and best_model is not None:
            self.load_state_dict(best_model.state_dict())

    @profile
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


def runRectangles(n_hidden, epochs, batch, lr, weight_decay, dropout, test_size, early_stop):
    train_data = pd.read_csv('data/E2.csv')
    e1a = pd.read_csv('data/E1a.csv')
    e1b = pd.read_csv('data/E1b.csv')
    e1c = pd.read_csv('data/E1c.csv')

    runExperiment(train_data, e1a, e1b, e1c, n_hidden, epochs,
                  batch, lr, weight_decay, dropout, test_size, early_stop)


def runCriminals(n_hidden, epochs, batch, lr, weight_decay, dropout, test_size, early_stop):
    train_data = pd.read_csv('data/E4.csv')
    e1a = pd.read_csv('data/E3a.csv')
    e1b = pd.read_csv('data/E3b.csv')
    e1c = pd.read_csv('data/E3c.csv')

    runExperiment(train_data, e1a, e1b, e1c, n_hidden, epochs,
                  batch, lr,  weight_decay, dropout, test_size, early_stop)


def runExperiment(train_data, e_a, e_b, e_c, n_hidden, epochs, batch, lr, weight_decay, dropout, test_size, early_stop):
    features = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    X = train_data
    y = (train_data.response.values - 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y.reshape(-1, 1), test_size=test_size) if test_size > 0 else (X, X[:1], y, y[:1])
    scaler = DummyScaler()

    model = MLBA_NN_M(6, 3, n_hidden=n_hidden, n_epochs=epochs, batch=batch, lr=lr,
                      weight_decay=weight_decay, dropout=dropout)
    model.fit(scaler.fit_transform(X_train[features].values), y_train,
              scaler.transform(X_val[features].values), y_val, early_stop)

    def evaluate(X, y, print_):
        resp_freq, _ = np.histogram(y, 3)
        probs = resp_freq / X.shape[0]
        probs1 = model.predict_proba(X).mean(0)
        # probs2 = model.predict_proba_mlba(X).mean(0)
        if print_:
            print("Actual:", probs)
            print("Predicted directly:", probs1,
                  "MSE:", mse(probs, probs1),
                  "Count:", X.shape[0])
        # print("Predicted simulated:", probs2,
        #       "MSE:", mse(probs, probs2))
        return mse(probs, probs1)

    def per_effect(X, y, scaler):
        effects = X.groupby('Effect')
        overall = 0
        counts = 0
        for e, d in effects:
            eff = X.Effect.str.startswith(e)
            overall += evaluate(scaler.transform(
                X[features].values[eff]), y[eff], False) * d.shape[0]
            counts += d.shape[0]
        print("Overall:", overall/counts)
        return overall/counts, counts

    print("train")
    evaluate(scaler.transform(X_train[features].values), y_train, True)
    per_effect(X_train, y_train, scaler)

    print("\nval")
    evaluate(scaler.transform(X_val[features].values), y_val, True)
    per_effect(X_val, y_val, scaler)

    print("\ne_a")
    mse_a, counts_a = per_effect(e_a, e_a.response.values - 1, scaler)

    print("e_b")
    mse_b, counts_b = per_effect(e_b, e_b.response.values - 1, scaler)

    print("e_c")
    mse_c, counts_c = per_effect(e_c, e_c.response.values - 1, scaler)

    print("Mean:", (mse_a * counts_a + mse_b * counts_b +
                    mse_c * counts_c) / (counts_a + counts_b + counts_c))

    print(model.m)


if __name__ == "__main__":
    runRectangles(n_hidden=50, epochs=200, batch=1024, lr=0.001,
                  weight_decay=0, dropout=0, test_size=.33, early_stop=True)
    # runCriminals(n_hidden=50, epochs=200, batch=1024, lr=0.001,
    #              weight_decay=0, dropout=0, test_size=0.33, early_stop=True)

    profiler.print_profile()