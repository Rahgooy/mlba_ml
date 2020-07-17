import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import r2_score
import pickle


class BayesianLogisticRegression:
    def __init__(self, epochs):
        self.W = None
        self.mean = None
        self.sigma = None
        self.epochs = epochs

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def predict_proba(self, x):
        x = np.array(x).tolist()
        x = torch.tensor(x)
        W = torch.tensor(self.W)
        net = torch.softmax(x @ W.T, 1).view(-1, 3)
        return net.detach().numpy()

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        def simple_model(x, y):
            x = torch.tensor(x)
            y = torch.tensor(y)
            W = pyro.sample('W', dist.Normal(torch.zeros(
                (3, x.shape[1])), torch.ones((3, x.shape[1]))).to_event(1))
            net = torch.softmax(x @ W.T, 1).view(-1, 3)
            with pyro.plate("data"):
                pyro.sample('resp', dist.Categorical(probs=net), obs=y)

        def guid(x, y):
            x = torch.tensor(x)
            mean = pyro.param('mean', torch.zeros((3, x.shape[1])))
            sigma = pyro.param('sigma', torch.ones(
                (3, x.shape[1])), constraint=constraints.positive)
            W = pyro.sample('W', dist.Normal(mean, sigma))

        pyro.clear_param_store()
        svi = pyro.infer.SVI(model=simple_model,
                             guide=guid,
                             optim=pyro.optim.Adam({"lr": 0.001}),
                             loss=pyro.infer.Trace_ELBO())

        losses, mean, sigma = [], [], []
        for t in range(self.epochs):
            loss = svi.step(X.tolist(), Y.tolist())
            losses.append(loss)
            mean.append(pyro.param("mean").detach().numpy())
            sigma.append(pyro.param("sigma").detach().numpy())
            if t % 500 == 0:
                print(f'Epoch {t+1} - Loss: {loss:0.3f}')

        self.W = mean[-1]
        self.mean = mean[-1]
        self.sigma = sigma[-1]
        return losses, mean, sigma

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            m = pickle.load(f)
            self.mean = m.mean
            self.sigma = m.sigma
            self.W = m.W
