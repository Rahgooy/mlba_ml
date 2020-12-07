import numpy as np
import pickle
from pathlib import Path
from scaler import CustomScaler, DummyScaler
import matplotlib.pyplot as plt


path = Path(f'out/temp/models/mlba_nn_rect_run_1.pkl')
with path.open('rb') as f:
    model, scaler = pickle.load(f)


def predict(m, x):
    x = scaler.transform(x)
    d = model.get_d(x).detach().numpy().reshape(-1)
    return d


A = [3, 4]
B = [4, 3]

x = np.arange(1, 6, 0.3)
y = np.arange(1, 6, 0.3)
x, y = np.meshgrid(x, y)
z = np.zeros((x.shape[0], x.shape[1], 3))
for i in range(x.shape[0]):
    print(i)
    for j in range(x.shape[1]):
        z[i, j] = predict(model, np.array(
            A + B + [x[i, j], y[i, j]]).reshape(-1, 6))

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(x, y, z[:, :, 0], label='A prob', color='green')
ax.plot_wireframe(x, y, z[:, :, 1], label='B prob', color='blue')
ax.plot_wireframe(x, y, z[:, :, 2], label='C prob', color='red')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('prob')
plt.show()
plt.legend()
print(model)
