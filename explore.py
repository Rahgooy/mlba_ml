import numpy as np
import pickle
from pathlib import Path
from scaler import CustomScaler, DummyScaler
import matplotlib.pyplot as plt


path = Path(f'out/temp/models/mlba_nn_crim_run_4.pkl')
with path.open('rb') as f:
    model, scaler = pickle.load(f)


def predict(m, x):
    x = scaler.transform(x)
    # d = model.get_d(x).detach().numpy().reshape(-1)
    # return d
    p = model.predict_proba(x)[0]
    return p[0] / (p[0] + p[1])


A = [3, 4]
B = [4, 3]

x = np.arange(1, 6, 0.1)
y = np.arange(1, 6, 0.1)
x, y = np.meshgrid(x, y)
z = np.zeros((x.shape[0], x.shape[1], 3))
for i in range(x.shape[0]):
    print(i)
    for j in range(x.shape[1]):
        z[i, j] = predict(model, np.array(
            A + B + [x[i, j], y[i, j]]).reshape(-1, 6))

fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(15, 5))
# ax = plt.axes(projection="3d")
CS = ax[0].contourf(x, y, z[:, :, 0], label='C prob', color='red')
CS = ax[1].contourf(x, y, z[:, :, 1], label='C prob', color='red')
CS = ax[2].contourf(x, y, z[:, :, 2], label='C prob', color='red')
ax[0].plot([1.1, 5.9], [5.9, 1.1], linewidth=1, c='black')
# ax[0].text(4.62, 5.5, 'Width + Height = 7')
ax[0].plot([5.9, 1.1], [1.1, 5.9], linewidth=1, c='black')
ax[1].plot([1.1, 5.9], [5.9, 1.1], linewidth=1, c='black')
ax[1].plot([5.9, 1.1], [1.1, 5.9], linewidth=1, c='black')
ax[2].plot([1.1, 5.9], [5.9, 1.1], linewidth=1, c='black')
ax[2].plot([5.9, 1.1], [1.1, 5.9], linewidth=1, c='black')
# fig.colorbar(CS)
ax[0].set_xlabel('Width of C')
ax[0].set_ylabel('Height of C')
ax[1].set_xlabel('Width of C')
ax[1].set_ylabel('Height of C')
ax[2].set_xlabel('Width of C')
ax[2].set_ylabel('Height of C')
ax[0].set_title('$d_A$ for A = [3, 4]')
ax[1].set_title('$d_B$ for B = [4, 3]')
ax[2].set_title('$d_C$')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(CS, cax=cbar_ax)
plt.show()
plt.legend()
print(model)
