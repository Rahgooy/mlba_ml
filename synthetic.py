import numpy as np


def generate_att(n):
    mu = [50, 80]
    sig = [[2, 0], [0, 2]]
    rnd = np.random.multivariate_normal(mu, sig, size=n)
    h1 = rnd[:, 0]
    w1 = rnd[:, 1]
    h2 = w1 + np.random.uniform(-2, 2, size=(n, ))
    w2 = (h1 * w1) / h2

    hAttR1 = h1 - np.random.uniform(7, 9, size=(n, ))
    wAttR1 = w1

    hAttR2 = h2
    wAttR2 = w2 - np.random.uniform(7, 9, size=(n, ))

    return h1, w1, h2, w2, hAttR1, wAttR1, hAttR2, wAttR2


def generate_sim(n):
    mu = [60, 60]
    sig = [[1, 0], [0, 1]]
    rnd = np.random.multivariate_normal(mu, sig, size=n)
    rnd = np.round(rnd)
    h_A = rnd[:, 0]
    w_A = rnd[:, 1]

    w_C = w_A + np.random.uniform(14, 16, size=(n, ))
    h_C = (h_A * w_A) / w_C

    h_SA = h_A - np.random.uniform(3, 5, size=(n, ))
    w_SA = (h_A * w_A) / h_SA

    h_SC = h_C - np.random.uniform(3, 5, size=(n, ))
    w_SC = (h_C * w_C) / h_SC

    return h_A, w_A, h_C, w_C, h_SA, w_SA, h_SC, w_SC


def generate_comp(n):
    h1, w1, h2, w2, _, _, _, _ = generate_att(n)
    h_C1 = h1 - np.random.uniform(9, 12, size=(n, ))
    w_C1 = (h1 * w1) / h_C1

    w_C2 = w2 - np.random.uniform(9, 12, size=(n, ))
    h_C2 = (h2 * w2) / w_C2

    return h1, w1, h2, w2, h_C1, w_C1, h_C2, w_C2


def write(f, h1, w1, h2, w2, h3, w3, effect):
    for i in range(len(h1)):
        f.write(f'{h1[i]},{w1[i]},{h2[i]},{w2[i]},{h3[i]},{w3[i]},{effect}\n')


def write_header(f):
    f.write(
        'Rect1Height,Rect1Width,Rect2Height,Rect2Width,Rect3Height,Rect3Width,Effect\n')


def generate_data(n_trials, n_problems_per_cond, path):
    with open(path, 'w') as f:
        write_header(f)

        h1, w1, h2, w2, hAttR1, wAttR1, hAttR2, wAttR2 = generate_att(
            n_problems_per_cond)
        for i in range(n_trials):
            write(f, h1, w1, h2, w2, hAttR1, wAttR1, 'AttR1')
            write(f, h1, w1, h2, w2, hAttR2, wAttR2, 'AttR2')

        h_A, w_A, h_C, w_C, h_SA, w_SA, h_SC, w_SC = generate_sim(
            n_problems_per_cond)
        for i in range(n_trials):
            write(f, h_A, w_A, h_C, w_C, h_SA, w_SA, 'SimA1')
            write(f, h_A, w_A, h_C, w_C, h_SC, w_SC, 'SimA2')

        h1, w1, h2, w2, h_C1, w_C1, h_C2, w_C2 = generate_comp(
            n_problems_per_cond)
        for i in range(n_trials):
            write(f, h1, w1, h2, w2, h_C1, w_C1, 'ComB1')
            write(f, h1, w1, h2, w2, h_C2, w_C2, 'ComB1')


if __name__ == "__main__":
    generate_data(100, 100, 'data/synth/synth_100_100.csv')
