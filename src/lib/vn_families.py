from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pylab as plt

from PerplexityLab.visualization import perplex_plot, one_line_iterator

SMALL_SIZE = 8 * 3
MEDIUM_SIZE = 10 * 3
BIGGER_SIZE = 12 * 3

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE * 2 / 3)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

K_MAX = 500
ZERO = 1e-15
Bounds = namedtuple("Bounds", "lower upper")
MWhere = namedtuple("MWhere", "m start")


# ========= ========== =========== ========== #
#                   Families                  #
# ========= ========== =========== ========== #
@dataclass(unsafe_hash=True)
class VnFamily:
    a: Tuple[float, float] = Bounds(lower=0, upper=1)
    b: Tuple[float, float] = Bounds(lower=1, upper=1)
    delta: Tuple[float, float] = Bounds(lower=0.5, upper=0.5)
    c: float = 0
    s: float = 0
    k_max: int = K_MAX

    def __eq__(self, other):
        return other.a == self.a and \
               other.b == self.b and \
               other.delta == self.delta and \
               other.c == self.c and \
               other.s == self.s

    @property
    def dim(self):
        return (self.a.lower < self.a.upper) + \
               (self.b.lower < self.b.upper) + \
               (self.delta.lower < self.delta.upper) + \
               (self.c > 0) * (self.k_max)

    def __repr__(self):
        vals = [f"{v}: [{getattr(self, v).lower}, {getattr(self, v).upper}]" for v in ["a", "b", "delta"]] + [
            f"c: {self.c}; s: {self.s}; k max: {self.k_max}"]
        return "; ".join(vals[:self.dim])


def get_k_eigenvalues(a: Union[np.ndarray, float], b: Union[np.ndarray, float], delta: Union[np.ndarray, float],
                      k_max: int, c: float = 0, s: float = -1):
    k = np.arange(1, k_max + 1)
    # pure sin cos
    # eigen_cos = np.sin(np.pi * k[np.newaxis, :] * delta[:, np.newaxis]) * np.cos(
    #     np.pi * k[np.newaxis, :] * (2 * a + delta)[:, np.newaxis])
    # eigen_sin = -np.sin(np.pi * k[np.newaxis, :] * delta[:, np.newaxis]) * np.sin(
    #     np.pi * k[np.newaxis, :] * (2 * a + delta)[:, np.newaxis])

    # eigenvalues = 2 * b[:, np.newaxis] / (np.pi * np.repeat(k, 2)[np.newaxis, :]) * \
    #               np.reshape([eigen_cos.T, eigen_sin.T], (-1, len(a)), order="F").T

    twopik = 2 * np.pi * k[np.newaxis, :]
    eigen_sin = -(np.cos(twopik * (a + delta)[:, np.newaxis]) - np.cos(twopik * a[:, np.newaxis]))
    eigen_cos = np.sin(twopik * (a + delta)[:, np.newaxis]) - np.sin(twopik * a[:, np.newaxis])

    eigenvalues = 2 * b[:, np.newaxis] / (2 * np.pi * np.repeat(k, 2)[np.newaxis, :]) * \
                  np.reshape([eigen_cos.T, eigen_sin.T], (-1, len(a)), order="F").T

    # add regular noise
    noise = c * np.reshape(np.random.uniform(-1, 1, size=(2,) + np.shape(eigen_sin)[::-1]), (-1, len(a)),
                           order="F").T * np.repeat(np.array(k, dtype=float), 2)[np.newaxis, :] ** s

    eigen0 = delta * b
    return np.hstack((eigen0[:, np.newaxis], eigenvalues)), np.hstack((np.zeros((np.shape(noise)[0], 1)), noise))


def vn_family_sampler(n, a_limits: Tuple[float, float], b_limits: Tuple[float, float],
                      delta_limits: Tuple[float, float], seed=42):
    np.random.seed(seed)
    a = np.random.uniform(*a_limits, size=n)
    b = np.random.uniform(*b_limits, size=n)
    delta = np.random.uniform(*delta_limits, size=n)
    return a, b, delta


def get_known_unknown_indexes(mwhere, learn_higher_modes_only=False, quantity=None):
    start = max((0, mwhere.start * 2 - 1))
    known_indexes = list(range(start, start + mwhere.m))
    unknown_indexes = [i for i in range(1 + 2 * K_MAX) if i not in known_indexes]
    if learn_higher_modes_only:
        change = np.where(np.diff(unknown_indexes) > 1)[0]
        change = change[0] if len(change) > 0 else -1
        unknown_indexes = unknown_indexes[change + 1:]
    return known_indexes, unknown_indexes[:quantity]


def get_k_values(negative=False):
    return np.append([0], np.repeat(np.arange(1, K_MAX + 1, dtype=float), 2) * np.array([-1, 1] * K_MAX) ** negative)


def learn_eigenvalues(model: Pipeline):
    def decorated_function(n_train, n_test, vn_family, mwhere: MWhere, k_decay_help, learn_higher_modes_only=True,
                           quantity=None):
        a, b, delta = vn_family_sampler(n_test + n_train, a_limits=vn_family.a, b_limits=vn_family.b,
                                        delta_limits=vn_family.delta)
        # shape(n, 1+2*k_max)
        known_indexes, unknown_indexes = get_known_unknown_indexes(mwhere, learn_higher_modes_only, quantity)
        eigenvalues, noise = get_k_eigenvalues(a, b, delta, K_MAX, vn_family.c, vn_family.s)
        k = get_k_values(negative=False)
        k[0] += 1  # +1 to avoid the 0 division
        k_known = k[known_indexes] ** k_decay_help
        k_unknown = k[unknown_indexes] ** k_decay_help
        model.fit((eigenvalues + noise)[n_test:, known_indexes] * k_known[np.newaxis, :],
                  eigenvalues[n_test:, unknown_indexes] * k_unknown[np.newaxis, :])
        predictions = model.predict((eigenvalues + noise)[:n_test, known_indexes] * k_known[np.newaxis, :])
        error = eigenvalues[:n_test, unknown_indexes] - predictions / k_unknown[np.newaxis, :]
        return {
            "error": error
        }

    decorated_function.__name__ = " ".join([s[0] for s in model.steps])
    return decorated_function


# ========= ========== =========== ========== #
#                   Plots                     #
# ========= ========== =========== ========== #
@perplex_plot()
def k_plot(fig, ax, error, model, mwhere, n_train, learn_higher_modes_only, vn_family, quantity, add_mwhere=False,
           color_dict=None, label_var="model"):
    error, mwhere, n_train, model, learn_higher_modes_only, vn_family, quantity = tuple(
        zip(*[(e, m, n, ex, l, vn, q) for e, m, n, ex, l, vn, q in
              zip(error, mwhere, n_train, model, learn_higher_modes_only,
                  vn_family, quantity)
              if
              e is not None and ex is not None])
    )

    mse = list(map(lambda e: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).squeeze(), error))
    k_full = get_k_values(negative=True)
    k_full[k_full > 0] = np.log10(k_full[k_full > 0])
    k_full[k_full < 0] = -np.log10(-k_full[k_full < 0])

    # for i, (exp_i, y_i, ms, lhmo, ntr) in enumerate(zip(model, mse, mwhere, learn_higher_modes_only, n_train)):
    for i, (ntr, y_i, ms, hmonly, mod, q) in enumerate(zip(n_train, mse, mwhere, learn_higher_modes_only, model, quantity)):
        y_i = np.reshape(y_i, (-1,))
        known_indexes, unknown_indexes = get_known_unknown_indexes(ms, hmonly, q)
        # change = np.where(np.diff(unknown_indexes) > 1)[0][0]
        # y_i = y_i[change:]
        k = k_full[unknown_indexes]  # [change:]
        # TODO: do it without an if
        if label_var == "model":
            label_i = mod
        elif label_var == "n_train":
            label_i = ntr
        elif label_var == "quantity":
            label_i = q
        else:
            raise Exception(f"label_var {label_var} not implemented.")

        if color_dict is None:
            c = sns.color_palette("colorblind")[i]
        else:
            c = color_dict[label_i]
        # m = [".", "*", ""][i]
        m = "o"
        ax.plot(k[(y_i > ZERO) & (k < 0)], y_i[(y_i > ZERO) & (k < 0)], "--", marker=m, c=c)
        ax.plot(k[(y_i > ZERO) & (k > 0)], y_i[(y_i > ZERO) & (k > 0)], "--", marker=m,
                label=str(label_i) + (f": start={ms.start}, m={ms.m}" if add_mwhere else ""), c=c)

    vn_family = vn_family[0]
    a, b, delta = vn_family_sampler(n=1000, a_limits=vn_family.a, b_limits=vn_family.b,
                                    delta_limits=vn_family.delta)
    eigenvalues, _ = get_k_eigenvalues(a, b, delta, K_MAX, vn_family.c, vn_family.s)
    eigenvalues_sd = eigenvalues.std(axis=0)
    ax.plot(k_full[(k_full < 0) & (eigenvalues_sd > ZERO)], eigenvalues_sd[(k_full < 0) & (eigenvalues_sd > ZERO)],
            ":k")
    ax.plot(k_full[(k_full > 0) & (eigenvalues_sd > ZERO)], eigenvalues_sd[(k_full > 0) & (eigenvalues_sd > ZERO)],
            ":k", label=r"$null model$")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend(loc='upper right')
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha_k$" + "\t\t\t\t   " + r"$\beta_k$")
    ax.set_ylabel("MSE")


@perplex_plot(plot_by_default=["vn_family"], axes_by_default=["mwhere", "learn_higher_modes_only"])
@one_line_iterator
def plot_reconstruction(fig, ax, error, n_train, n_test, vn_family, model, mwhere: MWhere, k_decay_help,
                        learn_higher_modes_only, i=1, num_points=K_MAX * 10):
    a, b, delta = vn_family_sampler(n_test + n_train, a_limits=vn_family.a, b_limits=vn_family.b,
                                    delta_limits=vn_family.delta)
    eigenvalues, noise = get_k_eigenvalues(a, b, delta, K_MAX, vn_family.c, vn_family.s)
    known_indexes, unknown_indexes = get_known_unknown_indexes(mwhere, learn_higher_modes_only)

    # k*x creates a matrix #k x #x but at the end we obtain #x x #k
    k = np.arange(1, K_MAX + 1)[:, np.newaxis]
    x = np.linspace(0, 1, num_points)[np.newaxis, :]
    eigenvectors = np.reshape([np.cos(2 * np.pi * k * x), np.sin(2 * np.pi * k * x)], (-1, num_points), order="F").T
    eigenvectors = np.hstack((np.ones((num_points, 1)), eigenvectors))

    # plot ground truth
    x = np.ravel(x)
    ground_truth = 0 + \
                   (delta[i] > (x - a[i])) * ((x - a[i]) > 0) * b[i] + \
                   (delta[i] > (1 + x - a[i])) * ((1 + x - a[i]) > 0) * b[i]
    # ground_truth = x * 0 + (((x - a[i]) % 1) > 0) * (((x - a[i] - delta[i]) % 1) < 0) * b[i]
    ax.plot(x, ground_truth, label=f"Ground truth")
    ax.plot(x, eigenvectors @ eigenvalues[:n_test][i, :], label=f"Ground truth threshold")
    # eigenvalues[:n_test][i, :]
    ax.plot(x, ground_truth + eigenvectors @ noise[i, :], label=f"Ground truth + regular")
    # plot approximation
    eigenvalues[:n_test][i, unknown_indexes] -= error[i, :]
    eigenvalues[:n_test][i, known_indexes] += noise[i, known_indexes]
    reconstruction = eigenvectors @ eigenvalues[:n_test][i, :]
    ax.plot(x, reconstruction, label=f"{model}: reconstruction of ground truth: ")


if __name__ == "__main__":
    print("Testing the correctness of eigenvalues")
    vn_family = VnFamily(a=(0, 1))

    a, b, delta = vn_family_sampler(n=3, a_limits=vn_family.a, b_limits=vn_family.b, delta_limits=vn_family.delta,
                                    seed=42)
    k_max = 10
    eigenvalues, _ = get_k_eigenvalues(a, b, delta, k_max)
    print(eigenvalues)
    assert np.shape(eigenvalues) == (len(a), 2 * k_max + 1)

    a, b, delta = vn_family_sampler(n=1, a_limits=vn_family.a, b_limits=vn_family.b, delta_limits=vn_family.delta,
                                    seed=42)
    eigenvalues, _ = get_k_eigenvalues(a, b, delta, k_max)
    print(f"a={a}, b={b}, delta={delta}")
    for k in range(1, k_max, 2):
        eigen_sin = b * np.cos(2 * np.pi * a) / (2 * np.pi * k)
        eigen_cos = b * np.sin(2 * np.pi * a) / (2 * np.pi * k)
        print(np.allclose(np.transpose([eigen_cos, eigen_sin]), eigenvalues[:, [2 * k - 1, 2 * k]]))
        # print(np.transpose([eigen_cos, eigen_sin]) - get_k_eigenvalues(a, b, delta, k))

    a, b, delta = vn_family_sampler(n=100000, a_limits=vn_family.a, b_limits=vn_family.b, delta_limits=vn_family.delta,
                                    seed=42)
    k_max = 5
    eigenvalues, _ = get_k_eigenvalues(a, b, delta, k_max=k_max)
    print(eigenvalues.mean(axis=0) * np.append(0, np.repeat(np.arange(1, k_max + 1), 2)))
    import matplotlib.pylab as plt

    plt.hist(eigenvalues[:, 1])
    plt.show()


    def unroll(eigenvalues):
        eigenvalues = eigenvalues[0] + eigenvalues[1]
        return np.concatenate((eigenvalues[:, 1::2][:, ::-1], [eigenvalues[:, 0]], eigenvalues[:, 2::2]),
                              axis=1).ravel()


    import pandas as pd
    import seaborn as sns

    k_max = 20
    c = 0.5
    a = 0.25124
    s = -1
    sns.barplot(pd.DataFrame(np.transpose([np.append(
        unroll(get_k_eigenvalues(a=np.array([a]), b=np.array([1]), delta=np.array([0.5]), k_max=k_max, c=0, s=s)),
        unroll(get_k_eigenvalues(a=np.array([a]), b=np.array([1]), delta=np.array([0.5]), k_max=k_max, c=c, s=s))),
        np.append(np.arange(2 * k_max + 1) - k_max, np.arange(2 * k_max + 1) - k_max),
        np.append(np.repeat(0, k_max * 2 + 1), np.repeat(c, k_max * 2 + 1))]),
        columns=["eigenvalues", "m", "family"]),
        x="m", y="eigenvalues", hue="family")
    plt.show()
