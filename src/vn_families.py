from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pylab as plt

from src.viz_utils import perplex_plot

SMALL_SIZE = 8*2
MEDIUM_SIZE = 10*2
BIGGER_SIZE = 12*2

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

K_MAX = 500
ZERO = 1e-15
Bounds = namedtuple("Bounds", "lower upper")
MWhere = namedtuple("MWhere", "m start")


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
            f"c: {self.c}; k max: {self.k_max}"]
        return "; ".join(vals[:self.dim])


def get_k_eigenvalues(a: Union[np.ndarray, float], b: Union[np.ndarray, float], delta: Union[np.ndarray, float],
                      k_max: int, c: float = 0, s: float = -1):
    k = np.arange(1, k_max + 1)
    # pure sin cos
    eigen_cos = np.sin(np.pi * k[np.newaxis, :] * delta[:, np.newaxis]) * np.cos(
        np.pi * k[np.newaxis, :] * (2 * a + delta)[:, np.newaxis])
    eigen_sin = -np.sin(np.pi * k[np.newaxis, :] * delta[:, np.newaxis]) * np.sin(
        np.pi * k[np.newaxis, :] * (2 * a + delta)[:, np.newaxis])

    eigenvalues = b[:, np.newaxis] / (np.pi * np.repeat(k, 2)[np.newaxis, :]) * \
                  np.reshape([eigen_cos.T, eigen_sin.T], (-1, len(a)), order="F").T

    # add regular noise
    if c > 0:
        eigenvalues += np.reshape(np.random.uniform(-c, c, size=(2,) + np.shape(eigen_sin)[::-1]), (-1, len(a)),
                                  order="F").T * np.repeat(np.array(k, dtype=float), 2)[np.newaxis, :] ** s

    eigen0 = delta * b
    return np.hstack((eigen0[:, np.newaxis], eigenvalues))


def vn_family_sampler(n, a_limits: Tuple[float, float], b_limits: Tuple[float, float],
                      delta_limits: Tuple[float, float], seed=42):
    np.random.seed(seed)
    a = np.random.uniform(*a_limits, size=n)
    b = np.random.uniform(*b_limits, size=n)
    delta = np.random.uniform(*delta_limits, size=n)
    return a, b, delta


def get_known_unknown_indexes(mwhere):
    start = max((0, mwhere.start * 2 - 1))
    known_indexes = list(range(start, start + mwhere.m))
    unknown_indexes = [i for i in range(1 + 2 * K_MAX) if i not in known_indexes]
    return known_indexes, unknown_indexes


def learn_eigenvalues(model: Pipeline):
    def decorated_function(n_train, n_test, vn_family, mwhere: MWhere):
        a, b, delta = vn_family_sampler(n_test + n_train, a_limits=vn_family.a, b_limits=vn_family.b,
                                        delta_limits=vn_family.delta)
        # shape(n, 1+2*k_max)
        known_indexes, unknown_indexes = get_known_unknown_indexes(mwhere)
        eigenvalues = get_k_eigenvalues(a, b, delta, K_MAX, vn_family.c, vn_family.s)
        model.fit(eigenvalues[n_test:, known_indexes],
                  eigenvalues[n_test:, unknown_indexes])
        predictions = model.predict(eigenvalues[:n_test, known_indexes])
        error = eigenvalues[:n_test, unknown_indexes] - predictions
        return {
            "error": error
        }

    decorated_function.__name__ = " ".join([s[0] for s in model.steps])
    return decorated_function


@perplex_plot
def k_plot(fig, ax, error, experiments, mwhere, add_mwhere=False):
    error, mwhere, experiments = tuple(
        zip(*[(e, m, ex) for e, m, ex in zip(error, mwhere, experiments) if
              e is not None and ex is not None]))

    mse = list(map(lambda e: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).squeeze(), error))
    k_full = np.append([0], np.repeat(np.arange(1, K_MAX + 1, dtype=float), 2) * np.array([-1, 1] * K_MAX))
    k_full[k_full > 0] = np.log10(k_full[k_full > 0])
    k_full[k_full < 0] = -np.log10(-k_full[k_full < 0])

    for i, (y_i, label_i, ms) in enumerate(zip(mse, experiments, mwhere)):
        _, unknown_indexes = get_known_unknown_indexes(ms)
        k = k_full[unknown_indexes]
        c = sns.color_palette("colorblind")[i]
        # m = [".", "*", ""][i]
        m = "."
        ax.scatter(k[(y_i > ZERO) & (k < 0)], y_i[(y_i > ZERO) & (k < 0)], marker=m, color=c)
        ax.scatter(k[(y_i > ZERO) & (k > 0)], y_i[(y_i > ZERO) & (k > 0)], marker=m,
                   label=label_i + (f": start={ms.start}, m={ms.m}" if add_mwhere else ""), color=c)
    k = np.sort(np.unique(np.ravel(k_full)))
    ax.plot(k[k < 0], 1.0 / 10 ** (-k[k < 0]), ":k")
    ax.plot(k[k > 0], 1.0 / 10 ** (k[k > 0]), ":k", label=r"$k^{-1}$")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Unknown k")
    ax.set_ylabel("MSE")


if __name__ == "__main__":
    print("Testing the correctness of eigenvalues")
    vn_family = VnFamily(a=(0, 1))

    a, b, delta = vn_family_sampler(n=3, a_limits=vn_family.a, b_limits=vn_family.b, delta_limits=vn_family.delta,
                                    seed=42)
    k_max = 10
    eigenvalues = get_k_eigenvalues(a, b, delta, k_max)
    print(eigenvalues)
    assert np.shape(eigenvalues) == (len(a), 2 * k_max + 1)

    a, b, delta = vn_family_sampler(n=1, a_limits=vn_family.a, b_limits=vn_family.b, delta_limits=vn_family.delta,
                                    seed=42)
    eigenvalues = get_k_eigenvalues(a, b, delta, k_max)
    print(f"a={a}, b={b}, delta={delta}")
    for k in range(1, k_max, 2):
        eigen_sin = b * np.cos(2 * np.pi * a) / (2 * np.pi * k)
        eigen_cos = b * np.sin(2 * np.pi * a) / (2 * np.pi * k)
        print(np.allclose(np.transpose([eigen_cos, eigen_sin]), eigenvalues[:, [2 * k - 1, 2 * k]]))
        # print(np.transpose([eigen_cos, eigen_sin]) - get_k_eigenvalues(a, b, delta, k))

    a, b, delta = vn_family_sampler(n=100000, a_limits=vn_family.a, b_limits=vn_family.b, delta_limits=vn_family.delta,
                                    seed=42)
    k_max = 5
    eigenvalues = get_k_eigenvalues(a, b, delta, k_max=k_max)
    print(eigenvalues.mean(axis=0) * np.append(0, np.repeat(np.arange(1, k_max + 1), 2)))
    import matplotlib.pylab as plt

    plt.hist(eigenvalues[:, 1])
    plt.show()


    def unroll(eigenvalues):
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
