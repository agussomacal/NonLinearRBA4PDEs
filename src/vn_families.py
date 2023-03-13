from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np

Bounds = namedtuple("Bounds", "lower upper")


@dataclass(unsafe_hash=True)
class VnFamily:
    a: Tuple[float, float] = Bounds(lower=0, upper=1)
    b: Tuple[float, float] = Bounds(lower=1, upper=1)
    delta: Tuple[float, float] = Bounds(lower=0.5, upper=0.5)
    c: float = 0
    s: float = 0
    k_max: int = 0

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
    # add noise
    # eigen_sin += np.random.uniform(-c, c, size=np.shape(eigen_sin)) * k[np.newaxis, :] ** s
    # eigen_cos += np.random.uniform(-c, c, size=np.shape(eigen_cos)) * k[np.newaxis, :] ** s

    eigenvalues = b[:, np.newaxis] / (np.pi * np.repeat(k, 2)[np.newaxis, :]) * \
                  np.reshape([eigen_cos.T, eigen_sin.T], (-1, len(a)), order="F").T

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
        return np.concatenate((eigenvalues[:, 1::2][:, ::-1], [eigenvalues[:, 0]], eigenvalues[:, 2::2]), axis=1).ravel()


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
