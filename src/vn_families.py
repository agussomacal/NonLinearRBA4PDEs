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
        return (self.a.lower == self.a.upper) + \
               (self.b.lower == self.b.upper) + \
               (self.delta.lower == self.delta.upper) + \
               (self.c > 0) * (self.k_max)


def get_k_eigenvalues(a: Union[np.ndarray, float], b: Union[np.ndarray, float], delta: Union[np.ndarray, float],
                      k_max: int):
    k = np.arange(1, k_max+1)
    if np.allclose(delta, 0.5):
        eigen_sin = -2 * np.cos(2 * np.pi * a[:, np.newaxis] * np.ones((1, len(k))))
        eigen_cos = -2 * np.sin(2 * np.pi * a[:, np.newaxis] * np.ones((1, len(k))))
    else:
        # Not so precise, error in 1e-2
        eigen_cos = np.sin(2 * np.pi * k[np.newaxis, :] * (a + delta)) - np.sin(2 * np.pi * k[np.newaxis, :] * a)
        eigen_sin = np.cos(2 * np.pi * k[np.newaxis, :] * (a + delta)) - np.cos(2 * np.pi * k[np.newaxis, :] * a)
        # eigen_cos = (np.cos(2 * np.pi * k * delta)-1)*np.sin(2 * np.pi * k * a)
        # - np.cos(2 * np.pi * k * a)*np.sin(2 * np.pi * k * delta)
        # eigen_sin = (np.cos(2 * np.pi * k * delta)-1)*np.cos(2 * np.pi * k * a)
        # - np.sin(2 * np.pi * k * a)*np.sin(2 * np.pi * k * delta)
    # order F to put in the correct k order, eigen_cos, eigen_sin, eigen_cos, eigen_sin ...
    # example: np.arange(10).reshape((2, 5)).reshape((-1, 1), order="F")
    eigenvalues = -b[:, np.newaxis] / (4 * np.pi * np.repeat(k, 2)[np.newaxis, :]) * \
                  np.reshape([eigen_cos.T, eigen_sin.T], (-1, len(a)), order="F").T
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
        print(np.allclose(np.transpose([eigen_cos, eigen_sin]), eigenvalues[:, [2*k-1, 2*k]]))
        # print(np.transpose([eigen_cos, eigen_sin]) - get_k_eigenvalues(a, b, delta, k))

