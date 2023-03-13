from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, scale, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot, generic_plot
from src.vn_families import VnFamily, get_k_eigenvalues, vn_family_sampler, Bounds


class NullModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return 0


def learn_eigenvalues(model: Pipeline):
    def decorated_function(n_train, n_test, k_max, vn_family, num_of_known_eigenvalues):
        a, b, delta = vn_family_sampler(n_test + n_train, a_limits=vn_family.a, b_limits=vn_family.b,
                                        delta_limits=vn_family.delta)
        # shape(n, 1+2*k_max)
        eigenvalues = get_k_eigenvalues(a, b, delta, k_max)
        model.fit(eigenvalues[n_test:, :num_of_known_eigenvalues * 2 - 1],
                  eigenvalues[n_test:, num_of_known_eigenvalues * 2 - 1:])
        predictions = model.predict(eigenvalues[:n_test, :num_of_known_eigenvalues * 2 - 1])
        error = eigenvalues[:n_test, num_of_known_eigenvalues * 2 - 1:] - predictions
        return {
            "error": error
        }

    decorated_function.__name__ = " ".join([s[0] for s in model.steps[1:]])
    return decorated_function


def null_error(args):
    num_of_known_eigenvalues, n_train, n_test, vn_family, k_max = args
    a, b, delta = vn_family_sampler(n_test + n_train, a_limits=vn_family.a, b_limits=vn_family.b,
                                    delta_limits=vn_family.delta)
    # shape(n, 1+2*k_max)
    eigenvalues = get_k_eigenvalues(a, b, delta, k_max)
    return np.abs(eigenvalues[:n_test, num_of_known_eigenvalues * 2 - 1:]).mean(axis=0)


@perplex_plot
def k_plot(fig, ax, error, k_max, experiments):
    error, k_max, experiments = tuple(
        zip(*[(e, k, ex) for e, k, ex in zip(error, k_max, experiments) if e is not None and ex is not None]))
    zero = 1e-15
    mse = list(map(lambda e: np.concatenate((np.sqrt(np.mean(e[:, ::2] ** 2, axis=0))[::-1],
                                             np.sqrt(np.mean(e[:, 1::2] ** 2, axis=0)))), error))

    k = list(map(lambda t: np.concatenate((
        -np.log10(t[0] - np.arange((np.shape(t[1])[1] + 1) // 2)[::-1])[::-1],
        np.log10(t[0] - np.arange(np.shape(t[1])[1] // 2)[::-1]))),
                 zip(k_max, error)))

    for i, (x_i, y_i, label_i) in enumerate(zip(k, mse, experiments)):
        c = sns.color_palette("colorblind")[i]
        # m = [".", "*", ""][i]
        m = "."
        ax.plot(x_i[(y_i > zero) & (x_i < 0)], y_i[(y_i > zero) & (x_i < 0)], "--", marker=m, c=c)
        ax.plot(x_i[(y_i > zero) & (x_i > 0)], y_i[(y_i > zero) & (x_i > 0)], "--", marker=m, label=label_i, c=c)
    k = np.sort(np.unique(np.ravel(k)))
    ax.plot(k[k < 0], 1.0 / 10**(-k[k < 0]), ":k")
    ax.plot(k[k > 0], 1.0 / 10**(k[k > 0]), ":k", label=r"$k^{-1}$")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Unknown k")
    ax.set_ylabel("MSE")

if __name__ == "__main__":
    name = "NonLinearRBA"
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.5)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.01, upper=0.5))
    ]
    # name = "NonLinearRBATest"
    # vn_family = [
    #     VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.5)),
    #     # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.25, upper=0.5)),
    #     # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.1, upper=0.5)),
    #     # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=1e-5, upper=0.5))
    # ]

    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB
    )
    data_manager.load()
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "experiments",
        # learn_eigenvalues(Pipeline([("Norm", StandardScaler()), ("Null", NullModel())])),
        # learn_eigenvalues(Pipeline([("Norm", StandardScaler()), ("LR", LinearRegression())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", StandardScaler()), ("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", StandardScaler()), ("Degree 4", PolynomialFeatures(degree=4)), ("LR", LinearRegression())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", StandardScaler()), ("Tree", DecisionTreeRegressor())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", StandardScaler()), ("RF", RandomForestRegressor(n_estimators=5))])),

        # learn_eigenvalues(Pipeline(
        # [("Norm", StandardScaler()), ("FNN", MLPRegressor(hidden_layer_sizes=(20, 20,), activation="logistic",
        #                                                   learning_rate_init=1, max_iter=1000,
        #                                                   n_iter_no_change=25))])),
        # learn_eigenvalues(Pipeline([("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())]))

        # learn_eigenvalues(Pipeline([("Norm", FunctionTransformer(lambda x: x)), ("Null", NullModel())])),
        # learn_eigenvalues(Pipeline([("Norm", FunctionTransformer(lambda x: x)), ("LR", LinearRegression())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", FunctionTransformer(lambda x: x)), ("Quadratic", PolynomialFeatures(degree=2)),
        #      ("LR", LinearRegression())])),
        learn_eigenvalues(Pipeline(
            [("Norm", FunctionTransformer(lambda x: x)), ("Degree 4", PolynomialFeatures(degree=4)),
             ("LR", LinearRegression())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", FunctionTransformer(lambda x: x)), ("Tree", DecisionTreeRegressor())])),
        # learn_eigenvalues(Pipeline(
        #     [("Norm", FunctionTransformer(lambda x: x)), ("RF", RandomForestRegressor(n_estimators=10))])),
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=3,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[1000, 10000],
        # n_train=[10000],
        # num_of_known_eigenvalues=[1, 2, 3, 5],
        num_of_known_eigenvalues=[1, 2, 3],
        k_max=[500],
        vn_family=vn_family,
        save_on_iteration=None
    )

    for vn_family in set(data_manager["vn_family"]):
        k_plot(
            data_manager,
            folder=str(vn_family),
            vn_family=vn_family,
            num_of_known_eigenvalues=[2, 3, 5],
            plot_by=["vn_family", "n_train"],
            axes_by="num_of_known_eigenvalues"
        )

    # # import matplotlib as mpl
    # # mpl.use('TkAgg')  # !IMPORTANT
    # generic_plot(
    #     data_manager, x="k", y="mse", label="experiments", log="y",
    #     plot_func=lambda ax, *args, **kwargs: ax.plot(marker=".", *args, **kwargs),
    #     other_plot_funcs=lambda ax, k: ax.plot(
    #         np.append(-np.log(np.sort(np.unique(np.ravel(k))))[::-1],
    #                   np.log(np.sort(np.unique(np.ravel(k))))),
    #         np.append(-1 / np.sort(np.unique(np.ravel(k)))[::-1],
    #                   1 / np.sort(np.unique(np.ravel(k)))), ":", label=r"$k^{-1}$"),
    #     mse=lambda error: np.concatenate((np.sqrt(np.mean(error[:, ::2] ** 2, axis=0))[::-1],
    #                                       np.sqrt(np.mean(error[:, 1::2] ** 2, axis=0)))),
    #     k=lambda k_max, num_of_known_eigenvalues: np.concatenate((
    #         -np.log(np.arange(num_of_known_eigenvalues // 2 + 2, k_max + 1)[::-1]),
    #         np.log(np.arange(num_of_known_eigenvalues // 2 + 2, k_max + 1)))),
    #     # k=lambda k_max, num_of_known_eigenvalues: np.append(0, np.repeat(np.arange(1, k_max + 1), 2))[
    #     #                                           num_of_known_eigenvalues * 2 - 1:],
    #     num_of_known_eigenvalues=3,
    #     plot_by=["vn_family", "num_of_known_eigenvalues"],
    #     axes_by="n_train"
    # )

    # generic_plot(
    #     data_manager, x="num_of_known_eigenvalues", y="mse", label="experiments", log="y",
    #     # plot_func=lambda ax, xi, yi, *args, **kwargs: ax.plot(xi, yi, marker=".", linewidth=2, linestyle="dashed",
    #     #                                                       *args, **kwargs),
    #     mse=lambda error, k_max: np.mean(np.sqrt(np.mean(error ** 2, axis=0))),
    #     plot_func=partial(sns.lineplot, markers="."),
    #     # mse_k=lambda error, k_max, num_of_known_eigenvalues:
    #     # np.sqrt(np.mean((error*np.append(1, np.repeat(np.arange(1, k_max + 1), 2))
    #     # [np.newaxis, num_of_known_eigenvalues * 2 - 1:]) ** 2)),
    #     plot_by="vn_family",
    #     axes_by="n_train"
    # )
