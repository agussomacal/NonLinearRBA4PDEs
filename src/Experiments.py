import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, scale, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot, generic_plot
from src.vn_families import VnFamily, get_k_eigenvalues, vn_family_sampler, Bounds


def learn_eigenvalues(model):
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

    decorated_function.__name__ = str(model)
    return decorated_function


if __name__ == "__main__":
    name = "NonLinearRBA_V1V2_Test"
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
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
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "experiments",
        # learn_eigenvalues(DecisionTreeRegressor()),
        # learn_eigenvalues(MLPRegressor(hidden_layer_sizes=(20, 20, ), activation="logistic",
        #                                learning_rate_init=1, max_iter=1000, n_iter_no_change=25)),
        # learn_eigenvalues(MLPRegressor(hidden_layer_sizes=(20, 20, ), activation="relu",
        #                                learning_rate_init=1, max_iter=1000, n_iter_no_change=25)),
        learn_eigenvalues(LinearRegression()),
        learn_eigenvalues(Pipeline([("Norm", StandardScaler()), ("LR", LinearRegression())])),
        learn_eigenvalues(Pipeline(
            [("Norm", StandardScaler()), ("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())])),
        learn_eigenvalues(Pipeline(
            [("Norm", StandardScaler()), ("Tree", DecisionTreeRegressor())])),
        learn_eigenvalues(Pipeline(
            [("Norm", StandardScaler()), ("FNN", MLPRegressor(hidden_layer_sizes=(20, 20,), activation="logistic",
                                                              learning_rate_init=1, max_iter=1000,
                                                              n_iter_no_change=25))])),
        # learn_eigenvalues(Pipeline([("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())]))
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=1,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[1000],
        # num_of_known_eigenvalues=[1, 2, 3, 5, 9, 21],
        num_of_known_eigenvalues=[1, 3, 5],
        k_max=[500],
        vn_family=vn_family,
        save_on_iteration=5
    )

    # import matplotlib as mpl
    # mpl.use('TkAgg')  # !IMPORTANT
    generic_plot(
        data_manager, x="k", y="mse", label="num_of_known_eigenvalues", log="xy",
        plot_func=lambda ax, *args, **kwargs: ax.plot(marker=".", *args, **kwargs),
        mse=lambda error: np.sqrt(np.mean(error ** 2, axis=0)),
        k=lambda k_max, num_of_known_eigenvalues: np.append(0, np.repeat(np.arange(1, k_max + 1), 2))[
                                                  num_of_known_eigenvalues * 2 - 1:],
        # num_of_known_eigenvalues=1,
        expeiments=str(LinearRegression()),
        plot_by="vn_family",
        axes_by="n_train"
    )

    generic_plot(
        data_manager, x="num_of_known_eigenvalues", y="mse", label="experiments", log="xy",
        # plot_func=lambda ax, *args, **kwargs: ax.plot(marker=".", *args, **kwargs),
        mse=lambda error: np.sqrt(np.mean(error ** 2)),
        plot_by="vn_family",
        axes_by="n_train"
    )
