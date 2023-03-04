import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

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


@perplex_plot
def plot_error(fig, ax, num_of_known_eigenvalues, error, model):
    data = pd.DataFrame.from_dict({
        "num_of_known_eigenvalues": num_of_known_eigenvalues,
        "error": list(map(lambda e: np.nanmean(e ** 2), error)),
        "model": list(map(str, model))
    }).sort_values(by="num_of_known_eigenvalues")
    data.plot(x="num_of_known_eigenvalues", y="error", ax=ax, marker=".", linestyle="dashed")


if __name__ == "__main__":
    # name="NonLinearRBA_V1V2"
    # vn_family = [
    #     VnFamily(a=Bounds(lower=0, upper=1)),
    #     VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
    # ]
    name = "NonLinearRBA"
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.5)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.25, upper=0.5)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.1, upper=0.5)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=1e-5, upper=0.5))
    ]

    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB
    )
    lab = LabPipeline()
    lab.define_new_block_of_functions("experiments", learn_eigenvalues(LinearRegression()))

    lab.execute(
        datamanager=data_manager,
        num_cores=3,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[1000],
        num_of_known_eigenvalues=[1, 2, 3, 5, 9, 21],
        k_max=[500],
        vn_family=vn_family
    )

    # import matplotlib as mpl
    # mpl.use('TkAgg')  # !IMPORTANT
    generic_plot(
        data_manager, x="k", y="mse", label="num_of_known_eigenvalues", log="xy",
        plot_func=lambda ax, *args, **kwargs: ax.plot(marker=".", *args, **kwargs),
        mse=lambda error: np.sqrt(np.mean(error ** 2, axis=0)),
        k=lambda k_max, num_of_known_eigenvalues: np.append(0, np.repeat(np.arange(1, k_max + 1), 2))[
                                                  num_of_known_eigenvalues * 2 - 1:],
        num_of_known_eigenvalues=1,
        axes_by="vn_family"
    )
