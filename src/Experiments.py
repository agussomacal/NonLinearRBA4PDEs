import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot
from src.vn_families import VnFamily, get_k_eigenvalues, vn_family_sampler, Bounds


def learn_eigenvalues(n_train, n_test, k_max, vn_family, model, num_of_known_eigenvalues):
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


@perplex_plot
def plot_error(fig, ax, num_of_known_eigenvalues, error, model):
    data = pd.DataFrame.from_dict({
        "num_of_known_eigenvalues": num_of_known_eigenvalues,
        "error": list(map(lambda e: np.nanmean(e**2), error)),
        "model": list(map(str, model))
    }).sort_values(by="num_of_known_eigenvalues")
    data.plot(x="num_of_known_eigenvalues", y="error", ax=ax, marker=".", linestyle="dashed")


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name="NonLinearRBA",
        format=JOBLIB
    )
    lab = LabPipeline()
    lab.define_new_block_of_functions("experiments", learn_eigenvalues)

    lab.execute(
        datamanager=data_manager,
        num_cores=1,
        forget=True,
        recalculate=False,
        n_test=[100],
        n_train=[100, 1000],
        num_of_known_eigenvalues=[3, 5, 10],
        k_max=[100],
        vn_family=[VnFamily(a=Bounds(lower=0, upper=1))],
        model=[LinearRegression()]
    )

    # import matplotlib as mpl
    # mpl.use('TkAgg')  # !IMPORTANT
    plot_error(data_manager, axes_by=["n_train"])
