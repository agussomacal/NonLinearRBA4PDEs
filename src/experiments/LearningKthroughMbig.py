import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import perplex_plot, correlation_plot
from lib.vn_families import VnFamily, Bounds, learn_eigenvalues, MWhere, get_known_unknown_indexes, get_k_eigenvalues, \
    vn_family_sampler, K_MAX, k_plot
from lib.vn_families import get_k_values

ZERO = 5e-4

if __name__ == "__main__":
    start = 11
    name = f"CRB{start}_help"
    # name = f"CRB{start}_NN_help"
    vn_family = [
        # VnFamily(a=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.6)),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.01, upper=0.99))
    ]

    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB
    )
    # data_manager.load()
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "model",
        # learn_eigenvalues(Pipeline([("Null", NullModel())])),
        learn_eigenvalues(Pipeline([("RF", RandomForestRegressor(n_estimators=10))])),
        # learn_eigenvalues(Pipeline([("NN", FNNModel(hidden_layer_sizes=(20, 20,), activation="sigmoid", epochs=100))]))
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[1000, 10000, 25000],
        mwhere=[
            MWhere(start=start, m=(start - 1) * 2),
            # MWhere(start=start, m=start // 2),
            # MWhere(start=start, m=start // 2 + 2),
            MWhere(start=start, m=(start - 1)),
            # MWhere(start=0, m=4 * start + 1),
        ],
        k_decay_help=[False, True],
        vn_family=vn_family,
        learn_higher_modes_only=[True, False],
        save_on_iteration=None,
    )

    k_plot(
        data_manager,
        plot_by=["vn_family", "model", "learn_higher_modes_only"],
        m=lambda mwhere: mwhere.m,
        axes_by=["m", "k_decay_help"],
        add_mwhere=False,
    )

    correlation_plot(data_manager, axes_var="k_decay_help", val_1=True, val_2=False,
                     value_var="mse",
                     mse=lambda error: np.sqrt(np.mean(np.array(error) ** 2, axis=0)),
                     plot_by=["vn_family", "model", "learn_higher_modes_only"],
                     m=lambda mwhere: mwhere.m,
                     log="xy",
                     axes_by=["m", "n_train"])
