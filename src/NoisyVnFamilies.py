import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src import config
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot
from src.vn_families import VnFamily, get_k_eigenvalues, vn_family_sampler, Bounds, ZERO, K_MAX, \
    get_known_unknown_indexes, MWhere, learn_eigenvalues, k_plot

if __name__ == "__main__":
    name = "NonLinearRBA_noisy_family"
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.01, s=-4),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.01, s=-3),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.01, s=-2),

        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.1, s=-4),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.1, s=-3),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.1, s=-2),

        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=1, s=-4),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=1, s=-3),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=1, s=-2),
    ]

    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB
    )
    data_manager.load()
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "experiments",
        learn_eigenvalues(Pipeline([("RF", RandomForestRegressor(n_estimators=10))])),
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=3,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[10000],
        mwhere=[
            MWhere(start=0, m=5),
            MWhere(start=50, m=10),
            MWhere(start=200, m=10),
        ],
        vn_family=vn_family,
        save_on_iteration=None
    )

    k_plot(
        data_manager,
        c=lambda vn_family: vn_family.c,
        s=lambda vn_family: vn_family.s,
        plot_by=["c", "n_train"],
        axes_by="s",
        add_mwhere=True
    )
