import matplotlib.pylab as plt
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from config import paper_path
from lib.vn_families import VnFamily, Bounds, learn_eigenvalues, MWhere, k_plot
from src import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline


class NullModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return 0


if __name__ == "__main__":
    name = f"NonLinearCRB"
    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB,
        country_alpha_code="FR",
        trackCO2=True
    )

    # ----------- Experiments on families with first eigenvalues ----------- #
    # Parameters for experiment
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.6)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.01, upper=0.99))
    ]
    mwhere = [MWhere(start=1, m=2), MWhere(start=0, m=3), MWhere(start=0, m=5)]
    models = [
        Pipeline([("Null", NullModel())]),
        Pipeline([("LR", LinearRegression())]),
        Pipeline([("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())]),
        Pipeline([("Degree 4", PolynomialFeatures(degree=4)), ("LR", LinearRegression())]),
        Pipeline([("Tree", DecisionTreeRegressor())]),
        Pipeline([("RF", RandomForestRegressor(n_estimators=10))]),
        # Pipeline([("NN", FNNModel(hidden_layer_sizes=(20, 20,), activation="sigmoid"))]))
    ]

    # Experiment
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "experiments",
        *list(map(learn_eigenvalues, models)),
    )
    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        n_test=[1000],
        n_train=[1000, 10000],
        mwhere=mwhere,
        vn_family=vn_family,
        k_decay_help=[False],
        learn_higher_modes_only=[True],
    )

    # Plots
    palette = sns.color_palette("colorblind")
    k_plot(
        data_manager,
        folder=paper_path,
        vn_family=vn_family,
        plot_by=["vn_family", "n_train", "m"],
        m=lambda mwhere: mwhere.m,
        mwhere=mwhere,
        axes_by="m",
        add_mwhere=False,
        color_dict={"RF": palette[0], "Tree": palette[2], "LR": palette[4], "Null": palette[5],
                    "Quadratic LR": palette[1], "Degree 4 LR": palette[3]},
    )

    # ----------- Experiments on family with higher eigenvalues ----------- #
    # Parameters for experiment
    vn_family = [VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1))]
    mwhere = [MWhere(start=19, m=10), MWhere(start=19, m=20)]
    models = [Pipeline([("RF", RandomForestRegressor(n_estimators=10))])]

    # Experiment
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "experiments",
        *list(map(learn_eigenvalues, models)),
    )
    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        n_test=[1000],
        n_train=[1000, 10000, 25000],
        mwhere=mwhere,
        vn_family=vn_family,
        k_decay_help=[False],
        learn_higher_modes_only=[True],
    )

    # Plot
    palette = sns.color_palette("colorblind")
    k_plot(
        data_manager,
        folder=paper_path,
        vn_family=vn_family,
        plot_by=["vn_family", "d"],
        d=lambda mwhere: int(mwhere.m / 2 - 1),
        mwhere=mwhere,
        label_var="n_train",
        axes_by="d",
        add_mwhere=False,
        color_dict={"RF": palette[0], "Tree": palette[2], "LR": palette[4], "Null": palette[5],
                    "Quadratic LR": palette[1], "Degree 4 LR": palette[3]},
    )

    print(f"CO2 emissions: {data_manager.CO2kg:.4f}kg")
    print(f"Power consumption: {data_manager.electricity_consumption_kWh:.4f}kWh")
