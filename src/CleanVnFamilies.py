import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.vn_families import VnFamily, Bounds, learn_eigenvalues, k_plot, MWhere


class NullModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return 0


if __name__ == "__main__":
    name = "NonLinearCRB"
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.6)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.01, upper=0.99))
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
        learn_eigenvalues(Pipeline([("Null", NullModel())])),
        learn_eigenvalues(Pipeline([("LR", LinearRegression())])),
        learn_eigenvalues(Pipeline([("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())])),
        learn_eigenvalues(Pipeline([("Degree 4", PolynomialFeatures(degree=4)), ("LR", LinearRegression())])),
        learn_eigenvalues(Pipeline([("Tree", DecisionTreeRegressor())])),
        learn_eigenvalues(Pipeline([("RF", RandomForestRegressor(n_estimators=10))])),
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[1000, 10000],
        mwhere=[
            MWhere(start=0, m=2),
            MWhere(start=0, m=3),
            MWhere(start=0, m=5),
        ],
        vn_family=vn_family,
        save_on_iteration=None
    )
    palette = sns.color_palette("colorblind")
    k_plot(
        data_manager,
        vn_family=vn_family,
        plot_by=["vn_family", "n_train"],
        m=lambda mwhere: mwhere.m,
        mwhere=[MWhere(start=0, m=3), MWhere(start=0, m=5)],
        axes_by="m",
        add_mwhere=False,
        color_dict={"RF": palette[0], "Tree": palette[2], "LR": palette[4], "Null": palette[3],
                    "Quadratic LR": palette[1], "Degree 4 LR": palette[6]},
    )
