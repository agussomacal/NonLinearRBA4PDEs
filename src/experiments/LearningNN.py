import numpy as np
import torch
from sklearn.pipeline import Pipeline

from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from experiments.LearningKthroughMbig import k_plot
from lib.skkeras import SKKerasFNN
from lib.sktorch import FNNModel
from lib.vn_families import VnFamily, Bounds, learn_eigenvalues, MWhere
from src import config


# create a function (this my favorite choice)
def RelativeMSELoss(yhat, y):
    return torch.mean((yhat - y) ** 2) / torch.mean((y) ** 2)


if __name__ == "__main__":
    name = f"NNlearning"
    vn_family = [
        # VnFamily(a=Bounds(lower=0, upper=1)),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1)),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1), delta=Bounds(lower=0.4, upper=0.6)),
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
        # learn_eigenvalues(Pipeline([("RF", RandomForestRegressor(n_estimators=10))])),
        # learn_eigenvalues(Pipeline([("Zscore", StandardScaler()), ("RF", RandomForestRegressor(n_estimators=10))])),
        learn_eigenvalues(Pipeline([
            # ("Zscore", StandardScaler()),
            ("NN",
             FNNModel(hidden_layer_sizes=(100, 20,), activation="relu", epochs=1000,  # loss_func=RelativeMSELoss,
                      solver=torch.optim.Adam, lr=1))])),
        # learn_eigenvalues(Pipeline([
        #     # ("Zscore", StandardScaler()),
        #     ("NN",
        #      SKKerasFNN(hidden_layer_sizes=(100, 20,), activation="relu", epochs=1000, restarts=1,
        #                 max_time4fitting=np.Inf, validation_size=0.2, batch_size=None, criterion='mse', solver='adam',
        #                 lr=None,
        #                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=50, random_state=42,
        #                 workers=1))]))

    )

    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        n_test=[1000],
        n_train=[20000],  # 25000
        mwhere=[
            # MWhere(start=0, m=5),
            MWhere(start=0, m=11),
        ],
        k_decay_help=[False],
        vn_family=vn_family,
        learn_higher_modes_only=[True],
        save_on_iteration=1,
        quantity=[1, 2, 4, 8, 16]
    )

    k_plot(
        data_manager,
        plot_by=["vn_family", "model", "learn_higher_modes_only"],
        m=lambda mwhere: mwhere.m,
        axes_by=["m", "k_decay_help"],
        add_mwhere=False,
    )

    # correlation_plot(data_manager, axes_var="k_decay_help", val_1=True, val_2=False,
    #                  value_var="mse",
    #                  mse=lambda error: np.sqrt(np.mean(np.array(error) ** 2, axis=0)),
    #                  plot_by=["vn_family", "experiments", "learn_higher_modes_only"],
    #                  m=lambda mwhere: mwhere.m,
    #                  log="xy",
    #                  axes_by=["m", "n_train"])
