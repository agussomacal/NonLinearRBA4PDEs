import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import perplex_plot, correlation_plot
from lib.vn_families import VnFamily, Bounds, learn_eigenvalues, MWhere, get_known_unknown_indexes, get_k_eigenvalues, \
    vn_family_sampler, K_MAX
from lib.vn_families import get_k_values

ZERO = 5e-4


@perplex_plot
def k_plot(fig, ax, error, experiments, mwhere, n_train, learn_higher_modes_only, vn_family, add_mwhere=False,
           color_dict=None):
    error, mwhere, n_train, experiments, learn_higher_modes_only, vn_family = tuple(
        zip(*[(e, m, n, ex, l, vn) for e, m, n, ex, l, vn in
              zip(error, mwhere, n_train, experiments, learn_higher_modes_only,
                  vn_family)
              if
              e is not None and ex is not None])
    )

    mse = list(map(lambda e: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).squeeze(), error))
    k_full = get_k_values(negative=True)
    k_full[k_full > 0] = np.log10(k_full[k_full > 0])
    k_full[k_full < 0] = -np.log10(-k_full[k_full < 0])

    for i, (label_i, y_i, ms, hmonly) in enumerate(zip(n_train, mse, mwhere, learn_higher_modes_only)):
        known_indexes, unknown_indexes = get_known_unknown_indexes(ms, hmonly)
        # change = np.where(np.diff(unknown_indexes) > 1)[0][0]
        # y_i = y_i[change:]
        k = k_full[unknown_indexes]  # [change:]
        if color_dict is None:
            c = sns.color_palette("colorblind")[i]
        else:
            c = color_dict[label_i]
        # m = [".", "*", ""][i]
        m = "o"
        ax.plot(k[(y_i > ZERO) & (k < 0)], y_i[(y_i > ZERO) & (k < 0)], "--", marker=m, c=c)
        ax.plot(k[(y_i > ZERO) & (k > 0)], y_i[(y_i > ZERO) & (k > 0)], "--", marker=m,
                label=str(label_i) + (f": start={ms.start}, m={ms.m}" if add_mwhere else ""), c=c)

    vn_family = vn_family[0]
    a, b, delta = vn_family_sampler(n=1000, a_limits=vn_family.a, b_limits=vn_family.b,
                                    delta_limits=vn_family.delta)
    eigenvalues, _ = get_k_eigenvalues(a, b, delta, K_MAX, vn_family.c, vn_family.s)
    eigenvalues_sd = eigenvalues.std(axis=0)
    ax.plot(k_full[(k_full < 0) & (eigenvalues_sd > ZERO)], eigenvalues_sd[(k_full < 0) & (eigenvalues_sd > ZERO)],
            ":k")
    ax.plot(k_full[(k_full > 0) & (eigenvalues_sd > ZERO)], eigenvalues_sd[(k_full > 0) & (eigenvalues_sd > ZERO)],
            ":k", label=r"$null model$")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend(loc='upper right')
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha_k$" + "\t\t\t\t   " + r"$\beta_k$")
    ax.set_ylabel("MSE")


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
        "experiments",
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
        plot_by=["vn_family", "experiments", "learn_higher_modes_only"],
        m=lambda mwhere: mwhere.m,
        axes_by=["m", "k_decay_help"],
        add_mwhere=False,
    )

    correlation_plot(data_manager, axes_var="k_decay_help", val_1=True, val_2=False,
                     value_var="mse",
                     mse=lambda error: np.sqrt(np.mean(np.array(error) ** 2, axis=0)),
                     plot_by=["vn_family", "experiments", "learn_higher_modes_only"],
                     m=lambda mwhere: mwhere.m,
                     log="xy",
                     axes_by=["m", "n_train"])
