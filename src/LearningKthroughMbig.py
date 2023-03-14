import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src import config
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot
from src.vn_families import VnFamily, Bounds, learn_eigenvalues, MWhere, K_MAX, get_known_unknown_indexes, ZERO


@perplex_plot
def k_plot(fig, ax, error, experiments, mwhere, n_train, add_mwhere=False, color_dict=None):
    error, mwhere, n_train, experiments = tuple(
        zip(*[(e, m, n, ex) for e, m, n, ex in zip(error, mwhere, n_train, experiments) if
              e is not None and ex is not None])
    )

    mse = list(map(lambda e: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).squeeze(), error))
    k_full = np.append([0], np.repeat(np.arange(1, K_MAX + 1, dtype=float), 2) * np.array([-1, 1] * K_MAX))
    k_full[k_full > 0] = np.log10(k_full[k_full > 0])
    k_full[k_full < 0] = -np.log10(-k_full[k_full < 0])

    for i, (label_i, y_i, ms) in enumerate(zip(n_train, mse, mwhere)):
        known_indexes, unknown_indexes = get_known_unknown_indexes(ms)
        change = np.where(np.diff(unknown_indexes) > 1)[0][0]
        y_i = y_i[change:]
        k = k_full[unknown_indexes][change:]
        if color_dict is None:
            c = sns.color_palette("colorblind")[i]
        else:
            c = color_dict[label_i]
        # m = [".", "*", ""][i]
        m = "o"
        ax.plot(k[(y_i > ZERO) & (k < 0)], y_i[(y_i > ZERO) & (k < 0)], "--", marker=m, c=c)
        ax.plot(k[(y_i > ZERO) & (k > 0)], y_i[(y_i > ZERO) & (k > 0)], "--", marker=m,
                label=str(label_i) + (f": start={ms.start}, m={ms.m}" if add_mwhere else ""), c=c)
    k = np.sort(np.unique(np.ravel(k_full)))
    ax.plot(k[k < 0], 1.0 / 10 ** (-k[k < 0]), ":k")
    ax.plot(k[k > 0], 1.0 / 10 ** (k[k > 0]), ":k", label=r"$k^{-1}$")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend(loc='upper right')
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha_k$" + "\t\t\t\t   " + r"$\beta_k$")
    ax.set_ylabel("MSE")


if __name__ == "__main__":
    start = 11
    name = f"CRB{start}"
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
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=True,
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
        vn_family=vn_family,
        save_on_iteration=None
    )

    k_plot(
        data_manager,
        plot_by=["vn_family", "experiments"],
        m=lambda mwhere: mwhere.m,
        axes_by="m",
        add_mwhere=False,
    )
