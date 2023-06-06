import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from PerplexityLab.visualization import perplex_plot, one_line_iterator
from config import paper_path
from src import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from lib.vn_families import VnFamily, Bounds, MWhere, learn_eigenvalues, get_k_eigenvalues, vn_family_sampler, K_MAX, \
    get_known_unknown_indexes
from experiments.LearningKthroughMbig import k_plot


@perplex_plot(plot_by_default=["vn_family"], axes_by_default=["mwhere", "learn_higher_modes_only"])
@one_line_iterator
def plot_reconstruction(fig, ax, error, n_train, n_test, vn_family, model, mwhere: MWhere, k_decay_help,
                        learn_higher_modes_only, i=0, num_points=K_MAX * 10):
    a, b, delta = vn_family_sampler(n_test + n_train, a_limits=vn_family.a, b_limits=vn_family.b,
                                    delta_limits=vn_family.delta)
    eigenvalues, noise = get_k_eigenvalues(a, b, delta, K_MAX, vn_family.c, vn_family.s)
    known_indexes, unknown_indexes = get_known_unknown_indexes(mwhere, learn_higher_modes_only)

    # k*x creates a matrix #k x #x but at the end we obtain #x x #k
    k = np.arange(1, K_MAX + 1)[:, np.newaxis]
    x = np.linspace(0, 1, num_points)[np.newaxis, :]
    eigenvectors = np.reshape([np.cos(2 * np.pi * k * x), np.sin(2 * np.pi * k * x)], (-1, num_points), order="F").T
    eigenvectors = np.hstack((np.ones((num_points, 1)), eigenvectors))

    x = np.ravel(x)
    ax.plot(x, eigenvectors @ eigenvalues[:n_test][i, :], label=f"Ground truth")
    ax.plot(x, eigenvectors @ (eigenvalues[:n_test][i, :] + noise[i, :]), label=f"Ground truth + regular")
    eigenvalues[:n_test][i, unknown_indexes] -= error[i, :]
    eigenvalues[:n_test][i, known_indexes] += noise[i, known_indexes]
    reconstruction = eigenvectors @ eigenvalues[:n_test][i, :]
    ax.plot(x, reconstruction, label=f"{model}: reconstruction of ground truth: ")


if __name__ == "__main__":
    # name = "NonLinearCRB_plusRegular"
    name = "NonLinearCRB_plusRegular_test"
    vn_family = [
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.01, s=-4),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.01, s=-3),
        VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
                 delta=Bounds(lower=0.4, upper=0.6), c=0.01, s=-2),

        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
        #          delta=Bounds(lower=0.4, upper=0.6), c=0.1, s=-4),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
        #          delta=Bounds(lower=0.4, upper=0.6), c=0.1, s=-3),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
        #          delta=Bounds(lower=0.4, upper=0.6), c=0.1, s=-2),
        #
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
        #          delta=Bounds(lower=0.4, upper=0.6), c=1, s=-4),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
        #          delta=Bounds(lower=0.4, upper=0.6), c=1, s=-3),
        # VnFamily(a=Bounds(lower=0, upper=1), b=Bounds(lower=0, upper=1),
        #          delta=Bounds(lower=0.4, upper=0.6), c=1, s=-2),
    ]

    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB,
        country_alpha_code="FR",
        trackCO2=True
    )
    data_manager.load()
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "model",
        # learn_eigenvalues(Pipeline([("Zscore", StandardScaler()), ("RF", RandomForestRegressor(n_estimators=10))])),
        learn_eigenvalues(Pipeline([("RF", RandomForestRegressor(n_estimators=10))])),
    )

    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        n_test=[1000],
        n_train=[10000],
        mwhere=[
            # MWhere(start=0, m=5),
            # MWhere(start=50, m=5),
            MWhere(start=200, m=5),
        ],
        vn_family=vn_family,
        k_decay_help=[False],
        learn_higher_modes_only=[False, True],
    )

    k_plot(
        data_manager,
        # folder=paper_path,
        c=lambda vn_family: vn_family.c,
        s=lambda vn_family: vn_family.s,
        plot_by=["c", "n_train"],
        axes_by=["s", "model"],
        add_mwhere=True
    )

    plot_reconstruction(data_manager)
