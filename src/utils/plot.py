import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr, ttest_ind
from scipy.optimize import minimize_scalar
from time import time
from tqdm import tqdm

COLORS = ["#0077BB", "#EE7733", "#009988"]


def plot_parameter_recovery(
    y_true,
    y_pred,
    y_fit,
    r_squared,
    fname,
    param_label=None,
    fpath=None,
):
    """
    Plot the comparison between true and inferred parameter values.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    fig, ax = plt.subplots()
    sns.set_style("white")
    n_user = y_true.shape[0]
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    dist_val = (max_val - min_val) * 0.1

    label = f"$({y_fit[0]:.2f})x + ({y_fit[1]:.2f}), R^2={r_squared:.2f}$"
    sns.regplot(
        x=y_true,
        y=y_pred,
        scatter_kws={"color": "black", "alpha": 0.3},
        line_kws={"color": "red", "lw": 2.5},
    )
    custom_lines = [Line2D([0], [0], color="red", lw=2.5)]
    plt.plot(
        [min_val - dist_val, max_val + dist_val],
        [min_val - dist_val, max_val + dist_val],
        color="gray",
        linestyle="--"
    )
    plt.xlabel(f"True {param_label}")
    plt.ylabel(f"Inferred {param_label}")
    plt.xlim([min_val - dist_val, max_val + dist_val])
    plt.ylim([min_val - dist_val, max_val + dist_val])
    plt.legend(custom_lines, [label,], fontsize=9, loc="lower right")
    plt.grid(linestyle="--", linewidth=0.5)

    ax.set_aspect("equal")
    plt.tight_layout()
    fig_path = fpath if fpath is not None else "./"
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        os.path.join(fig_path, fname + ".pdf"),
        dpi=300,
    )
    plt.show()
    plt.close(fig)


def pair_plot(
    sampled_params,
    param_labels,
    limits=None,
    gt_params=None,
    fname="sample",
    fpath=None,
):
    """
    Plot the pair plot of the sampled parameters.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.linewidth"] = 2
    n_param = len(param_labels)
    ret = list()
    for i in range(n_param):
        kde = gaussian_kde(sampled_params[:, i])
        opt_x = minimize_scalar(lambda x: -kde(x), method="golden").x
        if hasattr(opt_x, "__len__"):
            ret.append(opt_x[0])
        else:
            ret.append(opt_x)
    mode_v = np.array(ret)
    mean_v = np.mean(sampled_params, axis=0)

    df = pd.DataFrame(dict(zip(param_labels, sampled_params.transpose())))
    g = sns.pairplot(
        df,
        kind="kde",
        diag_kind="kde", #"hist"
        plot_kws=dict(levels=5, fill=True, color="#0077BB"),
        diag_kws=dict(color="#0077BB"), #bins=50, kde=True),
        corner=True
    )
    for i in range(n_param):
        custom_lines = [Line2D([0], [0], color="red", lw=3),]
        custom_labels = [f"{mode_v[i]:.3f}",]
        custom_anot_labels = ["Our MAP fit",]
        g.axes[i][i].axvline(x=mode_v[i], color="red", lw=3)
        if gt_params is not None:
            g.axes[i][i].axvline(x=gt_params[i], color="#EE7733", lw=3)
            custom_lines.append(Line2D([0], [0], color="#EE7733", lw=3))
            custom_labels.append(f"{gt_params[i]:.3f}")
            custom_anot_labels.append("Baseline")
        if limits is not None:
            g.axes[i][i].set_xlim([limits[i][0], limits[i][1]])
        g.axes[i][i].legend(custom_lines, custom_labels, handlelength=0.6, handletextpad=0.4)
        
        for j in range(i):
            g.axes[i][j].scatter(x=mode_v[j], y=mode_v[i], color="red", s=50.0)
            if gt_params is not None:
                g.axes[i][j].scatter(x=gt_params[j], y=gt_params[i], color="#EE7733", s=50.0)
            if limits is not None:
                g.axes[i][j].set_xlim([limits[j][0], limits[j][1]])
                g.axes[i][j].set_ylim([limits[i][0], limits[i][1]])
    
    g.axes[1][0].legend(custom_lines, custom_anot_labels, handlelength=0.6, handletextpad=0.4)

    plt.tight_layout()
    fig_path = fpath if fpath is not None else "./"
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        os.path.join(fig_path, fname + ".pdf"),
        dpi=300,
    )
    plt.show()
    plt.close(g.figure)


def inference_time_by_trial_size(
    trainer,
    n_trial_arr,
    n_user=100,
):
    assert trainer.task_name in ["typing", "pnc"]
    trainer.amortizer.eval()
    fig_path = f"{trainer.result_path}/{trainer.name}/iter{trainer.iter:03d}/"
    n_sample = 1000

    def _infer(n_trial):
        assert n_trial > 1
        infer_time_rep = list()
        _, valid_data = trainer.valid_dataset.sample(n_trial)

        for user_i in range(n_user):
            start_t = time()
            if trainer.task_name == "pnc":
                stat_i, traj_i = valid_data[user_i]
                _ = trainer.amortizer.infer(
                    stat_i,
                    traj_i,
                    n_sample=n_sample,
                    type="mode"
                )
            else:
                stat_i = valid_data[user_i]
                _ = trainer.amortizer.infer(
                    stat_i,
                    n_sample=n_sample,
                    type="mode"
                )
            infer_time = time() - start_t
            infer_time_rep.append(infer_time * 1000.0) # ms

        return infer_time_rep
    
    # across different trial sizes
    mean_time_across_size, std_time_across_size = list(), list()
    for n_trial in tqdm(n_trial_arr):
        infer_time_rep = _infer(n_trial)
        mean_time_across_size.append(np.mean(infer_time_rep))
        std_time_across_size.append(np.std(infer_time_rep))

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2

    x_arr = n_trial_arr
    fig = plt.figure()
    plt.plot(x_arr, mean_time_across_size, marker="o", color=COLORS[0])
    plt.fill_between(
        x_arr,
        np.array(mean_time_across_size) - np.array(std_time_across_size),
        np.array(mean_time_across_size) + np.array(std_time_across_size),
        alpha=0.3,
        facecolor=COLORS[0]
    )
    plt.xlabel("No. of observed trials")
    plt.ylabel("Inference time [ms]")
    plt.title("Inference time per dataset")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"infer_time_across_size.pdf"), dpi=300)
    plt.show()
    plt.close(fig)


def parameter_recovery_by_trial_size(
    trainer,
    n_trial_arr,
    repitition=100,
):
    assert trainer.task_name in ["typing", "pnc"]
    trainer.amortizer.eval()
    fig_path = f"{trainer.result_path}/{trainer.name}/iter{trainer.iter:03d}/"
    n_sample = 1000
    p_list = trainer.targeted_params

    def _r2(n_trial):
        assert n_trial > 1
        r2_dict = dict(zip(p_list, [list() for _ in range(len(p_list))]))
        for _ in range(repitition):
            gt_params, valid_data = trainer.valid_dataset.sample(n_trial)
            res = dict()
            trainer.parameter_recovery(
                res,
                gt_params,
                valid_data,
                n_sample=n_sample,
                infer_type="mode",
                plot=False,
            )
            for p_label in p_list:
                r2_dict[p_label].append(res["Parameter_Recovery/r2_" + p_label])
        return r2_dict
    
    # across different trial sizes
    r2_obs_mean_dict = dict(zip(p_list, [list() for _ in range(len(p_list))]))
    r2_obs_std_dict = dict(zip(p_list, [list() for _ in range(len(p_list))]))
    for n_trial in tqdm(n_trial_arr):
        r2_obs = _r2(n_trial)
        for p_label in p_list:
            r2_obs_mean_dict[p_label].append(np.mean(r2_obs[p_label]))
            r2_obs_std_dict[p_label].append(np.std(r2_obs[p_label]))
            
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2

    x_arr = n_trial_arr
    fig = plt.figure()
    for p_i, p_label in enumerate(p_list):
        plt.plot(
            x_arr,
            r2_obs_mean_dict[p_label],
            marker="o",
            color=COLORS[p_i],
            label=trainer.param_symbol[p_i]
        )
        plt.fill_between(
            x_arr,
            np.array(r2_obs_mean_dict[p_label]) - np.array(r2_obs_std_dict[p_label]),
            np.array(r2_obs_mean_dict[p_label]) + np.array(r2_obs_std_dict[p_label]),
            alpha=0.3,
            facecolor=COLORS[p_i]
        )
    plt.xlabel("No. of observed trials")
    plt.ylabel("$R^2$")
    plt.title("Parameter recovery per dataset")
    plt.ylim([0., 1.])
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"parameter_recovery_across_size.pdf"), dpi=300)
    plt.show()
    plt.close(fig)


def significance_by_dataset_size(
    trainer,
    n_user_arr,
    repitition=100
):
    assert trainer.task_name in ["typing"]
    trainer.amortizer.eval()
    fig_path = f"{trainer.result_path}/{trainer.name}/iter{trainer.iter:03d}/"
    n_sample = 10000

    user_data, user_info  = trainer.user_dataset.indiv_sample() 
    cols = ["p_id", "age", "gender"] + trainer.targeted_params
    indiv_user_df = pd.DataFrame(columns=cols).astype(
        dict(zip(cols, [int, int, str, float, float, float]))
    )
    for user_i in tqdm(range(len(user_data))):
        inferred_params = trainer.amortizer.infer(
            user_data[user_i],
            n_sample=n_sample,
            type="mode",
            return_samples=False
        )
        inferred_params = trainer._clip_params(inferred_params)
        indiv_user_df = indiv_user_df.append(dict(zip(
            cols,
            [
                user_info[user_i]["id"],
                user_info[user_i]["age"],
                user_info[user_i]["gender"],
                inferred_params[0],
                inferred_params[1],
                inferred_params[2],
            ]
        )), ignore_index=True)
    
    max_n_user = indiv_user_df.shape[0]
    if max_n_user > n_user_arr[-1]:
        n_user_arr = np.append(n_user_arr, max_n_user)

    p_mean_dict, p_std_dict = dict(), dict()
    for i, n_user in enumerate(n_user_arr):
        p_values = dict()
        for rep in range(repitition):
            flag = True
            while flag:
                indiv_user_df_n = indiv_user_df.sample(n_user)
                flag = (sum(indiv_user_df_n["gender"] == "male") < 3 \
                    or sum(indiv_user_df_n["gender"] == "female") < 3)

            for x_col in ["age", "gender"]:
                for y_col in trainer.targeted_params:

                    if x_col == "gender":
                        t, p = ttest_ind(
                            indiv_user_df_n.groupby("gender").get_group("male")[y_col].to_numpy(),
                            indiv_user_df_n.groupby("gender").get_group("female")[y_col].to_numpy(),
                        )
                    else:
                        r, p = pearsonr(
                            indiv_user_df_n[x_col].to_numpy(), 
                            indiv_user_df_n[y_col].to_numpy()
                        )
                    if rep == 0:
                        p_values[f"{x_col}_{y_col}"] = list()
                    p_values[f"{x_col}_{y_col}"].append(p)
            
        for x_col in ["age", "gender"]:
            for y_col in trainer.targeted_params:
                if i == 0:
                    p_mean_dict[f"{x_col}_{y_col}"] = list()
                    p_std_dict[f"{x_col}_{y_col}"] = list()
                p_mean_dict[f"{x_col}_{y_col}"].append(np.nanmean(p_values[f"{x_col}_{y_col}"]))
                p_std_dict[f"{x_col}_{y_col}"].append(np.nanstd(p_values[f"{x_col}_{y_col}"]))

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2

    for x_col in ["age", "gender"]:
        for y_label, y_col in zip(trainer.param_symbol, trainer.targeted_params):
            fig = plt.figure()
            if x_col == "age":
                title = f"Correlation signifi. (age-{y_label})"
            else:
                title = f"$t$-test signif. (gender-{y_label})"
            plt.plot(n_user_arr, p_mean_dict[f"{x_col}_{y_col}"], marker="o", color=COLORS[0])
            plt.fill_between(
                n_user_arr,
                np.array(p_mean_dict[f"{x_col}_{y_col}"]) - np.array(p_std_dict[f"{x_col}_{y_col}"]),
                np.array(p_mean_dict[f"{x_col}_{y_col}"]) + np.array(p_std_dict[f"{x_col}_{y_col}"]),
                alpha=0.3,
                facecolor=COLORS[0]
            )
            plt.axhline(y=0.05, color="r", linestyle="--", label="$p$=.05")
            plt.legend()
            plt.xlabel("No. of participants")
            plt.ylabel("$p$-value")
            plt.xscale("log")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_path, f"sample_size_{x_col}_{y_col}.pdf"), dpi=300)
            plt.show()
            plt.close(fig)