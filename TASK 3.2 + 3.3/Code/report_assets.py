"""Sinh đúng bộ hình tuyển chọn đang được nhúng trong báo cáo LaTeX.

Các mẫu 5.000/1.000 quan sát chỉ phục vụ trực quan hóa. Mọi bảng và kiểm định
trong :mod:`anova_utils` vẫn dùng toàn bộ tập KDDTrain+ sạch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


GROUP_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]
FEATURES = ["count", "serror_rate", "src_bytes", "duration"]
PALETTE = {
    "Normal": "#4477AA", "DoS": "#CC6677", "Probe": "#DDCC77",
    "R2L": "#117733", "U2R": "#AA4499",
}
RANDOM_STATE = 18015
ECDF_SAMPLE_PER_GROUP = 5000
QQ_SAMPLE_PER_GROUP = 1000
REPORT_FIGURE_STEMS = [
    "wilson_95ci_five_groups", "anova_ecdf_four_features",
    "games_howell_heatmaps", "qq_log_features", "duration_zero_wilson",
]


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def generate_report_figures(
    data: pd.DataFrame, posthoc: pd.DataFrame, zero_duration: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Tạo 5 hình báo cáo ở cả PNG 300 DPI và PDF vector."""

    output_dir = Path(output_dir)
    sns.set_theme(style="whitegrid", context="notebook")
    rng = np.random.default_rng(RANDOM_STATE)
    generated: list[Path] = []

    counts = data["attack_category"].value_counts(sort=False).reindex(GROUP_ORDER)
    low, high = proportion_confint(counts.to_numpy(), len(data), alpha=0.05, method="wilson")
    proportions = counts.to_numpy() / len(data)
    y = np.arange(len(GROUP_ORDER))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.errorbar(proportions, y, xerr=np.vstack([proportions-low, high-proportions]),
                fmt="none", color="black", capsize=5, lw=1.4)
    ax.scatter(proportions, y, s=90, c=[PALETTE[g] for g in GROUP_ORDER],
               edgecolor="black", zorder=3)
    for index, value in enumerate(proportions):
        ax.annotate(f"{100*value:.3f}%  (n={counts.iloc[index]:,})", (value, index),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_yticks(y, GROUP_ORDER)
    ax.invert_yaxis()
    ax.set(xlabel="Tỷ lệ trong KDDTrain+ (trục logarit)", ylabel="Nhóm kết nối",
           title="Ước lượng tỷ lệ và khoảng tin cậy Wilson 95%")
    fig.tight_layout()
    generated.extend(_save(fig, output_dir, "wilson_95ci_five_groups"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    specs = [("count", "count (kết nối)", False), ("serror_rate", "serror_rate (0--1)", False),
             ("src_bytes", "ln(1 + src_bytes)", True), ("duration", "ln(1 + duration)", True)]
    for ax, (feature, xlabel, use_log) in zip(axes.flat, specs):
        for group in GROUP_ORDER:
            values = data.loc[data["attack_category"] == group, feature].to_numpy(float)
            if len(values) > ECDF_SAMPLE_PER_GROUP:
                values = rng.choice(values, ECDF_SAMPLE_PER_GROUP, replace=False)
            if use_log:
                values = np.log1p(values)
            sns.ecdfplot(x=values, label=group, color=PALETTE[group], lw=1.5, ax=ax)
        ax.set(xlabel=xlabel, ylabel="Tỷ lệ tích lũy", title=feature)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=5, frameon=False)
    fig.suptitle("Phân phối thực nghiệm theo năm nhóm", y=0.995, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    generated.extend(_save(fig, output_dir, "anova_ecdf_four_features"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    for ax, feature in zip(axes.flat, FEATURES):
        part = posthoc.loc[posthoc["feature"] == feature]
        matrix = pd.DataFrame(np.nan, index=GROUP_ORDER, columns=GROUP_ORDER)
        annotations = pd.DataFrame("--", index=GROUP_ORDER, columns=GROUP_ORDER)
        for row in part.itertuples():
            p_value = float(row.adjusted_p_value)
            matrix.loc[row.group1, row.group2] = matrix.loc[row.group2, row.group1] = p_value
            label = "<.001*" if p_value < 0.001 else f"{p_value:.3f}" + ("*" if p_value < 0.05 else "")
            annotations.loc[row.group1, row.group2] = annotations.loc[row.group2, row.group1] = label
        np.fill_diagonal(matrix.values, 0.05)
        sns.heatmap(matrix, annot=annotations, fmt="", cmap="viridis_r", vmin=0, vmax=0.05,
                    cbar=False, square=True, linewidths=0.4, ax=ax, annot_kws={"fontsize": 8})
        ax.set(title=feature, xlabel="", ylabel="")
    fig.suptitle("Hậu kiểm Games--Howell: p-value đã hiệu chỉnh (*: p < 0,05)", y=0.99, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    generated.extend(_save(fig, output_dir, "games_howell_heatmaps"))

    fig, axes = plt.subplots(2, 5, figsize=(13.5, 6.2))
    for row_index, feature in enumerate(["src_bytes", "duration"]):
        for column_index, group in enumerate(GROUP_ORDER):
            values = np.log1p(data.loc[data["attack_category"] == group, feature].to_numpy(float))
            if len(values) > QQ_SAMPLE_PER_GROUP:
                values = rng.choice(values, QQ_SAMPLE_PER_GROUP, replace=False)
            stats.probplot(values, dist="norm", plot=axes[row_index, column_index])
            axes[row_index, column_index].set_title(group if row_index == 0 else "")
            axes[row_index, column_index].set(xlabel="", ylabel="")
            for line in axes[row_index, column_index].lines:
                line.set_markersize(2.2)
        axes[row_index, 0].set_ylabel(f"log1p({feature})\nPhân vị quan sát")
    fig.suptitle("Q--Q plot trên thang phân tích (tối đa 1.000 quan sát/nhóm)", y=0.99, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    generated.extend(_save(fig, output_dir, "qq_log_features"))

    zero = zero_duration.set_index("group").loc[GROUP_ORDER]
    proportions = zero["zero_proportion"].to_numpy()
    errors = np.vstack([proportions-zero["wilson_ci95_lower"].to_numpy(),
                        zero["wilson_ci95_upper"].to_numpy()-proportions])
    positions = np.arange(len(GROUP_ORDER))
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.bar(positions, proportions, color=[PALETTE[g] for g in GROUP_ORDER],
           edgecolor="black", alpha=0.86)
    ax.errorbar(positions, proportions, yerr=errors, fmt="none", color="black", capsize=5)
    ax.set_xticks(positions, GROUP_ORDER)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set(xlabel="Nhóm kết nối", ylabel="Tỷ lệ duration = 0",
           title="Tỷ lệ duration = 0 và khoảng tin cậy Wilson 95%")
    fig.tight_layout()
    generated.extend(_save(fig, output_dir, "duration_zero_wilson"))
    return generated

