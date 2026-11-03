"""Phan tich ANOVA tai lap cho bon dac trung NSL-KDD.

Module nay chi su dung tap train sach o thang do goc. Cac phep bien doi log1p
duoc tao trong bo nho va duoc gan nhan ro trong tat ca bang ket qua.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from report_assets import generate_report_figures


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    ROOT
    / "00_DU_LIEU_CHUNG_CHO_TEAM"
    / "00_DU_LIEU_CHUNG"
    / "data"
    / "processed"
    / "train_clean.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
GROUP_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]
FEATURES = ["count", "serror_rate", "src_bytes", "duration"]
RQ_BY_FEATURE = {
    "count": "RQ1",
    "serror_rate": "RQ2",
    "src_bytes": "RQ3",
    "duration": "RQ4",
}
LABELS = {
    "count": "Số kết nối gần đây (kết nối)",
    "serror_rate": "Tỷ lệ lỗi SYN (0–1)",
    "src_bytes": "Byte từ nguồn đến đích (byte)",
    "duration": "Thời gian kết nối (giây)",
}
PALETTE = {
    "Normal": "#4477AA",
    "DoS": "#CC6677",
    "Probe": "#DDCC77",
    "R2L": "#117733",
    "U2R": "#AA4499",
}
ALPHA = 0.05
RANDOM_STATE = 18015
QQ_SAMPLE_PER_GROUP = 1000
PLOT_SAMPLE_PER_GROUP = 5000


@dataclass(frozen=True)
class AnalysisSpec:
    """Quy tac phan tich duoc xac dinh truoc cho mot research question."""

    feature: str
    analysis_scale: str
    primary_method: str
    reason: str


SPECS = {
    "count": AnalysisSpec(
        "count",
        "original",
        "Welch one-way ANOVA",
        "Biến đếm lệch và phương sai/cỡ mẫu rất khác nhau; cỡ mẫu bốn nhóm lớn, "
        "nhưng U2R nhỏ nên dùng Welch và báo cảnh báo độ bền.",
    ),
    "serror_rate": AnalysisSpec(
        "serror_rate",
        "original",
        "Welch one-way ANOVA",
        "Mục tiêu là so sánh trung bình tỷ lệ trên thang 0–1; phương sai khác nhau. "
        "Khối lượng lớn ở biên khiến đây là xấp xỉ, được kiểm tra nhạy bằng Kruskal–Wallis.",
    ),
    "src_bytes": AnalysisSpec(
        "src_bytes",
        "log1p",
        "Welch one-way ANOVA on log1p",
        "Thang gốc lệch phải và có cực trị rất lớn; log1p giảm ảnh hưởng nhưng đổi estimand "
        "thành trung bình log(1 + src_bytes).",
    ),
    "duration": AnalysisSpec(
        "duration",
        "log1p",
        "Welch one-way ANOVA on log1p",
        "Dữ liệu toàn phần zero-inflated và lệch mạnh; log1p giữ số 0. Kết quả được bổ sung "
        "bằng phân tích hai phần, không coi ANOVA là mô hình hoàn hảo.",
    ),
}


def _analysis_values(values: np.ndarray, scale: str) -> np.ndarray:
    return np.log1p(values) if scale == "log1p" else values


def p_display(value: float) -> str:
    """Dinh dang p-value, khong bao gio hien thi 'p = 0'."""

    if not np.isfinite(value):
        return "NA"
    if value < 1e-300:
        return "< 1e-300"
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def load_and_validate_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Doc cac cot can thiet va kiem tra data contract truoc khi suy luan."""

    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {path}. Hay dat train_clean.csv dung vi tri ghi trong README."
        )
    required = ["attack_name", "attack_category", "label_binary", *FEATURES]
    data = pd.read_csv(path, usecols=required, low_memory=False)
    audit: list[dict[str, object]] = []
    audit.append({"check": "row_count", "value": len(data), "status": "PASS"})
    audit.append(
        {
            "check": "required_columns",
            "value": ", ".join(required),
            "status": "PASS" if set(required).issubset(data.columns) else "FAIL",
        }
    )
    missing = data[required].isna().sum()
    audit.append(
        {
            "check": "missing_values",
            "value": int(missing.sum()),
            "status": "PASS" if missing.sum() == 0 else "FAIL",
        }
    )
    numeric = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    non_numeric = int(numeric.isna().sum().sum() - data[FEATURES].isna().sum().sum())
    audit.append(
        {
            "check": "numeric_coercion_failures",
            "value": non_numeric,
            "status": "PASS" if non_numeric == 0 else "FAIL",
        }
    )
    finite = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
    audit.append({"check": "finite_values", "value": finite, "status": "PASS" if finite else "FAIL"})
    observed = set(data["attack_category"].astype(str).unique())
    groups_ok = observed == set(GROUP_ORDER)
    audit.append(
        {
            "check": "five_expected_groups",
            "value": ", ".join(sorted(observed)),
            "status": "PASS" if groups_ok else "FAIL",
        }
    )
    domains_ok = bool(
        (numeric[["count", "src_bytes", "duration"]] >= 0).all().all()
        and numeric["serror_rate"].between(0, 1).all()
    )
    audit.append(
        {"check": "feature_domains", "value": domains_ok, "status": "PASS" if domains_ok else "FAIL"}
    )
    binary_ok = bool(
        ((data["attack_category"] == "Normal") == (data["label_binary"] == 0)).all()
    )
    audit.append(
        {"check": "binary_label_consistency", "value": binary_ok, "status": "PASS" if binary_ok else "FAIL"}
    )
    audit_df = pd.DataFrame(audit)
    if (audit_df["status"] == "FAIL").any():
        failed = audit_df.loc[audit_df["status"] == "FAIL", "check"].tolist()
        raise ValueError(f"Data validation failed: {failed}")
    data[FEATURES] = numeric
    data["attack_category"] = pd.Categorical(
        data["attack_category"], categories=GROUP_ORDER, ordered=True
    )
    return data, audit_df


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Tinh thong ke mo ta tren don vi goc; CI la t-CI cua trung binh."""

    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        for group in GROUP_ORDER:
            values = data.loc[data["attack_category"] == group, feature].to_numpy(float)
            n = len(values)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            se = sd / math.sqrt(n)
            critical = float(stats.t.ppf(0.975, n - 1))
            q1, q3 = np.quantile(values, [0.25, 0.75])
            rows.append(
                {
                    "rq": RQ_BY_FEATURE[feature],
                    "feature": feature,
                    "group": group,
                    "n": n,
                    "mean": mean,
                    "median": float(np.median(values)),
                    "std": sd,
                    "variance": float(np.var(values, ddof=1)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "zero_proportion": float(np.mean(values == 0)),
                    "one_proportion": float(np.mean(values == 1)),
                    "skewness": float(stats.skew(values, bias=False)),
                    "mean_ci95_lower": mean - critical * se,
                    "mean_ci95_upper": mean + critical * se,
                    "ci_method": "Two-sided t CI; large-sample CLT approximation",
                }
            )
    return pd.DataFrame(rows)


def assumption_checks(data: pd.DataFrame) -> pd.DataFrame:
    """Tong hop Brown-Forsythe, shape diagnostics, imbalance va outliers."""

    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, object]] = []
    for feature, spec in SPECS.items():
        samples = []
        group_sizes = []
        for group in GROUP_ORDER:
            raw = data.loc[data["attack_category"] == group, feature].to_numpy(float)
            values = _analysis_values(raw, spec.analysis_scale)
            samples.append(values)
            group_sizes.append(len(values))
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = q3 - q1
            outlier_prop = float(np.mean((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)))
            sample = rng.choice(values, size=min(5000, len(values)), replace=False)
            shapiro = stats.shapiro(sample)
            rows.append(
                {
                    "rq": RQ_BY_FEATURE[feature],
                    "feature": feature,
                    "analysis_scale": spec.analysis_scale,
                    "scope": group,
                    "diagnostic": "group_shape",
                    "statistic": float(stats.skew(values, bias=False)),
                    "p_value": float(shapiro.pvalue),
                    "n_or_sample_size": len(sample),
                    "variance": float(np.var(values, ddof=1)),
                    "excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
                    "iqr_outlier_proportion": outlier_prop,
                    "warning": "Shapiro dùng mẫu tối đa 5.000; không dùng một mình để chọn kiểm định.",
                }
            )
        bf = stats.levene(*samples, center="median")
        rows.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "analysis_scale": spec.analysis_scale,
                "scope": "all_groups",
                "diagnostic": "Brown-Forsythe (median-centered Levene)",
                "statistic": float(bf.statistic),
                "p_value": float(bf.pvalue),
                "n_or_sample_size": int(sum(group_sizes)),
                "variance": np.nan,
                "excess_kurtosis": np.nan,
                "iqr_outlier_proportion": np.nan,
                "warning": "Phương sai không đồng nhất" if bf.pvalue < ALPHA else "Chưa bác bỏ đồng nhất phương sai",
            }
        )
        rows.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "analysis_scale": spec.analysis_scale,
                "scope": "all_groups",
                "diagnostic": "sample-size imbalance",
                "statistic": max(group_sizes) / min(group_sizes),
                "p_value": np.nan,
                "n_or_sample_size": int(sum(group_sizes)),
                "variance": np.nan,
                "excess_kurtosis": np.nan,
                "iqr_outlier_proportion": np.nan,
                "warning": f"max(n)/min(n) = {max(group_sizes) / min(group_sizes):.1f}; U2R nhỏ.",
            }
        )
    return pd.DataFrame(rows)


def _classic_effect_sizes(samples: Iterable[np.ndarray]) -> tuple[float, float]:
    """Eta-squared va omega-squared mo ta theo tong binh phuong co dien."""

    arrays = list(samples)
    all_values = np.concatenate(arrays)
    grand = float(np.mean(all_values))
    ss_between = sum(len(x) * (float(np.mean(x)) - grand) ** 2 for x in arrays)
    ss_within = sum(float(np.sum((x - np.mean(x)) ** 2)) for x in arrays)
    ss_total = ss_between + ss_within
    df_between = len(arrays) - 1
    df_within = len(all_values) - len(arrays)
    ms_within = ss_within / df_within
    eta2 = ss_between / ss_total if ss_total else np.nan
    omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    return float(eta2), float(max(0.0, omega2))


def games_howell(samples: dict[str, np.ndarray], alpha: float = ALPHA) -> pd.DataFrame:
    """Games-Howell voi p-value va CI family-wise tu studentized range."""

    rows = []
    k = len(samples)
    for i, group1 in enumerate(GROUP_ORDER[:-1]):
        for group2 in GROUP_ORDER[i + 1 :]:
            x, y = samples[group1], samples[group2]
            n1, n2 = len(x), len(y)
            v1, v2 = np.var(x, ddof=1) / n1, np.var(y, ddof=1) / n2
            diff = float(np.mean(x) - np.mean(y))
            se_q = math.sqrt(0.5 * (v1 + v2))
            df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
            q = abs(diff) / se_q if se_q else np.inf
            p = float(stats.studentized_range.sf(q, k, df))
            qcrit = float(stats.studentized_range.ppf(1 - alpha, k, df))
            margin = qcrit * se_q
            rows.append(
                {
                    "group1": group1,
                    "group2": group2,
                    "mean_difference": diff,
                    "ci95_lower": diff - margin,
                    "ci95_upper": diff + margin,
                    "adjusted_p_value": p,
                    "adjusted_p_display": p_display(p),
                    "df": df,
                    "reject_h0": p < alpha,
                }
            )
    return pd.DataFrame(rows)


def tukey_hsd(samples: dict[str, np.ndarray], alpha: float = ALPHA) -> pd.DataFrame:
    values = np.concatenate([samples[g] for g in GROUP_ORDER])
    labels = np.concatenate([[g] * len(samples[g]) for g in GROUP_ORDER])
    result = pairwise_tukeyhsd(values, labels, alpha=alpha)
    table = pd.DataFrame(result._results_table.data[1:], columns=result._results_table.data[0])
    table = table.rename(
        columns={
            "group1": "group1",
            "group2": "group2",
            "meandiff": "mean_difference",
            "lower": "ci95_lower",
            "upper": "ci95_upper",
            "p-adj": "adjusted_p_value",
            "reject": "reject_h0",
        }
    )
    table["adjusted_p_display"] = table["adjusted_p_value"].map(p_display)
    table["df"] = np.nan
    return table


def run_omnibus(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chay primary Welch, classical sensitivity va Kruskal cho bon RQ."""

    result_rows, posthoc_tables, sensitivity_rows = [], [], []
    for feature, spec in SPECS.items():
        raw_samples = {
            group: data.loc[data["attack_category"] == group, feature].to_numpy(float)
            for group in GROUP_ORDER
        }
        samples = {
            group: _analysis_values(values, spec.analysis_scale)
            for group, values in raw_samples.items()
        }
        arrays = [samples[group] for group in GROUP_ORDER]
        welch = anova_oneway(arrays, use_var="unequal", welch_correction=True)
        classic = anova_oneway(arrays, use_var="equal")
        kw = stats.kruskal(*arrays)
        eta2, omega2 = _classic_effect_sizes(arrays)
        primary = welch
        p = float(primary.pvalue)
        result_rows.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "sample_size": int(sum(map(len, arrays))),
                "analysis_variable": f"log1p({feature})" if spec.analysis_scale == "log1p" else feature,
                "selected_test": spec.primary_method,
                "F_statistic": float(primary.statistic),
                "df_numerator": float(primary.df_num),
                "df_denominator": float(primary.df_denom),
                "raw_p_value": p,
                "raw_p_display": p_display(p),
                "alpha": ALPHA,
                "reject_h0_raw": p < ALPHA,
                "eta_squared_descriptive": eta2,
                "omega_squared_descriptive": omega2,
                "effect_size_note": "Classic sums-of-squares effect size on analysis scale; descriptive, not derived from Welch F.",
                "selection_reason": spec.reason,
            }
        )
        sensitivity_rows.extend(
            [
                {
                    "rq": RQ_BY_FEATURE[feature],
                    "feature": feature,
                    "analysis_variable": f"log1p({feature})" if spec.analysis_scale == "log1p" else feature,
                    "method": "Classical one-way ANOVA (sensitivity)",
                    "statistic": float(classic.statistic),
                    "df_numerator": float(classic.df_num),
                    "df_denominator": float(classic.df_denom),
                    "p_value": float(classic.pvalue),
                    "p_display": p_display(float(classic.pvalue)),
                    "estimand_note": "Equality of means on the named analysis scale.",
                },
                {
                    "rq": RQ_BY_FEATURE[feature],
                    "feature": feature,
                    "analysis_variable": f"log1p({feature})" if spec.analysis_scale == "log1p" else feature,
                    "method": "Kruskal-Wallis (sensitivity)",
                    "statistic": float(kw.statistic),
                    "df_numerator": len(GROUP_ORDER) - 1,
                    "df_denominator": np.nan,
                    "p_value": float(kw.pvalue),
                    "p_display": p_display(float(kw.pvalue)),
                    "estimand_note": "Rank-distribution test; not an equivalent test of raw means/medians.",
                },
            ]
        )
        if p < ALPHA:
            posthoc = games_howell(samples)
            posthoc.insert(0, "method", "Games-Howell")
            posthoc.insert(0, "analysis_variable", result_rows[-1]["analysis_variable"])
            posthoc.insert(0, "feature", feature)
            posthoc.insert(0, "rq", RQ_BY_FEATURE[feature])
            posthoc_tables.append(posthoc)
    results = pd.DataFrame(result_rows)
    _, adjusted, _, _ = multipletests(results["raw_p_value"], alpha=ALPHA, method="holm")
    results["holm_adjusted_p_value"] = adjusted
    results["holm_adjusted_p_display"] = [p_display(x) for x in adjusted]
    results["reject_h0_holm"] = adjusted < ALPHA
    posthoc_all = pd.concat(posthoc_tables, ignore_index=True) if posthoc_tables else pd.DataFrame()
    return results, posthoc_all, pd.DataFrame(sensitivity_rows)


def duration_two_part(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mô tả P(duration=0) và phân tích duration>0, không kiểm định Chi-square."""

    contingency = pd.crosstab(data["attack_category"], data["duration"].eq(0)).reindex(GROUP_ORDER)
    contingency = contingency.reindex(columns=[False, True], fill_value=0)
    zero_rows = []
    for group in GROUP_ORDER:
        zero = int(contingency.loc[group, True])
        total = int(contingency.loc[group].sum())
        low, high = stats.binomtest(zero, total).proportion_ci(confidence_level=0.95, method="wilson")
        zero_rows.append(
            {
                "group": group,
                "n": total,
                "zero_count": zero,
                "zero_proportion": zero / total,
                "wilson_ci95_lower": low,
                "wilson_ci95_upper": high,
            }
        )
    positive = data.loc[data["duration"] > 0]
    samples = {
        group: np.log1p(
            positive.loc[positive["attack_category"] == group, "duration"].to_numpy(float)
        )
        for group in GROUP_ORDER
    }
    welch = anova_oneway([samples[g] for g in GROUP_ORDER], use_var="unequal", welch_correction=True)
    kw = stats.kruskal(*[samples[g] for g in GROUP_ORDER])
    eta2, omega2 = _classic_effect_sizes(samples.values())
    positive_result = pd.DataFrame(
        [
            {
                "scope": "duration > 0 only",
                "analysis_variable": "log1p(duration)",
                "group_sizes": json.dumps({g: len(samples[g]) for g in GROUP_ORDER}),
                "method": "Welch one-way ANOVA",
                "F_statistic": float(welch.statistic),
                "df_numerator": float(welch.df_num),
                "df_denominator": float(welch.df_denom),
                "p_value": float(welch.pvalue),
                "p_display": p_display(float(welch.pvalue)),
                "eta_squared_descriptive": eta2,
                "omega_squared_descriptive": omega2,
                "kruskal_H_sensitivity": float(kw.statistic),
                "kruskal_p_sensitivity": float(kw.pvalue),
                "warning": "Conditional analysis; does not describe all connections and is not interchangeable with full-data ANOVA.",
            }
        ]
    )
    return pd.DataFrame(zero_rows), positive_result


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    data: pd.DataFrame, descriptive: pd.DataFrame, posthoc: pd.DataFrame, zero_duration: pd.DataFrame
) -> pd.DataFrame:
    """Tao bo hinh hoc thuat; sampling chi dung cho hinh, khong dung cho kiem dinh."""

    sns.set_theme(style="whitegrid", context="notebook")
    rng = np.random.default_rng(RANDOM_STATE)
    captions: list[dict[str, str]] = []
    for feature, spec in SPECS.items():
        out = FIGURE_DIR / feature
        # Boxplot original scale: outliers hidden visually, never removed from tests.
        fig, ax = plt.subplots(figsize=(9, 5.4))
        sns.boxplot(
            data=data,
            x="attack_category",
            y=feature,
            order=GROUP_ORDER,
            hue="attack_category",
            palette=PALETTE,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        ax.set(xlabel="Nhóm kết nối", ylabel=LABELS[feature], title=f"{RQ_BY_FEATURE[feature]} — {feature} theo nhóm (thang gốc)")
        fig.tight_layout()
        _save(fig, out, "01_boxplot_original")
        captions.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "figure": f"figures/{feature}/01_boxplot_original.png",
                "caption": "Boxplot thang gốc; ẩn điểm ngoại lệ chỉ để dễ đọc, không loại khỏi dữ liệu/kiểm định. Cho thấy vị trí và độ phân tán, không tự chứng minh khác biệt trung bình hay quan hệ nhân quả.",
            }
        )
        if feature in {"count", "src_bytes", "duration"}:
            plot_data = data[["attack_category", feature]].copy()
            plot_data[f"log1p_{feature}"] = np.log1p(plot_data[feature])
            fig, ax = plt.subplots(figsize=(9, 5.4))
            sns.boxplot(
                data=plot_data,
                x="attack_category",
                y=f"log1p_{feature}",
                order=GROUP_ORDER,
                hue="attack_category",
                palette=PALETTE,
                legend=False,
                showfliers=False,
                ax=ax,
            )
            ax.set(xlabel="Nhóm kết nối", ylabel=f"ln(1 + {feature})", title=f"{RQ_BY_FEATURE[feature]} — phân phối {feature} sau log1p")
            fig.tight_layout()
            _save(fig, out, "02_boxplot_log1p")
            captions.append(
                {
                    "rq": RQ_BY_FEATURE[feature],
                    "feature": feature,
                    "figure": f"figures/{feature}/02_boxplot_log1p.png",
                    "caption": "Boxplot log1p giúp thấy cấu trúc trung tâm của dữ liệu lệch; khác biệt trên hình là ở thang biến đổi, không phải chênh lệch byte/giây trung bình trên thang gốc.",
                }
            )
        # Means and 95% CI from the complete data.
        desc = descriptive.loc[descriptive["feature"] == feature].set_index("group").loc[GROUP_ORDER]
        x = np.arange(len(GROUP_ORDER))
        fig, ax = plt.subplots(figsize=(9, 5.4))
        y = desc["mean"].to_numpy(float)
        yerr = np.vstack([y - desc["mean_ci95_lower"], desc["mean_ci95_upper"] - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color="black", ecolor="black", capsize=5)
        ax.scatter(x, y, s=70, c=[PALETTE[g] for g in GROUP_ORDER], edgecolor="black", zorder=3)
        for idx, group in enumerate(GROUP_ORDER):
            ax.annotate(f"n={int(desc.loc[group, 'n']):,}", (idx, y[idx]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, GROUP_ORDER)
        ax.set(xlabel="Nhóm kết nối", ylabel=LABELS[feature], title=f"{RQ_BY_FEATURE[feature]} — trung bình và 95% CI theo nhóm")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 5))
        fig.tight_layout()
        _save(fig, out, "03_mean_ci95")
        captions.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "figure": f"figures/{feature}/03_mean_ci95.png",
                "caption": "Điểm là trung bình thang gốc; thanh sai số là t-CI 95% cho từng trung bình (CLT), không phải SD và không phải CI của chênh lệch cặp. Outlier nặng có thể làm CI của mean kém ổn định.",
            }
        )
        # ECDF with fixed sampling for legibility.
        fig, ax = plt.subplots(figsize=(9, 5.4))
        for group in GROUP_ORDER:
            values = data.loc[data["attack_category"] == group, feature].to_numpy(float)
            if len(values) > PLOT_SAMPLE_PER_GROUP:
                values = rng.choice(values, size=PLOT_SAMPLE_PER_GROUP, replace=False)
            display_values = _analysis_values(values, spec.analysis_scale)
            sns.ecdfplot(x=display_values, label=group, color=PALETTE[group], ax=ax)
        x_label = f"ln(1 + {feature})" if spec.analysis_scale == "log1p" else LABELS[feature]
        ax.set(xlabel=x_label, ylabel="Tỷ lệ tích lũy", title=f"{RQ_BY_FEATURE[feature]} — ECDF theo nhóm")
        ax.legend(title="Nhóm")
        fig.tight_layout()
        _save(fig, out, "04_ecdf")
        captions.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "figure": f"figures/{feature}/04_ecdf.png",
                "caption": f"ECDF trên thang phân tích, lấy tối đa {PLOT_SAMPLE_PER_GROUP:,} quan sát/nhóm với random_state={RANDOM_STATE} chỉ để vẽ. Hình cho thấy toàn bộ dạng phân phối; kiểm định vẫn dùng dữ liệu đầy đủ.",
            }
        )
        # Per-group QQ plots.
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
        for ax, group in zip(axes, GROUP_ORDER):
            values = data.loc[data["attack_category"] == group, feature].to_numpy(float)
            values = _analysis_values(values, spec.analysis_scale)
            if len(values) > QQ_SAMPLE_PER_GROUP:
                values = rng.choice(values, size=QQ_SAMPLE_PER_GROUP, replace=False)
            stats.probplot(values, dist="norm", plot=ax)
            ax.set_title(group)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.suptitle(f"{RQ_BY_FEATURE[feature]} — Q–Q plots trên thang phân tích (mẫu cố định)", y=1.03)
        fig.tight_layout()
        _save(fig, out, "05_qq_diagnostics")
        captions.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "figure": f"figures/{feature}/05_qq_diagnostics.png",
                "caption": f"Q–Q plot dùng tối đa {QQ_SAMPLE_PER_GROUP:,} điểm/nhóm để chẩn đoán chuẩn tính; sai lệch đường thẳng cảnh báo mô hình ANOVA, nhưng không được dùng đơn lẻ để quyết định.",
            }
        )
        # Adjusted p-value heatmap; numbers make it readable without relying only on color.
        ph = posthoc.loc[posthoc["feature"] == feature]
        matrix = pd.DataFrame(np.nan, index=GROUP_ORDER, columns=GROUP_ORDER)
        labels = pd.DataFrame("—", index=GROUP_ORDER, columns=GROUP_ORDER)
        for _, row in ph.iterrows():
            matrix.loc[row["group1"], row["group2"]] = row["adjusted_p_value"]
            matrix.loc[row["group2"], row["group1"]] = row["adjusted_p_value"]
            label = row["adjusted_p_display"] + ("†" if row["reject_h0"] else "")
            labels.loc[row["group1"], row["group2"]] = label
            labels.loc[row["group2"], row["group1"]] = label
        np.fill_diagonal(matrix.values, 1.0)
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        sns.heatmap(
            matrix,
            annot=labels,
            fmt="",
            cmap="viridis_r",
            vmin=0,
            vmax=0.05,
            square=True,
            linewidths=1.25,
            linecolor="white",
            cbar_kws={"label": "p hiệu chỉnh (giá trị ≥ 0,05 bão hòa tại 0,05)"},
            ax=ax,
        )
        ax.set_title(f"{RQ_BY_FEATURE[feature]} — Games–Howell (†: p_adj < 0,05)")
        fig.tight_layout()
        _save(fig, out, "06_posthoc_heatmap")
        captions.append(
            {
                "rq": RQ_BY_FEATURE[feature],
                "feature": feature,
                "figure": f"figures/{feature}/06_posthoc_heatmap.png",
                "caption": "Heatmap p-value Games–Howell đã hiệu chỉnh trong 10 cặp; đường viền trắng phân tách rõ từng ô, còn thanh màu biểu diễn miền p từ 0 đến 0,05. Số in trong ô và ký hiệu † cho phép đọc không chỉ dựa vào màu; chênh lệch cụ thể và CI nằm trong bảng posthoc.",
            }
        )
    # Duration zero proportion is a distinct estimand.
    fig, ax = plt.subplots(figsize=(9, 5.4))
    z = zero_duration.set_index("group").loc[GROUP_ORDER]
    x = np.arange(len(GROUP_ORDER))
    y = z["zero_proportion"].to_numpy()
    err = np.vstack([y - z["wilson_ci95_lower"], z["wilson_ci95_upper"] - y])
    ax.bar(x, y, color=[PALETTE[g] for g in GROUP_ORDER], edgecolor="black", alpha=0.85)
    ax.errorbar(x, y, yerr=err, fmt="none", color="black", capsize=5)
    ax.set_xticks(x, GROUP_ORDER)
    ax.set(xlabel="Nhóm kết nối", ylabel="Tỷ lệ duration = 0", title="RQ4 — xác suất quan sát duration = 0 (Wilson 95% CI)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    _save(fig, FIGURE_DIR / "duration", "07_zero_proportion")
    captions.append(
        {
            "rq": "RQ4",
            "feature": "duration",
            "figure": "figures/duration/07_zero_proportion.png",
            "caption": "Tỷ lệ duration=0 và Wilson CI 95%; đây là phần nhị phân của phân tích hai phần, khác câu hỏi về trung bình thời lượng dương và không chứng minh nguyên nhân.",
        }
    )
    return pd.DataFrame(captions)


def _warnings(feature: str, assumptions: pd.DataFrame) -> str:
    group_shape = assumptions[(assumptions["feature"] == feature) & (assumptions["diagnostic"] == "group_shape")]
    max_skew = group_shape["statistic"].abs().max()
    bf = assumptions[(assumptions["feature"] == feature) & assumptions["diagnostic"].str.startswith("Brown")].iloc[0]
    special = {
        "count": "Biến đếm; U2R n=52;",
        "serror_rate": "Tỷ lệ dồn ở 0/1; ANOVA chỉ là xấp xỉ cho mean;",
        "src_bytes": "Ngoại lệ cực lớn; estimand chính ở thang log1p;",
        "duration": "Zero-inflated; estimand chính ở thang log1p; cần đọc phân tích hai phần;",
    }[feature]
    return f"{special} Brown–Forsythe p {p_display(float(bf['p_value']))}; max |skew|={max_skew:.2f}; độc lập phiên không kiểm chứng được."


def write_interpretation(
    data: pd.DataFrame,
    descriptive: pd.DataFrame,
    assumptions: pd.DataFrame,
    results: pd.DataFrame,
    posthoc: pd.DataFrame,
    zero_duration: pd.DataFrame,
    duration_positive: pd.DataFrame,
) -> pd.DataFrame:
    """Tao bang tong hop va bao cao tieng Viet tu ket qua that."""

    summary = results.copy()
    summary["degrees_of_freedom"] = summary.apply(
        lambda r: f"({r.df_numerator:.3f}, {r.df_denominator:.3f})", axis=1
    )
    summary["effect_size"] = summary.apply(
        lambda r: f"eta2_desc={r.eta_squared_descriptive:.4f}; omega2_desc={r.omega_squared_descriptive:.4f}", axis=1
    )
    summary["assumption_warnings"] = [
        _warnings(feature, assumptions) for feature in summary["feature"]
    ]
    summary["conclusion"] = summary.apply(
        lambda r: (
            f"Bác bỏ H0 sau Holm: có ít nhất một trung bình khác trên thang {r.analysis_variable}."
            if r.reject_h0_holm
            else f"Chưa bác bỏ H0 sau Holm trên thang {r.analysis_variable}; không có nghĩa H0 đúng."
        ),
        axis=1,
    )
    summary_out = summary[
        [
            "rq",
            "feature",
            "sample_size",
            "selected_test",
            "F_statistic",
            "degrees_of_freedom",
            "raw_p_value",
            "raw_p_display",
            "holm_adjusted_p_value",
            "holm_adjusted_p_display",
            "effect_size",
            "assumption_warnings",
            "conclusion",
        ]
    ]
    lines = [
        "# Diễn giải ANOVA cho NSL-KDD (tập train)",
        "",
        "## Phạm vi và giả thuyết",
        "",
        f"Phân tích dùng {len(data):,} bản ghi train ở thang gốc, không SMOTE, không scaling và không dùng test set. Với mỗi RQ, "
        "$H_0: \\mu_{Normal}=\\mu_{DoS}=\\mu_{Probe}=\\mu_{R2L}=\\mu_{U2R}$; $H_1$: ít nhất một trung bình khác, $\\alpha=0.05$.",
        "",
        "F là tỷ số giữa biến thiên giữa nhóm và biến thiên trong nhóm (Welch dùng trọng số theo phương sai). Bậc tự do mô tả lượng thông tin cho tử/mẫu số; "
        "p-value là xác suất thấy thống kê ít nhất cực đoan như quan sát nếu H0 và mô hình đúng. Bác bỏ khi p hiệu chỉnh < alpha. Không bác bỏ không chứng minh H0; "
        "ANOVA có ý nghĩa cũng không nói mọi cặp đều khác.",
        "",
        "## Kết quả bốn RQ",
        "",
    ]
    for _, row in summary.iterrows():
        feature = row["feature"]
        desc = descriptive.loc[descriptive["feature"] == feature].set_index("group")
        significant = posthoc[(posthoc["feature"] == feature) & posthoc["reject_h0"]]
        pair_text = ", ".join(f"{r.group1}–{r.group2}" for r in significant.itertuples()) or "không có cặp khẳng định"
        lines.extend(
            [
                f"### {row['rq']} — `{feature}`",
                "",
                f"Phương pháp chính: **{row['selected_test']}** trên `{row['analysis_variable']}`. {row['selection_reason']}",
                "",
                f"F({row['df_numerator']:.3f}, {row['df_denominator']:.3f}) = {row['F_statistic']:.4f}; p thô {row['raw_p_display']}; "
                f"p Holm {row['holm_adjusted_p_display']}. Eta² mô tả = {row['eta_squared_descriptive']:.4f}, omega² mô tả = {row['omega_squared_descriptive']:.4f}. "
                "Hai effect size này dựa trên tổng bình phương cổ điển ở thang phân tích, không suy ra trực tiếp từ Welch F.",
                "",
                f"Các cặp Games–Howell có ý nghĩa: {pair_text}. Bảng hậu kiểm chứa chênh lệch, CI đồng thời và p đã hiệu chỉnh; "
                "với `src_bytes`/`duration`, chênh lệch là ở thang log1p.",
                "",
                "Trung bình thang gốc theo thứ tự Normal/DoS/Probe/R2L/U2R: "
                + ", ".join(f"{g}={desc.loc[g, 'mean']:.4g}" for g in GROUP_ORDER)
                + ". Trung vị: "
                + ", ".join(f"{g}={desc.loc[g, 'median']:.4g}" for g in GROUP_ORDER)
                + ".",
                "",
                f"Cảnh báo: {_warnings(feature, assumptions)}",
                "",
            ]
        )
    z = zero_duration.set_index("group")
    dp = duration_positive.iloc[0]
    lines.extend(
        [
            "## RQ4 — mô tả zero và phân tích có điều kiện",
            "",
            "Tỷ lệ `duration=0` (mô tả, kèm Wilson CI trong bảng): "
            + ", ".join(f"{g}={z.loc[g, 'zero_proportion']:.2%}" for g in GROUP_ORDER)
            + ". Phần này không thực hiện Chi-square và không phải kiểm định khác biệt mean duration.",
            f"Trong các kết nối duration>0, Welch trên log1p: F({dp['df_numerator']:.3f}, {dp['df_denominator']:.3f})={dp['F_statistic']:.4f}, p {dp['p_display']}. "
            "Đây là phân tích có điều kiện, không thay thế kết quả toàn bộ dữ liệu.",
            "",
            "## Ý nghĩa và giới hạn",
            "",
            "Kết quả mô tả hành vi kết nối trong NSL-KDD: một số nhóm khác rõ về cường độ kết nối, lỗi SYN, byte nguồn và thời lượng. "
            "Chúng có thể hỗ trợ giải thích vì sao mô hình ML khai thác các biến này, nhưng p-value nhỏ không đo khả năng dự đoán: cỡ mẫu, chồng lấn phân phối, tương tác, "
            "mất cân bằng lớp và khả năng khái quát đều quan trọng. Eta² cũng không phải bảng xếp hạng predictor.",
            "",
            "NSL-KDD cũ và không đại diện trực tiếp cho lưu lượng Internet hiện đại. Metadata không có session/host/time identifier đủ để kiểm chứng độc lập; "
            "các bản ghi gần cùng chiến dịch có thể tương quan, làm độ không chắc chắn thực lớn hơn CI/p-value giả định độc lập. Phân tích quan sát không chứng minh nguyên nhân hay xác suất tấn công.",
        ]
    )
    (RESULT_DIR / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_out


def run_analysis(data_path: Path = DATA_PATH) -> dict[str, pd.DataFrame]:
    """Chay day du pipeline va ghi tat ca bang/hinh vao anova_analysis/."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data, audit = load_and_validate_data(Path(data_path))
    descriptive = descriptive_statistics(data)
    assumptions = assumption_checks(data)
    results, posthoc, sensitivity = run_omnibus(data)
    zero_duration, duration_positive = duration_two_part(data)
    captions = create_figures(data, descriptive, posthoc, zero_duration)
    summary = write_interpretation(
        data, descriptive, assumptions, results, posthoc, zero_duration, duration_positive
    )
    feature_comparison = results[
        [
            "rq",
            "feature",
            "analysis_variable",
            "reject_h0_holm",
            "holm_adjusted_p_display",
            "eta_squared_descriptive",
            "omega_squared_descriptive",
            "effect_size_note",
        ]
    ].copy()
    feature_comparison["interpretation_guardrail"] = feature_comparison["feature"].map(
        {
            "count": "Mean difference on original count scale; not a prediction score.",
            "serror_rate": "Bounded/endpoint-inflated mean comparison; model is approximate.",
            "src_bytes": "Effect is on log1p scale, not raw-byte mean difference.",
            "duration": "Effect is on full-data log1p scale; read with two-part analysis.",
        }
    )
    tables = {
        "data_validation": audit,
        "descriptive_statistics": descriptive,
        "assumption_checks": assumptions,
        "anova_results": results,
        "posthoc_results": posthoc,
        "sensitivity_analysis": sensitivity,
        "duration_zero_analysis": zero_duration,
        "duration_positive_analysis": duration_positive,
        "summary_results": summary,
        "feature_comparison": feature_comparison,
        "figure_captions": captions,
    }
    for name, table in tables.items():
        table.to_csv(RESULT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    report_figure_dir = FIGURE_DIR / "report"
    report_assets = generate_report_figures(data, posthoc, zero_duration, report_figure_dir)
    latex_figure_dir = ROOT / "TASK 3.2 + 3.3" / "Latex" / "figures"
    latex_assets = generate_report_figures(data, posthoc, zero_duration, latex_figure_dir)
    manifest = {
        "data_path": str(Path(data_path).resolve()),
        "n": len(data),
        "alpha": ALPHA,
        "random_state": RANDOM_STATE,
        "group_order": GROUP_ORDER,
        "features": FEATURES,
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
            "statsmodels": __import__("statsmodels").__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
        "generated_tables": [f"results/{name}.csv" for name in tables],
        "generated_figures": captions["figure"].tolist(),
        "generated_report_figures": [str(path.relative_to(OUTPUT_DIR)).replace("\\", "/") for path in report_assets],
        "latex_report_figures": [str(path.relative_to(ROOT)).replace("\\", "/") for path in latex_assets],
    }
    (RESULT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tables


if __name__ == "__main__":
    output = run_analysis()
    # Keep console output ASCII-safe on Windows shells using legacy code pages.
    print(
        json.dumps(
            {
                "status": "completed",
                "output_directory": str(OUTPUT_DIR),
                "research_questions": len(output["summary_results"]),
            },
            ensure_ascii=True,
        )
    )
