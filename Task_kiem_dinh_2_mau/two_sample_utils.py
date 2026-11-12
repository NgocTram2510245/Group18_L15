"""Kiểm định giả thuyết hai mẫu độc lập trên dữ liệu NSL-KDD.

Module này dùng toàn bộ ``train_clean.csv`` trên thang đo gốc. Việc lấy mẫu
chỉ phục vụ ECDF, Q-Q plot và Shapiro bổ sung; các kiểm định chính luôn dùng
toàn bộ quan sát hợp lệ.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


ALPHA = 0.05
RANDOM_STATE = 18015
GROUP_ORDER = ["Normal", "Attack"]
GROUP_DISPLAY = {"Normal": "Normal", "Attack": "Attack"}
PALETTE = {"Normal": "#3572A5", "Attack": "#D55E00"}
FEATURES = {
    "RQ1": {
        "feature": "srv_serror_rate",
        "label": "Tỷ lệ lỗi SYN theo dịch vụ",
        "unit": "tỷ lệ [0, 1]",
        "question": "Tỷ lệ lỗi SYN trung bình có khác biệt giữa Normal và Attack hay không?",
        "practical": "Trong mẫu NSL-KDD, nhóm Attack có tỷ lệ lỗi SYN theo dịch vụ cao hơn rõ rệt, phù hợp với hành vi tạo kết nối thất bại/không hoàn tất bắt tay TCP thường gặp ở một số kiểu tấn công.",
    },
    "RQ2": {
        "feature": "srv_rerror_rate",
        "label": "Tỷ lệ lỗi REJ theo dịch vụ",
        "unit": "tỷ lệ [0, 1]",
        "question": "Tỷ lệ lỗi REJ trung bình có khác biệt giữa Normal và Attack hay không?",
        "practical": "Trong mẫu NSL-KDD, nhóm Attack có tỷ lệ kết nối bị từ chối theo dịch vụ cao hơn; đặc trưng này phản ánh một khía cạnh khác của lỗi kết nối so với lỗi SYN.",
    },
    "RQ3": {
        "feature": "count",
        "label": "Số kết nối gần đây",
        "unit": "kết nối",
        "question": "Số lượng kết nối gần đây trung bình có khác biệt giữa Normal và Attack hay không?",
        "practical": "Nhóm Attack có số kết nối gần đây trung bình cao hơn, cho thấy cường độ/lặp lại kết nối là tín hiệu hành vi đáng chú ý trong bộ dữ liệu này.",
    },
}


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    module_dir: Path
    data_path: Path
    results_dir: Path
    figures_dir: Path


def resolve_paths(repo_root: Path | None = None) -> Paths:
    """Xác định đường dẫn tương đối trong repository, không hardcode máy người dùng."""
    module_dir = Path(__file__).resolve().parent
    root = Path(repo_root).resolve() if repo_root else module_dir.parent.resolve()
    preferred = (
        root
        / "00_DU_LIEU_CHUNG_CHO_TEAM"
        / "00_DU_LIEU_CHUNG"
        / "data"
        / "processed"
        / "train_clean.csv"
    )
    if preferred.exists():
        data_path = preferred
    else:
        matches = sorted(root.rglob("train_clean.csv"))
        if not matches:
            raise FileNotFoundError(
                "Không tìm thấy train_clean.csv. Xem two_sample_testing/README.md."
            )
        data_path = matches[0]
    results = module_dir / "results"
    figures = module_dir / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return Paths(root, module_dir, data_path, results, figures)


def load_and_validate_data(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc cột cần thiết và kiểm tra hợp đồng nhãn/dữ liệu.

    Không loại ngoại lệ. Bản ghi chỉ bị loại nếu một biến phân tích thiếu hoặc
    không hữu hạn; số lượng này được ghi rõ trong bảng kiểm tra.
    """
    feature_names = [cfg["feature"] for cfg in FEATURES.values()]
    required = ["attack_category", "label_binary", *feature_names]
    df = pd.read_csv(paths.data_path, usecols=required)
    checks: list[dict[str, object]] = []

    checks.append({"check": "source_rows", "value": len(df), "status": "PASS"})
    checks.append(
        {
            "check": "duplicate_on_analysis_columns_not_removed",
            "value": int(df.duplicated().sum()),
            "status": "INFO",
        }
    )
    allowed_categories = {"Normal", "DoS", "Probe", "R2L", "U2R"}
    observed = set(df["attack_category"].dropna().unique())
    checks.append(
        {
            "check": "attack_category_domain",
            "value": ", ".join(sorted(observed)),
            "status": "PASS" if observed == allowed_categories else "FAIL",
        }
    )
    label_values = set(pd.to_numeric(df["label_binary"], errors="coerce").dropna().unique())
    checks.append(
        {
            "check": "label_binary_domain",
            "value": ", ".join(map(str, sorted(label_values))),
            "status": "PASS" if label_values == {0, 1} else "FAIL",
        }
    )
    mapping_ok = (
        ((df["attack_category"] == "Normal") & (df["label_binary"] == 0))
        | ((df["attack_category"] != "Normal") & (df["label_binary"] == 1))
    ).all()
    checks.append({"check": "normal_attack_mapping", "value": bool(mapping_ok), "status": "PASS" if mapping_ok else "FAIL"})

    for feature in feature_names:
        converted = pd.to_numeric(df[feature], errors="coerce")
        missing = int(converted.isna().sum())
        nonfinite = int((~np.isfinite(converted.dropna())).sum())
        checks.extend(
            [
                {"check": f"{feature}_missing", "value": missing, "status": "PASS" if missing == 0 else "WARN"},
                {"check": f"{feature}_nonfinite", "value": nonfinite, "status": "PASS" if nonfinite == 0 else "WARN"},
            ]
        )
        df[feature] = converted

    for feature in ("srv_serror_rate", "srv_rerror_rate"):
        in_range = df[feature].dropna().between(0, 1).all()
        checks.append({"check": f"{feature}_range_0_1", "value": bool(in_range), "status": "PASS" if in_range else "FAIL"})
    count_valid = (df["count"].dropna() >= 0).all()
    checks.append({"check": "count_nonnegative", "value": bool(count_valid), "status": "PASS" if count_valid else "FAIL"})

    df["group"] = np.where(df["label_binary"].eq(0), "Normal", "Attack")
    before = len(df)
    valid = np.isfinite(df[feature_names]).all(axis=1) & df["group"].isin(GROUP_ORDER)
    df = df.loc[valid].copy()
    checks.append({"check": "rows_removed_invalid", "value": before - len(df), "status": "PASS" if before == len(df) else "WARN"})
    for group, n in df["group"].value_counts().items():
        checks.append({"check": f"n_{group}", "value": int(n), "status": "PASS" if n >= 30 else "WARN"})

    validation = pd.DataFrame(checks)
    if (validation["status"] == "FAIL").any():
        failed = validation.loc[validation["status"] == "FAIL", "check"].tolist()
        raise ValueError(f"Dữ liệu không đạt kiểm tra bắt buộc: {failed}")
    return df, validation


def mean_ci(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """Khoảng tin cậy t hai phía cho trung bình của một nhóm."""
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(np.mean(x))
    if n < 2:
        return np.nan, np.nan
    margin = stats.t.ppf(1 - alpha / 2, n - 1) * stats.sem(x)
    return float(mean - margin), float(mean + margin)


def format_p_value(value: float) -> str:
    """Định dạng p-value mà không báo sai rằng p bằng 0 do underflow số học."""
    return "<1e-300" if value < 1e-300 else f"{value:.6g}"


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê mô tả theo biến và nhóm, giữ số thực đầy đủ trong CSV."""
    rows: list[dict[str, object]] = []
    for rq, cfg in FEATURES.items():
        feature = cfg["feature"]
        for group in GROUP_ORDER:
            x = df.loc[df["group"].eq(group), feature].to_numpy(float)
            q1, q3 = np.quantile(x, [0.25, 0.75])
            ci_low, ci_high = mean_ci(x)
            rows.append(
                {
                    "rq": rq,
                    "feature": feature,
                    "group": group,
                    "n": x.size,
                    "mean": np.mean(x),
                    "median": np.median(x),
                    "std": np.std(x, ddof=1),
                    "variance": np.var(x, ddof=1),
                    "minimum": np.min(x),
                    "q1": q1,
                    "q3": q3,
                    "maximum": np.max(x),
                    "iqr": q3 - q1,
                    "zero_proportion": np.mean(x == 0),
                    "one_proportion": np.mean(x == 1) if feature.endswith("rate") else np.nan,
                    "skewness": stats.skew(x, bias=False),
                    "mean_ci95_low": ci_low,
                    "mean_ci95_high": ci_high,
                    "ci_method": "t interval; large-sample inference, subject to independence",
                }
            )
    return pd.DataFrame(rows)


def welch_test(x: Iterable[float], y: Iterable[float], alpha: float = ALPHA) -> dict[str, float]:
    """Welch t-test hai phía và CI cho chênh lệch mean(x)-mean(y)."""
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    n1, n2 = a.size, b.size
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    v1, v2 = s1 / n1, s2 / n2
    se = np.sqrt(v1 + v2)
    diff = m1 - m2
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    t_stat = diff / se
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    critical = stats.t.ppf(1 - alpha / 2, df)
    return {
        "mean_difference": float(diff),
        "standard_error": float(se),
        "t_statistic": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "ci95_low": float(diff - critical * se),
        "ci95_high": float(diff + critical * se),
    }


def effect_sizes(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Cohen d/Hedges g mô tả và CI xấp xỉ của g.

    Hai chỉ số dùng SD gộp nên được trình bày như chuẩn hóa mô tả; chúng không
    biến Welch thành mô hình phương sai bằng nhau.
    """
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(x) - np.mean(y)) / pooled
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    g = correction * d
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2 - 2)))
    se_g = correction * se_d
    return {
        "cohen_d": float(d),
        "hedges_g": float(g),
        "hedges_g_ci95_low_approx": float(g - 1.96 * se_g),
        "hedges_g_ci95_high_approx": float(g + 1.96 * se_g),
        "effect_ci_note": "Normal approximation; descriptive pooled-SD effect size",
    }


def assumption_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Brown-Forsythe, mất cân bằng, ngoại lệ, độ lệch và Shapiro bổ sung."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[dict[str, object]] = []
    for rq, cfg in FEATURES.items():
        feature = cfg["feature"]
        x = df.loc[df["group"].eq("Normal"), feature].to_numpy(float)
        y = df.loc[df["group"].eq("Attack"), feature].to_numpy(float)
        levene = stats.levene(x, y, center="median")
        var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
        row: dict[str, object] = {
            "rq": rq,
            "feature": feature,
            "n_normal": len(x),
            "n_attack": len(y),
            "n_ratio_max_min": max(len(x), len(y)) / min(len(x), len(y)),
            "variance_normal": var_x,
            "variance_attack": var_y,
            "variance_ratio_max_min": max(var_x, var_y) / min(var_x, var_y) if min(var_x, var_y) > 0 else np.inf,
            "brown_forsythe_statistic": levene.statistic,
            "brown_forsythe_p_value": levene.pvalue,
            "independence_verifiable": False,
            "independence_warning": "Không có session/host ID để kiểm chứng độc lập giữa các bản ghi.",
        }
        for group, values in (("normal", x), ("attack", y)):
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = q3 - q1
            outlier = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
            sample = values if len(values) <= 5000 else rng.choice(values, 5000, replace=False)
            shapiro = stats.shapiro(sample)
            row[f"skewness_{group}"] = stats.skew(values, bias=False)
            row[f"iqr_outlier_proportion_{group}"] = np.mean(outlier)
            row[f"shapiro_sample_n_{group}"] = len(sample)
            row[f"shapiro_statistic_{group}"] = shapiro.statistic
            row[f"shapiro_p_value_{group}"] = shapiro.pvalue
        row["method_warning"] = (
            "Biến tỷ lệ có khối xác suất tại biên; Welch suy luận về trung bình nhưng mô hình chuẩn chỉ là xấp xỉ."
            if feature.endswith("rate")
            else "Biến đếm lệch và có ngoại lệ; dùng Welch nhờ cỡ mẫu lớn, kèm phân tích độ nhạy robust/rank-based."
        )
        row["brown_forsythe_p_display"] = format_p_value(float(levene.pvalue))
        rows.append(row)
    return pd.DataFrame(rows)


def run_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chạy Welch, effect size và hai phân tích độ nhạy trên toàn bộ dữ liệu."""
    primary: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    for rq, cfg in FEATURES.items():
        feature = cfg["feature"]
        x = df.loc[df["group"].eq("Normal"), feature].to_numpy(float)
        y = df.loc[df["group"].eq("Attack"), feature].to_numpy(float)
        result = welch_test(x, y)
        primary.append(
            {
                "rq": rq,
                "feature": feature,
                "comparison": "Normal - Attack",
                "selected_test": "Welch independent two-sample t-test (two-sided)",
                "n_normal": len(x),
                "n_attack": len(y),
                "mean_normal": np.mean(x),
                "mean_attack": np.mean(y),
                **result,
                "alpha": ALPHA,
            }
        )
        effects.append({"rq": rq, "feature": feature, **effect_sizes(x, y)})

        mwu = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        rbc = 2 * mwu.statistic / (len(x) * len(y)) - 1
        sensitivity.append(
            {
                "rq": rq,
                "feature": feature,
                "method": "Mann-Whitney U (two-sided, asymptotic)",
                "statistic": mwu.statistic,
                "df": np.nan,
                "p_value": mwu.pvalue,
                "effect_or_difference": rbc,
                "estimand_note": "Khác biệt phân phối/probability of superiority; không phải kiểm định trung bình.",
            }
        )
        yuen = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided", trim=0.2)
        sensitivity.append(
            {
                "rq": rq,
                "feature": feature,
                "method": "Yuen-Welch t-test, 20% trimmed means",
                "statistic": yuen.statistic,
                "df": yuen.df,
                "p_value": yuen.pvalue,
                "effect_or_difference": stats.trim_mean(x, 0.2) - stats.trim_mean(y, 0.2),
                "estimand_note": "Chênh lệch trung bình cắt ngọn 20%; không đồng nhất với trung bình gốc.",
            }
        )

    primary_df = pd.DataFrame(primary)
    _, adjusted, _, _ = multipletests(primary_df["p_value"].to_numpy(), alpha=ALPHA, method="holm")
    primary_df["holm_adjusted_p_value"] = adjusted
    primary_df["reject_h0_raw"] = primary_df["p_value"] < ALPHA
    primary_df["reject_h0_holm"] = primary_df["holm_adjusted_p_value"] < ALPHA
    primary_df["p_value_underflow"] = primary_df["p_value"].eq(0)
    primary_df["holm_p_underflow"] = primary_df["holm_adjusted_p_value"].eq(0)
    primary_df["p_value_display"] = primary_df["p_value"].map(format_p_value)
    primary_df["holm_p_display"] = primary_df["holm_adjusted_p_value"].map(format_p_value)
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df["p_value_display"] = sensitivity_df["p_value"].map(format_p_value)
    sensitivity_df["p_value_underflow"] = sensitivity_df["p_value"].eq(0)
    return primary_df, pd.DataFrame(effects), sensitivity_df


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    df: pd.DataFrame,
    descriptive: pd.DataFrame,
    primary: pd.DataFrame,
    effects: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    """Tạo hình học thuật; sampling chỉ dùng cho ECDF và Q-Q plot."""
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    rng = np.random.default_rng(RANDOM_STATE)
    captions: list[dict[str, str]] = []

    for rq, cfg in FEATURES.items():
        feature, label, unit = cfg["feature"], cfg["label"], cfg["unit"]
        out = figures_dir / feature
        plot_df = df[["group", feature]].copy()

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        sns.boxplot(data=plot_df, x="group", y=feature, order=GROUP_ORDER, hue="group", palette=PALETTE, showfliers=False, legend=False, ax=ax)
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_DISPLAY[group] for group in GROUP_ORDER])
        ax.set(title=f"{label}: phân bố theo nhóm", xlabel="Nhóm kết nối", ylabel=f"{label} ({unit})")
        _save_figure(fig, out / "01_boxplot")
        captions.append({"rq": rq, "figure": f"figures/{feature}/01_boxplot.png", "caption": "Boxplot dùng toàn bộ dữ liệu; điểm ngoại lệ chỉ ẩn khỏi hình để dễ đọc, không bị loại khỏi kiểm định."})

        d = descriptive.loc[descriptive["feature"].eq(feature)].set_index("group").loc[GROUP_ORDER]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        means = d["mean"].to_numpy()
        lower = means - d["mean_ci95_low"].to_numpy()
        upper = d["mean_ci95_high"].to_numpy() - means
        display_groups = [GROUP_DISPLAY[group] for group in GROUP_ORDER]
        ax.errorbar(display_groups, means, yerr=np.vstack([lower, upper]), fmt="o", ms=8, capsize=6, color="#222222", ecolor="#444444")
        for i, (mean, n) in enumerate(zip(means, d["n"])):
            ax.annotate(f"n={int(n):,}", (i, mean), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=f"Trung bình và khoảng tin cậy 95%: {label}", xlabel="Nhóm kết nối", ylabel=f"Trung bình ({unit})")
        _save_figure(fig, out / "02_mean_ci95")
        captions.append({"rq": rq, "figure": f"figures/{feature}/02_mean_ci95.png", "caption": "Điểm là trung bình từng nhóm; thanh sai số là CI 95% của trung bình nhóm, không phải độ lệch chuẩn hay CI của chênh lệch."})

        sampled_parts = []
        for group in GROUP_ORDER:
            part = plot_df.loc[plot_df["group"].eq(group)]
            if len(part) > 5000:
                idx = rng.choice(part.index.to_numpy(), 5000, replace=False)
                part = part.loc[idx]
            sampled_parts.append(part)
        ecdf_df = pd.concat(sampled_parts, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        sns.ecdfplot(data=ecdf_df, x=feature, hue="group", hue_order=GROUP_ORDER, palette=PALETTE, ax=ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Nhóm kết nối")
            for text, group in zip(legend.get_texts(), GROUP_ORDER):
                text.set_text(GROUP_DISPLAY[group])
        ax.set(title=f"Hàm phân phối tích lũy thực nghiệm theo nhóm: {label}", xlabel=f"{label} ({unit})", ylabel="Xác suất tích lũy")
        _save_figure(fig, out / "03_ecdf")
        captions.append({"rq": rq, "figure": f"figures/{feature}/03_ecdf.png", "caption": "ECDF dùng tối đa 5.000 quan sát/nhóm với random_state=18015 để hình rõ và tệp gọn; kiểm định vẫn dùng toàn bộ dữ liệu."})

        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharex=False, sharey=False)
        for ax, group in zip(axes, GROUP_ORDER):
            values = plot_df.loc[plot_df["group"].eq(group), feature].to_numpy(float)
            if len(values) > 1000:
                values = rng.choice(values, 1000, replace=False)
            stats.probplot(values, dist="norm", plot=ax)
            sample_size = f"{len(values):,}".replace(",", ".")
            ax.set_title(f"{GROUP_DISPLAY[group]} ({sample_size} quan sát/nhóm)")
            ax.set_xlabel("Phân vị lý thuyết")
            ax.set_ylabel("Giá trị quan sát đã sắp xếp")
            ax.get_lines()[0].set_markerfacecolor(PALETTE[group])
            ax.get_lines()[0].set_markeredgecolor(PALETTE[group])
        fig.suptitle(f"Đồ thị Q-Q: {label}")
        fig.tight_layout()
        _save_figure(fig, out / "04_qq_plot")
        captions.append({"rq": rq, "figure": f"figures/{feature}/04_qq_plot.png", "caption": "Q-Q plot dùng tối đa 1.000 quan sát/nhóm để chẩn đoán hình dạng; sai lệch rõ không được che khuất bởi số điểm quá lớn."})

        mass_rows = []
        for group in GROUP_ORDER:
            values = plot_df.loc[plot_df["group"].eq(group), feature]
            mass_rows.append({"group": group, "mass": "Bằng 0", "proportion": (values == 0).mean()})
            if feature.endswith("rate"):
                mass_rows.append({"group": group, "mass": "Bằng 1", "proportion": (values == 1).mean()})
        mass_df = pd.DataFrame(mass_rows)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        sns.barplot(data=mass_df, x="group", y="proportion", hue="mass", order=GROUP_ORDER, ax=ax, edgecolor="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_DISPLAY[group] for group in GROUP_ORDER])
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Giá trị")
        ax.set(title=f"Khối xác suất tại biên: {label}", xlabel="Nhóm kết nối", ylabel="Tỷ lệ quan sát", ylim=(0, 1))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        _save_figure(fig, out / "05_boundary_mass")
        captions.append({"rq": rq, "figure": f"figures/{feature}/05_boundary_mass.png", "caption": "Tỷ lệ giá trị bằng 0 (và bằng 1 với biến tỷ lệ) cho thấy dữ liệu rời rạc/tập trung tại biên, là giới hạn của xấp xỉ ANOVA/t-test chuẩn."})

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for ax, (rq, cfg) in zip(axes, FEATURES.items()):
        row = primary.loc[primary["rq"].eq(rq)].iloc[0]
        diff = row["mean_difference"]
        ax.errorbar(diff, 0, xerr=[[diff - row["ci95_low"]], [row["ci95_high"] - diff]], fmt="o", capsize=6, color="#222222")
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
        rq_number = rq.removeprefix("RQ")
        ax.set_title(f"Câu hỏi {rq_number}: {cfg['feature']}")
        ax.set_xlabel(f"Trung bình (Normal) - Trung bình (Attack)\n({cfg['unit']})")
    fig.suptitle("Chênh lệch trung bình và khoảng tin cậy 95% theo Welch")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "comparison" / "mean_difference_forest")
    captions.append({"rq": "All", "figure": "figures/comparison/mean_difference_forest.png", "caption": "Mỗi panel dùng đơn vị riêng; đường 0 biểu thị không có chênh lệch trung bình. Không so độ dài tuyệt đối giữa các panel khác đơn vị."})

    ordered = effects.set_index("rq").loc[list(FEATURES)]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    g = ordered["hedges_g"].to_numpy()
    lo = ordered["hedges_g_ci95_low_approx"].to_numpy()
    hi = ordered["hedges_g_ci95_high_approx"].to_numpy()
    y = np.arange(len(ordered))
    ax.errorbar(g, y, xerr=np.vstack([g - lo, hi - g]), fmt="o", capsize=5, color="#222222")
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"Câu hỏi {rq.removeprefix('RQ')}: {FEATURES[rq]['feature']}" for rq in ordered.index])
    ax.set(title="Hedges g: chênh lệch chuẩn hóa Normal - Attack", xlabel="Hedges g (KTC 95% xấp xỉ)", ylabel="")
    _save_figure(fig, figures_dir / "comparison" / "effect_size_forest")
    captions.append({"rq": "All", "figure": "figures/comparison/effect_size_forest.png", "caption": "Hedges g là chỉ số mô tả chuẩn hóa bằng SD gộp; CI dùng xấp xỉ chuẩn và cần đọc cùng chênh lệch trên đơn vị gốc."})
    return pd.DataFrame(captions)


def build_summary(primary: pd.DataFrame, effects: pd.DataFrame, assumptions: pd.DataFrame) -> pd.DataFrame:
    """Bảng tóm tắt một dòng cho mỗi câu hỏi nghiên cứu."""
    summary = primary.merge(effects, on=["rq", "feature"]).merge(
        assumptions[["rq", "feature", "brown_forsythe_p_value", "variance_ratio_max_min", "method_warning"]],
        on=["rq", "feature"],
    )
    summary["conclusion"] = np.where(
        summary["reject_h0_holm"],
        "Bác bỏ H0 sau hiệu chỉnh Holm: trung bình Normal và Attack khác nhau.",
        "Chưa đủ bằng chứng bác bỏ H0 sau hiệu chỉnh Holm.",
    )
    return summary


def write_interpretation(
    paths: Paths,
    descriptive: pd.DataFrame,
    primary: pd.DataFrame,
    effects: pd.DataFrame,
    sensitivity: pd.DataFrame,
    assumptions: pd.DataFrame,
) -> None:
    """Sinh diễn giải tiếng Việt trực tiếp từ kết quả, không hardcode số liệu."""
    lines = [
        "# Diễn giải kiểm định hai mẫu độc lập",
        "",
        "Tập huấn luyện gốc được dùng cho suy luận; tập test không tham gia kiểm định. Chênh lệch được định nghĩa là mean(Normal) - mean(Attack).",
        "",
    ]
    for rq, cfg in FEATURES.items():
        f = cfg["feature"]
        p = primary.loc[primary["rq"].eq(rq)].iloc[0]
        e = effects.loc[effects["rq"].eq(rq)].iloc[0]
        a = assumptions.loc[assumptions["rq"].eq(rq)].iloc[0]
        d = descriptive.loc[descriptive["feature"].eq(f)].set_index("group")
        sens = sensitivity.loc[sensitivity["rq"].eq(rq)]
        lines.extend(
            [
                f"## {rq} — {f}",
                "",
                f"Normal có trung bình {d.loc['Normal', 'mean']:.6g} (n={int(d.loc['Normal', 'n']):,}), còn Attack có trung bình {d.loc['Attack', 'mean']:.6g} (n={int(d.loc['Attack', 'n']):,}). Chênh lệch Normal - Attack là {p['mean_difference']:.6g}, CI 95% [{p['ci95_low']:.6g}; {p['ci95_high']:.6g}].",
                f"Welch t({p['df']:.2f}) = {p['t_statistic']:.4f}, p {p['p_value_display']}; p hiệu chỉnh Holm {p['holm_p_display']}. Hedges g = {e['hedges_g']:.4f}. Brown-Forsythe p {a['brown_forsythe_p_display']}, tỷ số phương sai lớn/nhỏ={a['variance_ratio_max_min']:.3f}.",
                ("Quyết định: bác bỏ H0 sau hiệu chỉnh Holm. " if p["reject_h0_holm"] else "Quyết định: chưa đủ bằng chứng bác bỏ H0 sau hiệu chỉnh Holm. ") + cfg["practical"],
                f"Phân tích độ nhạy: {sens.iloc[0]['method']} có p {sens.iloc[0]['p_value_display']}; {sens.iloc[1]['method']} có p {sens.iloc[1]['p_value_display']}. Hai kết quả này hỗ trợ đánh giá độ bền nhưng không cùng estimand với trung bình gốc.",
                f"Cảnh báo: {a['method_warning']} Không có metadata phiên/host nên không thể kiểm chứng đầy đủ tính độc lập; vì vậy kết quả là liên hệ thống kê, không phải quan hệ nhân quả.",
                "",
            ]
        )

    pearson_file = paths.repo_root / "03_TASK_3" / "3.1.1_Pearson" / "outputs" / "high_correlation_pairs.csv"
    if pearson_file.exists():
        corr = pd.read_csv(pearson_file)
        lines.extend(
            [
                "## Liên hệ với Pearson, ANOVA và Machine Learning",
                "",
                "Pearson đo liên hệ tuyến tính giữa hai biến định lượng, còn Welch ở đây so sánh trung bình của một biến giữa nhãn Normal và Attack. Vì trả lời hai câu hỏi khác nhau, việc biến được chọn khác nhau không làm kết quả mâu thuẫn.",
            ]
        )
        for feature, peer in (("srv_serror_rate", "dst_host_srv_serror_rate"), ("srv_rerror_rate", "dst_host_srv_rerror_rate")):
            match = corr[((corr.iloc[:, 0] == feature) & (corr.iloc[:, 1] == peer)) | ((corr.iloc[:, 1] == feature) & (corr.iloc[:, 0] == peer))]
            if not match.empty:
                numeric_cols = match.select_dtypes(include=np.number).columns
                r_value = match.iloc[0][numeric_cols[0]] if len(numeric_cols) else np.nan
                lines.append(f"Trong kết quả Pearson hiện có, {feature} liên hệ rất mạnh với {peer} (r≈{r_value:.4f}); điều này cảnh báo tính dư thừa khi đưa đồng thời các biến tương quan cao vào mô hình.")
    anova_file = paths.repo_root / "TASK 3.2 + 3.3" / "Code" / "results" / "anova_results.csv"
    if anova_file.exists():
        anova = pd.read_csv(anova_file)
        count_row = anova.loc[anova["feature"].eq("count")]
        if not count_row.empty:
            row = count_row.iloc[0]
            lines.append(
                f"ANOVA hiện có so sánh count giữa năm nhóm Normal, DoS, Probe, R2L và U2R, cho Welch F({row['df_numerator']:.0f}, {row['df_denominator']:.2f})={row['F_statistic']:.2f}, p {row['raw_p_display'].replace(' ', '')}. Kiểm định hai mẫu ở đây gộp bốn nhóm tấn công thành Attack; do đó nó trả lời một tương phản khác và chịu ảnh hưởng bởi tỷ trọng từng loại tấn công. Hai phân tích dùng cùng bản ghi nên không độc lập về thông tin."
            )
    lines.extend(
        [
            "",
            "Ý nghĩa thống kê không đồng nghĩa khả năng dự đoán tốt: p-value chịu ảnh hưởng mạnh của cỡ mẫu, không phản ánh trực tiếp độ phân tách từng quan sát, tương tác, phi tuyến hay hiệu năng ngoài mẫu. Các kết quả này hỗ trợ giải thích dữ liệu và chọn đặc trưng có hiểu biết, nhưng hiệu năng Machine Learning vẫn phải đánh giá trên test set bằng metric phù hợp.",
            "",
            "Không diễn giải p-value hiển thị <1e-300 là bằng 0; đó là giới hạn biểu diễn số. Không kết luận nhãn tấn công gây ra thay đổi của đặc trưng.",
        ]
    )
    (paths.results_dir / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(repo_root: Path | None = None, create_plots: bool = True) -> dict[str, pd.DataFrame]:
    """Chạy toàn bộ pipeline, ghi bảng/hình và trả về các DataFrame kết quả."""
    paths = resolve_paths(repo_root)
    df, validation = load_and_validate_data(paths)
    descriptive = descriptive_statistics(df)
    assumptions = assumption_checks(df)
    primary, effects, sensitivity = run_tests(df)
    summary = build_summary(primary, effects, assumptions)
    captions = create_figures(df, descriptive, primary, effects, paths.figures_dir) if create_plots else pd.DataFrame()

    tables = {
        "data_validation": validation,
        "descriptive_statistics": descriptive,
        "assumption_checks": assumptions,
        "two_sample_results": primary,
        "effect_sizes": effects,
        "sensitivity_analysis": sensitivity,
        "summary_results": summary,
        "figure_captions": captions,
    }
    for name, table in tables.items():
        table.to_csv(paths.results_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_interpretation(paths, descriptive, primary, effects, sensitivity, assumptions)

    manifest = {
        "data_path": str(paths.data_path.relative_to(paths.repo_root)),
        "rows_analyzed": int(len(df)),
        "alpha": ALPHA,
        "random_state": RANDOM_STATE,
        "main_test": "Welch independent two-sample t-test, two-sided",
        "multiple_testing": "Holm correction across 3 primary RQs",
        "plot_sampling_only": {"ecdf_per_group_max": 5000, "qq_per_group_max": 1000},
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
    }
    (paths.results_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return tables


if __name__ == "__main__":
    results = run_analysis()
    console_columns = [
        "rq",
        "feature",
        "mean_normal",
        "mean_attack",
        "mean_difference",
        "t_statistic",
        "df",
        "p_value_display",
        "holm_p_display",
        "hedges_g",
    ]
    print(results["summary_results"][console_columns].to_string(index=False))

