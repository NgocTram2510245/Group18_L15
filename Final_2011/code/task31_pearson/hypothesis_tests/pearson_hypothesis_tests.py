"""Task 3.1.2: kiểm định hai phía Pearson trên NSL-KDD KDDTrain+.

Cách chạy (dùng dữ liệu gốc đã được nhóm cung cấp):
    python pearson_hypothesis_tests.py --raw-zip ../archive.zip

Hoặc (dùng đầu ra Task 2):
    python pearson_hypothesis_tests.py --input /path/to/train_clean.csv

Yêu cầu: pandas, numpy, scipy. Tự tạo các bảng CSV và LaTeX ở outputs/.
"""

from __future__ import annotations
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr

# 9 cặp minh họa: 4 cặp từ những cụm có tương quan rất cao (Task 3.1.1),
# 2 cặp tương quan trung bình/mạnh và 3 cặp đối chiếu âm, yếu, gần 0.
# Đây là phân tích khám phá; nhóm cặp được chọn sau khi khảo sát dữ liệu.
SELECTED_PAIRS = [
    ("P1", "num_compromised", "num_root", "Root access"),
    ("P2", "serror_rate", "srv_serror_rate", "Lỗi SYN"),
    ("P3", "rerror_rate", "srv_rerror_rate", "Lỗi REJ"),
    ("P4", "serror_rate", "dst_host_serror_rate", "Lỗi SYN - đích"),
    ("P5", "same_srv_rate", "dst_host_same_srv_rate", "Cùng dịch vụ"),
    ("P6", "count", "srv_count", "Số kết nối"),
    ("P7", "serror_rate", "rerror_rate", "SYN và REJ"),
    ("P8", "duration", "src_bytes", "Thời lượng/dữ liệu"),
    ("P9", "src_bytes", "dst_bytes", "Lưu lượng 2 chiều"),
]

META_COLS = {
    "protocol_type", "service", "flag", "attack_name", "difficulty",
    "attack_category", "label_binary", "class"
}


def read_data(input_csv: Path | None, raw_zip: Path | None) -> tuple[pd.DataFrame, str]:
    if input_csv is not None:
        if not input_csv.is_file():
            raise FileNotFoundError(f"Không tìm thấy CSV: {input_csv}")
        return pd.read_csv(input_csv), f"CSV: {input_csv.name}"

    if raw_zip is None or not raw_zip.is_file():
        raise FileNotFoundError("Cần truyền --input train_clean.csv hoặc --raw-zip archive.zip")

    # Lấy tên 41 đặc trưng từ header ARFF gốc, nhưng số liệu + nhãn từ TXT
    # vì TXT lưu attack_name và difficulty đúng theo hợp đồng dữ liệu nhóm.
    with zipfile.ZipFile(raw_zip) as zf:
        arff_header = zf.read("KDDTrain+.arff").decode("utf-8")
        names = []
        for line in arff_header.splitlines():
            if line.lower().startswith("@attribute"):
                found = re.match(r"@attribute\s+['\"]?([^\s'\"]+)", line, re.I)
                if found:
                    names.append(found.group(1))
        if len(names) != 42 or names[-1] != "class":
            raise ValueError("Header KDDTrain+.arff không có cấu trúc 41 features + class")
        columns = names[:-1] + ["attack_name", "difficulty"]
        with zf.open("KDDTrain+.txt") as handle:
            frame = pd.read_csv(handle, header=None, names=columns)
    if frame.shape[1] != 43:
        raise ValueError("KDDTrain+.txt phải có 43 cột")
    return frame, "KDDTrain+.txt (archive.zip)"


def fmt_p(p: float) -> str:
    # Với n rất lớn, thư viện số học máy tính có thể làm tròn về 0.
    # p_value=0.0 trong phép tính KHÔNG có nghĩa xác suất toán học =0.
    if p == 0.0:
        return "<1e-300"
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def holm_adjust(pvals: list[float]) -> list[float]:
    """Điều chỉnh Holm cho đúng 9 phép kiểm định báo cáo (không cho toàn bộ 666 cặp)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda k: pvals[k])
    out = [0.0] * m
    running_max = 0.0
    for k, idx in enumerate(order):
        running_max = max(running_max, (m - k) * pvals[idx])
        out[idx] = min(1.0, running_max)
    return out


def latex_p(text: str) -> str:
    if text == "<1e-300":
        return r"$<10^{-300}$"
    if "e" in text:
        a, power = text.split("e")
        return f"${a}\\times10^{{{int(power)}}}$"
    return f"${text}$"


def latex_mono(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description="NSL-KDD Pearson hypothesis testing")
    parser.add_argument("--input", type=Path, help="Đường dẫn train_clean.csv (Task 2)")
    parser.add_argument("--raw-zip", type=Path, help="Đường dẫn archive.zip chứa KDDTrain+.txt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()

    frame, source = read_data(args.input, args.raw_zip)
    numeric = frame.drop(columns=list(META_COLS), errors="ignore").select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("Không tìm được cột số")
    if numeric.isnull().values.any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Dữ liệu Pearson chứa NaN hoặc Inf; cần kiểm tra tiền xử lý")
    constants = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    numeric = numeric.drop(columns=constants)
    n = len(numeric)
    if n < 4:
        raise ValueError("Cỡ mẫu tối thiểu là 4")
    missing = sorted({x for _, a, b, _ in SELECTED_PAIRS for x in (a, b) if x not in numeric.columns})
    if missing:
        raise ValueError(f"Thiếu đặc trưng: {missing}")
    alpha = 0.05
    ci_z = norm.ppf(1 - alpha / 2)

    rows = []
    for pid, x_name, y_name, topic in SELECTED_PAIRS:
        x = numeric[x_name].to_numpy(dtype=float)
        y = numeric[y_name].to_numpy(dtype=float)
        r, p_raw = pearsonr(x, y)
        r, p_raw = float(r), float(p_raw)
        t_value = r * np.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
        z = np.arctanh(r)
        ci_lo = float(np.tanh(z - ci_z / np.sqrt(n - 3)))
        ci_hi = float(np.tanh(z + ci_z / np.sqrt(n - 3)))
        rows.append({
            "id": pid, "feature_x": x_name, "feature_y": y_name, "category": topic,
            "n": n, "r": r, "t_statistic": float(t_value), "df": n - 2,
            "p_raw_numeric": p_raw, "ci95_low": ci_lo, "ci95_high": ci_hi,
        })

    adjusted = holm_adjust([row["p_raw_numeric"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_value"] = fmt_p(row["p_raw_numeric"])
        row["p_holm_9_tests"] = fmt_p(p_adj)
        row["reject_h0_0_05"] = row["p_raw_numeric"] < alpha
        row["reject_h0_holm_0_05"] = p_adj < alpha
        row["p_value_underflow"] = row["p_raw_numeric"] == 0.0
    result = pd.DataFrame(rows)

    # Độ nhạy của P1: chỉ là phép chẩn đoán, KHÔNG thay dữ liệu nghiên cứu chính.
    root_all = numeric[["num_compromised", "num_root"]]
    subset = root_all[(root_all["num_compromised"] <= 10) & (root_all["num_root"] <= 10)]
    root_sensitivity = pd.DataFrame([
        {"scope": "Full KDDTrain+", "n": n,
         "removed_rows": 0, "r": float(root_all.corr().iloc[0, 1])},
        {"scope": "Both root counts <= 10 (diagnostic)", "n": len(subset),
         "removed_rows": n - len(subset), "r": float(subset.corr().iloc[0, 1])},
    ])

    # Kiểm tra lại các con số trong khảo sát ma trận Pearson ban đầu.
    corr = numeric.corr(method="pearson")
    high_pairs = sum(abs(corr.iloc[i, j]) > 0.95 for i in range(len(corr)) for j in range(i + 1, len(corr)))
    assert all(abs(corr.loc[row["feature_x"], row["feature_y"]] - row["r"]) < 1e-10 for row in rows)

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    result.to_csv(outdir / "pearson_selected_tests.csv", index=False, float_format="%.15g")
    root_sensitivity.to_csv(outdir / "pearson_root_sensitivity.csv", index=False, float_format="%.15g")
    checks = {
        "data_source": source, "n": n, "input_cols": frame.shape[1],
        "numerical_nonconstant_features": len(numeric.columns), "constant_features_removed": constants,
        "unique_numerical_pairs": len(numeric.columns) * (len(numeric.columns) - 1) // 2,
        "pairs_with_abs_r_gt_0_95": int(high_pairs),
        "num_missing": int(frame.isna().sum().sum()),
        "num_duplicate_rows": int(frame.duplicated().sum()),
        "number_tests_selected": len(rows),
        "significant_at_0_05": int(result["reject_h0_0_05"].sum()),
        "holm_significant_at_0_05": int(result["reject_h0_holm_0_05"].sum()),
        "note": "P-value numerically zero is display-only '<1e-300'; not exact p=0. Holm applies to 9 selected tests only."
    }
    (outdir / "pearson_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")

    tex_rows = []
    for row in rows:
        pair = r"\shortstack[l]{" + latex_mono(row["feature_x"]) + r"\\ " + latex_mono(row["feature_y"]) + "}"
        verdict = "Bác bỏ $H_0$" if row["reject_h0_0_05"] else "Chưa bác bỏ $H_0$"
        tex_rows.append(f'{row["id"]} & {pair} & {row["r"]:+.6f} & {latex_p(row["p_value"])} & {verdict} '+r"\\")
    (outdir / "pearson_test_table_rows.tex").write_text("\n".join(tex_rows) + "\n", encoding="utf-8")

    root_lines = []
    for _, item in root_sensitivity.iterrows():
        name = "Toàn bộ dữ liệu" if item["removed_rows"] == 0 else "Chỉ giữ 2 số đếm $\\leq 10$"
        root_lines.append(f'{name} & {int(item["n"]):,}'.replace(",", r"\,") +
                          f' & {int(item["removed_rows"]):,} & {item["r"]:.6f} '+r"\\")
    (outdir / "pearson_root_table_rows.tex").write_text("\n".join(root_lines) + "\n", encoding="utf-8")

    # Ghi trọn cấu trúc bảng LaTeX, tránh đặt \input các dòng giữa môi trường tabular.
    table_full = (
        r"\begin{table}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{Kiểm định $H_0:\rho=0$ trên 9 cặp đặc trưng ($\alpha=0.05$)}" + "\n"
        + r"\label{tab:pearson-tests}" + "\n"
        + r"\footnotesize" + "\n"
        + r"\setlength{\tabcolsep}{4pt}" + "\n"
        + r"\renewcommand{\arraystretch}{1.10}" + "\n"
        + r"\begin{tabular}{@{}cp{6.3cm}rcl@{}}" + "\n"
        + r"\toprule" + "\n"
        + r"Mã & Cặp đặc trưng $(X,Y)$ & $r$ & $p$ hai phía & Quyết định\\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(tex_rows) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table}" + "\n"
    )
    (outdir / "pearson_test_table.tex").write_text(table_full, encoding="utf-8")
    root_table_full = (
        r"\begin{table}[htbp]" + "\n"
        + r"\centering" + "\n"
        + r"\caption{Độ nhạy của tương quan \texttt{num\_compromised}--\texttt{num\_root}}" + "\n"
        + r"\label{tab:pearson-root-sensitivity}" + "\n"
        + r"\small" + "\n"
        + r"\begin{tabular}{lrrr}" + "\n"
        + r"\toprule" + "\n"
        + r"Phạm vi khảo sát & $n$ & Số dòng loại khi chẩn đoán & $r$\\" + "\n"
        + r"\midrule" + "\n"
        + "\n".join(root_lines) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table}" + "\n"
    )
    (outdir / "pearson_root_table.tex").write_text(root_table_full, encoding="utf-8")

    print("Data source:", source)
    print(f"n={n}; numerical features={len(corr)}; high pairs={high_pairs}; tests={len(rows)}")
    print(result[["id", "feature_x", "feature_y", "r", "t_statistic", "p_value", "p_holm_9_tests", "reject_h0_0_05"]].to_string(index=False))
    print("\nRoot sensitivity:\n", root_sensitivity.to_string(index=False))
    print("\nOutputs:", outdir.resolve())

if __name__ == "__main__":
    main()
