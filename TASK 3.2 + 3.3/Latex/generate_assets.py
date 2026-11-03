"""Wrapper tương thích: tái tạo hình báo cáo từ pipeline trong anova_analysis."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TASK 3.2 + 3.3" / "Code"))

from anova_utils import DATA_PATH, duration_two_part, load_and_validate_data, run_omnibus  # noqa: E402
from report_assets import generate_report_figures  # noqa: E402


def main() -> None:
    data, _ = load_and_validate_data(DATA_PATH)
    _, posthoc, _ = run_omnibus(data)
    zero_duration, _ = duration_two_part(data)
    output_dir = Path(__file__).parent / "figures"
    outputs = generate_report_figures(data, posthoc, zero_duration, output_dir)
    print(f"Generated {len(outputs)} report assets in {output_dir}")


if __name__ == "__main__":
    main()

