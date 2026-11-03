# ANOVA cho NSL-KDD

Phân tích độc lập này trả lời bốn câu hỏi về `count`, `serror_rate`,
`src_bytes` và `duration` giữa năm nhóm `Normal, DoS, Probe, R2L, U2R`.
Phân tích chỉ dùng `train_clean.csv`; không dùng test set, SMOTE, one-hot encoding
hoặc standardization.

## Phương pháp chính

| RQ | Biến phân tích | Kiểm định chính | Lý do chính |
|---|---|---|---|
| RQ1 | `count` | Welch one-way ANOVA | phương sai/cỡ mẫu khác nhau; biến đếm lệch |
| RQ2 | `serror_rate` | Welch one-way ANOVA | cần so sánh mean trên thang 0–1; dồn biên được cảnh báo |
| RQ3 | `log1p(src_bytes)` | Welch one-way ANOVA | lệch phải và cực trị rất lớn |
| RQ4 | `log1p(duration)` | Welch one-way ANOVA | zero-inflated và lệch mạnh |

Mỗi RQ còn có classical ANOVA và Kruskal–Wallis như phân tích độ nhạy.
Kruskal–Wallis là kiểm định phân phối theo hạng, không phải phép kiểm định tương
đương về trung bình. Khi omnibus có ý nghĩa, hậu kiểm dùng Games–Howell cho 10
cặp. Bốn p-value omnibus được hiệu chỉnh Holm riêng với hiệu chỉnh trong hậu kiểm.

RQ4 có thêm hai mô tả bổ sung: (1) tỷ lệ `duration=0` và Wilson CI 95% theo
nhóm, không thực hiện Chi-square trong phần ANOVA; (2) Welch trên
`log1p(duration)` với các kết nối dương. Phần (2) là kết quả có điều kiện,
không thay thế phân tích toàn bộ dữ liệu.

## Chạy lại

Từ thư mục gốc dự án:

```powershell
.\anova_analysis\run_analysis.ps1
```

Script trên kiểm tra môi trường, chạy toàn bộ pipeline, chạy unit test và xác
nhận các bảng/hình bắt buộc. Có thể chỉ chạy module Python bằng lệnh
`.\.venv\Scripts\python.exe .\anova_analysis\anova_utils.py`.

Hoặc mở `anova_analysis.ipynb`, chọn kernel dùng môi trường trong
`.venv`, rồi chạy **Restart Kernel and Run All**. Notebook gọi cùng một module nên
kết quả script và notebook là một nguồn duy nhất.

Nếu chưa có môi trường:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\anova_analysis\anova_utils.py
```

Dữ liệu phải ở:

```text
00_DU_LIEU_CHUNG_CHO_TEAM/00_DU_LIEU_CHUNG/data/processed/train_clean.csv
```

## Đầu ra

- `results/descriptive_statistics.csv`: thống kê mô tả theo 5 nhóm.
- `results/assumption_checks.csv`: Brown–Forsythe, shape, Shapiro mẫu, outlier và mất cân bằng.
- `results/anova_results.csv`: kết quả chính và Holm correction.
- `results/posthoc_results.csv`: Games–Howell, chênh lệch/CI/p hiệu chỉnh.
- `results/sensitivity_analysis.csv`: classical ANOVA và Kruskal–Wallis.
- `results/duration_zero_analysis.csv`: tỷ lệ zero và Wilson CI của RQ4 (mô tả).
- `results/duration_positive_analysis.csv`: phần dương có điều kiện của RQ4.
- `results/summary_results.csv`: bảng tổng hợp bốn RQ.
- `results/feature_comparison.csv`: so sánh ý nghĩa/effect size trên đúng thang, không xếp hạng dự đoán.
- `results/interpretation.md`: diễn giải tiếng Việt tự sinh từ kết quả.
- `results/figure_captions.csv`: caption, phạm vi kết luận và giới hạn từng hình.
- `figures/<feature>/`: PNG 300 DPI và PDF vector.
- `figures/report/`: đúng 5 hình tuyển chọn được dùng trong báo cáo LaTeX.
- `Latex/ANOVA/figures/`: bản đồng bộ của 5 hình báo cáo.

Pipeline dùng `random_state=18015`. Mẫu tối đa 5.000 quan sát/nhóm cho ECDF và
1.000 quan sát/nhóm cho Q--Q chỉ giúp hình gọn, ít chồng lấp và có kích thước
hợp lý; các nhóm nhỏ được giữ nguyên. Toàn bộ 125.973 bản ghi vẫn được dùng cho
thống kê mô tả, CI, kiểm định tổng thể, effect size và hậu kiểm.

Các CI trung bình trong bảng/hình là two-sided t-CI theo xấp xỉ CLT. Với phân
phối rất lệch/cực trị nặng, đặc biệt U2R chỉ có 52 quan sát, cần đọc CI cùng ECDF,
boxplot, Q–Q và cảnh báo giả định. Không có metadata phiên/host/time đủ để xác minh
tính độc lập; đây là hạn chế suy luận quan trọng.

