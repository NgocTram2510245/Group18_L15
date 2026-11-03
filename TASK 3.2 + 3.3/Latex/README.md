# Nội dung LaTeX cho mục 3.4, 3.5, 4.4 và 4.5

Thư mục này chứa phần báo cáo được tích hợp vào dự án LaTeX chính:

- `03_4_wilson_method.tex`: phương pháp khoảng tin cậy Wilson, gồm các mục 3.4.1--3.4.4.
- `03_5_anova_method.tex`: phương pháp ANOVA/Welch, gồm các mục 3.5.1--3.5.6.
- `04_4_wilson.tex`: kết quả khoảng tin cậy Wilson 95% cho năm nhóm.
- `04_5_anova.tex`: kết quả Welch ANOVA, kiểm tra giả định, effect size,
  Games-Howell và phân tích hai phần cho `duration`.
- `generate_assets.py`: tái tạo các hình đã tuyển chọn từ dữ liệu và CSV kết quả.
- `figures/`: hình PNG 300 DPI và PDF vector dùng trong báo cáo.
- `Group18_L15_full_report.pdf`: bản PDF toàn bộ bài tập lớn sau khi biên dịch.
- `anova_standalone_report.tex` và `anova_standalone_report.pdf`: bản độc lập chỉ dùng nội dung trong thư mục ANOVA.

Bốn tệp liên kết trong `Latex/LATEX_TASK2_(2.1-2.7+4.1-4.2)/parts/`
chỉ gọi `\input` tới các tệp trên. Do đó, nội dung được quản lý tập trung trong
thư mục này mà vẫn xuất hiện đúng thứ tự ở Chương 3 và Chương 4.

Để tái tạo hình từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe .\Latex\ANOVA\generate_assets.py
```

Lệnh chính cũng sinh các hình vào `anova_analysis/figures/report` và đồng bộ tới
thư mục này. Sau đó biên dịch `main.tex` bằng XeLaTeX/latexmk. Báo cáo dùng dữ liệu train
125.973 bản ghi; test set không được gộp vào suy luận.

