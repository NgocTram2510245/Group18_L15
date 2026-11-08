# Báo cáo LaTeX kiểm định hai mẫu

Thư mục này chứa báo cáo độc lập cho Mục 3.6 và 4.6:

- `main.tex`: tài liệu độc lập, kế thừa preamble của báo cáo nhóm.
- `chapters/chapter3_two_sample.tex`: phương pháp Welch hai mẫu độc lập.
- `chapters/chapter4_two_sample.tex`: kết quả thực tế, đồ thị Q-Q và kích thước ảnh hưởng.
- `two_sample_report.pdf`: PDF đã biên dịch.

Các hình được tham chiếu trực tiếp từ `../figures/` để tránh nhân bản. Các listing
được trích trực tiếp theo dòng từ `../two_sample_utils.py`.

## Tích hợp vào báo cáo tổng

Chèn `chapter3_two_sample.tex` ngay sau phần ANOVA trong Chương 3 và
`chapter4_two_sample.tex` ngay sau phần ANOVA trong Chương 4. Khi đó LaTeX sẽ tự
đánh số lại các mục Chi-square và Logistic phía sau. Hai tệp chương không chứa
`documentclass`, preamble hoặc `begin{document}`.

## Biên dịch

Template gốc dùng XeLaTeX. Từ thư mục này:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Hoặc dùng Tectonic tương thích XeTeX:

```powershell
tectonic main.tex --keep-logs --keep-intermediates
```

