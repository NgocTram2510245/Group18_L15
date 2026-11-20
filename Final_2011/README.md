# Phân tích và dự đoán tấn công mạng dựa trên đặc trưng lưu lượng mạng

Bài tập lớn môn Xác suất và Thống kê (MT2013), Trường Đại học Bách khoa, ĐHQG-HCM.
Lớp L15 và L21, Nhóm 18, học kỳ HK261 (năm học 2026 - 2027).
Giảng viên hướng dẫn: Thái Bá Ngọc.

| Thành viên | MSSV |
|---|---|
| Nguyễn Thị Mai Quyên | 2510213 |
| Nguyễn Thị Mỹ Uyên | 2510264 |
| Nguyễn Kim Ngọc Trâm | 2510245 |
| Trần Minh Khoa | 2510127 |
| Nguyễn Ngọc Đô | 2510043 |

Báo cáo phân tích bộ dữ liệu NSL-KDD bằng thống kê suy diễn (tương quan Pearson, khoảng tin cậy Wilson, kiểm định hai mẫu độc lập, ANOVA, Chi-square) và hồi quy Logistic nhị phân (Normal / Attack). Phân công chi tiết nằm trong Phụ lục A của báo cáo.

## Cấu trúc thư mục

```
.
├── latex/      Mã nguồn LaTeX của báo cáo (main.tex, parts/, bib/, figures/, ...) và main.pdf
└── code/       Mã Python, dữ liệu đã làm sạch, notebook và các tệp kết quả của từng phần
```

## 1. Báo cáo LaTeX (`latex/`)

Bản PDF dựng sẵn: [latex/main.pdf](latex/main.pdf) (62 trang).

**Yêu cầu:** XeLaTeX và Biber (MiKTeX hoặc TeX Live), có phông Times New Roman. Trên Overleaf, chọn trình biên dịch XeLaTeX.

**Dựng lại** (chạy trong thư mục `latex/`):

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

Nếu máy không có Perl cho `latexmk`, chạy tay theo thứ tự:

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

**Bố cục:** `main.tex` gọi các tệp `parts/*.tex` (dạng `subfiles`), mỗi tệp ứng với một cụm mục trong Chương 1 đến Chương 5. Phụ lục A là phân công, Phụ lục B là bảng số liệu, Phụ lục C là danh mục tệp kèm theo. Tài liệu tham khảo nằm trong `bib/` (biblatex, kiểu IEEE). Ba tệp `bib/task31.bib`, `task32.bib`, `task33.bib` không có mục nào nên Biber sẽ báo cảnh báo "không có mục", không ảnh hưởng PDF.

## 2. Mã nguồn và kết quả (`code/`)

**Yêu cầu:** Python 3.11 trở lên. Cài thư viện bằng tệp `requirements.txt` của từng thư mục, hoặc cài chung:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels pytest nbclient nbformat ipykernel
```

Dữ liệu đã làm sạch nằm sẵn trong `code/data/` (`train_clean.csv`, `test_clean.csv`, `data_dictionary.csv`, `attack_mapping.csv`, `SHA256_DU_LIEU.txt`). Các script của Task 3 và Task 4 đọc từ thư mục này, nên chạy trong thư mục chứa script.

| Phần của báo cáo | Thư mục | Lệnh chạy |
|---|---|---|
| Task 2: tiền xử lý, thống kê mô tả (mục 2.5 đến 2.7, 4.1, 4.2) | `code/` | `python preprocessing.py` (cần đặt `KDDTrain+.txt`, `KDDTest+.txt`, `KDDTrain+.arff` vào `code/data/raw/`, xem `code/data/raw/README.txt`) |
| Task 3.1: Pearson (mục 3.3, 4.3) | `code/task31_pearson/` | `python pearson_correlation.py`; `python hypothesis_tests/pearson_hypothesis_tests.py --input ../data/train_clean.csv` |
| Task 3.2 và 3.3: Wilson, ANOVA (mục 3.4, 3.5, 4.4, 4.5) | `code/task32_33_wilson_anova/` | `python anova_utils.py`; kiểm thử: `pytest test_anova_utils.py` |
| Kiểm định hai mẫu (mục 3.6, 4.6) | `code/task35_hai_mau/` | `python two_sample_utils.py`; kiểm thử: `pytest test_two_sample_utils.py` |
| Task 3.4: Chi-square (mục 3.7, 4.7) | `code/task34_chi_square/` | `python task34_chi_square.py` |
| Task 4: hồi quy Logistic (mục 3.8, 4.8) | `code/task4_logistic/` | `python task4_logistic.py` |

Kết quả đã nộp nằm cạnh script: `results/`, `outputs/`, `outputs_pearson/`, `ket_qua/`. Hình dùng trong báo cáo nằm ở `latex/figures/`.

**Tái lập:** đã chạy lại toàn bộ script trên một bản sao sạch. Số liệu của Pearson, Wilson, ANOVA, hai mẫu và Chi-square trùng bản đã nộp (sai số cỡ 1e-11 trở xuống). Hai chỗ lệch nhỏ:
- Hồi quy Logistic, mô hình B (scikit-learn) phụ thuộc phiên bản thư viện. Với scikit-learn 1.9.1, độ chính xác trên tập kiểm tra là 74,77% (bản đã nộp 74,78%) và độ nhạy là 61,44% (bản đã nộp 61,47%). Số trong báo cáo lấy từ `outputs/` đã nộp.
- Vài p-value hiệu chỉnh cực nhỏ ở hậu kiểm Games-Howell (đã nộp "< 1e-300") ra cỡ 1e-11 khi chạy với scipy 1.18.1. Kết luận bác bỏ H0 không đổi.

`preprocessing.py` tái hiện đúng `train_clean.csv` và `test_clean.csv` (khớp SHA256 sau khi quy dòng mới CRLF về LF). Từ điển `data_dictionary.csv` đã nộp được hoàn thiện thủ công, script chỉ sinh bản tự động `data_dictionary_auto.csv`.

Các tệp `run_analysis.ps1` dùng môi trường ảo `.venv` hoặc `.venv312`, người chạy cần tự tạo môi trường này trước.

## 3. Nguồn dữ liệu

NSL-KDD (Tavallaee et al., 2009): tập huấn luyện `KDDTrain+` (125 973 bản ghi) và tập kiểm tra `KDDTest+` (22 544 bản ghi), mỗi bản ghi có 41 đặc trưng, nhãn tấn công và độ khó. Phân tích mô tả, Pearson, Wilson, ANOVA, Chi-square và hai mẫu dùng `KDDTrain+`; Task 4 huấn luyện trên `KDDTrain+` và đánh giá trên `KDDTest+`.
