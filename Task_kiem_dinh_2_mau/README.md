# Kiểm định giả thuyết hai mẫu độc lập trên NSL-KDD

Phần này so sánh **Normal** và **Attack** trên ba biến `srv_serror_rate`,
`srv_rerror_rate`, `count`. Kiểm định chính là Welch independent two-sample
t-test hai phía. Ba p-value tổng thể được hiệu chỉnh Holm. Mann–Whitney U và
Yuen–Welch với 20% trimmed means chỉ là phân tích độ nhạy vì chúng không kiểm
định đúng cùng tham số với trung bình gốc.

Các khoảng tin cậy 95% cho từng RQ là khoảng riêng lẻ; chúng không tự động có
độ bao phủ đồng thời 95% cho cả ba câu hỏi.

## Nguồn dữ liệu và phạm vi

- Nguồn mặc định: `../00_DU_LIEU_CHUNG_CHO_TEAM/00_DU_LIEU_CHUNG/data/processed/train_clean.csv`.
- `label_binary=0` được kiểm tra phải tương ứng `attack_category=Normal`; mọi
  nhãn còn lại phải có `label_binary=1`.
- Chỉ tập train gốc, chưa SMOTE/StandardScaler, được dùng. Tập test không tham
  gia suy luận.
- Không xóa/winsorize ngoại lệ. Bất kỳ bản ghi thiếu hoặc không hữu hạn nào đều
  được đếm trong `results/data_validation.csv`.

## Cách chạy

Từ thư mục gốc repository bằng PowerShell:

```powershell
& "C:\Users\Ngoc Tram\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv312
& .\.venv312\Scripts\python.exe -m pip install -r .\two_sample_testing\requirements.txt
& .\two_sample_testing\run_analysis.ps1
```

Nếu đã có môi trường Python khác, có thể cài `requirements.txt`, chuyển vào
`two_sample_testing` rồi chạy `python two_sample_utils.py` và
`python execute_notebook.py`.

## Tái lập và sampling hình ảnh

Toàn bộ dữ liệu hợp lệ được dùng cho thống kê mô tả, Welch, Brown–Forsythe,
Mann–Whitney và Yuen. Để tránh hình quá nặng/chồng điểm, ECDF dùng tối đa 5.000
quan sát/nhóm và Q-Q plot dùng tối đa 1.000 quan sát/nhóm, với
`random_state=18015`. Sampling này chỉ thay đổi hình minh họa, không thay đổi
kết quả kiểm định.

## Đầu ra

- `two_sample_analysis.ipynb`: báo cáo thực thi bằng tiếng Việt.
- `results/`: kiểm tra dữ liệu, thống kê mô tả, giả định, Welch, effect size,
  độ nhạy, bảng tổng hợp, caption và diễn giải.
- `figures/<feature>/`: boxplot, mean-CI, ECDF, Q-Q và khối xác suất tại biên.
- `figures/comparison/`: forest plot chênh lệch trung bình và Hedges g.

Giới hạn quan trọng: NSL-KDD không cung cấp session/host identifier đủ để xác
minh độc lập giữa các dòng. Vì vậy các kết quả là liên hệ thống kê trên dữ liệu,
không chứng minh quan hệ nhân quả hoặc hiệu năng dự đoán ngoài mẫu.
