# Hợp đồng dữ liệu chung (data contract)

- Dataset: NSL-KDD; nguồn dùng trong nhóm là `archive.zip` đã cung cấp. Tệp văn bản có **125.973** hàng (`KDDTrain+.txt`) và **22.544** hàng (`KDDTest+.txt`), kiểm tra trực tiếp từ tệp đính kèm. Không cộng hai tập rồi gọi là dữ liệu huấn luyện.
- Mỗi hàng của bản `.txt` có **43 cột**: 41 đặc trưng, tên tấn công gốc, độ khó (difficulty). Bản `.arff` có cách biểu diễn nhãn khác; không giả định chúng tương đương từng cột nếu không kiểm tra.
- Thống nhất các biến đầu ra: `attack_name` (nhãn gốc: normal/neptune/...); `attack_category` (Normal, DoS, Probe, R2L, U2R); `label_binary` (Normal=0, Attack=1). Biến difficulty phải được xử lý riêng, **không đưa vào đặc trưng dự đoán**.
- Task 2 phải cung cấp file `data_dictionary.csv`, `train_clean.csv`, `test_clean.csv`, `preprocessing.py` (hoặc notebook/script chạy lại được), cùng thông báo các biến bị loại bỏ và lý do. File sạch không tự ý xóa hàng trước khi báo kết quả.
- Phân tích mô tả / Pearson / Wilson / ANOVA / Chi-square: sử dụng `KDDTrain+` như tập phân tích chính; nếu phân tích `KDDTest+`, ghi rõ riêng và không trộn tỷ lệ. Lý do: nhất quán và tránh làm test set trở thành cơ sở quyết định mô hình. Điều này là quy ước của nhóm, không phải bắt buộc của tài liệu môn học.
- Mô hình Task 4: fit bộ mã hoá, scaling, feature selection và mọi bước học tham số **chỉ trên train**; áp dụng phép biến đổi đã học lên test. Không chọn tham số/tinh chỉnh theo kết quả test nhiều lần.
- Nhóm năm loại: **Normal + bốn nhóm tấn công DoS, Probe, R2L, U2R**. Không gọi là 'năm nhóm tấn công'. Nhóm phải kiểm tra mapping đầy đủ tên tấn công ở cả train và test; giá trị chưa ánh xạ phải báo lỗi chứ không ngầm coi là Normal.
- Đầu ra thống kê: n, đơn vị tính, tập train/test, alpha=0,05, phương pháp, bảng kết quả, nhãn kết luận. CI Wilson tính trên kích thước mẫu tương ứng của tập đã chọn; không diễn giải thành tỷ lệ mọi lưu lượng thực trên Internet.
- Tên biến, nhãn và thứ tự nhóm dùng nhất quán; thứ tự mặc định: Normal, DoS, Probe, R2L, U2R.
- Công thức kiểm định tương quan Pearson phải nêu H0: **rho=0** (không viết p=0; p thường dùng chỉ p-value).

## Vị trí trong cấu trúc bàn giao Task 2 (mới)

- Các Task khác: chỉ tải `00_DU_LIEU_CHUNG/`, dùng các CSV trong `data/processed/` và `attack_mapping.csv`. Không cần tải raw hoặc các bảng thống kê nội bộ.
- Người phụ trách Task 2: code, raw và bảng kiểm tra riêng ở `../01_NOI_BO_TASK_2/`; xem `../01_NOI_BO_TASK_2/README_TASK2_NOI_BO.md`.
- LaTeX, bảng trình bày, hình biểu đồ chính thức ở file PRISM riêng, không nằm trong gói dữ liệu chung.
