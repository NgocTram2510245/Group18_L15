# TASK 3.1.1 – PEARSON CORRELATION ANALYSIS

## 1. Giới thiệu

Thư mục này chứa toàn bộ source code, kết quả phân tích và báo cáo của **Task 3.1.1 – Phân tích tương quan Pearson và xác định các đặc trưng có khả năng dư thừa**.

Mục tiêu của nhiệm vụ là phân tích mối quan hệ tuyến tính giữa các đặc trưng định lượng trong bộ dữ liệu NSL-KDD, xác định những cặp đặc trưng có tương quan cao và đề xuất các đặc trưng cần xem xét khi thực hiện những bước phân tích tiếp theo.

Kết quả của Task 3.1.1 có thể được sử dụng làm tài liệu tham khảo cho các nhiệm vụ tiếp theo liên quan đến phân tích đặc trưng, lựa chọn đặc trưng (Feature Selection), xây dựng mô hình và đánh giá dữ liệu.

**Lưu ý:** Kết quả tương quan cao chỉ cho thấy hai đặc trưng có mối quan hệ tuyến tính mạnh. Điều này chưa đủ để kết luận rằng một trong hai đặc trưng chắc chắn dư thừa hoặc có thể bị loại bỏ.

---

## 2. Cấu trúc thư mục

```text
3.1.1_Pearson/
│
├── outputs/
│   ├── pearson_matrix.csv
│   ├── high_correlation_pairs.csv
│   └── pearson_heatmap.png
│
├── pearson_correlation.py
├── pearson_report_full.tex
├── pearson_report_full.pdf
├── references.bib
└── README.md
```

### 2.1. Source code

| File | Chức năng |
|---|---|
| `pearson_correlation.py` | Chương trình Python thực hiện phân tích tương quan Pearson. File này đảm nhiệm quá trình đọc và xử lý dữ liệu, lựa chọn các đặc trưng phù hợp, tính toán ma trận tương quan, tìm các cặp tương quan cao và xuất kết quả. |

### 2.2. Thư mục outputs/

Đây là thư mục chứa các kết quả đầu ra chính của Task 3.1.1. Các nhiệm vụ tiếp theo nên sử dụng những file này khi cần tham khảo kết quả phân tích tương quan.

| File | Định dạng | Nội dung và mục đích sử dụng |
|---|---|---|
| `pearson_matrix.csv` | CSV | Chứa ma trận tương quan Pearson giữa 37 đặc trưng định lượng được lựa chọn. Mỗi ô thể hiện hệ số tương quan Pearson (r) giữa hai đặc trưng tương ứng. Có thể sử dụng file này để tra cứu mức độ tương quan giữa bất kỳ hai đặc trưng nào, thực hiện phân tích sâu hơn hoặc tạo các biểu đồ tương quan khác. |
| `high_correlation_pairs.csv` | CSV | Chứa danh sách 10 cặp đặc trưng có hệ số tương quan tuyệt đối lớn hơn 0.95 (|r| > 0.95). File này giúp xác định nhanh những cặp đặc trưng có mối quan hệ tuyến tính rất mạnh mà không cần đọc toàn bộ ma trận tương quan. Đây là kết quả quan trọng để xem xét khả năng dư thừa đặc trưng. |
| `pearson_heatmap.png` | PNG | Là biểu đồ Heatmap trực quan hóa toàn bộ ma trận tương quan Pearson. Mỗi ô biểu diễn hệ số tương quan giữa hai đặc trưng; màu sắc thể hiện mức độ và chiều hướng tương quan. Biểu đồ giúp nhận diện nhanh các nhóm đặc trưng có tương quan cao và hỗ trợ trình bày kết quả trong báo cáo. |

### 2.3. Các file báo cáo và tài liệu

| File | Chức năng |
|---|---|
| `pearson_report_full.tex` | Mã nguồn LaTeX của báo cáo Task 3.1.1. Dùng để chỉnh sửa nội dung, công thức, bảng biểu và định dạng báo cáo. |
| `pearson_report_full.pdf` | Báo cáo PDF đã biên dịch từ file LaTeX. Dùng để đọc, tham khảo và tổng hợp kết quả. |
| `references.bib` | Chứa thông tin các tài liệu tham khảo theo định dạng BibTeX, được sử dụng để tạo danh mục tài liệu tham khảo trong báo cáo. |
| `README.md` | Tài liệu hướng dẫn cấu trúc thư mục, chức năng từng file và cách sử dụng kết quả của Task 3.1.1. |

---

## 3. Phương pháp và phạm vi phân tích

Phân tích được thực hiện trên bộ dữ liệu `train_clean.csv` thuộc NSL-KDD.

Thông tin dữ liệu:
- Số lượng quan sát: 125,973.
- Tổng số cột ban đầu: 45.
- Số đặc trưng định lượng được đưa vào phân tích: 37.
- Kích thước ma trận tương quan: 37 × 37.
- Số cặp tương quan duy nhất giữa các đặc trưng: 666.
- Ngưỡng xác định tương quan cao: |r| > 0.95.

Các cột metadata, nhãn phân loại và các đặc trưng dạng categorical không được đưa trực tiếp vào phép tính Pearson. Đặc trưng `num_outbound_cmds` cũng được loại khỏi phép tính vì có phương sai bằng 0, khiến hệ số tương quan Pearson không xác định.

Công thức hệ số tương quan Pearson:

\[
r_{XY} =
\frac{\operatorname{Cov}(X,Y)}
{\sigma_X \sigma_Y}
\]

Trong đó:
- \(r_{XY}\): Hệ số tương quan Pearson giữa hai biến X và Y.
- \(\operatorname{Cov}(X,Y)\): Hiệp phương sai giữa X và Y.
- \(\sigma_X, \sigma_Y\): Độ lệch chuẩn của hai biến.

Giá trị của hệ số tương quan nằm trong khoảng [-1, 1]:
- r gần 1: Tương quan tuyến tính thuận mạnh.
- r gần -1: Tương quan tuyến tính nghịch mạnh.
- r gần 0: Không có mối tương quan tuyến tính đáng kể.

---

## 4. Kết quả phân tích chính

Với ngưỡng |r| > 0.95, phân tích xác định được **10 cặp đặc trưng có tương quan tuyến tính rất cao**.

Các cặp này được chia thành ba nhóm chính:

| Nhóm | Số cặp | Ý nghĩa |
|---|---:|---|
| Root access | 1 | Mối tương quan giữa số lượng điều kiện bị xâm phạm và số lần truy cập quyền root. |
| SYN error rate | 6 | Các đặc trưng liên quan đến tỷ lệ lỗi SYN ở những phạm vi thống kê khác nhau. |
| REJ error rate | 3 | Các đặc trưng liên quan đến tỷ lệ lỗi REJ ở những phạm vi thống kê khác nhau. |

Một số cặp có hệ số tương quan cao nhất:

| Cặp đặc trưng | Hệ số tương quan (r) |
|---|---:|
| `num_compromised` – `num_root` | 0.998833 |
| `serror_rate` – `srv_serror_rate` | 0.993289 |
| `rerror_rate` – `srv_rerror_rate` | 0.989008 |
| `srv_serror_rate` – `dst_host_srv_serror_rate` | 0.986252 |
| `dst_host_serror_rate` – `dst_host_srv_serror_rate` | 0.985052 |

Để xem đầy đủ cả 10 cặp, sử dụng file `outputs/high_correlation_pairs.csv`.

---

## 5. Cách sử dụng kết quả cho các nhiệm vụ tiếp theo

Các nhiệm vụ tiếp theo có thể tham khảo kết quả Task 3.1.1 theo những cách sau:

### 5.1. Khi cần kiểm tra tương quan giữa hai đặc trưng

Sử dụng `outputs/pearson_matrix.csv`.

Cách sử dụng:
1. Mở file bằng Excel, Python hoặc Pandas.
2. Tìm tên hai đặc trưng cần kiểm tra.
3. Đọc giá trị tại giao điểm của hàng và cột tương ứng.
4. Dựa vào hệ số Pearson để đánh giá mức độ tương quan tuyến tính.

Ví dụ: Nếu cần kiểm tra mối quan hệ giữa `num_compromised` và `num_root`, có thể tra cứu trực tiếp hệ số tương quan trong ma trận.

### 5.2. Khi cần xác định các đặc trưng có khả năng dư thừa

Sử dụng `outputs/high_correlation_pairs.csv`.

File này cung cấp danh sách các cặp có |r| > 0.95. Các nhiệm vụ liên quan đến Feature Selection có thể sử dụng danh sách này làm cơ sở để xem xét những đặc trưng có thông tin tương đồng.

Tuy nhiên, không nên tự động loại bỏ một đặc trưng chỉ dựa trên hệ số tương quan. Cần xem xét thêm ý nghĩa của đặc trưng, mục tiêu phân tích và hiệu quả của mô hình nếu có thực hiện đánh giá.

### 5.3. Khi cần trực quan hóa hoặc trình bày kết quả

Sử dụng `outputs/pearson_heatmap.png`.

Hình ảnh này có thể được:
- Chèn vào báo cáo để minh họa ma trận tương quan.
- Sử dụng trong slide thuyết trình.
- Tham khảo để nhận diện các nhóm đặc trưng có tương quan cao.

### 5.4. Khi cần kiểm tra hoặc tái tạo kết quả

Sử dụng `pearson_correlation.py`.

File Python là source code gốc của quá trình phân tích. Nếu cần kiểm tra cách tính toán, thay đổi ngưỡng tương quan hoặc chạy lại phân tích trên dữ liệu khác, có thể tham khảo và điều chỉnh chương trình này.

Lưu ý: Khi chạy lại chương trình, cần bảo đảm đường dẫn đến bộ dữ liệu đầu vào chính xác và các thư viện Python cần thiết đã được cài đặt.

---

## 6. Một số lưu ý quan trọng

1. `pearson_matrix.csv` chứa toàn bộ ma trận tương quan, trong khi `high_correlation_pairs.csv` chỉ chứa các cặp vượt ngưỡng |r| > 0.95. Hai file có mục đích khác nhau và không thay thế cho nhau.

2. `pearson_heatmap.png` chỉ là hình ảnh trực quan hóa kết quả. Nếu cần lấy chính xác hệ số tương quan để tính toán tiếp, hãy sử dụng file CSV thay vì đọc giá trị từ hình ảnh.

3. Tương quan cao không đồng nghĩa với quan hệ nhân quả.

4. Tương quan Pearson chỉ đo lường mối quan hệ tuyến tính. Hai biến có thể có quan hệ phi tuyến nhưng hệ số Pearson vẫn thấp.

5. Danh sách 10 cặp tương quan cao được sử dụng để đề xuất các đặc trưng cần xem xét, không phải danh sách các đặc trưng đã bị loại bỏ khỏi bộ dữ liệu.

6. Các kết quả trong thư mục `outputs/` được tạo từ dữ liệu và cấu hình của lần phân tích hiện tại. Nếu thay đổi dữ liệu đầu vào hoặc phương pháp xử lý, cần chạy lại chương trình và cập nhật các kết quả tương ứng.

---

## 7. Tổng kết

Task 3.1.1 đã hoàn thành việc phân tích tương quan Pearson trên 37 đặc trưng định lượng của bộ dữ liệu NSL-KDD, tạo ra ma trận tương quan, biểu đồ Heatmap và danh sách 10 cặp đặc trưng có tương quan cao.

Các file trong thư mục này được tổ chức nhằm phục vụ ba mục đích chính:

- **Tra cứu dữ liệu:** Sử dụng `pearson_matrix.csv` và `high_correlation_pairs.csv`.
- **Trực quan hóa kết quả:** Sử dụng `pearson_heatmap.png`.
- **Kiểm tra và tái tạo phân tích:** Sử dụng `pearson_correlation.py`.

Các thành viên thực hiện những nhiệm vụ tiếp theo có thể tham khảo trực tiếp các kết quả này để hỗ trợ quá trình phân tích và xây dựng mô hình, đồng thời cần kiểm tra xem kết quả có phù hợp với mục tiêu cụ thể của nhiệm vụ mình hay không.
