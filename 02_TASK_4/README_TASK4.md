# HƯỚNG DẪN VÀ TỔNG HỢP KẾT QUẢ TASK 4 (HỒI QUY LOGISTIC)

Người phụ trách theo phân công: **Ngọc Dư**  
Mục tiêu: Xây dựng mô hình hồi quy Logistic nhị phân phân loại lưu lượng mạng thành **Normal ($Y=0$)** hoặc **Attack ($Y=1$)** trên bộ dữ liệu **NSL-KDD**, tuân thủ nghiêm ngặt hợp đồng dữ liệu chung (`QUY_UOC_DU_LIEU.md`).

---

## 1. Cấu trúc thư mục Task 4

```
02_TASK_4/
├── task4_logistic.py           # Mã nguồn Python tái lập toàn bộ quy trình từ đầu đến cuối
├── requirements.txt            # Thư viện phụ thuộc (scikit-learn, statsmodels, matplotlib, ...)
├── outputs/                    # Bảng dữ liệu thống kê xuất ra dạng CSV & JSON
│   ├── summary.json            # Tóm tắt toàn bộ chỉ số mô hình
│   ├── cv_results_modelB.csv   # Kết quả Stratified 5-Fold CV trên KDDTrain+
│   ├── coefficients_modelA.csv # Bảng hệ số beta, SE, Wald z, p, OR, CI 95% (statsmodels Logit)
│   ├── vif_modelA.csv          # Hệ số VIF kiểm tra đa cộng tuyến (toàn bộ < 10)
│   ├── metrics.csv             # Accuracy, Precision, Recall, Specificity, F1, AUC + CI Wilson 95%
│   ├── recall_by_category.csv  # Độ nhạy phân rã theo 5 nhóm: Normal, DoS, Probe, R2L, U2R
│   ├── recall_seen_vs_unseen.csv # So sánh độ nhạy tấn công cũ (Train) vs tấn công mới (Test)
│   ├── missed_by_attack_name.csv # Top các cuộc tấn công bị mô hình bỏ sót nhiều nhất
│   ├── accuracy_by_difficulty.csv # Độ chính xác phân theo thang độ khó (difficulty 0-21)
│   └── run_log.txt             # Nhật ký chạy chi tiết
├── figures/                    # Biểu đồ (PDF xuất vào LaTeX + PNG xem nhanh)
│   ├── task4_confusion_test.pdf # Ma trận nhầm lẫn trên KDDTest+
│   ├── task4_roc.pdf           # Đường cong ROC Train vs Test và so sánh hai mô hình
│   ├── task4_or_modelA.pdf     # Biểu đồ Forest plot tỷ số Odds Ratio (thang log)
│   ├── task4_recall_by_category.pdf # Biểu đồ cột Recall theo 5 nhóm lưu lượng kèm thanh sai số
│   └── task4_prob_hist_test.pdf # Phân bố xác suất dự đoán p_hat của các ca FN
├── latex/                      # Toàn bộ mã nguồn LaTeX đã ghép hoàn chỉnh
│   ├── main.tex                # File chủ đạo biên dịch bằng XeLaTeX
│   ├── references.tex          # Danh mục tài liệu tham khảo đã cập nhật chuẩn
│   ├── figures/                # Thư mục hình ảnh chứa các biểu đồ PDF của Task 2 và Task 4
│   └── parts/
│       ├── 03_task4_logistic.tex # Phần Phương pháp (Chương 3)
│       └── 04_task4_logistic.tex # Phần Kết quả & Thảo luận (Chương 4)
└── latex.zip  # File zip hoàn chỉnh sẵn sàng tải lên Overleaf / Prism
```

---

## 2. Cách chạy lại toàn bộ quy trình phân tích
## Cách 1: Đối với máy chưa có sẵn các thư viện trong Python
Môi trường Python ảo đã được khởi tạo sẵn trong thư mục dự án (`.venv`). Để chạy lại toàn bộ thực nghiệm và tạo mới tất cả các bảng, biểu đồ:
```powershell
# Chạy từ thư mục gốc của repository:
.venv\Scripts\python 02_TASK_4\task4_logistic.py
```
## Cách 2: Đối với máy có sẵn các thư viện cần thiết trong mã nguồn:
Chạy trực tiếp trên Terminal.
---

## 3. Tóm tắt kết quả nghiên cứu nổi bật

### 3.1. Thiết kế hai mô hình giải quyết hai bài toán
- **Mô hình B (Dự đoán toàn diện):** Dùng `scikit-learn` với điều chuẩn $L_2$ trên toàn bộ 121 đặc trưng (sau khi biến đổi $\log(1+x)$, chuẩn hóa $z$ và mã hóa one-hot). Siêu tham số $C=10$ được xác lập bằng kiểm định chéo phân tầng 5 phần trên tập Train theo quy tắc một sai số chuẩn ($F_1\text{-CV} = 0{,}9844$).
- **Mô hình A (Khả giải - Diễn giải xác suất):** Dùng `statsmodels Logit` (Newton--Raphson) trên 11 biến tinh gọn. Toàn bộ $\text{VIF} < 10$ (cao nhất là 9,38), loại trừ hoàn toàn nguy cơ đa cộng tuyến nặng. Kiểm định tỷ số hợp lý $G = 134.954{,}4$ ($p < 0{,}001$), Pseudo-$R^2$ McFadden đạt $0{,}7755$.

### 3.2. Hiệu năng tổng thể (Train vs. Test)
| Độ đo | Mô hình B (Train) | Mô hình B (Test) | Khoảng tin cậy Wilson 95\% (Test) |
|---|---|---|---|
| **Accuracy** | 98,57\% | **74,78\%** | $[74,21\%;\ 75,35\%]$ |
| **Precision** | 98,61\% | **91,41\%** | $[90,80\%;\ 91,99\%]$ |
| **Recall** | 98,31\% | **61,47\%** | $[60,63\%;\ 62,31\%]$ |
| **Specificity** | 98,80\% | **92,37\%** | $[91,82\%;\ 92,88\%]$ |
| **$F_1$-score** | 0,9846 | **0,7351** | --- |
| **ROC-AUC** | 0,9980 | **0,7878** | --- |

*(Ngưỡng quyết định cố định $\hat\pi \geq 0{,}5$, không tinh chỉnh trên Test).*

### 3.3. Các phát hiện cốt lõi từ phân tích lỗi (Error Analysis)
1. **Điểm mù của nhóm R2L:** Độ nhạy với R2L trên Test chỉ đạt **2,70\%** (bỏ sót 2.807 / 2.885 cuộc tấn công). R2L chiếm đến **56,8\%** tổng số ca âm tính giả ($FN$). Lý do: các cuộc tấn công đoán mật khẩu (`guess_passwd`) hoạt động ở tầng ứng dụng, không làm đột biến lượng gói tin hay tạo cờ lỗi nên bộ phân loại tuyến tính dựa trên đặc trưng mào đầu mạng không thể phân biệt.
2. **Tác động của tấn công mới:** KDDTest+ có 17 loại tấn công mới (3.750 dòng). Tỷ lệ phát hiện tấn công mới chỉ đạt **27,31\%** (so với 75,58\% ở tấn công đã học). Các dạng tấn công mới chiếm **55,1\%** tổng số $FN$.
3. **Mức độ tự tin cực đoan:** Trung vị xác suất $\hat\pi$ của 4.944 ca $FN$ chỉ là **0,00064**, với **87,0\%** số ca $FN$ có $\hat\pi < 0{,}1$. Mô hình hoàn toàn tự tin gán chúng là Normal, do đó việc hạ ngưỡng phân loại không giải quyết được vấn đề mà sẽ làm bùng nổ báo động giả.
4. **Phân tích báo động giả ($FP = 741$):** 87,2\% số ca báo động giả ($646$ ca) tập trung ở cổng `service = private` (tỷ lệ $FP$ trên cổng này là 75,82\%). Điều này do trong tập Train, dịch vụ `private` có tỷ lệ tấn công áp đảo (20.871 dòng Attack vs 982 dòng Normal), khiến mô hình gán trọng số dương rất lớn ($OR = 33{,}3$).
5. **Tương quan với độ khó (`difficulty`):** Tỷ lệ chính xác tăng đơn điệu từ **8,64\%** ở nhóm độ khó $0-10$ lên **99,87\%** ở nhóm độ khó $21$.
