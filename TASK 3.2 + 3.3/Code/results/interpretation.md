# Diễn giải ANOVA cho NSL-KDD (tập train)

## Phạm vi và giả thuyết

Phân tích dùng 125,973 bản ghi train ở thang gốc, không SMOTE, không scaling và không dùng test set. Với mỗi RQ, $H_0: \mu_{Normal}=\mu_{DoS}=\mu_{Probe}=\mu_{R2L}=\mu_{U2R}$; $H_1$: ít nhất một trung bình khác, $\alpha=0.05$.

F là tỷ số giữa biến thiên giữa nhóm và biến thiên trong nhóm (Welch dùng trọng số theo phương sai). Bậc tự do mô tả lượng thông tin cho tử/mẫu số; p-value là xác suất thấy thống kê ít nhất cực đoan như quan sát nếu H0 và mô hình đúng. Bác bỏ khi p hiệu chỉnh < alpha. Không bác bỏ không chứng minh H0; ANOVA có ý nghĩa cũng không nói mọi cặp đều khác.

## Kết quả bốn RQ

### RQ1 — `count`

Phương pháp chính: **Welch one-way ANOVA** trên `count`. Biến đếm lệch và phương sai/cỡ mẫu rất khác nhau; cỡ mẫu bốn nhóm lớn, nhưng U2R nhỏ nên dùng Welch và báo cảnh báo độ bền.

F(4.000, 405.490) = 35907.4465; p thô < 1e-300; p Holm < 1e-300. Eta² mô tả = 0.4049, omega² mô tả = 0.4049. Hai effect size này dựa trên tổng bình phương cổ điển ở thang phân tích, không suy ra trực tiếp từ Welch F.

Các cặp Games–Howell có ý nghĩa: Normal–DoS, Normal–Probe, Normal–R2L, Normal–U2R, DoS–Probe, DoS–R2L, DoS–U2R, Probe–R2L, Probe–U2R. Bảng hậu kiểm chứa chênh lệch, CI đồng thời và p đã hiệu chỉnh; với `src_bytes`/`duration`, chênh lệch là ở thang log1p.

Trung bình thang gốc theo thứ tự Normal/DoS/Probe/R2L/U2R: Normal=22.52, DoS=178.1, Probe=77.05, R2L=1.297, U2R=5.808. Trung vị: Normal=4, DoS=172, Probe=1, R2L=1, U2R=1.

Cảnh báo: Biến đếm; U2R n=52; Brown–Forsythe p < 1e-300; max |skew|=5.56; độc lập phiên không kiểm chứng được.

### RQ2 — `serror_rate`

Phương pháp chính: **Welch one-way ANOVA** trên `serror_rate`. Mục tiêu là so sánh trung bình tỷ lệ trên thang 0–1; phương sai khác nhau. Khối lượng lớn ở biên khiến đây là xấp xỉ, được kiểm tra nhạy bằng Kruskal–Wallis.

F(4.000, 386.813) = 31976.1963; p thô < 1e-300; p Holm < 1e-300. Eta² mô tả = 0.6202, omega² mô tả = 0.6202. Hai effect size này dựa trên tổng bình phương cổ điển ở thang phân tích, không suy ra trực tiếp từ Welch F.

Các cặp Games–Howell có ý nghĩa: Normal–DoS, Normal–Probe, DoS–Probe, DoS–R2L, DoS–U2R, Probe–R2L. Bảng hậu kiểm chứa chênh lệch, CI đồng thời và p đã hiệu chỉnh; với `src_bytes`/`duration`, chênh lệch là ở thang log1p.

Trung bình thang gốc theo thứ tự Normal/DoS/Probe/R2L/U2R: Normal=0.01344, DoS=0.7485, Probe=0.04653, R2L=0.0119, U2R=0.03808. Trung vị: Normal=0, DoS=1, Probe=0, R2L=0, U2R=0.

Cảnh báo: Tỷ lệ dồn ở 0/1; ANOVA chỉ là xấp xỉ cho mean; Brown–Forsythe p < 1e-300; max |skew|=8.85; độc lập phiên không kiểm chứng được.

### RQ3 — `src_bytes`

Phương pháp chính: **Welch one-way ANOVA on log1p** trên `log1p(src_bytes)`. Thang gốc lệch phải và có cực trị rất lớn; log1p giảm ảnh hưởng nhưng đổi estimand thành trung bình log(1 + src_bytes).

F(4.000, 387.016) = 42494.9643; p thô < 1e-300; p Holm < 1e-300. Eta² mô tả = 0.5639, omega² mô tả = 0.5639. Hai effect size này dựa trên tổng bình phương cổ điển ở thang phân tích, không suy ra trực tiếp từ Welch F.

Các cặp Games–Howell có ý nghĩa: Normal–DoS, Normal–Probe, Normal–R2L, DoS–Probe, DoS–R2L, DoS–U2R, Probe–R2L, Probe–U2R, R2L–U2R. Bảng hậu kiểm chứa chênh lệch, CI đồng thời và p đã hiệu chỉnh; với `src_bytes`/`duration`, chênh lệch là ở thang log1p.

Trung bình thang gốc theo thứ tự Normal/DoS/Probe/R2L/U2R: Normal=1.313e+04, DoS=1176, Probe=3.857e+05, R2L=3.077e+05, U2R=906.2. Trung vị: Normal=233, DoS=0, Probe=1, R2L=334, U2R=273.

Cảnh báo: Ngoại lệ cực lớn; estimand chính ở thang log1p; Brown–Forsythe p < 1e-300; max |skew|=3.17; độc lập phiên không kiểm chứng được.

### RQ4 — `duration`

Phương pháp chính: **Welch one-way ANOVA on log1p** trên `log1p(duration)`. Dữ liệu toàn phần zero-inflated và lệch mạnh; log1p giữ số 0. Kết quả được bổ sung bằng phân tích hai phần, không coi ANOVA là mô hình hoàn hảo.

F(4.000, 386.205) = 1652.4421; p thô 2.161e-241; p Holm 2.161e-241. Eta² mô tả = 0.0412, omega² mô tả = 0.0412. Hai effect size này dựa trên tổng bình phương cổ điển ở thang phân tích, không suy ra trực tiếp từ Welch F.

Các cặp Games–Howell có ý nghĩa: Normal–DoS, Normal–Probe, Normal–R2L, Normal–U2R, DoS–Probe, DoS–R2L, DoS–U2R, Probe–R2L, Probe–U2R, R2L–U2R. Bảng hậu kiểm chứa chênh lệch, CI đồng thời và p đã hiệu chỉnh; với `src_bytes`/`duration`, chênh lệch là ở thang log1p.

Trung bình thang gốc theo thứ tự Normal/DoS/Probe/R2L/U2R: Normal=168.6, DoS=0.006227, Probe=2075, R2L=633.4, U2R=80.94. Trung vị: Normal=0, DoS=0, Probe=0, R2L=0, U2R=46.

Cảnh báo: Zero-inflated; estimand chính ở thang log1p; cần đọc phân tích hai phần; Brown–Forsythe p < 1e-300; max |skew|=38.03; độc lập phiên không kiểm chứng được.

## RQ4 — mô tả zero và phân tích có điều kiện

Tỷ lệ `duration=0` (mô tả, kèm Wilson CI trong bảng): Normal=88.18%, DoS=99.91%, Probe=86.31%, R2L=61.11%, U2R=28.85%. Phần này không thực hiện Chi-square và không phải kiểm định khác biệt mean duration.
Trong các kết nối duration>0, Welch trên log1p: F(4.000, 158.324)=315.8186, p 2.476e-74. Đây là phân tích có điều kiện, không thay thế kết quả toàn bộ dữ liệu.

## Ý nghĩa và giới hạn

Kết quả mô tả hành vi kết nối trong NSL-KDD: một số nhóm khác rõ về cường độ kết nối, lỗi SYN, byte nguồn và thời lượng. Chúng có thể hỗ trợ giải thích vì sao mô hình ML khai thác các biến này, nhưng p-value nhỏ không đo khả năng dự đoán: cỡ mẫu, chồng lấn phân phối, tương tác, mất cân bằng lớp và khả năng khái quát đều quan trọng. Eta² cũng không phải bảng xếp hạng predictor.

NSL-KDD cũ và không đại diện trực tiếp cho lưu lượng Internet hiện đại. Metadata không có session/host/time identifier đủ để kiểm chứng độc lập; các bản ghi gần cùng chiến dịch có thể tương quan, làm độ không chắc chắn thực lớn hơn CI/p-value giả định độc lập. Phân tích quan sát không chứng minh nguyên nhân hay xác suất tấn công.