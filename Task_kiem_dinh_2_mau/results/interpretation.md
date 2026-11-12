# Diễn giải kiểm định hai mẫu độc lập

Tập huấn luyện gốc được dùng cho suy luận; tập test không tham gia kiểm định. Chênh lệch được định nghĩa là mean(Normal) - mean(Attack).

## RQ1 — srv_serror_rate

Normal có trung bình 0.0120834 (n=67,343), còn Attack có trung bình 0.593072 (n=58,630). Chênh lệch Normal - Attack là -0.580988, CI 95% [-0.58501; -0.576967].
Welch t(61805.40) = -283.1566, p <1e-300; p hiệu chỉnh Holm <1e-300. Hedges g = -1.7070. Brown-Forsythe p <1e-300, tỷ số phương sai lớn/nhỏ=32.175.
Quyết định: bác bỏ H0 sau hiệu chỉnh Holm. Trong mẫu NSL-KDD, nhóm Attack có tỷ lệ lỗi SYN theo dịch vụ cao hơn rõ rệt, phù hợp với hành vi tạo kết nối thất bại/không hoàn tất bắt tay TCP thường gặp ở một số kiểu tấn công.
Phân tích độ nhạy: Mann-Whitney U (two-sided, asymptotic) có p <1e-300; Yuen-Welch t-test, 20% trimmed means có p <1e-300. Hai kết quả này hỗ trợ đánh giá độ bền nhưng không cùng estimand với trung bình gốc.
Cảnh báo: Biến tỷ lệ có khối xác suất tại biên; Welch suy luận về trung bình nhưng mô hình chuẩn chỉ là xấp xỉ. Không có metadata phiên/host nên không thể kiểm chứng đầy đủ tính độc lập; vì vậy kết quả là liên hệ thống kê, không phải quan hệ nhân quả.

## RQ2 — srv_rerror_rate

Normal có trung bình 0.0446291 (n=67,343), còn Attack có trung bình 0.209114 (n=58,630). Chênh lệch Normal - Attack là -0.164485, CI 95% [-0.168098; -0.160872].
Welch t(83489.74) = -89.2302, p <1e-300; p hiệu chỉnh Holm <1e-300. Hedges g = -0.5254. Brown-Forsythe p <1e-300, tỷ số phương sai lớn/nhỏ=3.999.
Quyết định: bác bỏ H0 sau hiệu chỉnh Holm. Trong mẫu NSL-KDD, nhóm Attack có tỷ lệ kết nối bị từ chối theo dịch vụ cao hơn; đặc trưng này phản ánh một khía cạnh khác của lỗi kết nối so với lỗi SYN.
Phân tích độ nhạy: Mann-Whitney U (two-sided, asymptotic) có p <1e-300; Yuen-Welch t-test, 20% trimmed means có p 4.905e-08. Hai kết quả này hỗ trợ đánh giá độ bền nhưng không cùng estimand với trung bình gốc.
Cảnh báo: Biến tỷ lệ có khối xác suất tại biên; Welch suy luận về trung bình nhưng mô hình chuẩn chỉ là xấp xỉ. Không có metadata phiên/host nên không thể kiểm chứng đầy đủ tính độc lập; vì vậy kết quả là liên hệ thống kê, không phải quan hệ nhân quả.

## RQ3 — count

Normal có trung bình 22.5179 (n=67,343), còn Attack có trung bình 154.85 (n=58,630). Chênh lệch Normal - Attack là -132.332, CI 95% [-133.418; -131.246].
Welch t(77661.21) = -238.8283, p <1e-300; p hiệu chỉnh Holm <1e-300. Hedges g = -1.4143. Brown-Forsythe p <1e-300, tỷ số phương sai lớn/nhỏ=5.296.
Quyết định: bác bỏ H0 sau hiệu chỉnh Holm. Nhóm Attack có số kết nối gần đây trung bình cao hơn, cho thấy cường độ/lặp lại kết nối là tín hiệu hành vi đáng chú ý trong bộ dữ liệu này.
Phân tích độ nhạy: Mann-Whitney U (two-sided, asymptotic) có p <1e-300; Yuen-Welch t-test, 20% trimmed means có p <1e-300. Hai kết quả này hỗ trợ đánh giá độ bền nhưng không cùng estimand với trung bình gốc.
Cảnh báo: Biến đếm lệch và có ngoại lệ; dùng Welch nhờ cỡ mẫu lớn, kèm phân tích độ nhạy robust/rank-based. Không có metadata phiên/host nên không thể kiểm chứng đầy đủ tính độc lập; vì vậy kết quả là liên hệ thống kê, không phải quan hệ nhân quả.

## Liên hệ với Pearson, ANOVA và Machine Learning

Pearson đo liên hệ tuyến tính giữa hai biến định lượng, còn Welch ở đây so sánh trung bình của một biến giữa nhãn Normal và Attack. Vì trả lời hai câu hỏi khác nhau, việc biến được chọn khác nhau không làm kết quả mâu thuẫn.
Trong kết quả Pearson hiện có, srv_serror_rate liên hệ rất mạnh với dst_host_srv_serror_rate (r≈0.9863); điều này cảnh báo tính dư thừa khi đưa đồng thời các biến tương quan cao vào mô hình.
Trong kết quả Pearson hiện có, srv_rerror_rate liên hệ rất mạnh với dst_host_srv_rerror_rate (r≈0.9702); điều này cảnh báo tính dư thừa khi đưa đồng thời các biến tương quan cao vào mô hình.
ANOVA hiện có so sánh count giữa năm nhóm Normal, DoS, Probe, R2L và U2R, cho Welch F(4, 405.49)=35907.45, p <1e-300. Kiểm định hai mẫu ở đây gộp bốn nhóm tấn công thành Attack; do đó nó trả lời một tương phản khác và chịu ảnh hưởng bởi tỷ trọng từng loại tấn công. Hai phân tích dùng cùng bản ghi nên không độc lập về thông tin.

Ý nghĩa thống kê không đồng nghĩa khả năng dự đoán tốt: p-value chịu ảnh hưởng mạnh của cỡ mẫu, không phản ánh trực tiếp độ phân tách từng quan sát, tương tác, phi tuyến hay hiệu năng ngoài mẫu. Các kết quả này hỗ trợ giải thích dữ liệu và chọn đặc trưng có hiểu biết, nhưng hiệu năng Machine Learning vẫn phải đánh giá trên test set bằng metric phù hợp.

Không diễn giải p-value hiển thị <1e-300 là bằng 0; đó là giới hạn biểu diễn số. Không kết luận nhãn tấn công gây ra thay đổi của đặc trưng.