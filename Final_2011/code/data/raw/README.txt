Đặt vào thư mục này ba tệp lấy từ archive.zip của NSL-KDD (nguồn dữ liệu của nhóm):
  KDDTrain+.txt   (125973 hàng, 43 cột)
  KDDTest+.txt    (22544 hàng, 43 cột)
  KDDTrain+.arff  (đọc danh sách giá trị của protocol_type, service, flag)
Sau đó chạy: python preprocessing.py (từ thư mục code/).
Kết quả train_clean.csv, test_clean.csv và data_dictionary_auto.csv được ghi vào code/data/;
data_dictionary.csv đã nộp được hoàn thiện thủ công nên script không ghi đè;
sau khi chuẩn hóa dòng mới CRLF/LF, hai tệp CSV khớp SHA256 đã nộp;
SHA256 đối chiếu nằm trong SHA256_DU_LIEU.txt.
