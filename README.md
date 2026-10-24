# Group18_L15
# DATA DÙNG CHUNG CHO TASK 3 / TASK 4

**Chỉ tải thư mục `00_DU_LIEU_CHUNG`.** Đây là dữ liệu đã chuẩn bị bởi Task 2, đủ để đọc và tự phân tích các nhiệm vụ tiếp theo. Không cần tải raw TXT/ARFF, code kiểm toán và CSV nội bộ của Task 2.

## File cần biết
| File | Dùng để làm gì |
|---|---|
| `data/processed/train_clean.csv` | Train: phân tích Pearson, Wilson, ANOVA, Chi-square; phát triển/huấn luyện mô hình |
| `data/processed/test_clean.csv` | Test: đánh giá mô hình ngoài tập huấn luyện |
| `data/processed/data_dictionary.csv` | Ý nghĩa, kiểu dữ liệu, đơn vị, lưu ý của **41 features** |
| `data/processed/metadata_dictionary.csv` | Bốn cột ngoài features: `attack_name`, `difficulty`, `attack_category`, `label_binary` |
| `attack_mapping.csv` | Tên tấn công gốc → 5 nhóm và nhị phân, cùng nguồn/ghi chú |
| `QUY_UOC_DU_LIEU.md` | Quy tắc phân tích và chống rò rỉ dữ liệu |

## Bắt đầu bằng Python
Từ bên trong thư mục **`00_DU_LIEU_CHUNG`**:
```python
import pandas as pd
train = pd.read_csv("data/processed/train_clean.csv")
test = pd.read_csv("data/processed/test_clean.csv")
print(train.shape, test.shape)
# (125973, 45) (22544, 45)
```
Không bắt buộc dùng Python: file CSV cũng đọc được bằng R/Excel (Excel có giới hạn hiển thị, các tệp này nằm trong số hàng giới hạn nhưng dễ nhầm kiểu dữ liệu).

## Task nào cần file nào?
- Pearson / ANOVA / Chi-square: dùng **Train**, chọn các biến theo giả thuyết và đúng loại dữ liệu.
- Wilson: đếm nhãn `attack_category` trong **Train**, tính CI từ số đếm và cỡ mẫu.
- Logistic: học từ **Train**, sử dụng **Test** ở giai đoạn đánh giá, không đưa cột nhãn/difficulty vào input.

**Không coi `train_clean.csv` là đã chuẩn hóa cho mô hình.** File có hai nhãn dẫn xuất nhưng 41 features vẫn ở thang gốc. Mọi thay đổi thêm phải được nêu rõ trong báo cáo của Task tương ứng.
