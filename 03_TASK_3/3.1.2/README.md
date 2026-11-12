# Code phan kiem dinh Pearson - phan cong noi bo 3.1.2

Thu muc nay chi chua code va outputs MOI cua phan kiem dinh, khong chep lai code hay ket qua cua 3.1.1.

## Noi dung
- `src/pearson_hypothesis_tests.py`: tinh r, thong ke t, p-value, Holm tren 9 cap, khoang tin cay Fisher, kiem tra do nhay cap root.
- `outputs/pearson_selected_tests.csv`: 9 cap kiem dinh.
- `outputs/pearson_root_sensitivity.csv`: kiem tra do nhay cap root.
- `outputs/pearson_checks.json`: thong tin va kiem tra so lieu.
- `outputs/*.tex`: bang ket qua tu dong xuat de doi chieu khi viet bao cao.

## Chay lai
Cai thu vien `pip install -r requirements.txt`

Dung archive.zip NSL-KDD (co san o du an chung, khong dong goi lap):
`python src/pearson_hypothesis_tests.py --raw-zip /duong_dan/archive.zip`

Hoac dung train_clean.csv tu phan tien xu ly chung:
`python src/pearson_hypothesis_tests.py --input /duong_dan/train_clean.csv`

Ket qua ghi de vao `outputs/`. Khong can code/outputs 3.1.1 de chay script nay.

## Pham vi
9 cap kiem dinh hai phia voi H0: rho = 0; alpha = 0.05.
Phan chinh thuc trong LaTeX gom muc 3.3 va 4.3, khong ghi nhan phan cong noi bo.
