import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess(file_path=None):
    if file_path is None:
        candidate_paths = [
            'train_clean.csv',
            os.path.join('data', 'processed', 'train_clean.csv'),
        ]
        file_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not file_path:
            raise FileNotFoundError(
                "Không tìm thấy file 'train_clean.csv'. Vui lòng kiểm tra lại đường dẫn dữ liệu!"
            )

    print(f"Đang đọc dữ liệu từ: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False, skipinitialspace=True)
    df.columns = df.columns.astype(str).str.strip().str.lower()

    if 'attack_category' not in df.columns:
        raise ValueError(
            "Không tìm thấy cột 'attack_category'. Các cột hiện có: "
            f"{list(df.columns)}"
        )

    # Chuẩn hóa cách viết nhãn theo quy ước nhóm
    canonical = {
        'dos': 'DoS', 'probe': 'Probe', 'u2r': 'U2R',
        'r2l': 'R2L', 'normal': 'Normal'
    }
    cat = df['attack_category'].astype(str).str.strip()
    df['attack_category'] = cat.str.lower().map(canonical).fillna(cat)
    return df

def run_chi_square_analysis(df, col1, col2, output_img='figures/task34_heatmap_residuals_protocol_type.png', order_col2=None):
    # Chạy toàn bộ kiểm định, in kết quả và vẽ/lưu Heatmap
    print(f"\n" + "="*50)
    print(f"PHÂN TÍCH CHI-SQUARE: {col1.upper()} × {col2.upper()}")
    print("="*50)

    # Bảng chéo tần số quan sát
    crosstab = pd.crosstab(df[col1], df[col2])
    if order_col2:
        crosstab = crosstab.reindex(columns=[c for c in order_col2 if c in crosstab.columns])

    print("\n[BẢNG CHÉO TẦN SỐ QUAN SÁT]")
    print(crosstab)

    # Kiểm định Chi-square
    chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)

    total_cells = expected.size
    cells_under_5 = np.sum(expected < 5)
    pct_under_5 = (cells_under_5 / total_cells) * 100
    e_min = np.min(expected)

    # Tính Cramér's V
    n = crosstab.to_numpy().sum()
    min_dim = min(crosstab.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))

    # In kết quả
    print("\n[KẾT QUẢ THỐNG KÊ]")
    print(f"- Điều kiện Cochran: {pct_under_5:.1f}% số ô có E < 5 ({cells_under_5}/{total_cells}), E_min = {e_min:.2f}")
    print(f"- Giá trị Chi-square thu được: {chi2:.4f}")
    print(f"- Bậc tự do (df): {dof}")
    print(f"- P-value: {p_value:.4e} (tiệm cận 0)")
    print(f"- Hệ số Cramér's V: {cramer_v:.4f}")

    # Heatmap phần dư chuẩn hóa
    residuals = (crosstab - expected) / np.sqrt(expected)
    print("\n[PHẦN DƯ PEARSON CHUẨN HÓA Z_ij]")
    print(residuals.round(2))
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        residuals,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={'label': 'Phần dư Pearson $Z_{ij}$'}
    )
    plt.title(f"Heatmap Phần dư chuẩn hóa: {col1.upper()} vs {col2.upper()}", fontsize=13, pad=12)
    plt.xlabel(col2, fontsize=11)
    plt.ylabel(col1, fontsize=11)
    plt.tight_layout()
    
    # Lưu ảnh vào thư mục figures
    if output_img:
        os.makedirs(os.path.dirname(output_img), exist_ok=True)
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"\n[ĐÃ LƯU BIỂU ĐỒ TẠI]: {output_img}")
    plt.show()

# ==========================================
# KHỐI THỰC THI (MAIN)
# ==========================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None, help='đường dẫn tới train_clean.csv')
    ap.add_argument('--out', default='figures/task34_heatmap_residuals_protocol_type.png', help='nơi lưu heatmap')
    args = ap.parse_args()

    dataset = load_and_preprocess(args.data)
    canonical_order = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    run_chi_square_analysis(dataset, 'protocol_type', 'attack_category',
                            output_img=args.out, order_col2=canonical_order)