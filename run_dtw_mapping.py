import sys
from pathlib import Path

# Thêm đường dẫn src
sys.path.append(str(Path(__file__).parent / 'src'))

from services.io_service import IOService
from services.dtw_service import DtwService

# CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

# Tên file đầu vào (Bạn sửa lại tên file cho đúng thực tế nếu cần)
USER_FILE = DATA_DIR / 'aligned_grading_data.json' # File người dùng (313 frames)
MODEL_FILE = DATA_DIR / 'model_data.json'         

def main():
    print("--- CHẠY THUẬT TOÁN DTW MAPPING ---")
    
    io = IOService()
    dtw = DtwService()

    try:
        # 1. Load dữ liệu
        print(f"[1] Đang đọc file dữ liệu...")
        user_data_raw = io.load_json(USER_FILE)
        model_data_raw = io.load_json(MODEL_FILE)

        # Trích xuất list toạ độ (giả sử cấu trúc file là list các dict có key 'data')
        # user_seq sẽ là list các mảng [13x3]
        user_seq = [item['data'] for item in user_data_raw]
        model_seq = [item['data'] for item in model_data_raw]

        print(f"    - User Frames: {len(user_seq)}")
        print(f"    - Model Frames: {len(model_seq)}")

        
        mapping_result = dtw.run_dtw_alignment(user_seq, model_seq)

       

        # 4. Lưu kết quả
        output_path = OUTPUT_DIR / 'dtw_mapping_result.json'
        
        # Đóng gói kết quả đẹp hơn để dễ debug
        final_output = {
            "total_frames": len(mapping_result),
            "mapping": mapping_result  # Mảng [0, 0, 1, 2, ...]
        }
        
        io.save_json(final_output, output_path)
        print(f"--- Đã lưu kết quả mapping vào: {output_path} ---")

    except Exception as e:
        print(f"[ERROR] Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()