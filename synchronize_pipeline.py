import sys
import numpy as np
from pathlib import Path

# Setup đường dẫn
sys.path.append(str(Path(__file__).parent / 'src'))

from services.io_service import IOService
from services.align_service import AlignmentService

# CẤU HÌNH
BASE_DIR = Path(__file__).parent
INPUT_DATA_FILE = BASE_DIR / 'data' / 'grading_data.json'  # File gộp 313 frame
MODEL_FILE = BASE_DIR / 'data' / 'frame0Model.json'              # File mẫu
OUTPUT_FILE = BASE_DIR / 'output' / 'aligned_grading_data.json' # Kết quả cuối cùng

def main():
    print("--- BẮT ĐẦU ĐỒNG BỘ HÓA CHUỖI DỮ LIỆU (SEQUENCE ALIGNMENT) ---")
    
    io = IOService()
    aligner = AlignmentService()

    try:
        # 1. Load dữ liệu
        print("[1] Đang đọc dữ liệu...")
        camera_sequence = io.load_json(INPUT_DATA_FILE) # List 313 frames
        model_ref = io.load_json(MODEL_FILE)            # List 1 frame mẫu
        
        # 2. Lấy dữ liệu Frame 0 để làm mốc ("Calibration")
        # Giả sử cấu trúc dữ liệu là list các object có key 'data'
        cam_frame0_points = camera_sequence[0]['data'] 
        model_frame0_points = model_ref[0]['data']

        print(f"    - Camera Frame 0 Points: {len(cam_frame0_points)}")
        print(f"    - Model Frame 0 Points: {len(model_frame0_points)}")

        # 3. Tính toán Ma trận biến đổi (CHỈ DÙNG FRAME 0)
        print("[2] Đang tính toán tham số biến đổi dựa trên Frame 0...")
        scale, R, t = aligner.calculate_umeyama_params(cam_frame0_points, model_frame0_points)
        
        print(f"    > Scale Factor cố định: {scale:.4f}")
        print(f"    > Translation cố định: {t}")

        # 4. Áp dụng biến đổi cho TOÀN BỘ 313 Frame
        print(f"[3] Đang áp dụng biến đổi cho {len(camera_sequence)} frames...")
        
        aligned_sequence = []
        
        for frame_obj in camera_sequence:
            original_points = frame_obj['data']
            frame_id = frame_obj['frame']
            
            # Gọi hàm apply với tham số cố định đã tính ở bước 3
            transformed_points = aligner.apply_transform(original_points, scale, R, t)
            
            aligned_sequence.append({
                "frame": frame_id,
                "data": transformed_points
            })

        # 5. Lưu kết quả
        print("[4] Lưu file kết quả...")
        io.save_json(aligned_sequence, OUTPUT_FILE)
        print("--- HOÀN TẤT ---")

    except Exception as e:
        print(f"[ERROR] Có lỗi xảy ra: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()