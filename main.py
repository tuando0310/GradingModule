import sys
from pathlib import Path

# Thêm thư mục src vào path để python tìm thấy modules
sys.path.append(str(Path(__file__).parent / 'src'))

from services.io_service import IOService
from services.align_service import AlignmentService

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
CONFIG_DIR = BASE_DIR / 'config'
OUTPUT_DIR = BASE_DIR / 'output'

def main():
    print("--- BẮT ĐẦU QUÁ TRÌNH ĐỒNG BỘ DỮ LIỆU ---")
    
    # 1. Khởi tạo Services
    io_service = IOService()
    align_service = AlignmentService()

    # 2. Load Dữ liệu & Config
    try:
        # Load Config (IndexMap) - Có thể dùng để validate dữ liệu nếu cần sau này
        index_map = io_service.load_json(CONFIG_DIR / 'indexMap.json')
        print(f"[INFO] Đã load IndexMap: {len(index_map['index_map'])} khớp.")

        # Load Data
        raw_camera_data = io_service.load_json(DATA_DIR / 'frame0.json')
        raw_model_data = io_service.load_json(DATA_DIR / 'frame0Model.json')
        
        # Trích xuất mảng toạ độ (giả sử cấu trúc file json là list object frame)
        # Lấy frame đầu tiên
        camera_points = raw_camera_data[0]['data']
        model_points = raw_model_data[0]['data']

        print(f"[INFO] Dữ liệu Camera: {len(camera_points)} điểm.")
        print(f"[INFO] Dữ liệu Model: {len(model_points)} điểm.")

    except Exception as e:
        print(f"[ERROR] Lỗi khi đọc dữ liệu: {e}")
        return

    # 3. Thực hiện thuật toán (Core Logic)
    print("[PROCESS] Đang thực hiện thuật toán Umeyama...")
    refined_points, transform_params = align_service.umeyama_transform(
        source_points=camera_points,
        target_points=model_points
    )

    # 4. Đánh giá kết quả
    rmse = align_service.calculate_rmse(refined_points, model_points)
    print(f"[RESULT] Đồng bộ hoàn tất.")
    print(f"   - Scale Factor: {transform_params['scale']:.4f}")
    print(f"   - RMSE (Sai số): {rmse:.4f}")

    # 5. Đóng gói và Lưu kết quả
    output_data = [
        {
            "frame": raw_camera_data[0]['frame'],
            "data": refined_points
        }
    ]
    
    io_service.save_json(output_data, OUTPUT_DIR / 'frame0Refine.json')
    print("--- KẾT THÚC CHƯƠNG TRÌNH ---")

if __name__ == "__main__":
    main()